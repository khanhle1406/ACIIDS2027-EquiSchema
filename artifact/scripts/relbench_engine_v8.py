"""Executed audit engine using the pinned RelBench GraphSAGE implementation."""

from __future__ import annotations

import copy
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from audit_v8 import esd
from numpy.typing import NDArray
from rewrite_v8 import frame_hash, inverse_r1, split_r1

SEEDS = (11, 23, 37, 51, 71)
EVAL_SEED = 20261028


def execute(raw: Path, vendor: Path, out: Path, deadline: float) -> None:
    """Fit all ten models under one deadline, saving evidence after each fit."""
    sys.path[:0] = [str(vendor), str(vendor / "examples")]
    from model import Model
    from relbench import load_dataset
    from relbench.base import Database, Table
    from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from text_embedder import GloveTextEmbedding
    from torch_frame.config.text_embedder import TextEmbedderConfig
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.seed import seed_everything

    def check_time() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("Global audit execution deadline reached")

    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    seed_everything(EVAL_SEED)
    dataset = load_dataset(raw)
    task = dataset.load_task("qualifying-position")
    tables = {
        s: task.get_table(s, mask_input_cols=False) for s in ("train", "val", "test")
    }
    masked_test = task.get_table("test", mask_input_cols=True)
    if task.target_col in masked_test.df:
        raise ValueError("Test target unexpectedly exposed")
    original = dataset.get_db(upto_test_timestamp=False)
    result_table = original.table_dict["results"]
    core, payload = split_r1(
        result_table.df,
        result_table.pkey_col,
        list(result_table.fkey_col_to_pkey_table),
        result_table.time_col,
    )
    restored = inverse_r1(
        core, payload, result_table.df, result_table.pkey_col, result_table.time_col
    )
    transformed = dict(original.table_dict)
    transformed["results"] = Table(
        core,
        dict(result_table.fkey_col_to_pkey_table),
        result_table.pkey_col,
        result_table.time_col,
    )
    transformed["results_payload"] = Table(
        payload, {"resultId": "results"}, "resultId", "date"
    )
    canonical_tables = dict(original.table_dict)
    canonical_tables["results"] = Table(
        restored,
        dict(result_table.fkey_col_to_pkey_table),
        result_table.pkey_col,
        result_table.time_col,
    )
    dbs = {
        "original": original,
        "R1": Database(transformed),
        "canonical": Database(canonical_tables),
    }
    types = get_stype_proposal(original)
    types_r1 = copy.deepcopy(types)
    types_r1["results"] = {k: v for k, v in types["results"].items() if k in core}
    types_r1["results_payload"] = {
        k: v for k, v in types["results"].items() if k in payload
    }
    text = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device="cpu"), batch_size=256
    )
    graphs, stats = {}, {}
    for view, database_view in dbs.items():
        check_time()
        graphs[view], stats[view] = make_pkey_fkey_graph(
            database_view,
            col_to_stype_dict=copy.deepcopy(types if view != "R1" else types_r1),
            text_embedder_cfg=text,
            remove_columns=task.hidden_columns(),
        )
        cols = [
            c
            for names in graphs[view][task.entity_table].tf.col_names_dict.values()
            for c in names
        ]
        if task.target_col in cols:
            raise ValueError("Hidden target present in model features")
    if graphs["original"].node_types != graphs["canonical"].node_types:
        raise ValueError("Canonical node-type mismatch")
    for edge, indices in graphs["original"].edge_index_dict.items():
        if not torch.equal(indices, graphs["canonical"][edge].edge_index):
            raise ValueError("Canonical topology mismatch")
    for name in graphs["original"].node_types:
        if (
            graphs["original"][name].tf.col_names_dict
            != graphs["canonical"][name].tf.col_names_dict
        ):
            raise ValueError("Canonical field mapping mismatch")
    test_frame = tables["test"].df
    entities = test_frame[task.entity_col].to_numpy()
    clusters = (
        original.table_dict[task.entity_table]
        .df.set_index(task.entity_col)
        .loc[entities, "raceId"]
        .to_numpy()
    )
    sigma = float(tables["train"].df[task.target_col].std(ddof=1))
    clamps = np.percentile(tables["train"].df[task.target_col], [2, 98])
    predictions: dict[tuple[str, int], NDArray] = {}
    contexts: dict[str, list[set[tuple]]] = {}
    metrics, canonical = [], []
    task_hash = frame_hash(test_frame, [task.entity_col, task.time_col])
    (out / "certificate.json").write_text(
        json.dumps(
            {
                "roundtrip_hash": frame_hash(restored, ["resultId"]),
                "original_hash": frame_hash(result_table.df, ["resultId"]),
                "evaluation_hash": task_hash,
                "hidden_columns": task.hidden_columns(),
                "status": "PASS_INSTANCE_AND_TASK_IDENTITY",
            },
            indent=2,
        )
    )

    def loader(graph: Any, table: Any, shuffle: bool = False) -> Any:
        inputs = get_node_train_table_input(table, task)
        return NeighborLoader(
            graph,
            num_neighbors=[32, 16],
            time_attr="time",
            input_nodes=inputs.nodes,
            input_time=inputs.time,
            transform=inputs.transform,
            batch_size=512,
            temporal_strategy="uniform",
            shuffle=shuffle,
            num_workers=0,
        )

    def infer(
        model: Any, graph: Any, table: Any, view: str, collect: bool = False
    ) -> NDArray:
        check_time()
        seed_everything(EVAL_SEED)
        model.eval()
        values = []
        facts = [set() for _ in range(len(table))] if collect else []
        with torch.no_grad():
            for batch in loader(graph, table):
                check_time()
                anchors = batch[task.entity_table].seed_time
                for name in batch.node_types:
                    store = batch[name]
                    if (
                        "time" in store
                        and "batch" in store
                        and bool((store.time > anchors[store.batch]).any())
                    ):
                        raise ValueError(f"Future context sampled in {name}")
                if collect:
                    input_ids = batch[task.entity_table].input_id.tolist()
                    for name in batch.node_types:
                        store = batch[name]
                        if "batch" not in store:
                            raise ValueError(
                                "Temporal disjoint batch assignment unavailable"
                            )
                        base_name = "results" if name == "results_payload" else name
                        source = dbs[view].table_dict[name]
                        fields = [
                            c
                            for c in source.df
                            if c != source.pkey_col
                            and (base_name, c) not in task.hidden_columns()
                        ]
                        if name == "results_payload":
                            fields = [c for c in fields if c != "resultId"]
                        for row, assignment in zip(
                            store.n_id.tolist(), store.batch.tolist(), strict=True
                        ):
                            target = input_ids[assignment]
                            facts[target].add((base_name, row, "__exists__"))
                            facts[target].update((base_name, row, c) for c in fields)
                values.append(
                    torch.clamp(
                        model(
                            batch.to(next(model.parameters()).device), task.entity_table
                        ).view(-1),
                        float(clamps[0]),
                        float(clamps[1]),
                    ).cpu()
                )
        if collect:
            contexts[view] = facts
        return torch.cat(values).numpy()

    for view in ("original", "R1"):
        for seed in SEEDS:
            check_time()
            seed_everything(seed)
            model = Model(
                data=graphs[view],
                col_stats_dict=stats[view],
                num_layers=2,
                channels=32,
                out_channels=1,
                aggr="sum",
                norm="batch_norm",
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            best, best_metric = None, float("inf")
            began = time.monotonic()
            for epoch in range(10):
                seed_everything(seed + epoch * 1000)
                model.train()
                for batch in loader(graphs[view], tables["train"], True):
                    check_time()
                    optimizer.zero_grad()
                    batch = batch.to(device)
                    loss = torch.nn.functional.l1_loss(
                        model(batch, task.entity_table).view(-1),
                        batch[task.entity_table].y.float(),
                    )
                    loss.backward()
                    optimizer.step()
                val = infer(model, graphs[view], tables["val"], view)
                score = float(task.evaluate(val, tables["val"])["nmae"])
                if score < best_metric:
                    best_metric, best = score, copy.deepcopy(model.state_dict())
                print(view, seed, epoch + 1, score, flush=True)
                if view == "original" and seed == SEEDS[0] and epoch == 0:
                    projected = (time.monotonic() - began) * 100 + 120
                    if projected > deadline - time.monotonic():
                        raise TimeoutError(
                            "Pilot projection exceeds remaining fixed budget"
                        )
            if best is None:
                raise ValueError("No finite validation checkpoint")
            model.load_state_dict(best)
            torch.save(
                {"state_dict": best, "seed": seed, "view": view},
                out / f"{view}_{seed}.pt",
            )
            pred = infer(model, graphs[view], masked_test, view, seed == SEEDS[0])
            predictions[view, seed] = pred
            output = test_frame.copy()
            output["raceId"], output["view"], output["seed"], output["prediction"] = (
                clusters,
                view,
                seed,
                pred,
            )
            output.to_csv(out / f"predictions_{view}_{seed}.csv", index=False)
            metrics.append(
                {
                    "view": view,
                    "seed": seed,
                    "nmae": float(task.evaluate(pred, tables["test"])["nmae"]),
                    "validation_nmae": best_metric,
                    "seconds": time.monotonic() - began,
                }
            )
            if view == "original":
                # Reconstructed canonical view has identical typed rows, topology and features.
                # CPU inference is actually executed; numerical differences are recorded.
                model.cpu()
                canon = infer(model, graphs["canonical"], masked_test, "canonical")
                np.save(out / f"canonical_cpu_{seed}.npy", canon)
                canonical.append(
                    {
                        "seed": seed,
                        "mean_absolute_difference": float(np.abs(canon - pred).mean()),
                        "max_absolute_difference": float(np.abs(canon - pred).max()),
                    }
                )
            del model, optimizer, best
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    a = np.stack([predictions["original", s] for s in SEEDS])
    b = np.stack([predictions["R1", s] for s in SEEDS])
    drift = esd(a, b, clusters=clusters, classification=False, scale=sigma)
    coverage = [
        len(a & b) / len(a | b) if a | b else 1.0
        for a, b in zip(contexts["original"], contexts["R1"], strict=True)
    ]
    pd.DataFrame(metrics).to_csv(out / "performance.csv", index=False)
    pd.DataFrame({"entity_id": entities, "jaccard": coverage}).to_csv(
        out / "coverage.csv", index=False
    )
    report = {
        "status": "COMPLETE",
        "drift": drift,
        "canonical_cpu": canonical,
        "mean_jaccard": float(np.mean(coverage)),
        "train_std": sigma,
        "seconds": time.monotonic() - start,
        "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20
        if torch.cuda.is_available()
        else 0,
        "context_scope": "per-target sampled base-table fields plus row existence; auxiliary R1 link key excluded; copied timestamps deduplicated",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
