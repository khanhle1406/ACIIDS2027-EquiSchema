"""Corrected controlled study of storage invariance and role identifiability (v8).

Key corrections over legacy v7 scripts:
1. LightGBM imported before PyTorch with strict single-thread environment to
   prevent macOS OpenMP runtime conflicts.
2. Train-only normalization computed strictly on the training partition of the
   reference view after fixed stratified splitting.
3. Role swap executed at the raw transaction table level (buyer_id <-> seller_id)
   followed by incident reaggregation and frozen preprocessing.
4. Oracle contrast evaluated using the generator's full-population normalizer
   rather than the training scaler.
5. Role-keyed loader vs anonymous physical slot ordering actually evaluated
   through serialized channel representations rather than hardcoded zero.
6. Explicit logical fact identification and Jaccard computation, decoupled
   from single-view full-context fractions.
7. Bootstrap confidence intervals computed via audit_v8.esd.
8. Durable per-entity predictions, certificate reports, and summaries saved
   to the controlled v8 output directory without mutating results/current.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Enforce single-threaded execution for reproducibility and macOS libomp safety
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# LightGBM MUST be imported BEFORE torch to avoid macOS OpenMP runtime segfault
import lightgbm as _lgb  # noqa: F401
import torch

torch.set_num_threads(1)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

SCRIPTS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPTS_DIR.parents[1]
DEFAULT_OUTDIR = WORKSPACE_ROOT / "results" / "v8" / "controlled"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_v8 import esd, jaccard

# Import legacy generator and model helpers without executing legacy main
_legacy_path = SCRIPTS_DIR / "run_physical_sampler_study.py"
_spec = importlib.util.spec_from_file_location("base_sampler", _legacy_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load module spec from {_legacy_path}")
base = importlib.util.module_from_spec(_spec)
sys.modules["base_sampler"] = base
_spec.loader.exec_module(base)

# Re-enforce single-threaded torch after legacy module load
torch.set_num_threads(1)

SEEDS: tuple[int, ...] = (11, 23, 37, 51, 71)


@dataclass
class SerializedRoleChannels:
    """Serialized relational channels with physical slot ordering and semantic role metadata."""

    slots: list[tuple[str, NDArray]]
    entity_feat: NDArray


def extract_logical_facts(
    view: str,
    hop_budget: int,
    cust_id: int,
    ids_b: Sequence[int],
    ids_s: Sequence[int],
    merchant_ids: NDArray,
) -> set[tuple[str, ...]]:
    """Return explicit set of logical fact tuples visible for a customer under view and hop budget.

    Logical-fact ID convention:
    1. Base entity: ('customer', cust_id)
    2. Structural role: ('tx_struct', tid, 'buyer' | 'seller')
    3. Event payload amount: ('tx_amount', tid)
    4. Joined risk contribution: ('tx_merchant_risk_contrib', tid, merchant_id)
       Labelled at the joined contribution level so repeated merchant dimension
       values across multiple transactions are not conflated with distinct base facts.
    """
    d = base.DIST[view]
    facts: set[tuple[str, ...]] = {("customer", str(cust_id))}
    if d["struct"] <= hop_budget:
        for tid in ids_b:
            facts.add(("tx_struct", str(tid), "buyer"))
        for tid in ids_s:
            facts.add(("tx_struct", str(tid), "seller"))
    if d["amount"] <= hop_budget:
        for tid in ids_b:
            facts.add(("tx_amount", str(tid)))
        for tid in ids_s:
            facts.add(("tx_amount", str(tid)))
    if d["risk"] <= hop_budget:
        for tid in ids_b:
            facts.add(
                ("tx_merchant_risk_contrib", str(tid), str(int(merchant_ids[tid])))
            )
        for tid in ids_s:
            facts.add(
                ("tx_merchant_risk_contrib", str(tid), str(int(merchant_ids[tid])))
            )
    return facts


def sample_view_with_facts(
    view: str,
    hop_budget: int,
    cust: pd.DataFrame,
    merchants: pd.DataFrame,
    tx: pd.DataFrame,
    buy: Sequence[Sequence[int]],
    sell: Sequence[Sequence[int]],
    frac: float = 1.0,
    sample_seed: int = 707,
) -> tuple[NDArray, list[set[tuple[str, ...]]], float]:
    """Sample a physical view, computing raw features, logical fact sets, and context fraction."""
    n = len(cust)
    x_mat = np.zeros((n, 7), dtype=float)
    x_mat[:, 0] = cust.customer_x.to_numpy()
    mr = merchants.merchant_risk.to_numpy()
    mer = tx.merchant_id.to_numpy()
    amt = tx.amount.to_numpy()
    d = base.DIST[view]
    facts_per_entity: list[set[tuple[str, ...]]] = []
    context_fractions: list[float] = []

    for cid in range(n):
        ids_b = base.choose(buy[cid], frac, sample_seed, cid, 1)
        ids_s = base.choose(sell[cid], frac, sample_seed, cid, 2)
        total_tx = len(buy[cid]) + len(sell[cid])
        visible_components = 0
        if d["struct"] <= hop_budget:
            x_mat[cid, 3] = len(ids_b)
            x_mat[cid, 4] = len(ids_s)
            visible_components += len(ids_b) + len(ids_s)
        if d["amount"] <= hop_budget:
            x_mat[cid, 1] = amt[ids_b].sum() if len(ids_b) else 0.0
            x_mat[cid, 2] = amt[ids_s].sum() if len(ids_s) else 0.0
            visible_components += len(ids_b) + len(ids_s)
        if d["risk"] <= hop_budget:
            x_mat[cid, 5] = mr[mer[ids_b]].sum() if len(ids_b) else 0.0
            x_mat[cid, 6] = mr[mer[ids_s]].sum() if len(ids_s) else 0.0
            visible_components += len(ids_b) + len(ids_s)

        cf = 1.0 if total_tx == 0 else visible_components / (3.0 * total_tx)
        context_fractions.append(cf)
        facts_per_entity.append(
            extract_logical_facts(view, hop_budget, cid, ids_b, ids_s, mer)
        )

    return x_mat, facts_per_entity, float(np.mean(context_fractions))


def build_raw_collapsed(raw: NDArray) -> NDArray:
    """Build collapsed representation from raw features before normalization.

    Sums buyer and seller channels: amount (1+2), counts (3+4), risks (5+6).
    """
    tot_amt = raw[:, 1] + raw[:, 2]
    tot_cnt = raw[:, 3] + raw[:, 4]
    tot_risk = raw[:, 5] + raw[:, 6]
    return np.column_stack([raw[:, 0], tot_amt, tot_cnt, tot_risk])


def fit_collapsed_normalizer(
    raw_coll: NDArray, train_idx: NDArray
) -> tuple[NDArray, NDArray]:
    """Train-only normalizer for collapsed representations."""
    x_tr = raw_coll[train_idx].copy()
    x_tr[:, 1] = np.log1p(x_tr[:, 1])
    mu = x_tr[:, 1:].mean(axis=0)
    sd = x_tr[:, 1:].std(axis=0) + 1e-9
    return mu, sd


def apply_collapsed_normalizer(raw_coll: NDArray, mu: NDArray, sd: NDArray) -> NDArray:
    """Apply collapsed normalizer parameters (mu, sigma) to collapsed feature array."""
    x = raw_coll.copy()
    x[:, 1] = np.log1p(x[:, 1])
    x[:, 1:] = (x[:, 1:] - mu) / sd
    return x.astype(np.float32)


def build_symmetric_mpnn_channels(x_coll_norm: NDArray) -> NDArray:
    """Construct symmetric 7-feature matrix for TypedMPNN(aware=False).

    Both buyer and seller channels receive identical halves of the collapsed
    aggregates so b + s equals the collapsed total and channels are symmetric.
    """
    cx = x_coll_norm[:, [0]]
    half_amt = 0.5 * x_coll_norm[:, [1]]
    half_cnt = 0.5 * x_coll_norm[:, [2]]
    half_risk = 0.5 * x_coll_norm[:, [3]]
    return np.column_stack(
        [
            cx,
            half_amt,
            half_amt,
            half_cnt,
            half_cnt,
            half_risk,
            half_risk,
        ]
    ).astype(np.float32)


def fit_train_normalizer(
    raw_ref: NDArray, train_idx: NDArray
) -> tuple[NDArray, NDArray]:
    """Compute normalizer mean and standard deviation strictly on the training partition."""
    x_tr = raw_ref[train_idx].copy()
    x_tr[:, 1] = np.log1p(x_tr[:, 1])
    x_tr[:, 2] = np.log1p(x_tr[:, 2])
    mu = x_tr[:, 1:].mean(axis=0)
    sd = x_tr[:, 1:].std(axis=0) + 1e-9
    return mu, sd


def apply_normalizer(raw: NDArray, mu: NDArray, sd: NDArray) -> NDArray:
    """Apply frozen preprocessing parameters (mu, sigma) to feature array."""
    x = raw.copy()
    x[:, 1] = np.log1p(x[:, 1])
    x[:, 2] = np.log1p(x[:, 2])
    x[:, 1:] = (x[:, 1:] - mu) / sd
    return x.astype(np.float32)


def fit_population_normalizer(raw_ref: NDArray) -> tuple[NDArray, NDArray]:
    """Full-population normalizer matching the data generator's latent standardizer."""
    x_pop = raw_ref.copy()
    x_pop[:, 1] = np.log1p(x_pop[:, 1])
    x_pop[:, 2] = np.log1p(x_pop[:, 2])
    pop_mu = x_pop[:, 1:].mean(axis=0)
    pop_sd = x_pop[:, 1:].std(axis=0) + 1e-9
    return pop_mu, pop_sd


def compute_oracle_score(
    raw: NDArray, pop_mu: NDArray, pop_sd: NDArray, signal_scale: float
) -> NDArray:
    """Evaluate ground truth oracle function using full-population normalizer."""
    f_mat = raw.copy()
    f_mat[:, 1] = np.log1p(f_mat[:, 1])
    f_mat[:, 2] = np.log1p(f_mat[:, 2])
    f_mat[:, 1:] = (f_mat[:, 1:] - pop_mu) / pop_sd
    score = signal_scale * (
        0.9 * f_mat[:, 0]
        + 1.25 * f_mat[:, 1]
        - 1.25 * f_mat[:, 2]
        + 0.45 * f_mat[:, 3]
        - 0.45 * f_mat[:, 4]
        + 0.75 * f_mat[:, 5]
        - 0.75 * f_mat[:, 6]
    )
    return score


def swap_transaction_roles(tx: pd.DataFrame) -> pd.DataFrame:
    """Perform role swap at the raw transaction table level (buyer_id <-> seller_id)."""
    tx_swapped = tx.copy()
    tx_swapped["buyer_id"] = tx["seller_id"]
    tx_swapped["seller_id"] = tx["buyer_id"]
    return tx_swapped


def serialize_channels_permuted(features: NDArray) -> SerializedRoleChannels:
    """Serialize channels with permuted physical slot ordering while preserving role keys.

    Physical slots are serialized in order [seller, buyer]. Role metadata
    explicitly annotates the semantic identity of each slot.
    """
    buyer_channel = features[:, [1, 3, 5]]
    seller_channel = features[:, [2, 4, 6]]
    slots = [("seller", seller_channel), ("buyer", buyer_channel)]
    return SerializedRoleChannels(slots=slots, entity_feat=features[:, [0]])


def role_keyed_loader(payload: SerializedRoleChannels) -> NDArray:
    """Canonicalize channels by role metadata tags, invariant to physical slot ordering."""
    role_map = dict(payload.slots)
    b = role_map["buyer"]
    s = role_map["seller"]
    return np.column_stack(
        [
            payload.entity_feat,
            b[:, [0]],
            s[:, [0]],
            b[:, [1]],
            s[:, [1]],
            b[:, [2]],
            s[:, [2]],
        ]
    ).astype(np.float32)


def anonymous_slot_loader(payload: SerializedRoleChannels) -> NDArray:
    """Load channels strictly by physical slot position, ignoring role metadata tags."""
    slot0 = payload.slots[0][1]
    slot1 = payload.slots[1][1]
    return np.column_stack(
        [
            payload.entity_feat,
            slot0[:, [0]],
            slot1[:, [0]],
            slot0[:, [1]],
            slot1[:, [1]],
            slot0[:, [2]],
            slot1[:, [2]],
        ]
    ).astype(np.float32)


def predict_model(
    model: Any, model_name: str, features: NDArray, eval_idx: NDArray
) -> NDArray:
    """Evaluate fitted model on evaluation indices."""
    if model_name == "LightGBM":
        preds = model.predict_proba(features[eval_idx])[:, 1]
        return np.asarray(preds, dtype=float)
    x_t, b_t, s_t = map(torch.tensor, base.parts(features))
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(x_t, b_t, s_t)).numpy()[eval_idx]
    return np.asarray(preds, dtype=float)


def compute_pairwise_jaccard(
    facts_a: list[set[tuple[str, ...]]],
    facts_b: list[set[tuple[str, ...]]],
    eval_idx: NDArray,
) -> float:
    """Compute mean pairwise Jaccard similarity between two fact representations."""
    scores = [jaccard(facts_a[idx], facts_b[idx]) for idx in eval_idx]
    return float(np.mean(scores))


def bootstrap_paired_drift(
    orig_preds: NDArray,
    view_preds: NDArray,
    nboot: int = 2000,
    seed: int = 20261028,
) -> tuple[float, float, float]:
    """Bootstrap matched-seed view disagreement over seed units and test entities."""
    a = np.asarray(orig_preds) >= 0.5
    b = np.asarray(view_preds) >= 0.5
    diff = (a != b).astype(float)
    n_s, n_e = diff.shape
    mean_val = float(diff.mean())
    rng = np.random.default_rng(seed)
    boot_means = np.empty(nboot)
    for i in range(nboot):
        s_idx = rng.integers(0, n_s, n_s)
        e_idx = rng.integers(0, n_e, n_e)
        boot_means[i] = diff[s_idx][:, e_idx].mean()
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return mean_val, float(lo), float(hi)


def run_domain_study(
    cfg: base.DomainCfg,
    data_seed: int,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Execute corrected controlled study for a single synthetic domain."""
    cust, merchants, tx, labels = base.generate(cfg, data_seed)
    cert = base.certify(merchants, tx)
    buy, sell = base.incident_lists(tx, len(cust))

    # All 9 views under examination
    specs: list[tuple[str, str, int]] = [
        ("ORIG_H1", "ORIG", 1),
        ("ORIG_H2", "ORIG", 2),
        ("R1_H1", "R1", 1),
        ("R1_H2", "R1", 2),
        ("R2_H1", "R2", 1),
        ("R2_H2", "R2", 2),
        ("R3_H1", "R3", 1),
        ("R3_H2", "R3", 2),
        ("R3_H3", "R3", 3),
    ]

    raw_features: dict[str, NDArray] = {}
    facts_per_view: dict[str, list[set[tuple[str, ...]]]] = {}
    context_fraction: dict[str, float] = {}

    for nm, v_type, h_val in specs:
        raw_feat, fact_sets, cf = sample_view_with_facts(
            v_type, h_val, cust, merchants, tx, buy, sell
        )
        raw_features[nm] = raw_feat
        facts_per_view[nm] = fact_sets
        context_fraction[nm] = cf

    # Fixed train / validation / test splits
    y = labels.target.to_numpy()
    tr0, te = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=2026, stratify=y
    )
    tr, va = train_test_split(tr0, test_size=0.20, random_state=2027, stratify=y[tr0])

    # 1. Train-only normalizer on ORIG_H2
    mu_train, sd_train = fit_train_normalizer(raw_features["ORIG_H2"], tr)
    norm_features = {
        k: apply_normalizer(v, mu_train, sd_train) for k, v in raw_features.items()
    }

    # 2. Population normalizer for oracle contrast
    pop_mu, pop_sd = fit_population_normalizer(raw_features["ORIG_H2"])
    score_orig = labels.oracle_score.to_numpy()

    # Fit models and evaluate performance
    perf_rows: list[dict[str, Any]] = []
    preds: dict[tuple[str, str, int], NDArray] = {}
    models: dict[tuple[str, str, int], Any] = {}

    for mdl in ["LightGBM", "TypedMPNN"]:
        for view_name, x_norm in norm_features.items():
            for s0 in seeds:
                if mdl == "LightGBM":
                    model, p_te = base.train_gbdt(x_norm, y, tr, te, s0)
                else:
                    model, p_te, _ = base.train_nn(
                        x_norm, y, tr, va, te, s0, aware=True
                    )
                preds[(mdl, view_name, s0)] = p_te
                models[(mdl, view_name, s0)] = model

                perf_rows.append(
                    {
                        "domain": cfg.name,
                        "model": mdl,
                        "view": view_name,
                        "seed": s0,
                        "full_context_fraction": context_fraction[view_name],
                        "auc": float(roc_auc_score(y[te], p_te)),
                        "accuracy": float(accuracy_score(y[te], p_te >= 0.5)),
                        "brier": float(brier_score_loss(y[te], p_te)),
                    }
                )

    # 3. Drift and ESD with audit_v8.esd
    drift_transitions: list[tuple[str, str, str]] = [
        ("mid-to-low (R1)", "ORIG_H1", "R1_H1"),
        ("mid-to-full (R2)", "ORIG_H1", "R2_H1"),
        ("full-to-mid (R3)", "ORIG_H2", "R3_H2"),
        ("full-to-full (R1)", "ORIG_H2", "R1_H2"),
        ("full-to-full (R2)", "ORIG_H2", "R2_H2"),
        ("mid-to-low (R3)", "ORIG_H1", "R3_H1"),
        ("full-to-full (R3)", "ORIG_H2", "R3_H3"),
    ]

    drift_rows: list[dict[str, Any]] = []
    for mdl in ["LightGBM", "TypedMPNN"]:
        for label_trans, ref_view, tgt_view in drift_transitions:
            orig_p_mat = np.stack([preds[(mdl, ref_view, s0)] for s0 in seeds], axis=0)
            view_p_mat = np.stack([preds[(mdl, tgt_view, s0)] for s0 in seeds], axis=0)
            stat = esd(orig_p_mat, view_p_mat, classification=True, nboot=2000)
            jacc = compute_pairwise_jaccard(
                facts_per_view[ref_view], facts_per_view[tgt_view], te
            )
            drift_rows.append(
                {
                    "domain": cfg.name,
                    "model": mdl,
                    "transition": label_trans,
                    "reference": ref_view,
                    "view": tgt_view,
                    "jaccard": jacc,
                    "full_context_fraction_ref": context_fraction[ref_view],
                    "full_context_fraction_view": context_fraction[tgt_view],
                    "coverage_ref": context_fraction[ref_view],
                    "coverage_view": context_fraction[tgt_view],
                    "delta_seed_ind": stat["delta_seed_ind"],
                    "delta_view_ind": stat["delta_view_ind"],
                    "esd_ind": stat["esd_ind"],
                    "esd_ci_lo": stat["esd_ci_lo"],
                    "esd_ci_hi": stat["esd_ci_hi"],
                    "delta_view_paired": stat["delta_view_paired"],
                }
            )

    # 4. Role swap at transaction level & serialization loader comparison
    tx_swap = swap_transaction_roles(tx)
    buy_swap, sell_swap = base.incident_lists(tx_swap, len(cust))
    raw_swap, _, _ = sample_view_with_facts(
        "ORIG", 2, cust, merchants, tx_swap, buy_swap, sell_swap
    )
    x_swap_norm = apply_normalizer(raw_swap, mu_train, sd_train)

    # Oracle contrast evaluated with population normalizer
    oracle_score_swap = compute_oracle_score(raw_swap, pop_mu, pop_sd, cfg.signal_scale)
    od = oracle_score_swap[te] - score_orig[te]
    info = np.abs(od) > 1e-10

    # Build collapsed representation from RAW features before normalization
    raw_coll = build_raw_collapsed(raw_features["ORIG_H2"])
    raw_swap_coll = build_raw_collapsed(raw_swap)
    np.testing.assert_array_equal(raw_coll, raw_swap_coll)

    # Train-only collapsed normalizer
    mu_coll, sd_coll = fit_collapsed_normalizer(raw_coll, tr)
    x_coll_norm = apply_collapsed_normalizer(raw_coll, mu_coll, sd_coll)
    x_swap_coll_norm = apply_collapsed_normalizer(raw_swap_coll, mu_coll, sd_coll)
    np.testing.assert_allclose(x_coll_norm, x_swap_coll_norm)

    # Symmetric channels for TypedMPNN(aware=False)
    x_mpnn_coll = build_symmetric_mpnn_channels(x_coll_norm)
    x_swap_mpnn_coll = build_symmetric_mpnn_channels(x_swap_coll_norm)
    np.testing.assert_allclose(x_mpnn_coll, x_swap_mpnn_coll)

    role_rows: list[dict[str, Any]] = []
    role_preds_log: dict[str, NDArray] = {}

    x_complete = norm_features["ORIG_H2"]
    payload_permuted = serialize_channels_permuted(x_complete)
    x_role_loaded = role_keyed_loader(payload_permuted)
    x_anon_loaded = anonymous_slot_loader(payload_permuted)

    for mdl in ["LightGBM", "TypedMPNN"]:
        # Role-keyed and anonymous-slot sharing the trained role-aware models
        for s0 in seeds:
            m = models[(mdl, "ORIG_H2", s0)]
            p0 = preds[(mdl, "ORIG_H2", s0)]
            ps = predict_model(m, mdl, x_swap_norm, te)
            p_role_perm = predict_model(m, mdl, x_role_loaded, te)
            p_anon_perm = predict_model(m, mdl, x_anon_loaded, te)

            d = ps - p0
            concordance = float(np.mean(np.sign(d[info]) == np.sign(od[info])))
            response = float(np.mean(np.abs(d)))

            # Disagreement computed from actual loader model evaluation
            ser_disagree_role = float(np.mean((p0 >= 0.5) != (p_role_perm >= 0.5)))
            ser_disagree_anon = float(np.mean((p0 >= 0.5) != (p_anon_perm >= 0.5)))

            role_rows.append(
                {
                    "domain": cfg.name,
                    "model": mdl,
                    "encoding": "role-keyed",
                    "seed": s0,
                    "swap_concordance": concordance,
                    "swap_response": response,
                    "serialization_disagreement": ser_disagree_role,
                }
            )
            role_rows.append(
                {
                    "domain": cfg.name,
                    "model": mdl,
                    "encoding": "anonymous-slot",
                    "seed": s0,
                    "swap_concordance": concordance,
                    "swap_response": response,
                    "serialization_disagreement": ser_disagree_anon,
                }
            )

            role_preds_log[f"{mdl}_s{s0}_p0"] = p0
            role_preds_log[f"{mdl}_s{s0}_ps"] = ps
            role_preds_log[f"{mdl}_s{s0}_p_role_perm"] = p_role_perm
            role_preds_log[f"{mdl}_s{s0}_p_anon_perm"] = p_anon_perm

            # Save role predictions for durable tabular provenance
            preds[(mdl, "role_orig", s0)] = p0
            preds[(mdl, "role_swap", s0)] = ps
            preds[(mdl, "role_perm", s0)] = p_role_perm
            preds[(mdl, "anon_perm", s0)] = p_anon_perm

        # Collapsed baseline (negative control)
        for s0 in seeds:
            if mdl == "LightGBM":
                m_coll, p0_coll = base.train_gbdt(x_coll_norm, y, tr, te, s0)
                ps_coll = m_coll.predict_proba(x_swap_coll_norm[te])[:, 1]
            else:
                m_coll, p0_coll, _ = base.train_nn(
                    x_mpnn_coll, y, tr, va, te, s0, aware=False
                )
                x_sw, b_sw, s_sw = map(torch.tensor, base.parts(x_swap_mpnn_coll))
                m_coll.eval()
                with torch.no_grad():
                    ps_coll = torch.sigmoid(m_coll(x_sw, b_sw, s_sw)).numpy()[te]

            d_coll = ps_coll - p0_coll
            assert np.allclose(d_coll, 0.0, atol=1e-6)
            role_rows.append(
                {
                    "domain": cfg.name,
                    "model": mdl,
                    "encoding": "collapsed",
                    "seed": s0,
                    "swap_concordance": 0.0,
                    "swap_response": float(np.mean(np.abs(d_coll))),
                    "serialization_disagreement": float(
                        np.mean((p0_coll >= 0.5) != (ps_coll >= 0.5))
                    ),
                }
            )

            preds[(mdl, "collapsed_orig", s0)] = p0_coll
            preds[(mdl, "collapsed_swap", s0)] = ps_coll

    # 5. Coverage ablation: varying fanout on R1 with H=2
    cov_rows: list[dict[str, Any]] = []
    fanout_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    for frac in fanout_fractions:
        raw_cov, facts_cov, cf = sample_view_with_facts(
            "R1", 2, cust, merchants, tx, buy, sell, frac=frac, sample_seed=707
        )
        x_cov_norm = apply_normalizer(raw_cov, mu_train, sd_train)
        jacc = compute_pairwise_jaccard(facts_per_view["ORIG_H2"], facts_cov, te)
        v_ablation_name = f"R1_H2_f{frac:.1f}"
        for mdl in ["LightGBM", "TypedMPNN"]:
            cov_p_list: list[NDArray] = []
            for s0 in seeds:
                if mdl == "LightGBM":
                    _, p_cov = base.train_gbdt(x_cov_norm, y, tr, te, s0)
                else:
                    _, p_cov, _ = base.train_nn(
                        x_cov_norm, y, tr, va, te, s0, aware=True
                    )
                cov_p_list.append(p_cov)
                preds[(mdl, v_ablation_name, s0)] = p_cov
            cov_p_mat = np.stack(cov_p_list, axis=0)
            orig_h2_mat = np.stack(
                [preds[(mdl, "ORIG_H2", s0)] for s0 in seeds], axis=0
            )
            cov_stat = esd(orig_h2_mat, cov_p_mat, classification=True, nboot=2000)
            p_mean, p_lo, p_hi = bootstrap_paired_drift(
                orig_h2_mat,
                cov_p_mat,
                nboot=2000,
                seed=20261028 + int(frac * 100) + sum(map(ord, mdl + cfg.name)),
            )
            cov_rows.append(
                {
                    "domain": cfg.name,
                    "model": mdl,
                    "fanout_fraction": frac,
                    "full_context_fraction": cf,
                    "coverage": cf,
                    "jaccard": jacc,
                    "paired_view_drift": p_mean,
                    "paired_ci_lo": p_lo,
                    "paired_ci_hi": p_hi,
                    "ci_lo": p_lo,
                    "ci_hi": p_hi,
                    "delta_seed_ind": cov_stat["delta_seed_ind"],
                    "delta_view_ind": cov_stat["delta_view_ind"],
                    "esd_ind": cov_stat["esd_ind"],
                    "esd_ci_lo": cov_stat["esd_ci_lo"],
                    "esd_ci_hi": cov_stat["esd_ci_hi"],
                }
            )

    # Collect raw prediction dict for durable export
    raw_preds_dict: dict[str, Any] = {
        f"test_indices_{cfg.name}": te,
        f"targets_{cfg.name}": y[te],
        f"oracle_diff_{cfg.name}": od,
        f"oracle_score_{cfg.name}": score_orig[te],
        f"oracle_score_swap_{cfg.name}": oracle_score_swap[te],
    }
    for (mdl, v_name, s0), p_arr in preds.items():
        raw_preds_dict[f"pred_{cfg.name}_{mdl}_{v_name}_s{s0}"] = p_arr

    for k_r, arr_r in role_preds_log.items():
        raw_preds_dict[f"role_{cfg.name}_{k_r}"] = arr_r

    scalers_dict: dict[str, Any] = {
        "domain": cfg.name,
        "train_mu": mu_train.tolist(),
        "train_sd": sd_train.tolist(),
        "collapsed_mu": mu_coll.tolist(),
        "collapsed_sd": sd_coll.tolist(),
        "pop_mu": pop_mu.tolist(),
        "pop_sd": pop_sd.tolist(),
    }

    return (
        pd.DataFrame(perf_rows),
        pd.DataFrame(drift_rows),
        pd.DataFrame(role_rows),
        pd.DataFrame(cov_rows),
        cert,
        raw_preds_dict,
        scalers_dict,
    )


def run_all_controls(
    outdir: Path,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Execute complete controlled study across domains and export all artifacts."""
    outdir.mkdir(parents=True, exist_ok=True)

    domain_configs = [
        (
            base.DomainCfg(
                "Synth-A-uniform",
                heavy_tail=False,
                amount_sigma=0.95,
                signal_scale=1.0,
            ),
            3107,
        ),
        (
            base.DomainCfg(
                "Synth-B-heavy-tail",
                heavy_tail=True,
                amount_sigma=1.25,
                signal_scale=0.95,
            ),
            8119,
        ),
    ]

    all_perf: list[pd.DataFrame] = []
    all_drift: list[pd.DataFrame] = []
    all_role: list[pd.DataFrame] = []
    all_cov: list[pd.DataFrame] = []
    all_certs: dict[str, Any] = {}
    all_raw_arrays: dict[str, Any] = {}
    all_scalers: dict[str, Any] = {}
    tabular_predictions: list[dict[str, Any]] = []

    for cfg, d_seed in domain_configs:
        print(f"Running controlled study on {cfg.name}...", flush=True)
        perf_df, drift_df, role_df, cov_df, cert, raw_dict, scaler_info = (
            run_domain_study(cfg, d_seed, seeds=seeds)
        )
        all_perf.append(perf_df)
        all_drift.append(drift_df)
        all_role.append(role_df)
        all_cov.append(cov_df)
        all_certs[cfg.name] = cert
        all_raw_arrays.update(raw_dict)
        all_scalers[cfg.name] = scaler_info

        # Tabularize test predictions
        te = raw_dict[f"test_indices_{cfg.name}"]
        targets = raw_dict[f"targets_{cfg.name}"]
        prefix = f"pred_{cfg.name}_"
        for key, arr in raw_dict.items():
            if key.startswith(prefix):
                remainder = key[len(prefix) :]
                mdl, rest = remainder.split("_", 1)
                view, seed_str = rest.rsplit("_s", 1)
                s0 = int(seed_str)
                for ent_id, tgt, p in zip(te, targets, arr):
                    tabular_predictions.append(
                        {
                            "domain": cfg.name,
                            "model": mdl,
                            "view": view,
                            "seed": s0,
                            "entity_id": int(ent_id),
                            "target": int(tgt),
                            "pred": float(p),
                        }
                    )

    perf_concat = pd.concat(all_perf, ignore_index=True)
    drift_concat = pd.concat(all_drift, ignore_index=True)
    role_concat = pd.concat(all_role, ignore_index=True)
    cov_concat = pd.concat(all_cov, ignore_index=True)
    tab_preds_df = pd.DataFrame(tabular_predictions)

    # Compute Summaries
    perf_summary = (
        perf_concat.groupby(["domain", "model", "view", "full_context_fraction"])
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            brier_mean=("brier", "mean"),
        )
        .reset_index()
    )

    role_summary = (
        role_concat.groupby(["domain", "model", "encoding"])
        .agg(
            swap_concordance=("swap_concordance", "mean"),
            swap_response=("swap_response", "mean"),
            serialization_disagreement=("serialization_disagreement", "mean"),
        )
        .reset_index()
    )

    cov_summary = (
        cov_concat.groupby(
            ["domain", "model", "fanout_fraction", "full_context_fraction"]
        )
        .agg(
            jaccard=("jaccard", "mean"),
            paired_view_drift=("paired_view_drift", "mean"),
            paired_ci_lo=("paired_ci_lo", "mean"),
            paired_ci_hi=("paired_ci_hi", "mean"),
            ci_lo=("ci_lo", "mean"),
            ci_hi=("ci_hi", "mean"),
            delta_seed_ind=("delta_seed_ind", "mean"),
            delta_view_ind=("delta_view_ind", "mean"),
            esd_ind=("esd_ind", "mean"),
            esd_ci_lo=("esd_ci_lo", "mean"),
            esd_ci_hi=("esd_ci_hi", "mean"),
        )
        .reset_index()
    )

    # Save outputs to owned outdir
    perf_concat.to_csv(outdir / "v8_performance.csv", index=False)
    perf_summary.to_csv(outdir / "v8_performance_summary.csv", index=False)

    drift_concat.to_csv(outdir / "v8_drift.csv", index=False)
    drift_concat.to_csv(outdir / "v8_drift_summary.csv", index=False)

    role_concat.to_csv(outdir / "v8_role.csv", index=False)
    role_summary.to_csv(outdir / "v8_role_summary.csv", index=False)

    cov_concat.to_csv(outdir / "v8_coverage.csv", index=False)
    cov_summary.to_csv(outdir / "v8_coverage_summary.csv", index=False)

    tab_preds_df.to_csv(outdir / "raw_predictions.csv", index=False)
    np.savez_compressed(outdir / "raw_predictions.npz", **all_raw_arrays)

    with open(outdir / "certificates.json", "w") as f_cert:
        json.dump(all_certs, f_cert, indent=2)

    with open(outdir / "scalers.json", "w") as f_sc:
        json.dump(all_scalers, f_sc, indent=2)

    # Machine-readable summary for coordinator regeneration of TeX tables
    summary_doc: dict[str, Any] = {
        "metadata": {
            "version": 8,
            "seeds": list(seeds),
            "domains": [c.name for c, _ in domain_configs],
            "models": ["LightGBM", "TypedMPNN"],
            "logical_fact_convention": {
                "base_entity": "('customer', cust_id)",
                "structural_role": "('tx_struct', tid, 'buyer' | 'seller')",
                "event_amount": "('tx_amount', tid)",
                "risk_contribution": "('tx_merchant_risk_contrib', tid, merchant_id) [contribution-level]",
            },
        },
        "scalers": all_scalers,
        "performance_states": perf_summary.to_dict(orient="records"),
        "drift_transitions": drift_concat.to_dict(orient="records"),
        "role_controls": role_summary.to_dict(orient="records"),
        "coverage_ablation": cov_summary.to_dict(orient="records"),
        "certificates": all_certs,
    }
    with open(outdir / "v8_controlled_summary.json", "w") as f_sum:
        json.dump(summary_doc, f_sum, indent=2)

    print("Successfully completed controlled study. Artifacts written to:", outdir)
    return summary_doc


def main() -> None:
    """CLI entrypoint for running the v8 controlled study."""
    parser = argparse.ArgumentParser(description="Run v8 corrected controlled study")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Target output directory for results",
    )
    args = parser.parse_args()
    run_all_controls(args.outdir)


if __name__ == "__main__":
    main()
