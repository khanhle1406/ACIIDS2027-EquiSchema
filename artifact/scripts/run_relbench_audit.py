"""Pinned v8 CLI: prepare, bounded train, then validate/analyze actual predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from audit_v8 import esd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 23, 37, 51, 71)


def analyze(out: Path) -> dict:
    """Reject incomplete or misaligned matrices before computing report statistics."""
    evidence = out / "evidence" if (out / "evidence").is_dir() else out
    arrays: dict[str, list] = {"original": [], "R1": []}
    reference = None
    for view, view_predictions in arrays.items():
        for seed in SEEDS:
            frame = pd.read_csv(evidence / f"predictions_{view}_{seed}.csv")
            keys = ["qualifyId", "date", "position", "raceId"]
            if frame.duplicated(["qualifyId", "date"]).any():
                raise ValueError("Duplicate evaluation entity/time keys")
            if not (frame["seed"] == seed).all() or not (frame["view"] == view).all():
                raise ValueError("Prediction metadata disagrees with registered matrix")
            if reference is None:
                reference = frame[keys]
            elif not frame[keys].equals(reference):
                raise ValueError(
                    "Keys, timestamps, labels or race clusters differ across fits"
                )
            view_predictions.append(frame.prediction.to_numpy())
    if reference is None:
        raise ValueError("No prediction evidence")
    sigma = json.loads((evidence / "summary.json").read_text())["train_std"]
    result = esd(
        np.stack(arrays["original"]),
        np.stack(arrays["R1"]),
        clusters=reference.raceId.to_numpy(),
        classification=False,
        scale=sigma,
    )
    performance = pd.read_csv(evidence / "performance.csv")
    by_view = performance.groupby("view").nmae.mean()
    report = {
        "status": "COMPLETE_PAIRED_MATRIX",
        "seed_count": 5,
        "test_rows": len(reference),
        "drift": result,
        "mean_nmae": by_view.to_dict(),
        "worst_view_nmae": float(by_view.max()),
    }
    (out / "analysis_verified.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "analyze"])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/v8/relbench")
    parser.add_argument("--backend", choices=["local", "modal"], default="modal")
    parser.add_argument("--budget-usd", type=float, default=1.0)
    args = parser.parse_args()
    if args.stage == "prepare":
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("prepare_relbench_v8.py"))],
            check=True,
        )
    elif args.stage == "analyze":
        print(json.dumps(analyze(args.out_dir), indent=2))
    elif args.backend == "modal":
        from modal_bounded_v8 import main as remote_main

        remote_main(args.out_dir, args.budget_usd)
        print(json.dumps(analyze(args.out_dir), indent=2))
    else:
        from relbench_engine_v8 import execute

        execute(
            ROOT / "results/v8/preparation/raw/rel-f1",
            ROOT / "artifact/relbench_vendor",
            args.out_dir,
            time.monotonic() + 24 * 3600,
        )
        print(json.dumps(analyze(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
