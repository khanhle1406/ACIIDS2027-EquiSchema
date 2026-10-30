"""Admission tests for actual CSV contracts; no fabricated training claims."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_relbench_audit import SEEDS, analyze


def make_matrix(path: Path) -> None:
    for view in ("original", "R1"):
        for seed in SEEDS:
            pd.DataFrame(
                {
                    "qualifyId": [0, 1, 2, 3],
                    "date": ["2001-01-01"] * 4,
                    "position": [1, 2, 3, 4],
                    "raceId": [1, 1, 2, 2],
                    "seed": seed,
                    "view": view,
                    "prediction": [1 if view == "original" else 3] * 4,
                }
            ).to_csv(path / f"predictions_{view}_{seed}.csv", index=False)
    (path / "summary.json").write_text(json.dumps({"train_std": 2}))
    pd.DataFrame({"view": ["original", "R1"], "nmae": [0.1, 0.4]}).to_csv(
        path / "performance.csv", index=False
    )


def test_aligned_regression_and_metric_direction(tmp_path: Path) -> None:
    make_matrix(tmp_path)
    result = analyze(tmp_path)
    assert result["drift"]["esd_ind"] == 1
    assert result["worst_view_nmae"] == 0.4


def test_corrupted_label_or_order_rejected(tmp_path: Path) -> None:
    make_matrix(tmp_path)
    p = tmp_path / "predictions_R1_11.csv"
    df = pd.read_csv(p)
    df.loc[0, "position"] = 999
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="labels"):
        analyze(tmp_path)


def test_missing_seed_cannot_be_reported_complete(tmp_path: Path) -> None:
    make_matrix(tmp_path)
    (tmp_path / "predictions_R1_71.csv").unlink()
    with pytest.raises(FileNotFoundError):
        analyze(tmp_path)
