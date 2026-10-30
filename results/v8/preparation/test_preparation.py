"""EquiSchema v8: Pinned Contract Tests for RelBench Preparation.

Validates the data, schema, split, and masking contracts for:
- Dataset: stanford-star/relbench-v1/rel-f1 @ d500f4d5eb581b6d768a1f5ac5a20db14d61c280
- Task: qualifying-position (autocomplete regression)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PREPARATION_DIR = Path(__file__).resolve().parent
RAW_DIR = PREPARATION_DIR / "raw" / "rel-f1"
MATERIALIZED_DIR = PREPARATION_DIR / "materialized"
HASHES_FILE = PREPARATION_DIR / "hashes.json"
FEASIBILITY_FILE = PREPARATION_DIR / "pyg_feasibility.md"
OFFICIAL_MANIFEST = (
    WORKSPACE_ROOT
    / "artifact"
    / "external_manifests"
    / "rel-f1_qualifying-position_official.yaml"
)
VENDOR_DIR = WORKSPACE_ROOT / "artifact" / "relbench_vendor"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def audit_data() -> dict:
    assert HASHES_FILE.exists(), f"Audit file missing at {HASHES_FILE}"
    with open(HASHES_FILE, encoding="utf-8") as f:
        return json.load(f)


def test_audit_file_integrity(audit_data: dict) -> None:
    assert audit_data["dataset"] == "rel-f1"
    assert audit_data["task"] == "qualifying-position"
    assert audit_data["hf_revision"] == "d500f4d5eb581b6d768a1f5ac5a20db14d61c280"
    assert audit_data["vendor_commit"] == "91c1f6420a03feb61c279aaeb194d7268435f278"
    assert audit_data["manifest_audit"]["status"] == "PASS"

    # Validate actual vendor git HEAD matches audit
    actual_vendor_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=VENDOR_DIR, text=True
    ).strip()
    assert actual_vendor_head == "91c1f6420a03feb61c279aaeb194d7268435f278"

    # Verify cross-check regeneration flags
    cross_check = audit_data["metadata"]["regeneration_cross_check"]
    assert cross_check["train_exact_match"] is True
    assert cross_check["val_exact_match"] is True
    assert cross_check["test_multiset_match"] is True

    # Verify contiguous entity ID mapping
    entity_mapping = audit_data["metadata"]["entity_key_mapping"]
    assert entity_mapping["is_contiguous_0_based"] is True
    assert entity_mapping["total_entities"] == 9815


def test_raw_snapshot_files_and_checksums(audit_data: dict) -> None:
    raw_files = audit_data["raw_files"]
    assert len(raw_files) == 14

    for rel_path, meta in raw_files.items():
        local_path = RAW_DIR / rel_path
        assert local_path.exists(), f"Missing raw file: {local_path}"
        assert local_path.stat().st_size == meta["size_bytes"]
        assert _sha256(local_path) == meta["sha256"]


def test_task_manifest_contract() -> None:
    task_manifest_path = RAW_DIR / "tasks" / "qualifying-position" / "manifest.yaml"
    assert task_manifest_path.exists()

    with open(task_manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    assert manifest["name"] == "qualifying-position"
    assert manifest["kind"] == "autocomplete"
    assert manifest["task_type"] == "regression"
    assert manifest["entity_table"] == "qualifying"
    assert manifest["target_col"] == "position"

    # Must match official manifest verbatim
    assert _sha256(task_manifest_path) == _sha256(OFFICIAL_MANIFEST)


def test_qualifying_table_schema_and_keys() -> None:
    qual_path = RAW_DIR / "db" / "qualifying.parquet"
    df = pd.read_parquet(qual_path)

    assert len(df) == 9815
    assert "qualifyId" in df.columns
    assert "date" in df.columns
    assert "position" in df.columns

    # qualifyId must be unique primary key and contiguous 0..N-1
    assert df["qualifyId"].nunique() == len(df)
    assert not df["qualifyId"].isna().any()
    assert (df["qualifyId"].values == list(range(len(df)))).all()

    # Foreign keys present
    for fk in ["raceId", "driverId", "constructorId"]:
        assert fk in df.columns
        assert not df[fk].isna().any()


def test_materialized_split_counts_and_conservation() -> None:
    train_df = pd.read_parquet(MATERIALIZED_DIR / "train.parquet")
    val_df = pd.read_parquet(MATERIALIZED_DIR / "val.parquet")
    test_df = pd.read_parquet(MATERIALIZED_DIR / "test.parquet")
    test_masked_df = pd.read_parquet(MATERIALIZED_DIR / "test_masked.parquet")

    assert len(train_df) == 2228
    assert len(val_df) == 1854
    assert len(test_df) == 5733
    assert len(test_masked_df) == 5733

    # Total row conservation
    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_split_rows == 9815

    # Entity key partition disjointness
    train_keys = set(train_df["qualifyId"])
    val_keys = set(val_df["qualifyId"])
    test_keys = set(test_df["qualifyId"])

    assert len(train_keys & val_keys) == 0
    assert len(val_keys & test_keys) == 0
    assert len(train_keys & test_keys) == 0
    assert len(train_keys | val_keys | test_keys) == 9815


def test_split_temporal_boundaries() -> None:
    train_df = pd.read_parquet(MATERIALIZED_DIR / "train.parquet")
    val_df = pd.read_parquet(MATERIALIZED_DIR / "val.parquet")
    test_df = pd.read_parquet(MATERIALIZED_DIR / "test.parquet")

    val_boundary = pd.Timestamp("2005-01-01 00:00:00")
    test_boundary = pd.Timestamp("2010-01-01 00:00:00")

    assert (train_df["date"] <= val_boundary).all()
    assert ((val_df["date"] > val_boundary) & (val_df["date"] <= test_boundary)).all()
    assert (test_df["date"] > test_boundary).all()


def test_label_and_masking_contract() -> None:
    train_df = pd.read_parquet(MATERIALIZED_DIR / "train.parquet")
    val_df = pd.read_parquet(MATERIALIZED_DIR / "val.parquet")
    test_df = pd.read_parquet(MATERIALIZED_DIR / "test.parquet")
    test_masked_df = pd.read_parquet(MATERIALIZED_DIR / "test_masked.parquet")

    # Visible columns
    assert train_df.columns.tolist() == ["date", "qualifyId", "position"]
    assert val_df.columns.tolist() == ["date", "qualifyId", "position"]
    assert test_df.columns.tolist() == ["date", "qualifyId", "position"]

    # Official masking contract: position is concealed in test_masked
    assert test_masked_df.columns.tolist() == ["date", "qualifyId"]
    assert "position" not in test_masked_df.columns

    # Label validity in ground truth tables
    for df in [train_df, val_df, test_df]:
        assert not df["position"].isna().any()
        assert (df["position"] >= 1).all()

    # Verify train standard deviation (normalizer for NMAE)
    train_std = train_df["position"].std(ddof=1)
    assert 6.63 < train_std < 6.65


def test_regeneration_cross_checks_all_splits() -> None:
    """Verifies that duckdb SQL regeneration yields exact train/val matches and test multiset match."""
    import relbench

    ds = relbench.load_dataset(RAW_DIR)
    task_hosted = ds.load_task("qualifying-position", regenerate=False)
    task_regen = ds.load_task("qualifying-position", regenerate=True)

    # 1. Train exact match
    h_train = task_hosted.get_table("train", mask_input_cols=False).df
    r_train = task_regen.get_table("train", mask_input_cols=False).df
    assert h_train.equals(r_train)

    # 2. Val exact match
    h_val = task_hosted.get_table("val", mask_input_cols=False).df
    r_val = task_regen.get_table("val", mask_input_cols=False).df
    assert h_val.equals(r_val)

    # 3. Test multiset match
    h_test = task_hosted.get_table("test", mask_input_cols=False).df
    r_test = task_regen.get_table("test", mask_input_cols=False).df
    h_sorted = h_test.sort_values(["qualifyId"]).reset_index(drop=True)
    r_sorted = r_test.sort_values(["qualifyId"]).reset_index(drop=True)
    assert h_sorted.equals(r_sorted)


def test_feasibility_report_exists_and_untested_marked() -> None:
    assert FEASIBILITY_FILE.exists()
    content = FEASIBILITY_FILE.read_text(encoding="utf-8")
    assert "UNTESTED on this host" in content
    assert "UNTESTED in this preparation phase" in content
    assert "Apple Silicon (Mac M1, ARM64" in content
