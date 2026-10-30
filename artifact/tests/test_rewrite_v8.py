import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rewrite_v8 import frame_hash, inverse_r1, split_r1


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [2, 1],
            "fk": pd.array([None, 7], dtype="Int64"),
            "date": pd.to_datetime(["2001-01-01", "2002-01-01"]),
            "value": [2.0, 3.0],
        }
    )


def test_roundtrip_preserves_null_fk_dtype_and_row_identity() -> None:
    original = fixture()
    core, payload = split_r1(original, "id", ["fk"], "date")
    restored = inverse_r1(core.iloc[::-1], payload, original, "id", "date")
    assert frame_hash(restored, ["id"]) == frame_hash(original, ["id"])
    assert str(restored.fk.dtype) == "Int64"


def test_timestamp_corruption_rejected() -> None:
    original = fixture()
    core, payload = split_r1(original, "id", ["fk"], "date")
    payload.loc[0, "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Temporal"):
        inverse_r1(core, payload, original, "id", "date")


def test_missing_rows_and_duplicate_keys_rejected() -> None:
    original = fixture()
    core, payload = split_r1(original, "id", ["fk"], "date")
    with pytest.raises(ValueError, match="lost rows"):
        inverse_r1(core, payload.iloc[:1], original, "id", "date")
    with pytest.raises(ValueError, match="unique"):
        split_r1(pd.concat([original, original]), "id", ["fk"], "date")
