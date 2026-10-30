"""Instance-level vertical decomposition with temporal metadata preservation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import pandas as pd


def frame_hash(frame: pd.DataFrame, keys: Sequence[str]) -> str:
    """Hash typed keyed rows independently of their serialization order."""
    if frame[list(keys)].isna().any().any() or frame.duplicated(list(keys)).any():
        raise ValueError("Certificate requires unique non-null primary keys")
    ordered = frame.sort_values(list(keys)).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(
        json.dumps([(str(c), str(t)) for c, t in ordered.dtypes.items()]).encode()
    )
    digest.update(pd.util.hash_pandas_object(ordered, index=False).values.tobytes())
    return digest.hexdigest()


def split_r1(
    frame: pd.DataFrame, key: str, foreign_keys: Sequence[str], time_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain FK topology in core; payload's key references core.

    Duplicated time metadata is required for temporal filtering at either node.
    It maps to the same logical fact and must be deduplicated by coverage audits.
    """
    frame_hash(frame, [key])
    core_cols = list(dict.fromkeys([key, *foreign_keys, time_col]))
    payload_cols = [key, time_col] + [c for c in frame if c not in core_cols]
    return frame[core_cols].copy(), frame[payload_cols].copy()


def inverse_r1(
    core: pd.DataFrame,
    payload: pd.DataFrame,
    original: pd.DataFrame,
    key: str,
    time_col: str,
) -> pd.DataFrame:
    """Reject missing rows, altered time metadata, and dtype changes."""
    frame_hash(core, [key])
    frame_hash(payload, [key])
    joined = core.merge(
        payload, on=key, validate="one_to_one", suffixes=("", "_payload")
    )
    if len(joined) != len(original):
        raise ValueError("Rewrite lost rows")
    if not joined[time_col].equals(joined[f"{time_col}_payload"]):
        raise ValueError("Temporal metadata differs across projections")
    restored = joined[list(original.columns)]
    if frame_hash(restored, [key]) != frame_hash(original, [key]):
        raise ValueError("Typed round-trip mismatch")
    return restored
