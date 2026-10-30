import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from budget_v8 import T4_CAPPED_USD_PER_SECOND, reserve


def test_uncertain_attempt_cannot_be_retried_for_free(tmp_path: Path) -> None:
    ledger = tmp_path / "budget.json"
    reserve(ledger, 0.7, "first invocation")
    with pytest.raises(ValueError, match="remaining"):
        reserve(ledger, 0.7, "retry after uncertain termination")
    assert reserve(ledger, 0.15, "setup")["reserved_usd"] == 0.85


def test_cap_cannot_be_raised_or_nan_bypassed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        reserve(tmp_path / "budget.json", float("nan"), "invalid")
    with pytest.raises(ValueError):
        reserve(tmp_path / "budget.json", 0.5, "invalid cap", 2)
    assert 3000 * T4_CAPPED_USD_PER_SECOND < 0.7


def test_corrupt_ledger_fails_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "budget.json"
    ledger.write_text('{"cap_usd": 1, "reservations": [{"amount_usd": -1}]}')
    with pytest.raises(ValueError, match="Invalid persisted"):
        reserve(ledger, 0.7, "launch")
