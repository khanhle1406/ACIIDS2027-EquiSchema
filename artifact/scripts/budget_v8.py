"""Fail-closed, durable cost reservation before any Modal launch.

Reservations are never automatically refunded on timeout or process failure.
All billable startup, execution, idle and shutdown time must fit the reservation.
Rates recorded from modal.com/pricing on 2026-10-28; CPU and RAM must be capped.
"""

from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path

T4_CAPPED_USD_PER_SECOND = 0.000164 + 2 * 0.0000131 + 8 * 0.00000222


def reserve(path: Path, amount_usd: float, purpose: str, cap_usd: float = 1.0) -> dict:
    """Reserve a conservative upper bound; an uncertain attempt stays charged."""
    if not all(math.isfinite(x) and x > 0 for x in (amount_usd, cap_usd)):
        raise ValueError("Budget and reservation must be finite positive amounts")
    if cap_usd > 1:
        raise ValueError("Author authorized at most USD 1 before credits")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        raw = stream.read()
        ledger = json.loads(raw) if raw else {"cap_usd": cap_usd, "reservations": []}
        entries = [
            ledger["cap_usd"],
            *(row["amount_usd"] for row in ledger["reservations"]),
        ]
        if not all(math.isfinite(value) and value > 0 for value in entries):
            raise ValueError("Invalid persisted budget ledger")
        cap = min(cap_usd, ledger["cap_usd"])
        used = sum(row["amount_usd"] for row in ledger["reservations"])
        if used + amount_usd > cap:
            raise ValueError("Reservation exceeds remaining authorized budget")
        ledger["reservations"].append({"amount_usd": amount_usd, "purpose": purpose})
        ledger["reserved_usd"] = used + amount_usd
        stream.seek(0)
        stream.truncate()
        json.dump(ledger, stream, indent=2)
        stream.flush()
        import os

        os.fsync(stream.fileno())
        return ledger
