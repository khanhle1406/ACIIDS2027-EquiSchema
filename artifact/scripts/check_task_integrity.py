#!/usr/bin/env python3
"""Lightweight task-contract preflight for EquiSchema external audits.

This is deliberately a conservative *linter*, not a SQL theorem prover. It catches a
specific high-risk pattern: a forecast task that uses a historical lookback subquery
for entity eligibility but fails to upper-bound that subquery at the anchor time.
Autocomplete tasks are classified as row-mask contracts and do not use this forecast
eligibility check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.lower()).strip()


def extract_in_subqueries(sql: str) -> list[str]:
    """Extract balanced parenthesized bodies immediately following IN ( ... )."""
    low = sql.lower()
    out: list[str] = []
    for m in re.finditer(r"\bin\s*\(", low):
        start = m.end()
        depth = 1
        i = start
        while i < len(sql) and depth:
            if sql[i] == "(":
                depth += 1
            elif sql[i] == ")":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(sql[start : i - 1])
    return out


def audit(path: Path) -> dict:
    manifest = yaml.safe_load(path.read_text())
    kind = manifest.get("kind")
    report = {
        "path": str(path),
        "sha256": sha256(path),
        "name": manifest.get("name"),
        "kind": kind,
        "task_type": manifest.get("task_type"),
        "findings": [],
    }

    if kind == "autocomplete":
        report["contract_type"] = "row-mask"
        report["status"] = "PASS_STATIC_PREFLIGHT"
        report["notes"] = [
            "No future-pointing task SQL is present; external runs must still hash evaluation keys/labels and hidden-column masks."
        ]
        return report

    if kind != "forecast":
        report["contract_type"] = "other"
        report["status"] = "REVIEW"
        report["notes"] = ["No automated temporal rule is implemented for this task kind."]
        return report

    report["contract_type"] = "forecast"
    sql = manifest.get("sql") or ""
    norm = normalize_sql(sql)
    findings = report["findings"]

    # Declared future label window sanity check.
    future_lower = bool(re.search(r"(?:\w+\.)?date\s*>\s*t\.timestamp", norm))
    future_upper = bool(re.search(r"(?:\w+\.)?date\s*<=\s*t\.timestamp\s*\+\s*interval", norm))
    if not (future_lower and future_upper):
        findings.append({
            "severity": "review",
            "code": "FORECAST_WINDOW_NOT_RECOGNIZED",
            "detail": "Could not recognize both lower and upper bounds for the declared future label window."
        })

    # Inspect eligibility subqueries independently of the future label join.
    for idx, body in enumerate(extract_in_subqueries(sql)):
        b = normalize_sql(body)
        has_lookback = bool(re.search(r"date\s*>\s*t\.timestamp\s*-\s*interval", b))
        if not has_lookback:
            continue
        has_anchor_upper = bool(re.search(r"date\s*<=\s*t\.timestamp(?:\s|$|\))", b))
        if not has_anchor_upper:
            findings.append({
                "severity": "risk",
                "code": "UNBOUNDED_HISTORICAL_ELIGIBILITY",
                "subquery_index": idx,
                "detail": "Historical eligibility has a lookback lower bound but no explicit date <= anchor-time upper bound. Future activity can therefore affect entity eligibility."
            })

    report["status"] = "REVIEW" if any(f["severity"] in {"risk", "review"} for f in findings) else "PASS_STATIC_PREFLIGHT"
    report["notes"] = [
        "PASS_STATIC_PREFLIGHT is only a syntax-level preflight; full EquiSchema admission also requires evaluation-key/label hashes, split/cutoff checks, and the rewrite certificate."
    ]
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    reports = [audit(p) for p in args.manifests]
    text = json.dumps({"task_contract_audits": reports}, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
