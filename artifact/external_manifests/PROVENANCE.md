# External task-manifest provenance

The files suffixed `official` are pinned local copies matching the published RelBench v1 `rel-f1` task-manifest content exposed at Hugging Face commit `d500f4d5eb581b6d768a1f5ac5a20db14d61c280` (Stanford STAR). Their SHA-256 values in `results/task_integrity_audit.json` identify these local audit inputs; they are not claimed to be remote object hashes.

The `driver-dnf_causal_variant` file is **not an official RelBench task**. EquiSchema adds only `AND date <= t.timestamp` inside the one-year eligibility subquery so the historical eligibility set cannot depend on post-anchor activity. It is provided solely as a separately labelled sensitivity contract and must never be pooled with official-task scores.
