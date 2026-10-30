# Reproducibility notes

The protocol in `results/v8/protocol.json` was frozen before the real-data runs. The RelBench data revision and digests are recorded in `results/v8/preparation/hashes.json`; the upstream source revision is pinned by the Git submodule. Controlled predictions and summaries are in `results/v8/controlled/`. Real-data predictions, canonical CPU outputs, and measurements are in `results/v8/relbench/evidence/`; the ten checkpoints remain in the original local workspace and are not included in this repository.

The final RelBench matrix uses the official `rel-f1/qualifying-position` task, original and R1 views, and seeds 11, 23, 37, 51, and 71. This is a budgeted GraphSAGE configuration, not a leaderboard reproduction. `analysis_verified.json` contains the independently recomputed summary. The setup and library versions are documented in the README and evidence files.

The preparation certificate's local dataset path is expressed relative to the repository. Provider-specific receipts and logs are not included. The original workspace retains those records.
