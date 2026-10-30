# PyG Dependency Feasibility and Platform Status (v8 Preparation)

Date: 2026-10-28
Scope: Preparation only (rel-f1 qualifying-position); no model training performed.

## 1. Observed Local Platform (Mac M1)
- Hardware: Apple Silicon (Mac M1, ARM64, Darwin 25.6.0).
- Python Runtime: Python 3.13.9 in isolated `.venv` (created via `python3 -m venv` and `pip` only).
- Verified CPU Operations:
  - Hugging Face Hub snapshot download (revision `d500f4d5eb581b6d768a1f5ac5a20db14d61c280`): VERIFIED.
  - Parquet I/O (`pyarrow` 26.0.0, `pandas` 3.1.0): VERIFIED.
  - DuckDB SQL query execution for task label regeneration (`duckdb` 2.0.0): VERIFIED.
  - Contract linting and test suite (`pytest` 9.2.0, `ruff` 0.17.3): VERIFIED (13/13 tests passing).
- Documented Pitfalls (EquiSchema `BUILD.md`):
  - Concurrency conflicts: `torch` and `scikit-learn` bundle independent `libomp` libraries on macOS. On Python 3.13, uncoordinated imports or multi-threading can trigger OpenMP thread initialization failures (`OMP: Error #179: pthread_mutex_init failed`).
- Hardware Acceleration Status:
  - Apple Silicon MPS and PyG C++ extensions (`torch-scatter`, `torch-sparse`, `pyg-lib`): UNTESTED on this host. Pre-compiled wheels are not provided by upstream PyG for macOS ARM64; source builds under Python 3.13 are unverified.

## 2. Remote Linux Platform Status (CUDA GPU)
- Status: UNTESTED in this preparation phase.
- Scope Limitation: No remote VMs, GPU instances, Modal, or paid compute services were invoked.
- Literature/Upstream Specification:
  - RelBench documentation references Linux x86_64 with NVIDIA CUDA 12.x for GNN baselines.
  - End-to-end execution of rel-f1 qualifying-position GNN training on Linux GPU remains unverified in this workspace and must be evaluated under a future training dispatch.
