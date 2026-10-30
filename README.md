# EquiSchema

EquiSchema audits whether relational prediction pipelines remain stable across certified storage rewrites and respond to changes in foreign-key roles. The current anonymous manuscript is [FINAL.pdf](FINAL.pdf); its LaTeX source is in `paper/`.

The repository contains the experiment and audit code (`artifact/scripts/`), tests (`artifact/tests/`), public task manifests, and the v8 experimental evidence (`results/v8/`). The pinned upstream RelBench implementation is included as a Git submodule at commit `91c1f6420a03feb61c279aaeb194d7268435f278` and retains its own license.

## Get the code

```bash
git clone --recurse-submodules https://github.com/khanhle1406/ACIIDS2027-EquiSchema.git
cd ACIIDS2027-EquiSchema
```

For an existing clone, run `git submodule update --init --recursive`. Run the commands below from the repository root.

## Reproduce and check

Use Python 3.10+ in a virtual environment with pip. The preparation environment is recorded in `results/v8/preparation/requirements-locked.txt`; versions used for controlled experiments are in `results/v8/controlled/environment.json`. Model training also requires LightGBM and PyTorch. The Linux environment used for the real-data run is recorded in `results/v8/relbench/evidence/requirements-linux.txt`.

```bash
python -m pytest -q artifact/tests
python artifact/scripts/run_relbench_audit.py analyze
python artifact/scripts/render_v8_results.py
tectonic --keep-logs --keep-intermediates --outdir qa/v8/build paper/paper.tex
```

The analysis command checks prediction alignment and recomputes the real-data drift interval from the stored predictions. Rendering regenerates the manuscript tables and figure. `prepare_relbench_v8.py` validates the pinned dataset and submodule, while `run_v8_controls.py` regenerates the controlled experiments; these are computational experiments, so library and hardware differences may affect fitted models.

The Modal training entry point may incur cloud charges. It refuses another paid invocation when `results/v8/relbench/analysis_verified.json` is present. The committed results include per-target predictions, controlled outputs, and the pinned public dataset snapshot. The ten model checkpoints remain in the local workspace and are not included in this repository. Local cloud logs, account-specific receipts, virtual environments, and duplicate archives are also excluded. The `dataset_source` path in the preparation certificate was made relative for portability.

## Scope of results

The five-seed RelBench audit measures a sampled-context mismatch. Its excess schema drift estimate is 0.0157 with a 95% seed/race-cluster interval of [-0.0721, 0.1236], which includes zero. The budgeted GraphSAGE models also underperform a training-median reference. The experiment therefore does not establish excess predictive drift or production-model robustness. See the manuscript for the controlled results, assumptions, and limitations.
