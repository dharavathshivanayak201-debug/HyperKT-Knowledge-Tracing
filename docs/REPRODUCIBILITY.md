# Reproduction guide

The repository supports several levels of reproducibility with different inputs. Public checks and regenerated figures are immediately runnable, while learner-level evaluation and full training require matching privately obtained data and original intermediate packages.

## Environment

Use Python 3.11 or 3.12 for the public CI path, a virtual environment and `requirements-audit.txt`; install CPU PyTorch 2.11.0 from the official wheel index for model tests. `requirements.txt` additionally lists PyArrow and scikit-learn for data/notebook work; its ranges are compatibility guidance, not an exact historical lockfile.

The supplied final evaluation protocol records Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.4.6 and CUDA. The independent CPU verification environment is captured in `analysis/environment.json`; changing a device, library version or numerical kernel can change probabilities and threshold-sensitive metrics.

## Public checks and plots

```bash
python -m pip install -r requirements-audit.txt
python scripts/verify_public.py
python -m unittest discover -s tests -v
python scripts/make_figures.py
```

Without PyTorch, model-specific tests are explicitly skipped rather than reported as passed. The GitHub workflow installs CPU PyTorch and runs all public tests for both supported CI Python versions.

- **Integrity:** original public files retain supplied SHA-256 hashes; the final model class ASTs must match the Stage 10 notebook.
- **Tables:** all five-seed final metric means/sample SDs are reconstructed from public per-seed rows.
- **Figures:** four new neutral-labelled figures are regenerated from public aggregate tables; original supplied stage plots are preserved separately.
- **Boundary:** public graph NPZ files must not expose learner sequence or prediction arrays.
- **Model probes:** synthetic checks cover output shape, finite values, causal future perturbation, fixed assignments and pre-deserialization hash rejection.

These public tests do not recompute learner-bootstrap intervals or recreate individual prediction rows without their private inputs.

## Frozen checkpoint verification

Follow [the checkpoint guide](CHECKPOINTS.md) to download and hash-check model-only release assets. All 15 final weights can be strict-loaded and causally probed without private data; historical states require their own notebook architecture.

With authorized local copies of the exact private test sequence NPZ and Stage 10 prediction directory, run:

```bash
python scripts/verify_checkpoints.py \
  --checkpoints checkpoints/final \
  --sequences private/hyperkt_stage2/test_sequences.npz \
  --predictions private/hyperkt_stage10 \
  --output analysis/local-cpu-inference.json
```

The script performs no training. It writes only aggregate metrics, hashes, numerical differences and agreement flags, not learner IDs or prediction vectors; values exceeding the 1e-5 agreement check remain visible rather than causing the tolerance to be widened.

## Full private-artifact reconstruction

The original three input ZIPs remain private. To rerun `scripts/audit_private_artifacts.py`, extract the original source/results package into one local source folder, place the six large inner ZIPs under its `checkpoint_and_large_artifacts/` directory, and extract each small stage ZIP under its original stem in a separate nested folder. Retain supplied stems such as `hyperkt_stage2(2)` and `hyperkt_stage3b(1)`; the audit expects the fixed sequences specifically under `hyperkt_stage2(2)`.

```bash
python -m pip install pyarrow
python scripts/audit_private_artifacts.py \
  --source private/source \
  --nested private/nested \
  --output analysis/local-private-artifact-audit.json \
  --bootstrap 1000
```

The source tree must contain the original manifest, original notebooks and large archives for the 45-entry integrity audit. The nested tree must include Stage 1 cleaned Parquet files, fixed Stage 2 sequences and Stage 10 prediction rows; none is supplied in the public clone.

The audit reconstructs sequences against cleaned records, metrics and whole-learner bootstrap intervals without model training. Its saved-output subsection explicitly does not load checkpoint bytes; run the separate checkpoint script for those checks.

## Original training workflow

Open the unchanged notebooks in the order described by the [stage walkthrough](RESEARCH_WALKTHROUGH.md), not by running every historical variant indiscriminately. Begin with Stage 1 preprocessing and the fixed Stage 2 builder, inspect/reproduce repaired Stage 3B graph learning, and carry the resulting packages through the relevant model stages; later notebooks frequently request uploaded ZIPs via `google.colab.files` and/or Google Drive rather than using the public folder layout automatically.

Use fresh local/Drive output directories, document exact input hashes and do not overwrite the historical result tables. Reproducing original training requires notebook-specific dependencies and human upload/Drive setup; the generic audit environment is not a tested one-command training launcher.

The final Stage 9E settings are batch 32, learning rate 0.0002, weight decay 0.0001, at most 20 epochs, patience 5, hidden size 128 and dropout 0.1. HyperKT has two causal blocks and eight attention heads; seed-specific selection is validation-based and the pretrained assignments are reused across seeds.

If conducting a new experiment, declare it as a new protocol with new artifacts before evaluating an appropriate untouched test cohort. Rerunning the already published frozen models is a reproducibility check, not another opportunity for test-based architecture selection.
