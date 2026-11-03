# HyperKT — Objective 01 Research Artifacts

This package contains the original research notebooks, stage result archives, recovered model checkpoints, and LaTeX manuscript supplied in the SHIVA-LLM research conversation. The files are preserved as originally provided.

## Research finding
On the held-out ASSISTments 2009 test split, five-seed mean ROC-AUC was 0.744119 for the independently implemented DKT comparator, 0.741595 for original HyperKT, and 0.741153 for gated HyperKT. Do **not** claim that HyperKT outperformed DKT on ROC-AUC.

## Package layout
- `notebooks/`: stage notebooks, including corrected variants where supplied.
- `manuscript/`: LaTeX drafts, including latest `HyperKT_SCI_Q1_Ready (1)(1).tex`.
- `stage_results/`: original stage output archives (including predictions and figures).
- `checkpoint_and_large_artifacts/`: large training archives, checkpoints and outputs, supplied as a separate ZIP for practical GitHub handling.
- `SHA256_MANIFEST.json`: source-file hashes and byte sizes.

## Reproducibility and limitations
Stage 9F reports event-level train/validation integrity; Stage 9G reports checkpoint validation reproducibility. The independently implemented baselines have not been demonstrated equivalent to official reference implementations. Stage 10 evaluated frozen checkpoints. Some early notebooks are superseded by corrected notebooks; do not treat every historical notebook as canonical.

**Before publishing:** verify data redistribution rights for ASSISTments-derived files, scan notebook outputs for private data and tokens, and verify that manuscript claims match actual test results. Large `.pt` files should be published via Git LFS or a release/archive rather than normal Git history. This is an archival package, not a newly executed end-to-end reproducibility certification.
