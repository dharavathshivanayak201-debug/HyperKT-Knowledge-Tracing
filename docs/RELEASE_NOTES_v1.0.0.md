# HyperKT Objective 1: Audited Research Artifacts

This release preserves 22 original notebooks, stage-by-stage explanations, aggregate experimental results and graphs, exact exported final model classes, privacy-aware audit utilities and selected checkpoints. It is an evidence release for one ASSISTments2009 study, not a claim of HyperKT ROC-AUC superiority, official baseline equivalence or journal readiness.

## Model-only assets

- **Final Stage 9E checkpoints:** 15 selected states used by Stage 10, covering DKT_matched, HyperKT_matched and HyperKT_gated for five seeds.
- **Historical Stage 4–8A checkpoints:** eight selected early temporal/fusion, uncertainty-head, fine-tuning and single-seed ablation states.
- **Historical Stage 8B checkpoints:** 20 selected validation-era states across four variants and five seeds.
- **SHA256SUMS:** whole-archive hashes; individual original checkpoint hashes appear in `provenance/checkpoint-release-manifest.json`.

The source archive contains no model binaries or learner-level data. Original selected checkpoint bytes are unchanged, and all 43 states passed restricted weights-only loading and finite tensor-tree inspection; historical architectures must be loaded with their own notebooks.

## Verified findings

All 45 original manifested artifacts match their hashes; all three sequence splits reconstruct against supplied cleaned records; all 15 saved final prediction metric sets and 25 bootstrap intervals reproduce to 1e-12. All 15 final checkpoint hashes match Stage 9G, strict state loading and causal probes pass, and frozen CPU inference was rerun without training.

Final test mean ROC-AUC is 0.744119 for DKT, 0.741595 for original HyperKT and 0.741153 for gated HyperKT. Most paired learner-bootstrap intervals include zero; no robust population-level superiority conclusion is supported.

CPU/CUDA DKT probability differences reach about 0.00124723 and are not relabelled exact agreement; HyperKT differences are below 3e-7. See the aggregate verification report for per-run flags and metrics.

## Exclusions and limitations

Learner interaction rows, source IDs, split-ID lists, sequence arrays, learner-linked predictions, embedded input archives and optimizer/resume states are excluded. This filtering is not differential-privacy certification, and exact private-data reconstruction cannot be rerun from a public clone alone.

Three unchanged historical manuscript drafts are included only with prominent unsupported-claim warnings. No software/model license was supplied or inferred, and this release does not grant redistribution rights for the underlying dataset.
