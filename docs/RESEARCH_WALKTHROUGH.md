# Research walkthrough

This study developed HyperKT through data preparation, structural diagnostics, causal temporal modelling, uncertainty experiments, repeated-seed comparisons, fairness checks and a frozen held-out evaluation. The supplied notebooks are a research history with corrections and negative findings, not 22 interchangeable steps in one automatically executable training command.

## Problem and final model

At each valid position the model receives the preceding skill and response, then predicts correctness for the target skill without seeing that target's answer. Training minimizes masked binary cross-entropy, and evaluation covers ROC-AUC, NLL, Brier score, F1, calibration error and other threshold metrics.

- **Temporal history:** skill, preceding-response and position embeddings feed two causal transformer blocks with eight heads and hidden size 128.
- **Decay:** a learned nonnegative per-head penalty reduces attention with interaction-position distance; it does not use measured time gaps.
- **Structure:** a repaired training-derived hypergraph supplies fixed skill-to-edge assignments; learnable edge embeddings and fusion augment the target-skill representation.
- **Output:** a concatenation of temporal and target features feeds a sigmoid/BCE predictor; there is no final Beta posterior.
- **Comparator:** an independently implemented DKT uses skill-response embeddings and a one-layer LSTM with target-skill logit selection.
- **Negative ablation:** HyperKT_gated multiplies the structural residual by a learned scalar sigmoid gate, without improving final observed AUC.

The unchanged Stage 10 class bodies are exported mechanically to [models.py](../src/hyperkt/models.py), with an AST-identity check in public verification. Causal attention has no explicit key-padding mask: only right-padded, prefix-valid windows are supported, and output masks exclude padded positions.

## Stage map

Each linked page explains its question, inputs, implementation, outputs and limits. Original reports remain unchanged, even when later audits refine or correct their scope.

| Stage | Purpose and principal evidence | Detailed explanation |
| --- | --- | --- |
| 1 | Single-source cleaning, learner split, train-only vocabulary and co-occurrence | [Data preparation](stages/stage1.md) |
| 2 | Fixed next-response window construction and mask alignment | [Sequence construction](stages/stage2.md) |
| 3 | Original differentiable graph fit, later diagnosed as near-constant | [Original graph](stages/stage3.md) |
| 3B | Repaired edge prediction and degree-product baseline | [Graph correction](stages/stage3b.md) |
| 4 | Early temporal-only versus structural-fusion comparison | [Temporal model](stages/stage4.md) |
| 5 | Beta-style head and MC-dropout diagnostic, with identifiability caveat | [Uncertainty](stages/stage5.md) |
| 6 | End-to-end fine-tuning attempt selecting epoch 0 | [Fine-tuning](stages/stage6.md) |
| 7 | Artifact and sequence audit, not model training | [Reproducibility audit](stages/stage7.md) |
| 8A | Causal checks and single-seed hypergraph/decay ablations | [Controlled ablations](stages/stage8a.md) |
| 8B | Four variants × five seeds on validation | [Repeated-seed ablations](stages/stage8b.md) |
| 9A | Validation statistics and an explicitly incomplete draft protocol lock | [Draft selection](stages/stage9a.md) |
| 9B | Historical checkpoint/causality/sequence audit | [Checkpoint audit](stages/stage9b.md) |
| 9C | Independent DKT, DKVMN, SAKT and AKT-inspired baselines | [Baseline experiments](stages/stage9c.md) |
| 9D | Performance gap and fairness audit, diagnostic-only decision | [Fairness](stages/stage9d.md) |
| 9E | Matched batch/budget final three-model training study | [Controlled comparison](stages/stage9e.md) |
| 9F | Event reconstruction against cleaned train/validation records | [Event integrity](stages/stage9f.md) |
| 9G | Frozen-checkpoint verification and conditional protocol decision | [Conditional freeze](stages/stage9g.md) |
| 10 | Frozen held-out test and whole-learner bootstrap | [Final evaluation](stages/stage10.md) |
| 11 | Supplied publication-analysis code, without a supplied result archive | [Publication analysis](stages/stage11.md) |

## Notebook versions and execution order

There are two Stage 2 notebooks and two Stage 3B notebooks. Use the fixed Stage 2 constructor for the arrays audited here; retain the earlier version as provenance, and inspect both repaired-graph versions before choosing the Colab/GPU-safe path for a new run.

The umbrella `HyperKT_Objective01_Colab.ipynb` is historical research scaffolding, not a replacement for the staged final protocol. All 22 notebooks are supplied without stored execution outputs, so results are established by the separate result artifacts, not by claiming a clean end-to-end notebook rerun.

For a new training reproduction, start at Stage 1, build fixed Stage 2 windows, reproduce repaired Stage 3B structure, then follow the relevant model and final controlled-study notebooks. Do not concatenate exploratory architecture changes into a retrospectively “preregistered” protocol; freeze a new study prospectively before an appropriate untouched test.

## Reading results responsibly

Only Stage 10 represents final learner-response test evaluation for the selected Stage 9E states. Stage 3B edge test AUC measures a different task on training-derived structure, and earlier validation scores drove research choices.

The supported conclusion is that original HyperKT did not exceed the independent DKT baseline in final observed ROC-AUC, while showing a slightly higher fixed-threshold F1. Structural interpretability, calibrated Bayesian uncertainty, multiple-dataset validation and educational intervention benefits remain unestablished.
