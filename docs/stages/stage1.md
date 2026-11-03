# Stage 1: Data preparation

The question is whether a single ASSISTments source can be converted into leakage-aware, chronological learner records before any vocabulary or graph is fit. The input is the separately obtained CSV; the original [Stage 1 notebook](../../notebooks/HyperKT_Stage1_ASSIST2009_Data_Preparation.ipynb) records its hash and selected columns.

## Method and outputs

- **Cleaning:** required fields must be present, responses binary and ordering usable; ambiguous multi-skill tokens trigger an error rather than an undocumented policy.
- **Split:** learners are separated 80/10/10 with seed 42 before fitting the vocabulary or co-occurrence.
- **Structure:** skills are fit on train only, with unknown 0 and padding 1; unordered skill pairs are counted from five-interaction windows.
- **Evidence:** 283,105 cleaned rows, 4,163 learners and 149 training skills were checked; public [vocabulary and graph aggregates](../../results/hyperkt_stage1/) remain available.

The source's 346,860 raw rows and raw hash are recorded provenance, not an independently redownloaded source check. Learner records and exact split-ID manifests are withheld; see [data boundaries](../DATA_AND_PRIVACY.md).
