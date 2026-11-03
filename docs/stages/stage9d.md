# Stage 9D: Performance-gap and fairness audit

The [Stage 9D notebook](../../notebooks/HyperKT_Stage9D_Performance_Gap_Fairness_Audit.ipynb) inspects the validation gap and comparability rather than adding a new favorable experiment. Its [recommendation](../../results/hyperkt_stage9d/protocol_recommendation.json) says `DIAGNOSTIC_ONLY_NOT_TEST_READY`.

## Findings

- **Gap:** the saved comparison reports DKT above the candidate by about 0.006486 validation AUC.
- **Fairness:** batch sizes and architecture capacities were not fully matched in the earlier comparisons.
- **Open questions:** baseline fidelity and cross-window event uniqueness still required attention.
- **Next stage:** controlled retraining was to be separately labelled and completed before frozen held-out testing.

The [statistics, fairness audit and plot](../../results/hyperkt_stage9d/) are diagnostic evidence. They must not be described as final test success or erased simply because a later repository is public.
