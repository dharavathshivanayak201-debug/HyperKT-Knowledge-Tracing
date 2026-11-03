# Stage 7: Reproducibility audit

The [audit notebook](../../notebooks/HyperKT_Stage7_Reproducibility_Audit.ipynb) checks artifact structure, sequence keys, masks, alignment and learner splits rather than training a new model. The [audit report and checks](../../results/hyperkt_stage7_audit/) distinguish counts of windows, raw split learners and learners that actually yield predictions.

## Interpretation

- **Useful checks:** valid array keys/shapes, learner-disjoint split membership, expected counts and local next-response shifts.
- **Count correction:** 3,222 train, 403 validation and 402 test learners have prediction windows; raw split totals are larger because of singletons.
- **Limits:** local sequence consistency alone cannot authenticate unique original event identities across windows.

Later event reconstruction and the release's all-split audit strengthen that last check using supplied cleaned records. Passing this stage does not establish baseline fidelity, raw-data rights or scientific superiority.
