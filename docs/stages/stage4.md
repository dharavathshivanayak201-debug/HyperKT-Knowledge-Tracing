# Stage 4: Early temporal encoder and graph fusion

The [Stage 4 notebook](../../notebooks/HyperKT_Stage4_Forgetting_Aware_Transformer.ipynb) compares a temporal-only predictor with a hypergraph-fusion predictor using repaired structure and fixed Stage 2 windows. Its attention decay uses positional distance rather than actual elapsed time.

## Evidence and use

- **Temporal-only:** best validation ROC-AUC is 0.756825778599 at epoch 6.
- **Fusion:** best validation ROC-AUC is 0.757776949969 at epoch 6.
- **Outputs:** [reports and histories](../../results/hyperkt_stage4/) are public; two selected historical state files are in the historical checkpoint release.
- **Scope:** this is an early single-seed validation comparison, not the final five-seed test or a controlled capacity-matched benchmark.

Do not strict-load these historical states with the final Stage 10 classes and assume equivalence. Use the original stage architecture, and distinguish this preliminary validation gain from the later negative final AUC comparison.
