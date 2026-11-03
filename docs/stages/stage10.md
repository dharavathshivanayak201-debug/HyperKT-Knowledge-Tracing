# Stage 10: Frozen held-out test evaluation

The [final evaluation notebook](../../notebooks/HyperKT_Stage10_Frozen_Heldout_Test_Evaluation_Colab.ipynb) applies 15 frozen Stage 9E states to the same learner-disjoint test split. Its saved protocol states no retraining or test-based model selection, but artifacts cannot independently authenticate the original access history.

## Final evidence

- **Cohort:** 402 eligible learners, 484 windows and 32,522 next-response targets.
- **AUC means:** DKT 0.744119363463, original HyperKT 0.741595208243 and gated HyperKT 0.741152939946.
- **Uncertainty:** 1,000-resample whole-learner bootstrap intervals per checkpoint and paired DKT-minus-comparator comparisons within seed.
- **Independent checks:** saved metrics/intervals reproduce to 1e-12; all final hashes match and all models underwent frozen CPU inference, with DKT numerical differences explicitly retained.

The [full result tables and original plot](../../results/hyperkt_stage10/) are public, but learner-linked prediction rows are withheld. See [results and interpretation](../RESULTS.md) rather than quoting unsupported manuscript AUC around 0.810 or claiming HyperKT discrimination superiority.
