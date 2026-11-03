# Stage 5: Uncertainty-head experiment

The [uncertainty notebook](../../notebooks/HyperKT_Stage5_Uncertainty_Colab.ipynb) explores a Beta-style head and repeated dropout predictions using the early model backbone. This is an exploratory validation-stage diagnostic, not the uncertainty mechanism of the final Stage 10 predictor.

## Findings

- **Deterministic validation:** ROC-AUC 0.758522858528 for the selected epoch 1.
- **MC30 validation:** ROC-AUC 0.758921370317, with variance summarized in the [report](../../results/hyperkt_stage5/report.json).
- **Identifiability:** binary BCE constrains the predictive mean, not Beta concentration; the report explicitly rejects calibrated concentration claims.
- **Artifacts:** a reliability plot and aggregate metrics are public, while validation prediction rows are withheld and the head state is in the historical release.

MC dropout is an approximate diagnostic rather than proof of calibrated Bayesian uncertainty. The final model uses a sigmoid probability and BCE, not a Beta posterior.
