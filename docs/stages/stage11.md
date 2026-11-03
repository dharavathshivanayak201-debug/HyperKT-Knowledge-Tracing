# Stage 11: Publication-analysis notebook

The [supplied publication-analysis notebook](../../notebooks/HyperKT_Stage11_Publication_Analysis_Colab.ipynb) is intended to recheck Stage 10 prediction/metric alignment and generate paper-facing summaries. No separate Stage 11 output archive was supplied, and this notebook was not executed end-to-end for the release.

## Release treatment

- **Preservation:** the notebook remains unchanged and passes the limited Python syntax inspection.
- **Independent replacement checks:** release utilities recreate final tables/figures and recompute the private saved-output statistics without claiming the Stage 11 notebook itself produced them.
- **Wording correction:** statements that all hypergraph embeddings were fixed should instead distinguish fixed assignments from learnable edge embeddings and fusion.
- **Manuscripts:** even the draft labelled `CODE_ALIGNED` retains unsupported claims and is preserved only with an explicit warning.

Use [the claim audit](../MANUSCRIPT_CLAIM_AUDIT.md) and verified [final interpretation](../RESULTS.md) when drafting a revised paper. This stage does not certify journal readiness or turn a negative AUC comparison into a successful superiority study.
