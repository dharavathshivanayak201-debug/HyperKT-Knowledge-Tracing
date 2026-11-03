# Stage 6: End-to-end fine-tuning attempt

The [fine-tuning notebook](../../notebooks/HyperKT_Stage6_End_to_End_Fine_Tuning.ipynb) tests whether further end-to-end optimization improves the early uncertainty-stage model. Its selection criterion is deterministic validation ROC-AUC, with dropout summaries treated separately.

## Outcome and artifacts

- **Selected epoch:** 0, meaning no fine-tuned candidate improved the initial model under the criterion.
- **Validation result:** the selected deterministic ROC-AUC remains 0.758522858528.
- **Structure:** hypergraph assignments remain fixed while fusion-related weights may be optimized.
- **Evidence:** [report, history and validation curve](../../results/hyperkt_stage6/) remain public; the selected historical checkpoint is separately packaged.

This negative outcome is not a fine-tuning gain or a held-out test result. The dropout and concentration caveats from Stage 5 continue to apply.
