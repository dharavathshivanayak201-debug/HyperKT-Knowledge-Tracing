# Stage 3B: Graph validation and correction

Stage 3B diagnoses the near-constant original reconstruction and evaluates a repaired graph-learning procedure against an explicit structural baseline. The supplied versions are [validation/correction](../../notebooks/HyperKT_Stage3B_Hypergraph_Validation_Correction.ipynb) and [GPU-safe v2](../../notebooks/HyperKT_Stage3B_GPU_SAFE_v2.ipynb).

## Outputs and limitations

- **Task:** positive/negative edge prediction on held-out pairs of a graph derived from training learner records, not the learner-held-out next-response task.
- **Repaired result:** edge test ROC-AUC is 0.867122645453; the degree-product baseline is higher at 0.886975788631.
- **Artifact:** [report.json](../../results/hyperkt_stage3b/report.json) and `hypergraph_repaired.npz` preserve the diagnostic and selected structure.
- **Reuse:** final models keep these assignments fixed across training seeds, but learn edge embeddings and fusion parameters.

The graph captures associations and has no supplied expert validation or causal intervention evidence. Its edge AUC must not be promoted as HyperKT's final knowledge-tracing ROC-AUC.
