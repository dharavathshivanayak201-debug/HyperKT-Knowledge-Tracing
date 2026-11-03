# Stage 8A: Causal checks and controlled ablations

The [Stage 8A notebook](../../notebooks/HyperKT_Stage8A_Causal_Audit_Controlled_Ablations.ipynb) compares temporal models with/without positional decay and repaired hypergraph features. It also probes whether changing future inputs affects earlier predictions.

## Validation evidence

- **Transformer without decay:** best validation ROC-AUC 0.756625347691.
- **Transformer with decay:** 0.756825755331.
- **Hypergraph without decay:** 0.757675297741.
- **Hypergraph with decay:** 0.757776946091.

The [report and histories](../../results/hyperkt_stage8a/) are preserved, with four states in the historical release. These small single-seed validation differences motivate repeated-seed diagnostics but do not prove a general benefit, final test improvement or causal graph interpretation.
