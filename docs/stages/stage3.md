# Stage 3: Original differentiable hypergraph

This stage asks whether training-derived skill co-occurrence can be represented by learned soft skill-to-hyperedge memberships. The [original notebook](../../notebooks/HyperKT_Stage3_Differentiable_Hypergraph.ipynb) fits 32 hyperedges for 149 skills and exports memberships, histories and diagnostic plots.

## Findings and interpretation

- **Artifact:** [original graph report](../../results/hyperkt_stage3/stage3_report.json) records 500 epochs and an off-diagonal reconstruction MSE around 0.000162803.
- **Failure diagnostic:** later Stage 3B analysis finds this reconstruction much worse than a constant baseline and predictions nearly constant.
- **Preservation:** [loss and membership plots](../../results/hyperkt_stage3/) and learned arrays are retained unchanged as evidence of the early approach.

This is not a successful causal prerequisite discovery result or a learner-response test score. The final knowledge-tracing models instead use the repaired Stage 3B structure.
