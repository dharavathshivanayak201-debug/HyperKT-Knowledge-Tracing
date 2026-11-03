# Stage 9A: Statistical analysis and draft selection

The [Stage 9A notebook](../../notebooks/HyperKT_Stage9A_Statistical_Analysis_Protocol_Lock.ipynb) summarizes Stage 8B validation runs and proposes the hypergraph-plus-decay architecture. Its lock file is explicitly a draft with `final_test_ready: false`.

## Evidence and blockers

- **Selection:** highest observed five-seed mean validation AUC among four candidate variants.
- **Outputs:** [verified result tables, paired-seed statistics and draft lock](../../results/hyperkt_stage9a/) preserve the rationale.
- **Historical blockers:** recovering selected checkpoint bytes, causal/padding inspection, cross-window event uniqueness and a predefined final baseline protocol.

Some blockers were investigated in later stages and in this release, but the old draft is not silently upgraded to a preregistration. Validation selection among explored models also does not make later claims of prospective untouched-test history independently verifiable.
