# Stage 9C: Independent baseline experiments

The [baseline notebook](../../notebooks/HyperKT_Stage9C_Five_Seed_Baselines_Colab.ipynb) implements DKT, DKVMN, SAKT and an AKT-inspired comparator and runs five-seed validation experiments. These are independently written implementations, not certified reproductions of the original authors' code.

## Artifacts and interpretation

- **Evidence:** [per-run metrics, histories, summaries and validation plot](../../results/hyperkt_stage9c/) are preserved.
- **Comparison:** the observed validation gap motivates the subsequent fairness audit rather than a claim that HyperKT wins.
- **Availability:** no Stage 9C model states were supplied in the three input packages; the repository does not fabricate or reconstruct missing trained weights.

Architecture labels alone do not establish equivalence to published implementations, tuning budgets or benchmark preprocessing. Stage 9E later narrows the final comparison to DKT and two HyperKT variants with matched batch size and training budget.
