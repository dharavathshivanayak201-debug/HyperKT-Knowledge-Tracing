# Stage 8B: Five-seed validation ablations

The [repeated-seed notebook](../../notebooks/HyperKT_Stage8B_Five_Seed_Colab.ipynb) evaluates four variants across seeds 42, 123, 2024, 3407 and 7777. All runs use the same pretrained Stage 3B graph, so structure-learning variability is not represented.

## Outputs and reading

- **Variants:** transformer, transformer_decay, transformer_hypergraph and transformer_hypergraph_decay.
- **Protocol:** batch 16, hidden 128, two layers, eight heads, dropout 0.1, validation selection and patience 5 within a 20-epoch budget.
- **Evidence:** [20 per-seed metrics and histories](../../results/hyperkt_stage8b/), summary statistics and paired validation differences are public.
- **Weights:** all 20 selected states are a separate historical release asset; the original Drive ZIP's embedded input Stage 2 archive is excluded because it contains learner arrays.

This is a validation-era ablation study, not the final three-model Stage 9E/10 protocol. The [recreated plot](../../figures/hyperkt_stage8b_validation.png) is explicitly labelled validation.
