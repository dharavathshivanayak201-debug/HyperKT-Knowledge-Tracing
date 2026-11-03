# Stage 9E: Controlled final-model training study

The [controlled-study notebook](../../notebooks/HyperKT_Stage9E_Controlled_Improvement_Colab.ipynb) trains DKT_matched, HyperKT_matched and HyperKT_gated across five seeds with matched batch size and training budget. These 15 selected states, not earlier checkpoints, feed the frozen Stage 10 evaluation.

## Configuration and artifacts

- **Budget:** batch 32, learning rate 0.0002, weight decay 0.0001, max 20 epochs, patience 5 and validation-based selection.
- **Architecture:** hidden size 128 and dropout 0.1; HyperKT uses two blocks/eight heads and fixed repaired assignments.
- **Capacity:** DKT has 190,231 parameters, original HyperKT 512,529 and gated HyperKT 512,530, so “matched” does not mean parameter-matched.
- **Evidence:** [all metrics, histories and protocol](../../results/hyperkt_stage9e/) remain unchanged; selected `best.pt` states are in the final release ZIP.

Optimizer/resume files are not needed for frozen inference and are excluded. The [recreated plot](../../figures/hyperkt_stage9e_validation.png) shows validation selection, while the final held-out outcome is documented at Stage 10.
