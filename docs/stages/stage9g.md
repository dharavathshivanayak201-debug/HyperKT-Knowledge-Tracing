# Stage 9G: Checkpoint verification and conditional freeze

The [Stage 9G notebook](../../notebooks/HyperKT_Stage9G_Checkpoint_Verification_Protocol_Freeze.ipynb) inventories 15 final checkpoints, checks validation behaviour and declares the intended final comparison. It selects original HyperKT, with independent DKT as primary comparator and gated HyperKT as a negative ablation.

## Lock status and verification

- **Historical status:** [protocol_lock.json](../../results/hyperkt_stage9g/protocol_lock.json) explicitly states `CONDITIONAL_LOCK_PENDING_BASELINE_FIDELITY_REVIEW`.
- **Inventory:** [checkpoint_verification.csv](../../results/hyperkt_stage9g/checkpoint_verification.csv) records the hashes used for final checkpoint-byte verification.
- **Release check:** all 15 actual Stage 9E selected states match those hashes, strict-load into AST-identical final classes and pass finite/causal probes.

Recovering missing bytes and validating their consistency does not resolve official DKT fidelity or establish a fully unconditional prospective freeze. The frozen final comparison should retain that qualification.
