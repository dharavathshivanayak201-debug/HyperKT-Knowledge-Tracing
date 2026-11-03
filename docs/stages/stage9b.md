# Stage 9B: Historical checkpoint and sequence audit

The [Stage 9B notebook](../../notebooks/HyperKT_Stage9B_Checkpoint_Sequence_Audit.ipynb) audits the 20 Stage 8B states and their validation predictions. Its report records successful loading, causal probes, train/validation disjointness and near-exact saved-versus-recomputed metrics.

## Scope and limitations

- **Saved audit:** maximum metric delta is about 5.04e-8, as recorded in the [original report](../../results/hyperkt_stage9b/stage9b_report.json).
- **Sequence checks:** no duplicate identical windows were recorded, but original event identity uniqueness remained unresolved at this stage.
- **Padding:** no explicit key-padding mask is implemented; causal right-padded, prefix-valid windows are the supported layout.
- **Independent release check:** all historical selected state bytes now restricted-load with finite tensors, but Stage 9B inference itself was not independently rerun.

Keep the saved audit's claims separate from the release's later final-checkpoint CPU inference. Neither audit is a full training reproduction or official baseline certification.
