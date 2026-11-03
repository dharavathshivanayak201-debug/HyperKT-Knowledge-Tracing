# Audit scope and research limitations

This release preserves evidence and documents checks performed against the supplied artifacts. Passing a file-integrity or reconstruction test does not establish originality, publication readiness, prospective protocol compliance or educational benefit.

## Checks actually performed

- **Input integrity:** all 45 entries in the original manifest are present across the source/results package and both checkpoint parts, with matching lengths and SHA-256 hashes; ZIP CRC checks passed during private extraction.
- **Notebook inspection:** 22 unchanged notebooks have no stored outputs; Python code compiles after removing single-line Colab magics, but the notebooks were not run end-to-end.
- **Private sequence reconstruction:** all six stored arrays were reproduced exactly for train, validation and test against supplied cleaned records, including chronological next-target mapping and disjoint learner splits.
- **Saved-output results:** all 15 final prediction files reproduce point metrics and five-seed summaries to 1e-12.
- **Bootstrap:** all 15 individual and 10 paired 1,000-resample learner-cluster intervals reproduce to 1e-12.
- **Final checkpoint loading:** all 15 actual `best.pt` hashes match the Stage 9G inventory; CPU loading uses `weights_only=True`, strict state matching, finite tensor checks and deterministic synthetic future-perturbation tests.
- **Frozen CPU inference:** all 15 models were evaluated on the supplied private test windows without retraining; HyperKT agrees closely, while DKT's retained numerical differences exceed 1e-5 for probabilities and some threshold-sensitive metrics.
- **Historical model release:** 28 earlier selected checkpoints also load under restricted weights-only deserialization and contain finite tensor trees with primitive metadata; their architecture-specific inference and training were not independently rerun.
- **Public checks:** inherited bytes, original notebook syntax, exported class AST identity, final table summaries, graph-file learner-key exclusions and synthetic model causality are covered by automated checks.

The [saved-output audit](../analysis/private-artifact-audit.json) intentionally retains `checkpoint_bytes_present: false` inside its saved-prediction subsection: that routine never loads model bytes. The later, separate [checkpoint verification](../analysis/checkpoint-verification.json) establishes byte verification and CPU inference; do not mistake the older routine's scope marker for a missing upload or silently relabel its checks.

## What is not established

- **Training reproduction:** no optimization run, hyperparameter search or full Colab pipeline execution was performed for this release.
- **Original test-access history:** the Stage 10 protocol states one-time held-out evaluation and no test-based selection; artifact consistency cannot independently authenticate whether the original authors accessed the test earlier.
- **Protocol freeze:** Stage 9G explicitly remains `CONDITIONAL_LOCK_PENDING_BASELINE_FIDELITY_REVIEW`; publication of the repository does not resolve official baseline fidelity.
- **Statistical generalization:** one source dataset, one learner split and one fixed pretrained structure limit generalization; five training seeds are not five independent cohorts.
- **Capacity fairness:** matched batch size and training budget do not match parameter counts or architectures.
- **Causal interpretation:** co-occurrence-derived graph structure is associational, not a validated causal prerequisite graph.
- **Time and uncertainty:** positional distance is not actual elapsed time, and the final model is sigmoid/BCE rather than a calibrated Beta posterior.
- **Human evaluation:** no expert-validation study, institutional deployment, adaptive-learning outcome experiment or journal acceptance is substantiated by the supplied artifacts.
- **Security/privacy certification:** limited content and file-type checks are not a comprehensive security audit or differential-privacy guarantee.

## Reproducibility levels

Public-table reconstruction and figure generation are available from a clone. Frozen checkpoint verification additionally needs a release ZIP; full CPU prediction comparisons and learner-bootstrap recomputation need privately obtained matching learner-level artifacts. Original training requires the datasets, intermediate packages, Colab dependencies and notebook workflow, and remains unrerun here.

Numerical differences in CPU-versus-CUDA DKT inference are explicitly recorded, including false agreement flags at 1e-5. Their cause was not isolated; the unchanged saved-prediction metrics remain the primary reported historical numbers rather than being silently replaced with CPU rerun values.
