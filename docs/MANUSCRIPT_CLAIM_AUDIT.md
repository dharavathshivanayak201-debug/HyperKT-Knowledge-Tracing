# Historical manuscript claim audit

The three supplied manuscript drafts are preserved byte-for-byte under `archive/manuscript_drafts/`, with an adjacent warning. None is a validated final paper, including the file whose name says `CODE_ALIGNED`.

| Claim or framing in supplied drafts | Evidence-supported treatment |
| --- | --- |
| ROC-AUC around 0.810 or state-of-the-art HyperKT performance | Unsupported by the final artifacts: HyperKT_matched mean test AUC is 0.741595, versus DKT_matched 0.744119. |
| Validation across multiple named datasets | Only one ASSISTments2009 source is supplied and audited. Additional dataset claims require actual code, splits, checkpoints and results. |
| Interpretable causal skill prerequisites | Training-derived co-occurrence and learned memberships do not establish causal prerequisites or expert validity. |
| Calibrated Bayesian/Beta uncertainty | Early Beta concentration is not identifiable from binary BCE; MC dropout is an approximate diagnostic. Final models use sigmoid/BCE. |
| Elapsed-time forgetting | The implemented decay uses sequence-position distance, not actual elapsed-time intervals. |
| All hypergraph embeddings frozen across seeds | Assignments are fixed; edge embeddings and fusion are learned during model training. |
| Strong fine-tuning gain | Stage 6 selected epoch 0, preserving the initial validation result. |
| HyperKT superior to the DKT comparator | DKT has higher observed final AUC in each seed; F1 and ROC-AUC must not be conflated, and paired uncertainty is conditional. |
| Official published baseline reproduction | Implementations are independent and not certified equivalent to official code; the Stage 9G protocol lock is conditional. |
| Completed expert validation, broad deployment or proven teaching benefit | No corresponding study artifacts are supplied. Do not claim these outcomes. |
| “Q1 ready” or publication-ready filename | A filename is not evidence of peer review, journal acceptance or scientific adequacy. |

## Supported replacement framing

HyperKT can be described as a hypergraph-augmented causal sequence model evaluated in a single-source, learner-disjoint knowledge-tracing study, with stage-wise diagnostics and transparent negative findings. The final original HyperKT has slightly higher thresholded F1 but lower observed ROC-AUC than the independent DKT baseline; gated fusion does not improve discrimination.

Any manuscript revision should distinguish early architecture proposals from the final implemented model, graph-edge validation from learner-response prediction, and historical protocol assertions from independently checked artifact consistency. This repository does not silently rewrite the supplied manuscripts or claim that repository publication resolves their scientific problems.
