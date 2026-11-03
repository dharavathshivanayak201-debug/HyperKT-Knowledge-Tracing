# Stage 9F: Event-integrity reconstruction

The [event-integrity notebook](../../notebooks/HyperKT_Stage9F_Event_Integrity_Protocol_Lock.ipynb) checks sequence targets against cleaned Stage 1 records for train and validation. It addresses a limitation that skill/response arrays alone cannot prove original event identity uniqueness.

## Evidence and updated scope

- **Saved outputs:** [event_audit.json and audit_issues.json](../../results/hyperkt_stage9f/) record the historical train/validation reconstruction and no reported issues.
- **Public boundary:** the learner-level coverage CSV is withheld because it links source learners and target coverage.
- **New release audit:** exact reconstruction was additionally performed for all three splits, including test, and is recorded separately in [the aggregate audit](../../analysis/private-artifact-audit.json).

The new check must not be retroactively attributed to original Stage 10 execution. Because stored numeric sequences omit event IDs, uniqueness is conditional on the supplied cleaned records and declared stable chronological builder, not authentication of raw system logs.
