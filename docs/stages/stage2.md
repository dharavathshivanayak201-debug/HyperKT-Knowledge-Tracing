# Stage 2: Fixed sequence construction

The objective is to produce next-response targets exactly once without target-answer leakage or silently losing short learners. The [fixed constructor](../../notebooks/HyperKT_Stage2_Sequence_Construction_FIXED.ipynb) uses cleaned Stage 1 records; the earlier [original version](../../notebooks/HyperKT_Stage2_Sequence_Construction.ipynb) is retained as history.

## Method and evidence

- **Alignment:** target indices 1 through T−1 are partitioned into blocks of at most 200, using the immediately preceding skills/responses as inputs.
- **Padding:** sequences are prefix-valid and right-padded; padded skill ID is 1 and masks define scored positions.
- **Counts:** train has 3,706 windows/222,562 targets, validation 445/23,858 and test 484/32,522.
- **Reconstruction:** the release independently reproduced all six arrays for all splits against cleaned records, including chronological target uniqueness conditional on that builder.

Single-interaction learners cannot produce a target; the test contains 402 eligible learners, not 417 evaluated learners. Model histories restart per window, and the public [Stage 2 report](../../results/hyperkt_stage2/stage2_report.json) does not include withheld learner sequences.
