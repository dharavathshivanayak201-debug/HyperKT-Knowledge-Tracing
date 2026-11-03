# Data provenance and public-release boundaries

The supplied experiments concern one source described as ASSISTments2009. The originating project describes ASSIST 2009-10 as student interactions from the 2009–2010 school year and recommends the citation Feng, Heffernan and Koedinger (2009), “Addressing the assessment challenge in an Intelligent Tutoring System that tutors as it assesses,” User Modeling and User-Adapted Interaction 19, 243–266 ([ASSISTments dataset page](https://www.etrialstestbed.org/data-sets/publicly-available-data-sets)).

## Source and preprocessing

The recorded raw CSV SHA-256 is `162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da`, with 346,860 recorded raw rows. That raw file was not independently redownloaded or rehashed for this release; the supplied cleaned Stage 1 records were checked directly.

Stage 1 selects user, skill, binary correctness and order fields; removes missing/invalid required values; rejects ambiguous comma/`~~` multi-skill rows rather than silently treating them as one skill; and uses a stable chronological sort. It splits learners 80/10/10 with seed 42 before fitting the skill vocabulary or constructing co-occurrence. The 149 training skills use IDs 2–150, unknown is 0 and padding is 1; training-derived co-occurrence uses windows of five interactions.

| Split | Interactions | Raw split learners | Eligible learners | Windows | Scored targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 225,892 | 3,330 | 3,222 | 3,706 | 222,562 |
| Validation | 24,274 | 416 | 403 | 445 | 23,858 |
| Test | 32,939 | 417 | 402 | 484 | 32,522 |

There are 283,105 cleaned interactions and 4,163 learners in total. Singleton learners generate no next-response pair; that explains the raw-versus-eligible counts, without dropping an evaluated learner from the denominator.

## Sequence construction and leakage checks

The fixed Stage 2 builder partitions target indices 1 through T−1 into non-overlapping blocks of at most 200 per learner, aligns each target with its immediately preceding interaction and right-pads unused positions. Model state restarts for each window; a learner's history before that window is not retained as a full-history memory.

The release audit reconstructed all six arrays for every split exactly from the supplied cleaned records and confirmed learner-disjoint splits, target counts, padding, label alignment and uniqueness under that declared chronological builder. The saved sequence arrays do not contain original event IDs, so the reconstruction's interpretation remains conditional on the supplied cleaned records and builder; it is not authentication of a raw server log.

## Included versus excluded

- **Included:** unchanged code notebooks without stored outputs; aggregate metrics and histories; aggregate graph structures and vocabulary; selected finite model states; hash inventories, aggregate audit reports and historical drafts with warnings.
- **Excluded:** raw and cleaned interaction rows, source learner IDs, exact split-ID manifests, learner-level sequence NPZ files, learner coverage tables, prediction outcomes/probabilities linked to learners, embedded Stage 2 input ZIPs and original outer uploaded archives.
- **Resume states:** Stage 9E `last_state.pt` files contain optimizer/resume/RNG material and are excluded; selected `best.pt` checkpoints are retained separately.
- **Privacy scope:** removing learner rows reduces direct disclosure but does not certify anonymization, differential privacy, or absence of membership-inference risk in trained weights and aggregate structures.

Every excluded nested member is recorded by path, byte count, SHA-256 and reason in [release-exclusions.json](../provenance/release-exclusions.json). That inventory records file-level provenance, not learner values.

## Data access and rights

Obtain data through the originating project's download instructions and review the terms applicable to that exact version. The retrieved project page did not establish redistribution permission for this supplied CSV; licenses stated for newer datasets on the same page must not be transferred to ASSISTments2009 ([ASSISTments dataset page](https://www.etrialstestbed.org/data-sets/publicly-available-data-sets)).

Because the exact supplied learner-level files are withheld, a public clone alone cannot reproduce the private sequence, prediction or bootstrap audit. A newly obtained dataset may differ in version, row filtering or column encoding; compare the recorded source hash and resulting aggregate counts before claiming exact reproduction.
