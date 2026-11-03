# Publication and verification workflow

This repository is the Objective 1 research release under the requested owner and name. Code and aggregate evidence belong in Git, while the three curated model-only checkpoint ZIPs belong in a versioned GitHub release.

## Publication history

The first publication attempt stopped before upload because GitHub requested owner re-verification and Comet disconnected. The owner subsequently completed GitHub's own email verification, allowing the already approved temporary repository-only uploader invitation to be accepted.

Authentication codes are entered only on GitHub, never in research files or chat. The temporary upload account is an operational collaborator, not an inferred research author.

## Upload and verification procedure

- **Access:** add `raghudhumpatiprojects-pixel` to `dharavathshivanayak201-debug/HyperKT-Knowledge-Tracing` only, accept the invitation through the existing CLI and use that temporary write access.
- **Git upload:** preserve the existing initial commit, check commit identity, push the reviewed tree and inspect the resulting public files.
- **Release:** publish the three curated model-only ZIPs and `SHA256SUMS.txt` under tag `v1.0.0`; never upload the original outer ZIPs, learner data or resume states.
- **Verification:** fresh-clone the remote, run public checks and tests, inspect both CI jobs and verify downloaded release-asset hashes.
- **Cleanup:** remove the temporary collaborator, leaving the owner's repository intact; share the actual repository/release URLs only after checking their contents.

No thesis-wide research home is being created, and the Objective 3 and Navier–Stokes repositories remain outside this work. Local tests do not establish remote CI success or release-asset availability; check the actual public repository, Actions results and downloaded asset hashes before reporting completion.
