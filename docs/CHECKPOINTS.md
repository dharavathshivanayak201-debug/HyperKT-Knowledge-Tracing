# Checkpoints and safe loading

Selected model binaries are packaged as release assets, while normal Git contains only code, aggregate results and hash inventories. Original uploaded ZIPs are not published wholesale because some contain learner data and nested input archives.

## Curated assets

| Asset | Selected states | Scope |
| --- | ---: | --- |
| `hyperkt-final-stage9e-checkpoints-v1.zip` | 15 | Five seeds each for DKT_matched, HyperKT_matched and HyperKT_gated; these are the frozen models evaluated in Stage 10. |
| `hyperkt-historical-stage4-to8a-checkpoints-v1.zip` | 8 | Stage 4 temporal/fusion, Stage 5 head, Stage 6 selected model and four Stage 8A variants. |
| `hyperkt-historical-stage8b-checkpoints-v1.zip` | 20 | Four early ablation variants across five seeds; validation-era states, not the final models. |
| `SHA256SUMS.txt` | n/a | Whole-asset SHA-256 hashes; per-state hashes appear in the public manifest. |

The complete [checkpoint-release-manifest.json](../provenance/checkpoint-release-manifest.json) records original checkpoint hashes, archive entries, lengths, top-level fields and restricted-load checks. Original selected state bytes are unchanged; optimizer/resume duplicates are excluded.

## Verify final states

Install the public audit requirements and CPU PyTorch, download the final checkpoint ZIP from the repository's release page, verify its hash, and extract it into a local ignored directory:

```bash
python -m pip install -r requirements-audit.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
sha256sum -c SHA256SUMS.txt --ignore-missing
unzip hyperkt-final-stage9e-checkpoints-v1.zip -d checkpoints/final
python scripts/verify_checkpoints.py --checkpoints checkpoints/final \
  --output analysis/local-checkpoint-verification.json
```

This strict-load path checks every final checkpoint against the original Stage 9G hashes, verifies finite tensors, and runs synthetic causal probes. It does not need private sequence arrays and does not perform training.

For inference with one final checkpoint:

```python
import json
from pathlib import Path
from hyperkt.loading import load_checkpoint, read_structure

root = Path(".")
manifest = json.loads((root / "provenance/checkpoint-release-manifest.json").read_text())
entry = next(x for x in manifest["checkpoints"]
             if x["asset"] == "hyperkt-final-stage9e-checkpoints-v1.zip"
             and x["entry"] == "HyperKT_matched_seed42/best.pt")
H = read_structure(root / "results/hyperkt_stage3b/hypergraph_repaired.npz")
model, metadata = load_checkpoint(
    root / "checkpoints/final/HyperKT_matched_seed42/best.pt",
    "HyperKT_matched", H, entry["sha256"])
```

Run that snippet with `PYTHONPATH=src`. Inputs are integer tensors `(batch, time)` for preceding skill IDs, preceding correctness and target skill IDs; `model(hs, hc, ts)` returns logits, to which `torch.sigmoid` gives probabilities. Valid skills are 0–150, responses 0/1 and length at most 200; mask padded positions out of evaluation.

## Historical model limitations

Use each original stage notebook's exact architecture and dependencies for historical states; the portable Stage 10 module is not a universal loader for all older checkpoints. Historical restricted deserialization and finite values do not amount to complete architecture verification or rerun inference.

Never bypass a restricted-load error by switching to `weights_only=False` on an unreviewed artifact. The final loader requires the expected hash before deserialization, and selected public states have been tested with `weights_only=True`; excluded resume files contain extra RNG/optimizer material and are not needed for inference.

## Numerical caveats

The release CPU rerun retains the saved CUDA metrics as historical primary results and reports per-model discrepancies separately. All HyperKT probability differences were below 3e-7; DKT reached about 0.00124723, so the report does not claim all variants agree to 1e-5 or that hardware-independent exact reproduction is established.
