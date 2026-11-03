"""Verify final checkpoint bytes and optionally rerun frozen CPU inference.

Requires separately downloaded release weights. Private sequences and prediction
rows are optional, never copied into aggregate reports. No training is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hyperkt.loading import load_checkpoint, read_structure
from audit_private_artifacts import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--sequences", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/checkpoint-verification.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    structure = read_structure(ROOT / "results/hyperkt_stage3b/hypergraph_repaired.npz")
    inventory = pd.read_csv(ROOT / "results/hyperkt_stage9g/checkpoint_verification.csv")
    saved_metrics = pd.read_csv(ROOT / "results/hyperkt_stage10/per_seed_test_metrics.csv").set_index(["variant", "seed"])
    sequences = None
    if args.sequences:
        with np.load(args.sequences, allow_pickle=False) as data:
            sequences = {k: data[k] for k in data.files}
    rows = []
    start = time.monotonic()
    for row in inventory.itertuples(index=False):
        path = args.checkpoints / f"{row.variant}_seed{row.seed}" / "best.pt"
        model, checkpoint = load_checkpoint(path, row.variant, structure, row.sha256)
        # A new, deterministic synthetic future-perturbation check, not a new
        # evaluation cohort or evidence of learning effectiveness.
        generator = torch.Generator().manual_seed(4242)
        hs = torch.randint(2, 151, (2, 32), generator=generator)
        hc = torch.randint(0, 2, (2, 32), generator=generator)
        ts = torch.randint(2, 151, (2, 32), generator=generator)
        changed_hs, changed_hc, changed_ts = hs.clone(), hc.clone(), ts.clone()
        changed_hs[:, 16:] = 2
        changed_hc[:, 16:] = 1 - hc[:, 16:]
        changed_ts[:, 16:] = 3
        with torch.inference_mode():
            baseline = model(hs, hc, ts)
            changed = model(changed_hs, changed_hc, changed_ts)
        delta = float((baseline[:, :16] - changed[:, :16]).abs().max())
        assert delta < 1e-6
        output = {"variant": row.variant, "seed": int(row.seed), "checkpoint_sha256": row.sha256,
                  "strict_state_load": True, "all_tensors_finite": True,
                  "parameters": sum(v.numel() for v in model.parameters()), "causal_probe_max_delta": delta}
        if sequences:
            probs, truth = [], []
            with torch.inference_mode():
                for offset in range(0, len(sequences["mask"]), 16):
                    stop = offset + 16
                    tensors = [torch.as_tensor(sequences[k][offset:stop], dtype=torch.long)
                               for k in ("history_skill", "history_correct", "target_skill")]
                    p = torch.sigmoid(model(*tensors)).numpy()
                    m = sequences["mask"][offset:stop]
                    probs.append(p[m])
                    truth.append(sequences["target_correct"][offset:stop][m])
            probabilities, y = np.concatenate(probs), np.concatenate(truth)
            fresh = metrics(y, probabilities)
            saved = saved_metrics.loc[(row.variant, row.seed)]
            metric_delta = max(abs(fresh[key] - saved[key]) for key in ("auc", "nll", "brier", "f1", "ece15", "accuracy"))
            output.update({"frozen_cpu_inference_metrics": fresh, "max_metric_difference_vs_saved_cuda": float(metric_delta)})
            if args.predictions:
                with np.load(args.predictions / f"predictions_{row.variant}_seed{row.seed}.npz", allow_pickle=False) as data:
                    assert np.array_equal(y, data["y"])
                    probability_delta = float(np.max(np.abs(probabilities.astype(float) - data["p"].astype(float))))
                    output["max_probability_difference_vs_saved_cuda"] = probability_delta
                    output["probabilities_close_at_1e_5"] = probability_delta < 1e-5
            print("inference differences", {key: float(fresh[key] - saved[key]) for key in ("auc", "nll", "brier", "f1", "ece15", "accuracy")}, output.get("max_probability_difference_vs_saved_cuda"), flush=True)
            output["all_metrics_close_at_1e_5"] = bool(metric_delta < 1e-5)
        rows.append(output)
        print(row.variant, row.seed, "verified", f"elapsed {time.monotonic()-start:.1f}s", flush=True)
    result = {"torch_version": torch.__version__, "device": "CPU", "threads": args.threads,
              "checkpoints": rows, "count": len(rows),
              "frozen_test_inference_rerun": sequences is not None,
              "training_performed": False, "elapsed_seconds": time.monotonic()-start,
              "scope": "Exact byte hashes, weights_only loading, strict state compatibility, finite tensors and a synthetic causal probe. Optional inference uses supplied private test windows; aggregate output contains no learner rows. CPU/CUDA agreement at 1e-5 is reported per checkpoint, not forced by widening tolerances. False agreement flags are retained and require interpretation. No retraining or hyperparameter selection."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Wrote aggregate verification:", args.output)


if __name__ == "__main__":
    main()
