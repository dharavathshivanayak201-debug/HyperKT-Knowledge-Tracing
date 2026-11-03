"""Audit HyperKT source/results without training, torch.load, or pickle.

Inputs are privately retained extracted source/nested result folders. Only
aggregate statistics are written; original learner IDs never enter output.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

VARIANTS = ["DKT_matched", "HyperKT_matched", "HyperKT_gated"]
SEEDS = [42, 123, 2024, 3407, 7777]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, raw):
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(raw, dtype=np.float64), 1e-7, 1-1e-7)
    positive = y.sum()
    negative = len(y) - positive
    auc = (rankdata(p)[y == 1].sum() - positive*(positive+1)/2) / (positive*negative)
    prediction = p >= .5
    tp = np.sum(prediction & (y == 1))
    precision = tp / prediction.sum() if prediction.sum() else 0.
    recall = tp / positive
    bins = np.minimum(np.digitize(p, np.linspace(0, 1, 16)[1:-1]), 14)
    ece = sum(np.mean(bins == i) * abs(np.mean(y[bins == i]) - np.mean(p[bins == i]))
              for i in range(15) if np.any(bins == i))
    return {"auc": float(auc), "nll": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            "brier": float(np.mean((p-y)**2)),
            "f1": float(2*tp/(prediction.sum()+positive)),
            "accuracy": float(np.mean(prediction == y)),
            "precision": float(precision), "recall": float(recall),
            "ece15": float(ece), "n": int(len(y))}


def prepared_weighted_auc(y, p, cluster_index):
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    return y[order], cluster_index[order], starts


def weighted_auc(prepared, multiplicities):
    y, cluster_index, starts = prepared
    weight = multiplicities[cluster_index]
    pos = np.add.reduceat(weight*y, starts)
    neg = np.add.reduceat(weight*(1-y), starts)
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg)) / (pos.sum()*neg.sum()))


def audit_source(source):
    original = json.loads((source / "SHA256_MANIFEST.json").read_text())
    present, missing = [], []
    for row in original["files"]:
        path = source / row["group"] / row["name"]
        if not path.exists():
            missing.append(row)
            continue
        assert path.stat().st_size == row["bytes"], path.name
        assert digest(path) == row["sha256"], path.name
        present.append(row)
    notebooks = list((source / "notebooks").glob("*.ipynb"))
    for path in notebooks:
        notebook = json.loads(path.read_text())
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert not cell.get("outputs"), path.name
            code = "\n".join(line for line in "".join(cell["source"]).splitlines()
                             if not line.lstrip().startswith(("!", "%")))
            compile(code, f"{path.name}:cell{i}", "exec")
    return {"manifest_entries": len(original["files"]), "present_files_verified": len(present),
            "intentionally_separate_missing_artifacts": missing,
            "notebooks": len(notebooks), "stored_notebook_outputs": 0,
            "notebook_python_syntax": "Pass after omitting single-line Colab magics; not executed."}


def audit_sequences(nested):
    one = nested / "hyperkt_stage1"
    two = nested / "hyperkt_stage2(2)"
    vocab = json.loads((one / "skill_vocab.json").read_text())
    manifest = json.loads((one / "manifest.json").read_text())
    splits, users, flat_test = {}, {}, None
    for split in ("train", "val", "test"):
        frame = pd.read_parquet(one / f"{split}_interactions.parquet")
        frame = frame.sort_values(["user", "order"], kind="mergesort")
        with np.load(two / f"{split}_sequences.npz", allow_pickle=False) as archive:
            saved = {k: archive[k] for k in archive.files}
        ids = []
        values = {k: [] for k in ("history_skill", "history_correct", "target_skill", "target_correct", "mask")}
        target_events = []
        for user, group in frame.groupby("user", sort=False):
            skills = np.array([vocab.get(str(x).strip(), 0) for x in group.skill], dtype=np.int32)
            answers = group.correct.to_numpy(dtype=np.int8)
            rows = group["_row"].to_numpy()
            for start in range(1, len(group), 200):
                end = min(start+200, len(group))
                length = end-start
                hs, ts = np.full(200, 1, dtype=np.int32), np.full(200, 1, dtype=np.int32)
                hc, tc = np.zeros(200, dtype=np.int8), np.zeros(200, dtype=np.int8)
                mask = np.zeros(200, dtype=bool)
                hs[:length], hc[:length] = skills[start-1:end-1], answers[start-1:end-1]
                ts[:length], tc[:length], mask[:length] = skills[start:end], answers[start:end], True
                for key, val in zip(values, (hs, hc, ts, tc, mask)):
                    values[key].append(val)
                ids.append(str(user))
                target_events.extend(rows[start:end].tolist())
        for key in values:
            assert np.array_equal(np.stack(values[key]), saved[key]), (split, key)
        assert np.array_equal(np.asarray(ids), saved["user_id"]), split
        assert len(target_events) == len(set(target_events)) == int(saved["mask"].sum()), split
        users[split] = set(frame.user.astype(str))
        assert users[split] == set(manifest["splits"][split])
        counts = frame.groupby("user").size()
        splits[split] = {
            "interactions": len(frame), "raw_split_students": len(users[split]),
            "students_with_predictions": len(set(ids)), "single_interaction_students": int((counts == 1).sum()),
            "windows": len(ids), "targets": int(saved["mask"].sum()),
            "sequence_reconstruction": "Exact: all six stored arrays",
            "reconstructed_target_row_uniqueness": True,
            "unknown_targets": int(((saved["target_skill"] == 0) & saved["mask"]).sum()),
        }
        if split == "test":
            flat_test = (saved["target_correct"][saved["mask"]],
                         np.repeat(saved["user_id"], saved["mask"].sum(axis=1)))
    assert all(not (users[a] & users[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    return {
        "source_raw_csv_sha256_recorded_not_redownloaded": manifest["source_sha256"],
        "raw_rows_recorded": manifest["raw_rows"], "clean_rows_checked": sum(s["interactions"] for s in splits.values()),
        "students": sum(len(u) for u in users.values()), "skills_in_training_vocabulary": len(vocab),
        "learner_disjoint": True, "splits": splits,
        "event_scope": "New independent reconstruction against supplied Stage 1 cleaned records for all three splits, including test. Original Stage 10 did not perform this reconstruction. Numeric arrays omit event IDs; mapping is conditional on the declared chronological builder and supplied source records.",
    }, flat_test


def audit_predictions(nested, test_reference, replicates):
    directory = nested / "hyperkt_stage10"
    saved = pd.read_csv(directory / "per_seed_test_metrics.csv")
    intervals = pd.read_csv(directory / "student_cluster_auc_intervals.csv")
    paired = pd.read_csv(directory / "paired_auc_comparisons.csv")
    summary = pd.read_csv(directory / "five_seed_test_summary.csv", header=[0, 1], index_col=0)
    checkpoint_rows = pd.read_csv(nested / "hyperkt_stage9g/checkpoint_verification.csv")
    predictions, fresh = {}, []
    assert len(saved) == 15 and not saved.duplicated(["variant", "seed"]).any()
    for variant in VARIANTS:
        for seed in SEEDS:
            path = directory / f"predictions_{variant}_seed{seed}.npz"
            with np.load(path, allow_pickle=False) as archive:
                y, p, user = archive["y"], archive["p"], archive["user_id"]
            assert np.array_equal(y, test_reference[0])
            assert np.array_equal(user, test_reference[1])
            assert len(y) == 32522 and len(np.unique(user)) == 402
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            row = saved[(saved.variant == variant) & (saved.seed == seed)].iloc[0]
            chk = checkpoint_rows[(checkpoint_rows.variant == variant) & (checkpoint_rows.seed == seed)].iloc[0]
            assert row.checkpoint_sha256 == chk.sha256
            computed = metrics(y, p)
            for key, value in computed.items():
                assert np.isclose(value, row[key], rtol=0, atol=1e-12), (variant, seed, key, value, row[key])
            fresh.append({"variant": variant, "seed": seed, "checkpoint_sha256": row.checkpoint_sha256, **computed})
            predictions[variant, seed] = y, p.astype(float), user
    table = pd.DataFrame(fresh)
    means = {}
    for variant in VARIANTS:
        subset = table[table.variant == variant]
        means[variant] = {}
        for column in ("auc", "nll", "brier", "f1", "ece15", "accuracy", "precision", "recall"):
            mean, sd = subset[column].mean(), subset[column].std(ddof=1)
            assert np.isclose(mean, summary.loc[variant, (column, "mean")], rtol=0, atol=1e-12)
            assert np.isclose(sd, summary.loc[variant, (column, "std")], rtol=0, atol=1e-12)
            means[variant][column] = {"mean": float(mean), "sample_sd": float(sd)}
    checks, comparisons = [], []
    if replicates:
        assert replicates == 1000, "Historical comparison requires 1,000 exact RNG resamples."
        for seed in SEEDS:
            y, _, user = predictions["DKT_matched", seed]
            clusters, inv = np.unique(user, return_inverse=True)
            prepared = {v: prepared_weighted_auc(y, predictions[v, seed][1], inv) for v in VARIANTS}
            for paired_mode, offset in ((False, 10000), (True, 19000)):
                rng = np.random.default_rng(seed + offset)
                values = {variant: [] for variant in VARIANTS}
                for _ in range(replicates):
                    chosen = rng.integers(0, len(clusters), size=len(clusters))
                    multiplicity = np.bincount(chosen, minlength=len(clusters))
                    for variant in VARIANTS:
                        values[variant].append(weighted_auc(prepared[variant], multiplicity))
                if not paired_mode:
                    for variant in VARIANTS:
                        low, high = np.quantile(values[variant], [.025, .975])
                        row = intervals[(intervals.variant == variant) & (intervals.seed == seed)].iloc[0]
                        assert np.allclose([low, high], [row.auc_ci_low, row.auc_ci_high], rtol=0, atol=1e-12)
                        checks.append({"variant": variant, "seed": seed, "ci95": [float(low), float(high)]})
                else:
                    for variant in VARIANTS[1:]:
                        delta = np.asarray(values["DKT_matched"]) - np.asarray(values[variant])
                        low, high = np.quantile(delta, [.025, .975])
                        row = paired[(paired.comparator == variant) & (paired.seed == seed)].iloc[0]
                        assert np.allclose([low, high], [row.student_boot_ci_low, row.student_boot_ci_high], rtol=0, atol=1e-12)
                        comparisons.append({"comparator": variant, "seed": seed,
                                            "delta_auc": float(row.dkt_minus_comparator_auc),
                                            "ci95": [float(low), float(high)]})
    return {"point_metrics": fresh, "five_seed_summary": means,
            "prediction_arrays": 15, "aligned_predictions_each": 32522, "learners_each": 402,
            "positive_prevalence": float(test_reference[0].mean()),
            "checkpoint_hashes_match_stage9g_table": True,
            "checkpoint_bytes_present": False,
            "bootstrap_replicates_per_comparison": replicates,
            "individual_auc_intervals": checks, "paired_auc_intervals": comparisons,
            "bootstrap_method": "Whole-learner sampling; seed + 10000 for individual intervals and seed + 19000 for paired intervals. Weighted tied-score AUC is algebraically equivalent to repeating sampled learner rows; saved percentile endpoints checked to 1e-12.",
            "limitations": "Saved-output reconstruction, not model retraining, checkpoint-byte verification, official baseline certification, or independent authentication that the original test was accessed only once."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--nested", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    source = audit_source(args.source)
    sequence, test = audit_sequences(args.nested)
    predictions = audit_predictions(args.nested, test, args.bootstrap)
    output = {"source": source, "sequences": sequence, "test_results": predictions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Verified", source["present_files_verified"], "manifested input files;",
          source["notebooks"], "notebook syntax checks; all train/validation/test sequence arrays; 15 prediction files.")
    print("Bootstrap checks:", len(predictions["individual_auc_intervals"]), "individual and",
          len(predictions["paired_auc_intervals"]), "paired intervals.")
    print("Only aggregate output written to", args.output)


if __name__ == "__main__":
    main()
