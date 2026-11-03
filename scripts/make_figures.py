"""Regenerate clearly labelled figures using public aggregate tables only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"DKT_matched": "#334155", "HyperKT_matched": "#00796b", "HyperKT_gated": "#aa650d"}


def main():
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    data = pd.read_csv(ROOT / "results/hyperkt_stage10/per_seed_test_metrics.csv")
    order = list(COLORS)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for i, name in enumerate(order):
        values = data.loc[data.variant == name, "auc"].to_numpy()
        ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="o", color=COLORS[name], capsize=5, markersize=8)
        ax.scatter(i + np.linspace(-.09, .09, len(values)), values, s=22, alpha=.7, color=COLORS[name])
    ax.set(xticks=range(3), xticklabels=order, ylabel="Held-out test ROC-AUC",
           title="Final frozen evaluation: five seeds\nMean ± sample SD; points are individual seeds")
    ax.set_ylim(.739, .746)
    ax.grid(axis="y", alpha=.18)
    fig.savefig(out / "final_test_auc.png")
    plt.close(fig)
    pairs = pd.read_csv(ROOT / "results/hyperkt_stage10/paired_auc_comparisons.csv")
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    labels = []
    for i, row in enumerate(pairs.itertuples()):
        ax.plot([row.student_boot_ci_low, row.student_boot_ci_high], [i, i], color=COLORS[row.comparator], lw=2)
        ax.scatter(row.dkt_minus_comparator_auc, i, color=COLORS[row.comparator], s=35)
        labels.append(f"DKT − {row.comparator}, seed {row.seed}")
    ax.axvline(0, ls="--", lw=1, color="#64748b")
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Difference in test ROC-AUC",
           title="Within-seed paired whole-learner bootstrap\n95% percentile intervals; 1,000 resamples")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    fig.savefig(out / "paired_test_auc_intervals.png")
    plt.close(fig)
    for stage, title in (("hyperkt_stage8b", "Historical Stage 8B validation ablations"),
                         ("hyperkt_stage9e", "Stage 9E validation model selection")):
        filename = "all_results.csv"
        frame = pd.read_csv(ROOT / "results" / stage / filename)
        # The two historical result formats use different names for the AUC.
        group = "variant" if "variant" in frame.columns else "model"
        auc = next(k for k in ("auc", "val_auc", "best_val_auc") if k in frame.columns)
        summary = frame.groupby(group)[auc].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        ax.barh(summary.index, summary["mean"], xerr=summary["std"], color="#587b8a", capsize=4)
        ax.set_xlim(max(0, summary["mean"].min() - .015), summary["mean"].max() + .005)
        ax.set(xlabel="Validation ROC-AUC, mean ± sample SD", title=title + "\nNot held-out test results")
        fig.savefig(out / (stage + "_validation.png"))
        plt.close(fig)
    print("Regenerated four figures from public tables. No models trained and no learner records read.")


if __name__ == "__main__":
    main()
