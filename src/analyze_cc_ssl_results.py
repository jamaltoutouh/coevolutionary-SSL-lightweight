#!/usr/bin/env python3
"""
Analysis helper for CC-SSL OpenML experiments.

Reads the CSV produced by cc_ssl_openml.py and generates:
- summary tables (by dataset and by labeled fraction)
- average ranks across datasets
- Wilcoxon signed-rank tests (CC-SSL vs best baseline) per labeled fraction (optional)
- a few plots (macro-F1 vs labeled fraction, boxplots)

Usage:
  python analyze_cc_ssl_results.py --csv results_ccssl_openml.csv --out_dir ./analysis_out
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="analysis_out")
    ap.add_argument("--method_main", type=str, default="CC-SSL (coevolved)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["macroF1_test"])

    # Remove placeholder rows like "Label Spreading (skipped)"
    df = df[~df["method"].str.contains("skipped", case=False, na=False)].copy()

    # Aggregate over seeds: dataset x labeled_frac x method
    agg = (
        df.groupby(["dataset_id", "dataset", "labeled_frac", "method"])["macroF1_test"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    agg.to_csv(os.path.join(args.out_dir, "by_dataset_summary.csv"), index=False)

    # Overall aggregate (all datasets pooled): labeled_frac x method
    overall = (
        df.groupby(["labeled_frac", "method"])["macroF1_test"]
          .agg(["mean", "std", "count"])
          .reset_index()
          .sort_values(["labeled_frac", "mean"], ascending=[True, False])
    )
    overall.to_csv(os.path.join(args.out_dir, "overall_summary.csv"), index=False)

    # Average ranks across datasets (per labeled fraction)
    ranks_rows = []
    for frac, g in agg.groupby("labeled_frac"):
        piv = g.pivot_table(index=["dataset_id", "dataset"], columns="method", values="mean")
        # Higher is better => rank 1 = best
        ranks = piv.rank(axis=1, ascending=False, method="average")
        mean_ranks = ranks.mean(axis=0).sort_values()
        for method, r in mean_ranks.items():
            ranks_rows.append({
                "labeled_frac": frac,
                "method": method,
                "avg_rank": float(r),
                "n_datasets": int(piv.shape[0]),
            })
    ranks_df = pd.DataFrame(ranks_rows)
    ranks_df.to_csv(os.path.join(args.out_dir, "average_ranks.csv"), index=False)

    # Wilcoxon signed-rank: main method vs best baseline (paired across datasets)
    tests = []
    if HAVE_SCIPY:
        for frac, g in agg.groupby("labeled_frac"):
            piv = g.pivot_table(index=["dataset_id", "dataset"], columns="method", values="mean")
            if args.method_main not in piv.columns:
                continue

            main_scores = piv[args.method_main]
            baseline_cols = [c for c in piv.columns if c != args.method_main]
            if not baseline_cols:
                continue

            best_baseline = piv[baseline_cols].max(axis=1)

            # pairwise intersection (drop NaNs)
            common = main_scores.dropna().index.intersection(best_baseline.dropna().index)
            if len(common) < 5:
                continue

            stat, p = wilcoxon(main_scores.loc[common], best_baseline.loc[common], alternative="greater")
            tests.append({
                "labeled_frac": frac,
                "n_pairs": int(len(common)),
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
            })
        pd.DataFrame(tests).to_csv(os.path.join(args.out_dir, "wilcoxon_tests.csv"), index=False)

    # Plot: overall macro-F1 vs labeled fraction (mean ± std)
    methods = sorted(overall["method"].unique().tolist())
    fracs = sorted(overall["labeled_frac"].unique().tolist())

    plt.figure()
    for method in methods:
        sub = overall[overall["method"] == method].set_index("labeled_frac").reindex(fracs)
        plt.plot(fracs, sub["mean"].to_numpy(), marker="o", label=method)
        plt.fill_between(
            fracs,
            (sub["mean"] - sub["std"]).to_numpy(),
            (sub["mean"] + sub["std"]).to_numpy(),
            alpha=0.15,
        )
    plt.xlabel("Labeled fraction")
    plt.ylabel("Macro-F1 (test)")
    plt.title("Performance vs labeled fraction")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "macroF1_vs_labeled_fraction.png"), dpi=200)
    plt.close()

    # Boxplots across datasets per labeled fraction
    for frac, g in agg.groupby("labeled_frac"):
        piv = g.pivot_table(index=["dataset_id", "dataset"], columns="method", values="mean")
        cols = [c for c in methods if c in piv.columns]
        data = [piv[c].dropna().to_numpy() for c in cols]

        plt.figure(figsize=(max(6, 0.8 * len(cols)), 4))
        plt.boxplot(data, labels=cols)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Macro-F1 (test), mean over seeds")
        plt.title(f"Across-dataset distribution @ labeled_frac={frac}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"boxplot_frac_{frac}.png"), dpi=200)
        plt.close()

    print(f"Wrote analysis to: {args.out_dir}")
    print("Key files: by_dataset_summary.csv, overall_summary.csv, average_ranks.csv, macroF1_vs_labeled_fraction.png")
    if HAVE_SCIPY:
        print("Also wrote: wilcoxon_tests.csv (if enough paired datasets were available)")
    else:
        print("SciPy not found: skipping Wilcoxon tests. Install with: pip install scipy")


if __name__ == "__main__":
    main()
