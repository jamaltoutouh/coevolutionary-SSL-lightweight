#!/usr/bin/env python3
"""
Compare CC vs EA from results CSVs.

Expected columns in input CSV:
  dataset_id, labeled_frac, seed, method, macroF1_test

Typical usage:
  python compare_cc_vs_ea.py --csv results_cc.csv results_ea.csv \
     --method_cc "CC-SSL (coevolved)" --method_ea "EA-SSL (joint)" \
     --out_csv cc_vs_ea_summary.csv
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

def holm_bonferroni(pvals: List[float]) -> List[float]:
    """
    Holm–Bonferroni adjusted p-values.
    """
    pvals = np.array(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)

    # Holm step-down
    prev = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, float(pvals[idx] * factor))
        val = max(prev, val)  # ensure monotonicity
        adj[idx] = val
        prev = val

    return adj.tolist()

def rank_biserial_from_diffs(d: np.ndarray) -> float:
    """
    Rank-biserial correlation for paired comparisons.
    Uses ranks of |d|, ignoring zeros.
    r = (W+ - W-) / (n(n+1)/2)
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return np.nan

    absd = np.abs(d)
    # average ranks for ties
    ranks = pd.Series(absd).rank(method="average").to_numpy(dtype=float)
    W_plus = float(np.sum(ranks[d > 0]))
    W_minus = float(np.sum(ranks[d < 0]))
    denom = n * (n + 1) / 2.0
    return (W_plus - W_minus) / denom

def wilcoxon_pvalue(d: np.ndarray) -> float:
    """
    Wilcoxon signed-rank test p-value (two-sided) on paired differences d = cc - ea.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if d.size < 3:
        return np.nan
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return np.nan

def load_and_filter(csvs: List[str], method_cc: str, method_ea: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    needed = {"dataset_id", "labeled_frac", "seed", "method", "macroF1_test"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing required columns. Need: {sorted(needed)}")

    df = df[df["method"].isin([method_cc, method_ea])].copy()
    df["labeled_frac"] = df["labeled_frac"].astype(float)
    df["dataset_id"] = df["dataset_id"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["macroF1_test"] = pd.to_numeric(df["macroF1_test"], errors="coerce")
    df = df.dropna(subset=["macroF1_test"])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="One or more results CSVs (can include CC and EA)")
    ap.add_argument("--method_cc", type=str, default="CC-SSL (coevolved)")
    ap.add_argument("--method_ea", type=str, default="EA-SSL (joint)")
    ap.add_argument("--out_csv", type=str, default="cc_vs_ea_summary.csv")
    args = ap.parse_args()

    df = load_and_filter(args.csv, args.method_cc, args.method_ea)
    if df.empty:
        raise SystemExit("No rows after filtering by method names. Check --method_cc/--method_ea.")

    # Per-dataset aggregate across seeds: mean macroF1 per (dataset_id, labeled_frac, method)
    agg = (df.groupby(["dataset_id", "labeled_frac", "method"], as_index=False)
             .agg(macroF1_mean=("macroF1_test", "mean"),
                  macroF1_std=("macroF1_test", "std"),
                  n_seeds=("macroF1_test", "count")))

    fracs = sorted(agg["labeled_frac"].unique().tolist())
    rows = []
    pvals = []

    for frac in fracs:
        sub = agg[agg["labeled_frac"] == frac]
        cc = sub[sub["method"] == args.method_cc][["dataset_id", "macroF1_mean"]].rename(columns={"macroF1_mean": "cc"})
        ea = sub[sub["method"] == args.method_ea][["dataset_id", "macroF1_mean"]].rename(columns={"macroF1_mean": "ea"})

        merged = cc.merge(ea, on="dataset_id", how="inner")
        if merged.empty:
            continue

        d = (merged["cc"] - merged["ea"]).to_numpy(dtype=float)

        win = int(np.sum(d > 0))
        tie = int(np.sum(d == 0))
        loss = int(np.sum(d < 0))

        p = wilcoxon_pvalue(d)
        rbc = rank_biserial_from_diffs(d)

        pvals.append(p)
        rows.append({
            "labeled_frac": float(frac),
            "n_datasets": int(merged.shape[0]),
            "cc_mean_over_datasets": float(np.mean(merged["cc"])),
            "ea_mean_over_datasets": float(np.mean(merged["ea"])),
            "mean_diff_cc_minus_ea": float(np.mean(d)),
            "median_diff_cc_minus_ea": float(np.median(d)),
            "wins_cc": win,
            "ties": tie,
            "losses_cc": loss,
            "wilcoxon_p": float(p) if p == p else np.nan,
            "effect_rank_biserial": float(rbc) if rbc == rbc else np.nan,
        })

    if not rows:
        raise SystemExit("No comparable labeled_fracs found (no overlapping datasets across methods).")

    out = pd.DataFrame(rows).sort_values("labeled_frac").reset_index(drop=True)

    # Holm–Bonferroni across fractions
    adj = holm_bonferroni([float(x) if x == x else 1.0 for x in out["wilcoxon_p"].to_numpy()])
    out["wilcoxon_p_holm"] = adj

    out.to_csv(args.out_csv, index=False)

    # Print compact
    cols = [
        "labeled_frac", "n_datasets",
        "cc_mean_over_datasets", "ea_mean_over_datasets",
        "mean_diff_cc_minus_ea", "wins_cc", "ties", "losses_cc",
        "wilcoxon_p", "wilcoxon_p_holm", "effect_rank_biserial"
    ]
    print(out[cols].to_string(index=False))
    print(f"\nWrote: {args.out_csv}")

if __name__ == "__main__":
    main()
