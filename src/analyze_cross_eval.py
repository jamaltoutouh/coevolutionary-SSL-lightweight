#!/usr/bin/env python3
"""
Analyze cross-evaluation matrices produced by CC-SSL / EA runs.

Input:
  - one or more cross_eval_*.csv files
  - or a directory containing them

Outputs:
  - summary CSV with diagonal vs off-diagonal dominance stats + effect size
  - prints a compact table to stdout

Notes:
  - If scipy is available, computes Mann-Whitney U p-value (diag vs off).
  - Effect size: Cliff's delta (robust, distribution-free).
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta in [-1, 1].
    Positive means x tends to be larger than y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan
    # O(n*m) but n~10, m~90 per run -> trivial
    gt = 0
    lt = 0
    for a in x:
        gt += int(np.sum(a > y))
        lt += int(np.sum(a < y))
    return float((gt - lt) / (x.size * y.size))

def try_mannwhitneyu(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
        if x.size == 0 or y.size == 0:
            return np.nan
        # alternative='greater' tests diag > off
        res = mannwhitneyu(x, y, alternative="greater")
        return float(res.pvalue)
    except Exception:
        return np.nan

def load_files(paths: List[str], directory: str | None) -> List[str]:
    files = []
    files.extend(paths)
    if directory:
        files.extend(glob.glob(os.path.join(directory, "cross_eval_*.csv")))
    # de-dup preserve order
    seen = set()
    out = []
    for f in files:
        f = os.path.abspath(f)
        if f not in seen and os.path.isfile(f):
            seen.add(f); out.append(f)
    return out

def analyze_one(df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
    # expected columns: rankA, rankB, fitness
    if not {"rankA", "rankB", "fitness"}.issubset(df.columns):
        raise ValueError(f"{file_path}: missing required columns (rankA, rankB, fitness)")

    # Determine k from max rank
    kA = int(df["rankA"].max() + 1)
    kB = int(df["rankB"].max() + 1)
    k = min(kA, kB)

    diag = df[(df["rankA"] == df["rankB"]) & (df["rankA"] < k)]["fitness"].to_numpy(dtype=float)
    off  = df[(df["rankA"] != df["rankB"])]["fitness"].to_numpy(dtype=float)


    # If some files came from EA extraction, diag still meaningful in same way.
    diag_mean = float(np.mean(diag)) if diag.size else np.nan
    off_mean  = float(np.mean(off)) if off.size else np.nan
    gap = float(diag_mean - off_mean) if (diag_mean == diag_mean and off_mean == off_mean) else np.nan

    cd = cliffs_delta(diag, off)
    pval = try_mannwhitneyu(diag, off)

    # Extract tag if available (some runs store it in filename)
    base = os.path.basename(file_path)
    tag = os.path.splitext(base)[0].replace("cross_eval_", "")

    return {
        "file": base,
        "tag": tag,
        "kA": kA,
        "kB": kB,
        "diag_n": int(diag.size),
        "off_n": int(off.size),
        "diag_mean": diag_mean,
        "off_mean": off_mean,
        "diag_minus_off": gap,
        "cliffs_delta": cd,
        "mannwhitneyu_p_greater": pval,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=str, nargs="*", default=[],
                    help="One or more cross_eval_*.csv files")
    ap.add_argument("--dir", type=str, default=None,
                    help="Directory containing cross_eval_*.csv files")
    ap.add_argument("--out_csv", type=str, default="cross_eval_summary.csv")
    args = ap.parse_args()

    files = load_files(args.files, args.dir)
    if not files:
        raise SystemExit("No cross_eval_*.csv files found. Use --files or --dir.")

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        rows.append(analyze_one(df, fp))

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    # Print compact summary
    show = out.sort_values("diag_minus_off", ascending=False)
    cols = ["file", "kA", "kB", "diag_mean", "off_mean", "diag_minus_off", "cliffs_delta", "mannwhitneyu_p_greater"]
    print(show[cols].to_string(index=False))
    print(f"\nWrote: {args.out_csv}")

if __name__ == "__main__":
    main()
