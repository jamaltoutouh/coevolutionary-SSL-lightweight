#!/usr/bin/env python3
"""
Plot heatmaps from cross_eval_*.csv produced by CC-SSL / EA runs.

Usage:
  python plot_cross_eval_heatmap.py --dir runs/ccssl/<exp_id>/logs
  python plot_cross_eval_heatmap.py --files path/to/cross_eval_*.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_files(files: List[str], directory: Optional[str]) -> List[str]:
    out = []
    out.extend(files or [])
    if directory:
        out.extend(glob.glob(os.path.join(directory, "**", "cross_eval*.csv"), recursive=True))
    # de-dup
    seen = set()
    final = []
    for f in out:
        f = os.path.abspath(f)
        if f not in seen and os.path.isfile(f):
            seen.add(f)
            final.append(f)
    return final

def build_matrix(df: pd.DataFrame) -> np.ndarray:
    if not {"rankA", "rankB", "fitness"}.issubset(df.columns):
        raise ValueError("cross_eval CSV must have columns: rankA, rankB, fitness")

    kA = int(df["rankA"].max() + 1)
    kB = int(df["rankB"].max() + 1)
    M = np.full((kA, kB), np.nan, dtype=float)

    for _, r in df.iterrows():
        i = int(r["rankA"])
        j = int(r["rankB"])
        M[i, j] = float(r["fitness"])
    return M

def diag_off_stats(M: np.ndarray) -> tuple[float, float, float]:
    k = min(M.shape[0], M.shape[1])
    diag = np.array([M[i, i] for i in range(k)], dtype=float)
    off = M.copy()
    for i in range(k):
        if i < off.shape[1]:
            off[i, i] = np.nan
    off = off[~np.isnan(off)]
    diag_mean = float(np.nanmean(diag)) if diag.size else float("nan")
    off_mean = float(np.nanmean(off)) if off.size else float("nan")
    gap = diag_mean - off_mean if (diag_mean == diag_mean and off_mean == off_mean) else float("nan")
    return diag_mean, off_mean, float(gap)

def plot_one(csv_path: str, out_dir: str, dpi: int = 180) -> str:
    df = pd.read_csv(csv_path)
    M = build_matrix(df)
    diag_mean, off_mean, gap = diag_off_stats(M)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_png = os.path.join(out_dir, f"{base}.png")

    plt.figure()
    plt.imshow(M, aspect="auto")  # default colormap
    plt.colorbar(label="fitness")
    plt.xlabel("rankB (top-k policies)")
    plt.ylabel("rankA (top-k views)")
    plt.title(f"{base}\nmean(diag)={diag_mean:.4f}  mean(off)={off_mean:.4f}  gap={gap:.4f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    return out_png

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=[], help="One or more cross_eval*.csv")
    ap.add_argument("--dir", type=str, default=None, help="Directory to search recursively for cross_eval*.csv")
    ap.add_argument("--out_dir", type=str, default="cross_eval_plots", help="Output directory for PNGs")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    files = find_files(args.files, args.dir)
    if not files:
        raise SystemExit("No cross_eval*.csv found. Use --files or --dir.")

    for fp in files:
        out = plot_one(fp, args.out_dir, dpi=args.dpi)
        print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
