#!/usr/bin/env python3
"""
Plot fitness evolution from per-generation JSONL logs.

Your log line schema (example):
{
  "generation": 0,
  "gen_best": {"fitness": ...},
  ...
}

Filename pattern:
  gen_log_{algo}_did{dataset_id}_lf{lf}_s{seed}.jsonl

Output:
- One PNG per dataset id
- Each plot shows, for each labeled_frac:
    - median best fitness per generation (across seeds)
    - shaded area between Q1 and Q3 (IQR)

Example:
  python plot_fitness_evolution.py \
    --logs_dir runs/tuning-20gens/logs \
    --out_dir runs/tuning-20gens/plots \
    --algo cc
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


FNAME_RE = re.compile(
    r"gen_log_(?P<algo>[a-zA-Z0-9]+)_did(?P<did>\d+)_lf(?P<lf>[0-9.]+)_s(?P<seed>\d+)\.jsonl$"
)


def discover_jsonl_files(logs_dir: Path) -> List[Path]:
    return sorted([p for p in logs_dir.rglob("*.jsonl") if p.is_file()])


def parse_filename(fp: Path) -> Optional[Tuple[str, int, float, int]]:
    """
    Returns (algo, dataset_id, labeled_frac, seed) or None if filename doesn't match.
    """
    m = FNAME_RE.search(fp.name)
    if not m:
        return None
    algo = m.group("algo")
    did = int(m.group("did"))
    lf = float(m.group("lf"))
    seed = int(m.group("seed"))
    return algo, did, lf, seed


def read_fitness_series(fp: Path) -> List[Tuple[int, float]]:
    """
    Read (generation, gen_best.fitness) from JSONL file.
    If multiple lines for same generation exist, last one wins.
    """
    by_gen: Dict[int, float] = {}
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            g = obj.get("generation", None)
            if g is None:
                continue
            try:
                g = int(g)
            except Exception:
                continue

            gb = obj.get("gen_best", None)
            if not isinstance(gb, dict):
                continue
            fit = gb.get("fitness", None)
            if fit is None:
                continue
            try:
                fit = float(fit)
            except Exception:
                continue

            by_gen[g] = fit

    series = sorted(by_gen.items(), key=lambda x: x[0])
    return series


def stack_runs_with_padding(runs: List[np.ndarray]) -> np.ndarray:
    """
    Stack 1D arrays with NaN padding to max length. Shape = (n_runs, max_len)
    """
    maxlen = max(a.size for a in runs)
    M = np.full((len(runs), maxlen), np.nan, dtype=float)
    for i, a in enumerate(runs):
        M[i, :a.size] = a
    return M


def plot_dataset(
    dataset_id: int,
    by_frac: Dict[float, List[np.ndarray]],
    out_dir: Path,
) -> None:
    fracs = sorted(by_frac.keys())
    plt.figure(figsize=(9.5, 5.2))

    for lf in fracs:
        runs = by_frac[lf]
        M = stack_runs_with_padding(runs)

        q1 = np.nanpercentile(M, 25, axis=0)
        med = np.nanpercentile(M, 50, axis=0)
        q3 = np.nanpercentile(M, 75, axis=0)

        x = np.arange(M.shape[1])
        plt.fill_between(x, q1, q3, alpha=0.25)
        plt.plot(x, med, linewidth=2, label=f"labeled_frac={lf:g} (n={len(runs)})")

    plt.xlabel("Generation")
    plt.ylabel("Best-pair fitness (gen_best.fitness)")
    plt.title(f"Fitness evolution (median ± IQR) — dataset {dataset_id}")
    plt.legend(loc="best")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fitness_evolution_did{dataset_id}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--algo", default=None, type=str, help="Filter by algo in filename (e.g., cc, ea).")
    ap.add_argument("--min_runs", default=2, type=int, help="Min runs (seeds) required per (dataset, lf).")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)

    files = discover_jsonl_files(logs_dir)
    if not files:
        raise SystemExit(f"No .jsonl files found under {logs_dir}")

    # data[did][lf] -> list of arrays of best fitness over gens (one per seed/run)
    data: Dict[int, Dict[float, List[np.ndarray]]] = {}

    matched = 0
    for fp in files:
        meta = parse_filename(fp)
        if meta is None:
            continue
        algo, did, lf, seed = meta
        if args.algo is not None and algo.lower() != args.algo.lower():
            continue

        series = read_fitness_series(fp)
        if not series:
            continue

        gens = np.array([g for g, _ in series], dtype=int)
        fits = np.array([v for _, v in series], dtype=float)

        g0 = int(gens.min())
        gmax = int(gens.max())
        arr = np.full(gmax - g0 + 1, np.nan, dtype=float)
        arr[gens - g0] = fits

        data.setdefault(did, {}).setdefault(lf, []).append(arr)
        matched += 1

    if matched == 0:
        raise SystemExit("No usable log files matched the filename pattern and schema.")

    # Filter by min_runs per group
    filtered: Dict[int, Dict[float, List[np.ndarray]]] = {}
    for did, by_frac in data.items():
        for lf, runs in by_frac.items():
            if len(runs) >= args.min_runs:
                filtered.setdefault(did, {})[lf] = runs

    if not filtered:
        raise SystemExit(f"After min_runs={args.min_runs}, nothing left to plot.")

    for did, by_frac in sorted(filtered.items(), key=lambda x: x[0]):
        plot_dataset(did, by_frac, out_dir)

    print(f"Done. Plotted {len(filtered)} dataset(s) to: {out_dir}")


if __name__ == "__main__":
    main()
