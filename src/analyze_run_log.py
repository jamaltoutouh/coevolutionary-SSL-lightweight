#!/usr/bin/env python3
"""
Analyze CC-SSL run logs (JSONL) produced by cc_ssl_openml.py (--log_dir).

Input:
  <log_dir>/run_log.jsonl
Each line is a JSON object with:
  run.{dataset_id,dataset,labeled_frac,seed,n_classes,...}
  metrics.eval_distributions.{val_macroF1,val_acc,probe_drop,pseudo_added,seconds,...}

Outputs:
  - run_level.csv             (one row per run)
  - eval_seed_level.csv       (one row per eval seed inside each run)
  - summaries:
      summary_by_frac.csv
      summary_by_frac_and_task.csv (binary vs multiclass)
  - plots:
      box_val_macroF1_mean_by_frac.png
      box_val_acc_mean_by_frac.png
      box_probe_drop_mean_by_frac.png
      box_pseudo_added_mean_by_frac.png
      scatter_f1_vs_probe_drop.png
      (and the same split by task type)

Usage:
  python analyze_run_log.py --log_jsonl logs_ccssl/run_log.jsonl --out_dir analysis_logs

Optional:
  python analyze_run_log.py --log_jsonl logs_ccssl/run_log.jsonl --out_dir analysis_logs --only_frac 0.01
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def task_type_from_n_classes(n_classes: int) -> str:
    return "binary" if int(n_classes) == 2 else "multiclass"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def boxplot_by_group(df: pd.DataFrame, group_col: str, value_col: str, title: str, out_path: str) -> None:
    groups = sorted(df[group_col].dropna().unique().tolist())
    data = [df.loc[df[group_col] == g, value_col].dropna().to_numpy() for g in groups]

    plt.figure(figsize=(max(6, 1.0 * len(groups)), 4))
    plt.boxplot(data, labels=[str(g) for g in groups])
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def scatter(df: pd.DataFrame, x: str, y: str, title: str, out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x].to_numpy(), df[y].to_numpy(), s=18)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_jsonl", type=str, required=True, help="Path to run_log.jsonl")
    ap.add_argument("--out_dir", type=str, default="analysis_run_log")
    ap.add_argument("--only_frac", type=float, default=None, help="If set, filter to one labeled fraction")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    raw = read_jsonl(args.log_jsonl)
    if not raw:
        raise SystemExit(f"No rows found in {args.log_jsonl}")

    run_rows = []
    eval_seed_rows = []

    for obj in raw:
        run = obj.get("run", {})
        cfg = obj.get("cfg", {})
        best = obj.get("best", {})
        metrics = obj.get("metrics", {})
        ed = metrics.get("eval_distributions", {})

        labeled_frac = float(run.get("labeled_frac"))
        if args.only_frac is not None and abs(labeled_frac - float(args.only_frac)) > 1e-12:
            continue

        n_classes = int(run.get("n_classes", -1))
        task_type = task_type_from_n_classes(n_classes) if n_classes > 0 else "unknown"

        # Run-level row (one per dataset_id x frac x seed)
        r = {
            "dataset_id": int(run.get("dataset_id")),
            "dataset": str(run.get("dataset")),
            "labeled_frac": labeled_frac,
            "seed": int(run.get("seed")),
            "n_rows_used": int(run.get("n_rows_used", -1)),
            "n_features": int(run.get("n_features", -1)),
            "n_classes": n_classes,
            "task_type": task_type,
            # Fitness + configuration
            "best_fitness": float(best.get("fitness", np.nan)),
            "popA": int(cfg.get("popA", -1)),
            "popB": int(cfg.get("popB", -1)),
            "generations": int(cfg.get("generations", -1)),
            "teams_per_individual": int(cfg.get("teams_per_individual", -1)),
            "max_u_fit": int(cfg.get("max_u_fit", -1)),
            "probe_size": int(cfg.get("probe_size", -1)),
            # Logged distribution summaries (validation)
            "val_macroF1_mean": float(ed.get("val_macroF1_mean", np.nan)),
            "val_macroF1_std": float(ed.get("val_macroF1_std", np.nan)),
            "val_acc_mean": float(ed.get("val_acc_mean", np.nan)),
            "val_acc_std": float(ed.get("val_acc_std", np.nan)),
            "probe_drop_mean": float(ed.get("probe_drop_mean", np.nan)),
            "pseudo_added_mean": float(ed.get("pseudo_added_mean", np.nan)),
            "seconds_mean": float(ed.get("seconds_mean", np.nan)),
        }

        # Best view/policy metadata (handy for later correlation studies)
        view = best.get("view", {})
        pol = best.get("policy", {})
        r.update({
            "view_size1": int(view.get("view_size1", -1)),
            "view_size2": int(view.get("view_size2", -1)),
            "view_overlap": float(view.get("view_overlap", np.nan)),
            "policy_calibrate": bool(pol.get("calibrate", False)),
            "policy_tau_start": float(pol.get("tau_start", np.nan)),
            "policy_tau_end": float(pol.get("tau_end", np.nan)),
            "policy_max_iters": int(pol.get("max_iters", -1)),
            "policy_disagreement_veto": bool(pol.get("disagreement_veto", False)),
            "policy_class_balance": bool(pol.get("class_balance", False)),
            "policy_veto_min_other_proba": float(pol.get("veto_min_other_proba", np.nan)),
            "policy_max_add_total": int(pol.get("max_add_total", -1)),
            "policy_max_add_per_class": int(pol.get("max_add_per_class", -1)),
        })

        # Test metrics (final selected best)
        test = metrics.get("test", {})
        r.update({
            "test_macroF1": float(test.get("macroF1", np.nan)),
            "test_acc": float(test.get("acc", np.nan)),
            "test_probe_drop": float(test.get("probe_drop", np.nan)),
            "test_pseudo_added": float(test.get("pseudo_added", np.nan)),
        })

        run_rows.append(r)

        # Explode eval-seed distributions into per-seed rows
        eval_seeds = ed.get("eval_seeds", [])
        f1_list = ed.get("val_macroF1", [])
        acc_list = ed.get("val_acc", [])
        drop_list = ed.get("probe_drop", [])
        add_list = ed.get("pseudo_added", [])
        sec_list = ed.get("seconds", [])

        m = min(len(eval_seeds), len(f1_list), len(acc_list), len(drop_list), len(add_list), len(sec_list))
        for k in range(m):
            eval_seed_rows.append({
                "dataset_id": int(run.get("dataset_id")),
                "dataset": str(run.get("dataset")),
                "labeled_frac": labeled_frac,
                "seed": int(run.get("seed")),
                "eval_seed": int(eval_seeds[k]),
                "task_type": task_type,
                "val_macroF1": float(f1_list[k]),
                "val_acc": float(acc_list[k]),
                "probe_drop": float(drop_list[k]),
                "pseudo_added": int(add_list[k]),
                "seconds": float(sec_list[k]),
            })

    if not run_rows:
        raise SystemExit("No runs after filtering. Check --only_frac or log file contents.")

    df_run = pd.DataFrame(run_rows)
    df_eval = pd.DataFrame(eval_seed_rows)

    df_run.to_csv(os.path.join(args.out_dir, "run_level.csv"), index=False)
    df_eval.to_csv(os.path.join(args.out_dir, "eval_seed_level.csv"), index=False)

    # Summaries
    summary_by_frac = (
        df_run.groupby("labeled_frac")[["val_macroF1_mean","val_acc_mean","probe_drop_mean","pseudo_added_mean","seconds_mean","test_macroF1","test_acc"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_by_frac.columns = ["_".join([c for c in col if c]) for col in summary_by_frac.columns.values]
    summary_by_frac.to_csv(os.path.join(args.out_dir, "summary_by_frac.csv"), index=False)

    summary_by_frac_task = (
        df_run.groupby(["labeled_frac","task_type"])[["val_macroF1_mean","val_acc_mean","probe_drop_mean","pseudo_added_mean","seconds_mean","test_macroF1","test_acc"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_by_frac_task.columns = ["_".join([c for c in col if c]) for col in summary_by_frac_task.columns.values]
    summary_by_frac_task.to_csv(os.path.join(args.out_dir, "summary_by_frac_and_task.csv"), index=False)

    # Plots (overall)
    boxplot_by_group(
        df_run, "labeled_frac", "val_macroF1_mean",
        "Validation macro-F1 mean by labeled fraction (run-level)",
        os.path.join(args.out_dir, "box_val_macroF1_mean_by_frac.png")
    )
    boxplot_by_group(
        df_run, "labeled_frac", "val_acc_mean",
        "Validation accuracy mean by labeled fraction (run-level)",
        os.path.join(args.out_dir, "box_val_acc_mean_by_frac.png")
    )
    boxplot_by_group(
        df_run, "labeled_frac", "probe_drop_mean",
        "Probe-drop mean by labeled fraction (run-level)",
        os.path.join(args.out_dir, "box_probe_drop_mean_by_frac.png")
    )
    boxplot_by_group(
        df_run, "labeled_frac", "pseudo_added_mean",
        "Pseudo-labeled additions (mean) by labeled fraction (run-level)",
        os.path.join(args.out_dir, "box_pseudo_added_mean_by_frac.png")
    )
    scatter(
        df_run, "val_macroF1_mean", "probe_drop_mean",
        "Validation macro-F1 mean vs probe-drop mean (run-level)",
        os.path.join(args.out_dir, "scatter_f1_vs_probe_drop.png")
    )

    # Plots split by task type (binary vs multiclass)
    for tt in sorted(df_run["task_type"].unique().tolist()):
        sub = df_run[df_run["task_type"] == tt].copy()
        if sub.empty:
            continue
        boxplot_by_group(
            sub, "labeled_frac", "val_macroF1_mean",
            f"[{tt}] Validation macro-F1 mean by labeled fraction",
            os.path.join(args.out_dir, f"box_val_macroF1_mean_by_frac_{tt}.png")
        )
        boxplot_by_group(
            sub, "labeled_frac", "val_acc_mean",
            f"[{tt}] Validation accuracy mean by labeled fraction",
            os.path.join(args.out_dir, f"box_val_acc_mean_by_frac_{tt}.png")
        )
        boxplot_by_group(
            sub, "labeled_frac", "probe_drop_mean",
            f"[{tt}] Probe-drop mean by labeled fraction",
            os.path.join(args.out_dir, f"box_probe_drop_mean_by_frac_{tt}.png")
        )

    print(f"OK. Wrote outputs to: {args.out_dir}")
    print("Key files:")
    print("  run_level.csv")
    print("  eval_seed_level.csv")
    print("  summary_by_frac.csv")
    print("  summary_by_frac_and_task.csv")
    print("  (plus .png plots)")


if __name__ == "__main__":
    main()
