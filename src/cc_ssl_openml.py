#!/usr/bin/env python3
"""
CC-SSL (Simple Variant) for Tabular Data (Mixed Binary + Multiclass) using OpenML.

Features
- No neural networks: Logistic Regression base learners.
- Cooperative coevolution:
  Pop A: evolves two "views" (feature-subset masks over original columns)
  Pop B: evolves pseudo-label policy (threshold schedule, caps, disagreement veto, calibration flag)
- Fitness (single objective):
  mean(val_macroF1) - lam_std*std(val_macroF1) - lam_bias*mean(probe_drop) - lam_added*mean(pseudo_added)

Offline OpenML
- Set OPENML_CACHE_DIR to point to a prefetched OpenML cache directory.
  The code will call openml.config.set_root_cache_directory(cache_dir) before any OpenML access.

Logging
- If --log_dir is set, writes JSONL logs to: <log_dir>/run_log.jsonl
  One JSON line per (dataset_id, labeled_frac, seed) run with distributions over eval seeds:
    - val_macroF1 (list), val_acc (list), probe_drop (list), pseudo_added (list), seconds (list)
  plus best view/policy parameters and test metrics.

Install
  pip install openml scikit-learn pandas numpy matplotlib
  (optional) pip install scipy

Example (smoke test)
  python cc_ssl_openml.py --dataset_ids 61 --max_rows 5000 --labeled_fracs 0.05 --seeds 0 \
      --popA 6 --popB 6 --generations 5 --out_csv tmp.csv --log_dir logs_smoke

Main run
  export OPENML_CACHE_DIR=/path/to/openml_cache   # optional (offline)
  python cc_ssl_openml.py --openml_suite OpenML-CC18 --max_datasets 15 --max_rows 50000 \
      --labeled_fracs 0.01 0.05 0.10 --seeds 0 1 2 3 4 \
      --popA 12 --popB 12 --generations 12 --teams_per_individual 2 \
      --out_csv results_ccssl_openml.csv --log_dir logs_ccssl
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import argparse
import inspect
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelSpreading

OPENML_CACHE_DIR = "./openml_cache"


# -----------------------------
# Metrics / utilities
# -----------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))

def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))

def schedule_tau(t: int, T: int, tau_start: float, tau_end: float) -> float:
    if T <= 1:
        return float(tau_end)
    frac = t / (T - 1)
    return float(tau_start + (tau_end - tau_start) * frac)

def ensure_min_features(mask: np.ndarray, min_features: int, rng: np.random.Generator) -> np.ndarray:
    if int(mask.sum()) >= int(min_features):
        return mask
    d = mask.size
    idx = rng.choice(d, size=min_features, replace=False)
    out = np.zeros(d, dtype=bool)
    out[idx] = True
    return out

def safe_unique_count(y: np.ndarray) -> int:
    try:
        return int(np.unique(y).size)
    except Exception:
        return 0

def configure_openml_cache_from_env() -> None:
    """
    If OPENML_CACHE_DIR is set, configure OpenML to use it.
    Safe to call multiple times.
    """
    cache_dir = os.environ.get("OPENML_CACHE_DIR", "").strip()
    if not cache_dir:
        return
    try:
        import openml
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        openml.config.set_root_cache_directory(cache_dir)
    except Exception:
        # If OpenML isn't installed or config fails, let caller fail later with a clear error
        return


# -----------------------------
# OpenML loading
# -----------------------------
def load_openml_dataset(dataset_id: int) -> Tuple[pd.DataFrame, np.ndarray, str]:
    configure_openml_cache_from_env()
    import openml
    openml.config.set_root_cache_directory(OPENML_CACHE_DIR)

    ds = openml.datasets.get_dataset(dataset_id)
    target = ds.default_target_attribute
    if target is None:
        raise ValueError(f"OpenML dataset {dataset_id} has no default_target_attribute.")
    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=target)

    y_series = pd.Series(y)
    if not pd.api.types.is_numeric_dtype(y_series):
        y_codes, _ = pd.factorize(y_series, sort=True)
        y_arr = y_codes.astype(int)
    else:
        y_arr = y_series.to_numpy().astype(int)

    name = f"openml_{dataset_id}_{ds.name}"
    return X, y_arr, name


def select_openml_datasets_from_suite(
    suite_name: str,
    max_datasets: int,
    max_rows: int,
    seed: int
) -> List[int]:
    configure_openml_cache_from_env()
    import openml

    suite = openml.study.get_suite(suite_name)
    dataset_ids = list(suite.data)
    rng = np.random.default_rng(seed)
    rng.shuffle(dataset_ids)

    selected = []
    for did in dataset_ids:
        try:
            ds = openml.datasets.get_dataset(did)
            n = ds.qualities.get("NumberOfInstances", None)
            if n is None or int(float(n)) <= max_rows:
                selected.append(did)
        except Exception:
            continue
        if len(selected) >= max_datasets:
            break
    return selected


# -----------------------------
# Preprocessing per view
# -----------------------------
def build_preprocessor_for_columns(X: pd.DataFrame, cols: List[str]) -> ColumnTransformer:
    X_sub = X[cols]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X_sub[c])]
    cat_cols = [c for c in cols if c not in num_cols]

    # compatibility: sparse_output introduced later; older uses sparse
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_lr_pipeline(
    X_ref: pd.DataFrame,
    cols: List[str],
    calibrate: bool,
    y_for_calib: Optional[np.ndarray]
):
    """
    LogisticRegression pipeline with version-safe kwargs.
    Your sklearn build may not expose 'multi_class' in the signature; we detect it.
    """
    pre = build_preprocessor_for_columns(X_ref, cols)

    lr_sig = inspect.signature(LogisticRegression).parameters
    lr_kwargs = dict(solver="lbfgs", max_iter=400)
    if "multi_class" in lr_sig:
        lr_kwargs["multi_class"] = "auto"

    base = Pipeline([
        ("prep", pre),
        ("clf", LogisticRegression(**lr_kwargs))
    ])

    if not calibrate or y_for_calib is None:
        return base

    # Only calibrate if we have enough labeled data per class
    classes, counts = np.unique(y_for_calib, return_counts=True)
    if classes.size < 2:
        return base
    min_count = int(counts.min())
    n = int(y_for_calib.size)
    if n < 40 or min_count < 10:
        return base
    cv = min(3, min_count)
    if cv < 2:
        return base

    return CalibratedClassifierCV(base, method="sigmoid", cv=cv)


# -----------------------------
# Genomes
# -----------------------------
@dataclass
class ViewGenome:
    mask1: np.ndarray
    mask2: np.ndarray

    @staticmethod
    def random(d: int, rng: np.random.Generator, frac: float = 0.5, min_features: int = 3) -> "ViewGenome":
        k = max(min_features, int(round(d * frac)))
        idx1 = rng.choice(d, size=k, replace=False)
        idx2 = rng.choice(d, size=k, replace=False)
        m1 = np.zeros(d, dtype=bool); m1[idx1] = True
        m2 = np.zeros(d, dtype=bool); m2[idx2] = True
        m1 = ensure_min_features(m1, min_features, rng)
        m2 = ensure_min_features(m2, min_features, rng)
        return ViewGenome(m1, m2)

    def clone(self) -> "ViewGenome":
        return ViewGenome(self.mask1.copy(), self.mask2.copy())


@dataclass
class PolicyGenome:
    calibrate: bool
    tau_start: float
    tau_end: float
    max_iters: int
    max_add_total: int
    max_add_per_class: int
    disagreement_veto: bool
    class_balance: bool
    veto_min_other_proba: float

    @staticmethod
    def random(rng: np.random.Generator) -> "PolicyGenome":
        tau_start = float(rng.uniform(0.85, 0.99))
        tau_end = float(rng.uniform(0.60, min(0.95, tau_start)))
        max_iters = int(rng.integers(3, 10))
        max_add_total = int(rng.integers(20, 400))
        max_add_per_class = int(rng.integers(10, 150))
        calibrate = bool(rng.integers(0, 2))
        disagreement_veto = bool(rng.integers(0, 2))
        class_balance = bool(rng.integers(0, 2))
        veto_min_other_proba = float(rng.uniform(0.45, 0.70))
        return PolicyGenome(
            calibrate=calibrate,
            tau_start=tau_start,
            tau_end=tau_end,
            max_iters=max_iters,
            max_add_total=max_add_total,
            max_add_per_class=max_add_per_class,
            disagreement_veto=disagreement_veto,
            class_balance=class_balance,
            veto_min_other_proba=veto_min_other_proba
        )

    def clone(self) -> "PolicyGenome":
        return PolicyGenome(**self.__dict__)


# -----------------------------
# Variation operators
# -----------------------------
def crossover_view(a: ViewGenome, b: ViewGenome, rng: np.random.Generator) -> ViewGenome:
    child = a.clone()
    for name in ["mask1", "mask2"]:
        ma = getattr(a, name)
        mb = getattr(b, name)
        take_b = rng.random(ma.size) < 0.5
        mc = ma.copy()
        mc[take_b] = mb[take_b]
        setattr(child, name, mc)
    return child

def mutate_view(v: ViewGenome, rng: np.random.Generator, p_flip: float = 0.08, min_features: int = 3) -> ViewGenome:
    out = v.clone()
    for name in ["mask1", "mask2"]:
        m = getattr(out, name)
        flips = rng.random(m.size) < p_flip
        m = np.logical_xor(m, flips)
        m = ensure_min_features(m, min_features, rng)
        setattr(out, name, m)
    return out

def crossover_policy(a: PolicyGenome, b: PolicyGenome, rng: np.random.Generator) -> PolicyGenome:
    c = a.clone()
    alpha = float(rng.uniform(0.2, 0.8))
    c.tau_start = float(np.clip(alpha * a.tau_start + (1 - alpha) * b.tau_start, 0.75, 0.995))
    c.tau_end = float(np.clip(alpha * a.tau_end + (1 - alpha) * b.tau_end, 0.50, min(0.98, c.tau_start)))
    c.max_iters = int(np.clip(int(round(alpha * a.max_iters + (1 - alpha) * b.max_iters)), 2, 14))
    c.max_add_total = int(np.clip(int(round(alpha * a.max_add_total + (1 - alpha) * b.max_add_total)), 10, 2500))
    c.max_add_per_class = int(np.clip(int(round(alpha * a.max_add_per_class + (1 - alpha) * b.max_add_per_class)), 5, c.max_add_total))
    c.veto_min_other_proba = float(np.clip(alpha * a.veto_min_other_proba + (1 - alpha) * b.veto_min_other_proba, 0.40, 0.80))
    for name in ["calibrate", "disagreement_veto", "class_balance"]:
        setattr(c, name, bool(a.__dict__[name] if rng.random() < 0.5 else b.__dict__[name]))
    return c

def mutate_policy(p: PolicyGenome, rng: np.random.Generator) -> PolicyGenome:
    out = p.clone()
    out.tau_start = float(np.clip(out.tau_start + rng.normal(0, 0.03), 0.75, 0.995))
    out.tau_end = float(np.clip(out.tau_end + rng.normal(0, 0.06), 0.50, min(0.98, out.tau_start)))
    out.max_iters = int(np.clip(out.max_iters + int(rng.integers(-1, 2)), 2, 14))
    out.max_add_total = int(np.clip(out.max_add_total + int(rng.integers(-40, 41)), 10, 2500))
    out.max_add_per_class = int(np.clip(out.max_add_per_class + int(rng.integers(-20, 21)), 5, out.max_add_total))
    out.veto_min_other_proba = float(np.clip(out.veto_min_other_proba + rng.normal(0, 0.05), 0.40, 0.80))
    if rng.random() < 0.12:
        out.calibrate = not out.calibrate
    if rng.random() < 0.12:
        out.disagreement_veto = not out.disagreement_veto
    if rng.random() < 0.12:
        out.class_balance = not out.class_balance
    return out


# -----------------------------
# Pseudo-label selection
# -----------------------------
def select_pseudolabels(
    proba: np.ndarray,
    y_pred: np.ndarray,
    tau: float,
    max_add_total: int,
    max_add_per_class: int,
    class_balance: bool,
) -> np.ndarray:
    conf = np.max(proba, axis=1)
    idx = np.where(conf >= tau)[0]
    if idx.size == 0:
        return idx
    idx = idx[np.argsort(conf[idx])[::-1]]

    if not class_balance:
        return idx[:max_add_total]

    selected = []
    counts: Dict[int, int] = {}
    for i in idx:
        c = int(y_pred[i])
        if counts.get(c, 0) >= max_add_per_class:
            continue
        selected.append(i)
        counts[c] = counts.get(c, 0) + 1
        if len(selected) >= max_add_total:
            break
    return np.array(selected, dtype=int)


# -----------------------------
# Diversity + summaries + cross-eval helpers
# -----------------------------
def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def view_diversity(popA: List[ViewGenome], max_pairs: int = 200, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """
    Diversity for Pop A: average Jaccard distance over random pairs, for mask1 and mask2.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(popA)
    if n < 2:
        return {"pairs": 0, "mask1_jaccard_dist": 0.0, "mask2_jaccard_dist": 0.0}

    pairs = min(max_pairs, n * (n - 1) // 2)
    d1 = []
    d2 = []
    for _ in range(pairs):
        i, j = rng.choice(n, size=2, replace=False)
        vi = popA[int(i)]
        vj = popA[int(j)]
        d1.append(1.0 - _jaccard(vi.mask1, vj.mask1))
        d2.append(1.0 - _jaccard(vi.mask2, vj.mask2))

    return {
        "pairs": int(pairs),
        "mask1_jaccard_dist": float(np.mean(d1)) if d1 else 0.0,
        "mask2_jaccard_dist": float(np.mean(d2)) if d2 else 0.0,
    }

def policy_diversity(popB: List[PolicyGenome]) -> Dict[str, float]:
    """
    Diversity for Pop B: mixed numeric+boolean average distance to population mean.
    Numeric params are normalized by their ranges.
    """
    n = len(popB)
    if n < 2:
        return {"n": n, "mean_dist": 0.0, "bool_disagree_rate": 0.0}

    # Collect
    tau_start = np.array([p.tau_start for p in popB], dtype=float)
    tau_end   = np.array([p.tau_end for p in popB], dtype=float)
    max_iters = np.array([p.max_iters for p in popB], dtype=float)
    max_add_t = np.array([p.max_add_total for p in popB], dtype=float)
    max_add_c = np.array([p.max_add_per_class for p in popB], dtype=float)
    veto_p    = np.array([p.veto_min_other_proba for p in popB], dtype=float)

    cal  = np.array([p.calibrate for p in popB], dtype=int)
    dis  = np.array([p.disagreement_veto for p in popB], dtype=int)
    bal  = np.array([p.class_balance for p in popB], dtype=int)

    # Normalize numeric (use your allowed ranges from random/mutation clamps)
    def norm(x, lo, hi):
        if hi <= lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    ns = norm(tau_start, 0.75, 0.995)
    ne = norm(tau_end,   0.50, 0.98)
    ni = norm(max_iters, 2.0, 14.0)
    nt = norm(max_add_t, 10.0, 2500.0)
    nc = norm(max_add_c, 5.0, 2500.0)
    nv = norm(veto_p,    0.40, 0.80)

    Xnum = np.vstack([ns, ne, ni, nt, nc, nv]).T
    mu = Xnum.mean(axis=0, keepdims=True)
    num_dist = np.linalg.norm(Xnum - mu, axis=1).mean()

    # Boolean “disagreement” rate: average pairwise mismatch probability
    # Use p*(1-p) *2 approx? simpler: mean variance of each boolean.
    p_cal = cal.mean(); p_dis = dis.mean(); p_bal = bal.mean()
    bool_disagree_rate = float(2*(p_cal*(1-p_cal) + p_dis*(1-p_dis) + p_bal*(1-p_bal))/3.0)

    return {
        "n": int(n),
        "mean_dist": float(num_dist),
        "bool_disagree_rate": float(bool_disagree_rate),
    }

def summarize_view(v: ViewGenome) -> Dict[str, Any]:
    return {
        "view_size1": int(v.mask1.sum()),
        "view_size2": int(v.mask2.sum()),
        "view_overlap": float(np.mean(np.logical_and(v.mask1, v.mask2))),
    }

def summarize_policy(p: PolicyGenome) -> Dict[str, Any]:
    return {
        "calibrate": bool(p.calibrate),
        "tau_start": float(p.tau_start),
        "tau_end": float(p.tau_end),
        "max_iters": int(p.max_iters),
        "max_add_total": int(p.max_add_total),
        "max_add_per_class": int(p.max_add_per_class),
        "disagreement_veto": bool(p.disagreement_veto),
        "class_balance": bool(p.class_balance),
        "veto_min_other_proba": float(p.veto_min_other_proba),
    }

def _run_tag(dataset_id: int, labeled_frac: float, seed: int, algo: str) -> str:
    # filesystem-friendly tag
    lf = f"{labeled_frac:.4f}".rstrip("0").rstrip(".")
    return f"{algo}_did{dataset_id}_lf{lf}_s{seed}"

def cross_eval_kxk(
    *,
    splits: Dict[str, Any],
    columns: List[str],
    popA: List[ViewGenome],
    popB: List[PolicyGenome],
    fitA: np.ndarray,
    fitB: np.ndarray,
    eval_seeds: List[int],
    max_u_fit: int,
    use_probe_penalty: bool,
    lam_std: float,
    lam_bias: float,
    lam_added: float,
    k: int,
    seed: int,
    out_csv: str,
) -> None:
    """
    Post-run cross evaluation: choose top-k by final fitness arrays and evaluate all k×k pairs.
    IMPORTANT: popA/popB must be the FINAL EVALUATED populations.
    """
    kA = min(int(k), len(popA))
    kB = min(int(k), len(popB))
    topA = np.argsort(-fitA)[:kA]
    topB = np.argsort(-fitB)[:kB]

    rows = []
    for ra, ia in enumerate(topA):
        a = popA[int(ia)]
        for rb, ib in enumerate(topB):
            b = popB[int(ib)]
            f, st = evaluate_team(
                splits=splits,
                columns=columns,
                view=a,
                policy=b,
                eval_seeds=eval_seeds,
                max_u_fit=max_u_fit,
                use_probe_penalty=use_probe_penalty,
                lam_std=lam_std,
                lam_bias=lam_bias,
                lam_added=lam_added,
            )
            rows.append({
                "rankA": int(ra),
                "rankB": int(rb),
                "idxA_in_pop": int(ia),
                "idxB_in_pop": int(ib),
                "fitA_final": float(fitA[int(ia)]),
                "fitB_final": float(fitB[int(ib)]),
                "fitness": float(f),
                "mean_f1": float(st.mean_f1),
                "std_f1": float(st.std_f1),
                "mean_probe_drop": float(st.mean_probe_drop),
                "mean_added": float(st.mean_added),
                "mean_seconds": float(st.mean_seconds),
                **{f"view_{k}": v for k, v in summarize_view(a).items()},
                **{f"pol_{k}": v for k, v in summarize_policy(b).items()},
            })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)



# -----------------------------
# CC-SSL co-training (simple)
# Returns: macroF1_eval, acc_eval, probe_drop, total_added
# -----------------------------
def cotraining_ccssl(
    X_L: pd.DataFrame, y_L: np.ndarray,
    X_U: pd.DataFrame,
    X_eval: pd.DataFrame, y_eval: np.ndarray,
    X_probe: Optional[pd.DataFrame], y_probe: Optional[np.ndarray],
    columns: List[str],
    view: ViewGenome,
    policy: PolicyGenome,
    rng: np.random.Generator,
    max_u_fit: int = 8000,
) -> Tuple[float, float, float, int]:
    d = len(columns)
    cols1 = [columns[i] for i in range(d) if view.mask1[i]]
    cols2 = [columns[i] for i in range(d) if view.mask2[i]]

    if len(cols1) == 0 or len(cols2) == 0:
        return 0.0, 0.0, 1.0, 0

    if X_U.shape[0] > max_u_fit:
        take = rng.choice(X_U.shape[0], size=max_u_fit, replace=False)
        X_U_fit = X_U.iloc[take].reset_index(drop=True)
    else:
        X_U_fit = X_U.reset_index(drop=True)

    model1 = make_lr_pipeline(X_L, cols1, calibrate=policy.calibrate, y_for_calib=y_L)
    model2 = make_lr_pipeline(X_L, cols2, calibrate=policy.calibrate, y_for_calib=y_L)
    try:
        model1.fit(X_L[cols1], y_L)
        model2.fit(X_L[cols2], y_L)
    except Exception:
        return 0.0, 0.0, 1.0, 0

    # Probe baseline for drop computation (macro-F1 only, as bias proxy)
    f1_before = None
    if X_probe is not None and y_probe is not None and X_probe.shape[0] > 0:
        try:
            p1 = model1.predict_proba(X_probe[cols1])
            p2 = model2.predict_proba(X_probe[cols2])
            yb = np.argmax((p1 + p2) / 2.0, axis=1)
            f1_before = macro_f1(y_probe, yb)
        except Exception:
            f1_before = None

    U_idx = np.arange(X_U_fit.shape[0])
    total_added = 0

    for it in range(policy.max_iters):
        if U_idx.size == 0:
            break
        tau = schedule_tau(it, policy.max_iters, policy.tau_start, policy.tau_end)

        U1 = X_U_fit.iloc[U_idx][cols1]
        U2 = X_U_fit.iloc[U_idx][cols2]
        proba1 = model1.predict_proba(U1)
        proba2 = model2.predict_proba(U2)
        yhat1 = np.argmax(proba1, axis=1)
        yhat2 = np.argmax(proba2, axis=1)

        sel1_local = select_pseudolabels(
            proba1, yhat1, tau,
            policy.max_add_total, policy.max_add_per_class, policy.class_balance
        )
        sel2_local = select_pseudolabels(
            proba2, yhat2, tau,
            policy.max_add_total, policy.max_add_per_class, policy.class_balance
        )
        if sel1_local.size == 0 and sel2_local.size == 0:
            break

        if policy.disagreement_veto:
            def veto(sel_local: np.ndarray, yhat_src: np.ndarray, proba_other: np.ndarray) -> np.ndarray:
                if sel_local.size == 0:
                    return sel_local
                keep = []
                for j in sel_local:
                    lab = int(yhat_src[j])
                    if float(proba_other[j, lab]) >= policy.veto_min_other_proba:
                        keep.append(j)
                return np.array(keep, dtype=int)

            sel1_local = veto(sel1_local, yhat1, proba2)
            sel2_local = veto(sel2_local, yhat2, proba1)

        sel1 = U_idx[sel1_local]
        sel2 = U_idx[sel2_local]
        if sel1.size == 0 and sel2.size == 0:
            break

        # Cross-label: view1 -> labels for view2, view2 -> labels for view1
        y_add_for_2 = model1.predict(X_U_fit.iloc[sel1][cols1]) if sel1.size > 0 else np.array([], dtype=int)
        y_add_for_1 = model2.predict(X_U_fit.iloc[sel2][cols2]) if sel2.size > 0 else np.array([], dtype=int)

        X_L1 = pd.concat([X_L[cols1], X_U_fit.iloc[sel2][cols1]], axis=0, ignore_index=True)
        y_L1 = np.concatenate([y_L, y_add_for_1]) if sel2.size > 0 else y_L.copy()

        X_L2 = pd.concat([X_L[cols2], X_U_fit.iloc[sel1][cols2]], axis=0, ignore_index=True)
        y_L2 = np.concatenate([y_L, y_add_for_2]) if sel1.size > 0 else y_L.copy()

        if safe_unique_count(y_L1) < 2 or safe_unique_count(y_L2) < 2:
            break

        model1 = make_lr_pipeline(X_L, cols1, calibrate=policy.calibrate, y_for_calib=y_L1)
        model2 = make_lr_pipeline(X_L, cols2, calibrate=policy.calibrate, y_for_calib=y_L2)
        try:
            model1.fit(X_L1, y_L1)
            model2.fit(X_L2, y_L2)
        except Exception:
            break

        used = np.unique(np.concatenate([sel1, sel2]))
        total_added += int(used.size)
        keep = ~np.isin(U_idx, used)
        U_idx = U_idx[keep]
        if used.size < 5:
            break

    # Validation / Test evaluation
    try:
        p1 = model1.predict_proba(X_eval[cols1])
        p2 = model2.predict_proba(X_eval[cols2])
        ye = np.argmax((p1 + p2) / 2.0, axis=1)
        f1_eval = macro_f1(y_eval, ye)
        acc_eval = acc(y_eval, ye)
    except Exception:
        f1_eval, acc_eval = 0.0, 0.0

    probe_drop = 0.0
    if X_probe is not None and y_probe is not None and f1_before is not None and X_probe.shape[0] > 0:
        try:
            p1 = model1.predict_proba(X_probe[cols1])
            p2 = model2.predict_proba(X_probe[cols2])
            yp = np.argmax((p1 + p2) / 2.0, axis=1)
            f1_after = macro_f1(y_probe, yp)
            probe_drop = max(0.0, float(f1_before - f1_after))
        except Exception:
            probe_drop = 0.0

    return float(f1_eval), float(acc_eval), float(probe_drop), int(total_added)


# -----------------------------
# Baselines
# -----------------------------
def self_training_baseline(
    X_L: pd.DataFrame, y_L: np.ndarray,
    X_U: pd.DataFrame,
    X_T: pd.DataFrame, y_T: np.ndarray,
    columns: List[str],
    calibrate: bool,
    tau_start: float = 0.97,
    tau_end: float = 0.80,
    max_iters: int = 8,
    max_add_total: int = 250,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    model = make_lr_pipeline(X_L, columns, calibrate=calibrate, y_for_calib=y_L)
    model.fit(X_L[columns], y_L)

    U_idx = np.arange(X_U.shape[0])
    for it in range(max_iters):
        if U_idx.size == 0:
            break
        tau = schedule_tau(it, max_iters, tau_start, tau_end)
        proba = model.predict_proba(X_U.iloc[U_idx][columns])
        yhat = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
        sel_local = np.where(conf >= tau)[0]
        if sel_local.size == 0:
            break
        sel_local = sel_local[np.argsort(conf[sel_local])[::-1]][:max_add_total]
        sel = U_idx[sel_local]

        X_L = pd.concat([X_L, X_U.iloc[sel]], axis=0, ignore_index=True)
        y_L = np.concatenate([y_L, yhat[sel_local]])
        if safe_unique_count(y_L) < 2:
            break
        model = make_lr_pipeline(X_L, columns, calibrate=calibrate, y_for_calib=y_L)
        model.fit(X_L[columns], y_L)

        keep = ~np.isin(U_idx, sel)
        U_idx = U_idx[keep]

    y_pred = np.argmax(model.predict_proba(X_T[columns]), axis=1)
    return macro_f1(y_T, y_pred), acc(y_T, y_pred)


def heuristic_cotraining_baseline(
    X_L: pd.DataFrame, y_L: np.ndarray,
    X_U: pd.DataFrame,
    X_T: pd.DataFrame, y_T: np.ndarray,
    columns: List[str],
    rng: np.random.Generator
) -> float:
    d = len(columns)
    perm = rng.permutation(d)
    half = max(1, d // 2)
    m1 = np.zeros(d, dtype=bool); m1[perm[:half]] = True
    m2 = np.zeros(d, dtype=bool); m2[perm[half:]] = True
    if int(m2.sum()) == 0:
        m2[perm[:half]] = True

    view = ViewGenome(m1, m2)
    policy = PolicyGenome(
        calibrate=True, tau_start=0.97, tau_end=0.80, max_iters=7,
        max_add_total=250, max_add_per_class=120,
        disagreement_veto=True, class_balance=True,
        veto_min_other_proba=0.5
    )
    f1, _acc, _drop, _added = cotraining_ccssl(
        X_L, y_L, X_U, X_T, y_T, None, None, columns, view, policy, rng
    )
    return f1, _acc


def label_spreading_baseline(
    X_L: pd.DataFrame, y_L: np.ndarray,
    X_U: pd.DataFrame,
    X_T: pd.DataFrame, y_T: np.ndarray,
    columns: List[str],
    max_total: int,
    rng: np.random.Generator
) -> Optional[float]:
    X_all = pd.concat([X_L, X_U], axis=0, ignore_index=True)
    if X_all.shape[0] > max_total:
        return None

    pre = build_preprocessor_for_columns(X_all, columns)
    X_mat = pre.fit_transform(X_all[columns])

    d = X_mat.shape[1]
    gamma = 1.0 / (2.0 * d)   # robust default for standardized-ish features
    ls = LabelSpreading(kernel="rbf", gamma=gamma, max_iter=30)

    y_all = np.concatenate([y_L, -np.ones(X_U.shape[0], dtype=int)])
    #ls = LabelSpreading(kernel="rbf", gamma=20, max_iter=30)
    #ls = LabelSpreading(kernel="knn", n_neighbors=7, max_iter=30)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls.fit(X_mat, y_all)
    X_T_mat = pre.transform(X_T[columns])
    y_pred = ls.predict(X_T_mat)
    return macro_f1(y_T, y_pred), acc(y_T, y_pred)


# -----------------------------
# Splits
# -----------------------------
def make_ssl_splits(
    X: pd.DataFrame, y: np.ndarray,
    labeled_frac: float,
    seed: int,
    test_size: float = 0.30,
    val_frac_of_train: float = 0.20,
    probe_size: int = 120,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))
    X_tr = X.iloc[train_idx].reset_index(drop=True)
    y_tr = y[train_idx]
    X_te = X.iloc[test_idx].reset_index(drop=True)
    y_te = y[test_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_train, random_state=seed + 1)
    pool_idx, val_idx = next(sss2.split(X_tr, y_tr))
    X_pool = X_tr.iloc[pool_idx].reset_index(drop=True)
    y_pool = y_tr[pool_idx]
    X_V = X_tr.iloc[val_idx].reset_index(drop=True)
    y_V = y_tr[val_idx]

    n_pool = X_pool.shape[0]
    n_classes = int(np.unique(y).size)

    # Minimum labeled to keep multiclass LR stable
    min_labeled = 2 * n_classes

    # Desired labeled count from fraction
    n_L = max(min_labeled, int(round(n_pool * labeled_frac)))

    # Ensure we keep at least this many unlabeled points for SSL to matter
    min_unlabeled = 30
    if n_pool < (min_labeled + min_unlabeled + 1):
        # dataset too small after splits: degrade gracefully
        min_unlabeled = max(10, n_pool // 5)

    # Cap labeled to keep at least min_unlabeled
    n_L = min(n_L, max(min_labeled, n_pool - min_unlabeled))

    # --- STRATIFIED labeled selection to avoid 1-class labeled sets ---
    classes = np.unique(y_pool)
    min_per_class = 2  # ensures LR has both classes; for multiclass gives stability

    # Ensure enough labeled points to cover all classes
    min_needed = int(classes.size * min_per_class)
    if n_L < min_needed:
        n_L = min_needed

    labeled_parts = []
    all_idx = np.arange(n_pool)

    # Take min_per_class from each class (as available)
    for c in classes:
        idx_c = np.where(y_pool == c)[0]
        if idx_c.size == 0:
            continue
        take_c = min(min_per_class, idx_c.size)
        labeled_parts.append(rng.choice(idx_c, size=take_c, replace=False))

    if labeled_parts:
        L_idx = np.unique(np.concatenate(labeled_parts))
    else:
        L_idx = np.array([], dtype=int)

    # Fill remaining labeled budget with random picks from the remaining pool
    remaining = np.setdiff1d(all_idx, L_idx, assume_unique=False)
    need = int(n_L - L_idx.size)
    if need > 0 and remaining.size > 0:
        if need > remaining.size:
            need = remaining.size
        extra = rng.choice(remaining, size=need, replace=False)
        L_idx = np.concatenate([L_idx, extra])

    rng.shuffle(L_idx)

    # Rest of pool after labeled
    rest = np.setdiff1d(all_idx, L_idx, assume_unique=False)
    rng.shuffle(rest)



    # Probe size: take as much as possible but never steal from the unlabeled minimum
    remaining = rest.size
    n_P = min(probe_size, max(0, remaining - min_unlabeled))
    P_idx = rest[:n_P] if n_P > 0 else np.array([], dtype=int)
    U_idx = rest[n_P:]  # guaranteed >= min_unlabeled (unless tiny dataset)

    X_L = X_pool.iloc[L_idx].reset_index(drop=True)
    y_L = y_pool[L_idx]

    if np.unique(y_L).size < 2:
        raise ValueError("Labeled set has <2 classes even after stratified selection (class too rare).")


    X_P = X_pool.iloc[P_idx].reset_index(drop=True) if n_P > 0 else None
    y_P = y_pool[P_idx] if n_P > 0 else None
    X_U = X_pool.iloc[U_idx].reset_index(drop=True)

    return dict(X_L=X_L, y_L=y_L, X_U=X_U, X_V=X_V, y_V=y_V, X_P=X_P, y_P=y_P, X_T=X_te, y_T=y_te)


# -----------------------------
# Coevolution (fitness evaluation)
# -----------------------------
@dataclass
class EvalStats:
    mean_f1: float
    std_f1: float
    mean_probe_drop: float
    mean_added: float
    mean_seconds: float


def evaluate_team(
    splits: Dict[str, Any],
    columns: List[str],
    view: ViewGenome,
    policy: PolicyGenome,
    eval_seeds: List[int],
    max_u_fit: int,
    use_probe_penalty: bool,
    lam_std: float = 0.30,
    lam_bias: float = 0.80,
    lam_added: float = 0.0008,
) -> Tuple[float, EvalStats]:
    f1s = []
    drops = []
    adds = []
    t0 = time.time()
    for s in eval_seeds:
        rng = np.random.default_rng(s)
        f1, _acc, drop, added = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_V"], splits["y_V"],
            splits["X_P"], splits["y_P"],
            columns, view, policy, rng,
            max_u_fit=max_u_fit
        )
        f1s.append(f1); drops.append(drop); adds.append(added)
    elapsed = time.time() - t0

    f1s = np.array(f1s, dtype=float)
    drops = np.array(drops, dtype=float)
    adds = np.array(adds, dtype=float)

    stats = EvalStats(
        mean_f1=float(f1s.mean()),
        std_f1=float(f1s.std()),
        mean_probe_drop=float(drops.mean()),
        mean_added=float(adds.mean()),
        mean_seconds=float(elapsed)
    )

    lam_std = lam_std
    lam_bias = lam_bias if use_probe_penalty else 0.0
    lam_added = lam_added

    fitness = stats.mean_f1 - lam_std * stats.std_f1 - lam_bias * stats.mean_probe_drop - lam_added * stats.mean_added
    return float(fitness), stats


def evaluate_team_detailed(
    splits: Dict[str, Any],
    columns: List[str],
    view: ViewGenome,
    policy: PolicyGenome,
    eval_seeds: List[int],
    max_u_fit: int,
) -> Dict[str, Any]:
    """
    Returns distributions over eval seeds for:
      - val_macroF1, val_acc, probe_drop, pseudo_added, seconds
    plus their mean/std summaries.
    """
    f1s, accs, drops, adds, secs = [], [], [], [], []
    for s in eval_seeds:
        rng = np.random.default_rng(s)
        t0 = time.time()
        f1, a, drop, added = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_V"], splits["y_V"],
            splits["X_P"], splits["y_P"],
            columns, view, policy, rng,
            max_u_fit=max_u_fit
        )
        secs.append(time.time() - t0)
        f1s.append(f1); accs.append(a); drops.append(drop); adds.append(added)

    f1s = np.array(f1s, dtype=float)
    accs = np.array(accs, dtype=float)
    drops = np.array(drops, dtype=float)
    adds = np.array(adds, dtype=float)
    secs = np.array(secs, dtype=float)

    return dict(
        eval_seeds=list(map(int, eval_seeds)),
        val_macroF1=list(map(float, f1s)),
        val_acc=list(map(float, accs)),
        probe_drop=list(map(float, drops)),
        pseudo_added=list(map(int, adds)),
        seconds=list(map(float, secs)),
        val_macroF1_mean=float(f1s.mean()),
        val_macroF1_std=float(f1s.std()),
        val_acc_mean=float(accs.mean()),
        val_acc_std=float(accs.std()),
        probe_drop_mean=float(drops.mean()),
        pseudo_added_mean=float(adds.mean()),
        seconds_mean=float(secs.mean()),
    )


def tournament_select(pop: List[Any], fit: np.ndarray, rng: np.random.Generator, k: int = 3):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[np.argmax(fit[idx])]
    return pop[int(best)]

def coevolve_ccssl(
    splits: Dict[str, Any],
    columns: List[str],
    popA_size: int,
    popB_size: int,
    generations: int,
    teams_per_individual: int,
    seed: int,
    eval_seeds: List[int],
    max_u_fit: int,
    use_probe_penalty: bool,
    force_no_disagree: bool = False,
    lam_std: float = 0.20,
    lam_bias: float = 0.70,
    lam_added: float = 0.0005,
    # logging + cross-eval
    log_dir: Optional[str] = None,
    run_tag: Optional[str] = None,
    cross_k: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    CC-SSL coevolution with:
      - per-generation checkpoint logs
      - final-population-safe post-run cross-eval (k×k)
    Returns:
      dict(best=..., final_popA, final_popB, final_fitA, final_fitB, cross_eval_path)
    """
    rng = np.random.default_rng(seed)
    d = len(columns)

    popA = [ViewGenome.random(d, rng, frac=0.5, min_features=3) for _ in range(popA_size)]
    popB = [PolicyGenome.random(rng) for _ in range(popB_size)]

    # Global best across all teams evaluated
    best = dict(fitness=-1e18, view=None, policy=None, stats=None, gen=-1)

    gen_log_path = None
    cross_eval_path = None
    if log_dir and run_tag:
        os.makedirs(log_dir, exist_ok=True)
        gen_log_path = os.path.join(log_dir, f"gen_log_{run_tag}.jsonl")
        cross_eval_path = os.path.join(log_dir, f"cross_eval_{run_tag}.csv")

    final_popA = None
    final_popB = None
    final_fitA = None
    final_fitB = None

    for gen in range(int(generations)):
        gen_t0 = time.time()

        A_fit = np.zeros(len(popA), dtype=float)
        B_fit = np.zeros(len(popB), dtype=float)

        # best team *within this generation* (useful for per-gen logs)
        gen_best = dict(fitness=-1e18, view=None, policy=None, stats=None)

        # Evaluate A individuals with random B partners
        for i, a in enumerate(popA):
            vals = []
            for _ in range(int(teams_per_individual)):
                b = popB[int(rng.integers(0, len(popB)))]
                if force_no_disagree:
                    b = b.clone()
                    b.disagreement_veto = False

                f, st = evaluate_team(
                    splits=splits,
                    columns=columns,
                    view=a,
                    policy=b,
                    eval_seeds=eval_seeds,
                    max_u_fit=max_u_fit,
                    use_probe_penalty=use_probe_penalty,
                    lam_std=lam_std,
                    lam_bias=lam_bias,
                    lam_added=lam_added,
                )
                vals.append(f)

                if f > gen_best["fitness"]:
                    gen_best = dict(fitness=f, view=a.clone(), policy=b.clone(), stats=st)
                if f > best["fitness"]:
                    best = dict(fitness=f, view=a.clone(), policy=b.clone(), stats=st, gen=gen)

            A_fit[i] = float(np.mean(vals)) if vals else -1e18

        # Evaluate B individuals with random A partners
        for j, b in enumerate(popB):
            vals = []
            for _ in range(int(teams_per_individual)):
                a = popA[int(rng.integers(0, len(popA)))]
                bb = b
                if force_no_disagree:
                    bb = b.clone()
                    bb.disagreement_veto = False

                f, st = evaluate_team(
                    splits=splits,
                    columns=columns,
                    view=a,
                    policy=bb,
                    eval_seeds=eval_seeds,
                    max_u_fit=max_u_fit,
                    use_probe_penalty=use_probe_penalty,
                    lam_std=lam_std,
                    lam_bias=lam_bias,
                    lam_added=lam_added,
                )
                vals.append(f)

                if f > gen_best["fitness"]:
                    gen_best = dict(fitness=f, view=a.clone(), policy=bb.clone(), stats=st)
                if f > best["fitness"]:
                    best = dict(fitness=f, view=a.clone(), policy=bb.clone(), stats=st, gen=gen)

            B_fit[j] = float(np.mean(vals)) if vals else -1e18

        gen_seconds = float(time.time() - gen_t0)

        # --- Per-generation checkpoint log ---
        if gen_log_path and gen_best["view"] is not None and gen_best["policy"] is not None:
            divA = view_diversity(popA, max_pairs=200, rng=rng)
            divB = policy_diversity(popB)
            rec = {
                "run_tag": run_tag,
                "generation": int(gen),
                "seconds_generation": gen_seconds,
                "gen_best": {
                    "fitness": float(gen_best["fitness"]),
                    "view": summarize_view(gen_best["view"]),
                    "policy": summarize_policy(gen_best["policy"]),
                    "mean_f1": float(gen_best["stats"].mean_f1),
                    "std_f1": float(gen_best["stats"].std_f1),
                    "mean_probe_drop": float(gen_best["stats"].mean_probe_drop),
                    "mean_added": float(gen_best["stats"].mean_added),
                    "mean_seconds": float(gen_best["stats"].mean_seconds),
                },
                "global_best_so_far": {
                    "fitness": float(best["fitness"]),
                    "found_at_gen": int(best["gen"]),
                },
                "diversity_A": divA,
                "diversity_B": divB,
            }
            with open(gen_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # IMPORTANT: do NOT breed after final evaluated generation
        if gen == int(generations) - 1:
            final_popA = popA
            final_popB = popB
            final_fitA = A_fit.copy()
            final_fitB = B_fit.copy()
            break

        # --- Breed next generation (simple elitism + tournament) ---
        # Elitism: keep best components (global best so far)
        newA = [best["view"].clone()]
        while len(newA) < len(popA):
            p1 = tournament_select(popA, A_fit, rng, k=3)
            p2 = tournament_select(popA, A_fit, rng, k=3)
            child = p1.clone()
            if rng.random() < cfg["pcxA"]:
                child = crossover_view(p1, p2, rng)
            if rng.random() < cfg["pmutA"]:
                child = mutate_view(child, rng, p_flip=0.08, min_features=3)
            newA.append(child)

        popA = newA

        newB = [best["policy"].clone()]
        while len(newB) < len(popB):
            p1 = tournament_select(popB, B_fit, rng, k=3)
            p2 = tournament_select(popB, B_fit, rng, k=3)
            child = p1.clone()
            if rng.random() < cfg["pcxB"]:
                child = crossover_policy(p1, p2, rng)
            if rng.random() < cfg["pmutB"]:
                child = mutate_policy(child, rng)
            newB.append(child)
        popB = newB

    # --- Post-run cross-evaluation using FINAL evaluated populations ---
    if cross_eval_path and final_popA is not None and final_popB is not None:
        cross_eval_kxk(
            splits=splits,
            columns=columns,
            popA=final_popA,
            popB=final_popB,
            fitA=final_fitA,
            fitB=final_fitB,
            eval_seeds=eval_seeds,
            max_u_fit=max_u_fit,
            use_probe_penalty=use_probe_penalty,
            lam_std=lam_std,
            lam_bias=lam_bias,
            lam_added=lam_added,
            k=int(cross_k),
            seed=seed,
            out_csv=cross_eval_path,
        )

    return dict(
        best=best,
        final_popA=final_popA,
        final_popB=final_popB,
        final_fitA=final_fitA,
        final_fitB=final_fitB,
        cross_eval_path=cross_eval_path,
    )


# -----------------------------
# Single-population EA adaptation (Joint genome: view + policy)
# -----------------------------
@dataclass
class JointGenome:
    view: ViewGenome
    policy: PolicyGenome

    @staticmethod
    def random(d: int, rng: np.random.Generator) -> "JointGenome":
        return JointGenome(
            view=ViewGenome.random(d, rng, frac=0.5, min_features=3),
            policy=PolicyGenome.random(rng),
        )

    def clone(self) -> "JointGenome":
        return JointGenome(view=self.view.clone(), policy=self.policy.clone())

def crossover_joint(a: JointGenome, b: JointGenome, rng: np.random.Generator) -> JointGenome:
    return JointGenome(
        view=crossover_view(a.view, b.view, rng),
        policy=crossover_policy(a.policy, b.policy, rng),
    )

def mutate_joint(g: JointGenome, rng: np.random.Generator) -> JointGenome:
    return JointGenome(
        view=mutate_view(g.view, rng, p_flip=0.08, min_features=3),
        policy=mutate_policy(g.policy, rng),
    )

def evolve_singlepop_ssl(
    splits: Dict[str, Any],
    columns: List[str],
    pop_size: int,
    generations: int,
    seed: int,
    eval_seeds: List[int],
    max_u_fit: int,
    use_probe_penalty: bool,
    lam_std: float,
    lam_bias: float,
    lam_added: float,
    log_dir: Optional[str] = None,
    run_tag: Optional[str] = None,
    cross_k: int = 10,
) -> Dict[str, Any]:
    """
    EA baseline: one population where each individual encodes BOTH view and policy.
    Includes per-generation logs + post-run cross-eval using FINAL evaluated population by
    extracting top-k views and top-k policies from the final EA population.
    """
    rng = np.random.default_rng(seed)
    d = len(columns)
    pop = [JointGenome.random(d, rng) for _ in range(int(pop_size))]

    best = dict(fitness=-1e18, ind=None, stats=None, gen=-1)

    gen_log_path = None
    cross_eval_path = None
    if log_dir and run_tag:
        os.makedirs(log_dir, exist_ok=True)
        gen_log_path = os.path.join(log_dir, f"gen_log_{run_tag}.jsonl")
        cross_eval_path = os.path.join(log_dir, f"cross_eval_{run_tag}.csv")

    final_pop = None
    final_fit = None

    for gen in range(int(generations)):
        gen_t0 = time.time()
        fit = np.zeros(len(pop), dtype=float)

        gen_best = dict(fitness=-1e18, ind=None, stats=None)

        for i, ind in enumerate(pop):
            f, st = evaluate_team(
                splits=splits,
                columns=columns,
                view=ind.view,
                policy=ind.policy,
                eval_seeds=eval_seeds,
                max_u_fit=max_u_fit,
                use_probe_penalty=use_probe_penalty,
                lam_std=lam_std,
                lam_bias=lam_bias,
                lam_added=lam_added,
            )
            fit[i] = f
            if f > gen_best["fitness"]:
                gen_best = dict(fitness=f, ind=ind.clone(), stats=st)
            if f > best["fitness"]:
                best = dict(fitness=f, ind=ind.clone(), stats=st, gen=gen)

        gen_seconds = float(time.time() - gen_t0)

        # diversity over extracted components
        popA = [ind.view for ind in pop]
        popB = [ind.policy for ind in pop]
        divA = view_diversity(popA, max_pairs=200, rng=rng)
        divB = policy_diversity(popB)

        if gen_log_path and gen_best["ind"] is not None:
            rec = {
                "run_tag": run_tag,
                "generation": int(gen),
                "seconds_generation": gen_seconds,
                "gen_best": {
                    "fitness": float(gen_best["fitness"]),
                    "view": summarize_view(gen_best["ind"].view),
                    "policy": summarize_policy(gen_best["ind"].policy),
                    "mean_f1": float(gen_best["stats"].mean_f1),
                    "std_f1": float(gen_best["stats"].std_f1),
                    "mean_probe_drop": float(gen_best["stats"].mean_probe_drop),
                    "mean_added": float(gen_best["stats"].mean_added),
                    "mean_seconds": float(gen_best["stats"].mean_seconds),
                },
                "global_best_so_far": {
                    "fitness": float(best["fitness"]),
                    "found_at_gen": int(best["gen"]),
                },
                "diversity_A": divA,
                "diversity_B": divB,
            }
            with open(gen_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # stop BEFORE breeding if last generation
        if gen == int(generations) - 1:
            final_pop = pop
            final_fit = fit.copy()
            break

        # breed: elitism + tournament
        new_pop = [best["ind"].clone()]
        while len(new_pop) < len(pop):
            p1 = tournament_select(pop, fit, rng, k=3)
            p2 = tournament_select(pop, fit, rng, k=3)
            child = mutate_joint(crossover_joint(p1, p2, rng), rng)
            new_pop.append(child)
        pop = new_pop

    # Post-run cross-eval (extract top-k views/policies from FINAL evaluated EA pop)
    if cross_eval_path and final_pop is not None and final_fit is not None:
        k = int(cross_k)
        top = np.argsort(-final_fit)[:min(k, len(final_pop))]
        candA = [final_pop[int(i)].view for i in top]
        candB = [final_pop[int(i)].policy for i in top]

        # For EA, we do NOT have fitA/fitB arrays; we reuse the individual fitness to rank both lists.
        # Create dummy fit arrays aligned with cand lists (descending)
        fitA = np.array([final_fit[int(i)] for i in top], dtype=float)
        fitB = np.array([final_fit[int(i)] for i in top], dtype=float)

        # Wrap into fake "popA/popB" lists and do k×k evaluation
        cross_eval_kxk(
            splits=splits,
            columns=columns,
            popA=candA,
            popB=candB,
            fitA=fitA,
            fitB=fitB,
            eval_seeds=eval_seeds,
            max_u_fit=max_u_fit,
            use_probe_penalty=use_probe_penalty,
            lam_std=lam_std,
            lam_bias=lam_bias,
            lam_added=lam_added,
            k=min(k, len(top)),
            seed=seed,
            out_csv=cross_eval_path,
        )

    return dict(
        best=best,
        final_pop=final_pop,
        final_fit=final_fit,
        cross_eval_path=cross_eval_path,
    )


def coevolve_ccssl_olld(
    splits: Dict[str, Any],
    columns: List[str],
    popA_size: int,
    popB_size: int,
    generations: int,
    teams_per_individual: int,
    seed: int,
    eval_seeds: List[int],
    max_u_fit: int,
    use_probe_penalty: bool,
    force_no_disagree: bool = False,
    lam_std: float = 0.20,
    lam_bias: float = 0.70,
    lam_added: float = 0.0005,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    d = len(columns)
    popA = [ViewGenome.random(d, rng, frac=0.5, min_features=3) for _ in range(popA_size)]
    popB = [PolicyGenome.random(rng) for _ in range(popB_size)]
    best = dict(fitness=-1e9, view=None, policy=None, stats=None)

    for _gen in range(generations):
        A_fit = np.zeros(popA_size, dtype=float)
        B_fit = np.zeros(popB_size, dtype=float)

        for i, a in enumerate(popA):
            vals = []
            for _ in range(teams_per_individual):
                b = popB[int(rng.integers(0, popB_size))]
                if force_no_disagree:
                    b = b.clone(); b.disagreement_veto = False
                f, stats = evaluate_team(splits, columns, a, b, eval_seeds, max_u_fit, use_probe_penalty, lam_std=lam_std, lam_bias=lam_bias, lam_added=lam_added)
                vals.append(f)
                if f > best["fitness"]:
                    best = dict(fitness=f, view=a.clone(), policy=b.clone(), stats=stats)
            A_fit[i] = float(np.mean(vals))

        for j, b in enumerate(popB):
            vals = []
            for _ in range(teams_per_individual):
                a = popA[int(rng.integers(0, popA_size))]
                bb = b
                if force_no_disagree:
                    bb = b.clone(); bb.disagreement_veto = False
                f, stats = evaluate_team(splits, columns, a, bb, eval_seeds, max_u_fit, use_probe_penalty, lam_std=lam_std, lam_bias=lam_bias, lam_added=lam_added)
                vals.append(f)
                if f > best["fitness"]:
                    best = dict(fitness=f, view=a.clone(), policy=bb.clone(), stats=stats)
            B_fit[j] = float(np.mean(vals))

        # Elitism + tournament variation
        newA = [best["view"].clone()]
        while len(newA) < popA_size:
            p1 = tournament_select(popA, A_fit, rng, k=3)
            p2 = tournament_select(popA, A_fit, rng, k=3)
            newA.append(mutate_view(crossover_view(p1, p2, rng), rng, p_flip=0.08, min_features=3))
        popA = newA

        newB = [best["policy"].clone()]
        while len(newB) < popB_size:
            p1 = tournament_select(popB, B_fit, rng, k=3)
            p2 = tournament_select(popB, B_fit, rng, k=3)
            c = mutate_policy(crossover_policy(p1, p2, rng), rng)
            if force_no_disagree:
                c.disagreement_veto = False
            newB.append(c)
        popB = newB

    return best


# -----------------------------
# Experiment runner
# -----------------------------
def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_setting(
    dataset_id: int,
    dataset_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    labeled_frac: float,
    seed: int,
    cfg: Dict[str, Any]
) -> pd.DataFrame:
    # Cap dataset size if needed
    if X.shape[0] > cfg["max_rows"]:
        rng = np.random.default_rng(seed)
        take = rng.choice(X.shape[0], size=cfg["max_rows"], replace=False)
        X = X.iloc[take].reset_index(drop=True)
        y = y[take]

    if np.unique(y).size < 2:
        raise ValueError("Dataset has <2 classes after filtering.")

    splits = make_ssl_splits(X, y, labeled_frac=labeled_frac, seed=seed, probe_size=cfg["probe_size"])
    columns = list(X.columns)

    # Split sizes (debug/paper-friendly)
    nL = int(splits["X_L"].shape[0])
    nU = int(splits["X_U"].shape[0])
    nP = int(0 if splits["X_P"] is None else splits["X_P"].shape[0])
    nV = int(splits["X_V"].shape[0])
    nT = int(splits["X_T"].shape[0])


    # Seeds used for fitness evaluation
    eval_seeds = [seed + 101, seed + 103, seed + 107]


    # Choose algorithm via cfg["algo"]
    algo = cfg.get("algo", "cc")
    run_tag = _run_tag(dataset_id, labeled_frac, seed, algo)

    if algo == "cc":
        out = coevolve_ccssl(
            splits=splits,
            columns=columns,
            popA_size=cfg["popA"],
            popB_size=cfg["popB"],
            generations=cfg["generations"],
            teams_per_individual=cfg["teams_per_individual"],
            seed=seed,
            eval_seeds=eval_seeds,
            max_u_fit=cfg["max_u_fit"],
            use_probe_penalty=True,
            force_no_disagree=False,
            lam_std=cfg["lam_std"],
            lam_bias=cfg["lam_bias"],
            lam_added=cfg["lam_added"],
            log_dir=cfg.get("log_dir"),
            run_tag=run_tag,
            cross_k=int(cfg.get("cross_k", 10)),
            cfg=cfg,
        )
        best = out["best"]  # <-- NOTE: best is now under "best"
    elif algo == "ea":
        out = evolve_singlepop_ssl(
            splits=splits,
            columns=columns,
            pop_size=int(cfg.get("popEA", cfg["popA"] + cfg["popB"])),  # fair budget default
            generations=cfg["generations"],
            seed=seed,
            eval_seeds=eval_seeds,
            max_u_fit=cfg["max_u_fit"],
            use_probe_penalty=True,
            lam_std=cfg["lam_std"],
            lam_bias=cfg["lam_bias"],
            lam_added=cfg["lam_added"],
            log_dir=cfg.get("log_dir"),
            run_tag=run_tag,
            cross_k=int(cfg.get("cross_k", 10)),
        )
        # normalize to same shape as CC
        best = dict(
            fitness=out["best"]["fitness"],
            view=out["best"]["ind"].view,
            policy=out["best"]["ind"].policy,
            stats=out["best"]["stats"],
        )
    else:
        raise ValueError(f"Unknown algo={algo}. Use 'cc' or 'ea'.")


    # Final test evaluation with a separate RNG seed
    f1_test, acc_test, probe_drop_test, added_test = cotraining_ccssl(
        splits["X_L"], splits["y_L"], splits["X_U"],
        splits["X_T"], splits["y_T"],
        splits["X_P"], splits["y_P"],
        columns, best["view"], best["policy"],
        rng=np.random.default_rng(seed + 999),
        max_u_fit=cfg["max_u_final"]
    )

    # Baselines (macro-F1 only)
    st, acc_st = self_training_baseline(
        splits["X_L"].copy(), splits["y_L"].copy(),
        splits["X_U"].copy(),
        splits["X_T"], splits["y_T"],
        columns, calibrate=True,
        rng=np.random.default_rng(seed + 555)
    )
    hct, acc_hct = heuristic_cotraining_baseline(
        splits["X_L"], splits["y_L"],
        splits["X_U"],
        splits["X_T"], splits["y_T"],
        columns,
        rng=np.random.default_rng(seed + 777)
    )
    ls, acc_ls = label_spreading_baseline(
        splits["X_L"], splits["y_L"],
        splits["X_U"],
        splits["X_T"], splits["y_T"],
        columns,
        max_total=cfg["label_spreading_max_total"],
        rng=np.random.default_rng(seed + 888)
    )

    rows = []

    def add_row(method: str, f1: float, extra: Dict[str, Any]):
        base = dict(
            dataset_id=dataset_id,
            dataset=dataset_name,
            labeled_frac=float(labeled_frac),
            seed=int(seed),
            method=method,
            macroF1_test=float(f1) if f1 == f1 else np.nan,
        )
        base.update(extra)
        rows.append(base)

    method_name = "CC-SSL (coevolved)" if cfg.get("algo","cc") == "cc" else "EA-SSL (joint)"

    add_row(method_name, f1_test, dict(
        acc_test=float(acc_test),
        probe_drop_test=float(probe_drop_test),
        pseudo_added_test=int(added_test),
        best_fitness=float(best["fitness"]),
        policy_calibrate=bool(best["policy"].calibrate),
        policy_tau_start=float(best["policy"].tau_start),
        policy_tau_end=float(best["policy"].tau_end),
        policy_max_iters=int(best["policy"].max_iters),
        policy_disagree=bool(best["policy"].disagreement_veto),
        policy_balance=bool(best["policy"].class_balance),
        policy_veto_min_other=float(best["policy"].veto_min_other_proba),
        view_overlap=float(np.mean(np.logical_and(best["view"].mask1, best["view"].mask2))),
        view_size1=int(best["view"].mask1.sum()),
        view_size2=int(best["view"].mask2.sum()),
        n_features=int(len(columns)),
        n_classes=int(np.unique(y).size),
        n_rows_used=int(X.shape[0]),
        # Split sizes
        n_labeled=nL,
        n_unlabeled=nU,
        n_probe=nP,
        n_val=nV,
        n_test=nT,
    ))
    add_row("Self-training", st, dict(
        acc_test=float(acc_st),
    ))
    add_row("Heuristic co-training", hct, dict(
        acc_test=float(acc_hct),
    ))
    if ls is not None:
        add_row("Label Spreading", ls, dict(
            acc_test=float(acc_ls),
        ))
    else:
        add_row("Label Spreading (skipped)", np.nan, dict(reason="too_large"))

    # Per-run detailed logging (distributions over eval seeds)
    if cfg.get("log_dir"):
        detail = evaluate_team_detailed(
            splits=splits,
            columns=columns,
            view=best["view"],
            policy=best["policy"],
            eval_seeds=eval_seeds,
            max_u_fit=cfg["max_u_fit"],
        )

        log_obj = dict(
            run=dict(
                dataset_id=int(dataset_id),
                dataset=str(dataset_name),
                labeled_frac=float(labeled_frac),
                seed=int(seed),
                n_rows_used=int(X.shape[0]),
                n_features=int(len(columns)),
                n_classes=int(np.unique(y).size),
            ),
            cfg=dict(
                popA=int(cfg["popA"]),
                popB=int(cfg["popB"]),
                generations=int(cfg["generations"]),
                teams_per_individual=int(cfg["teams_per_individual"]),
                max_u_fit=int(cfg["max_u_fit"]),
                max_u_final=int(cfg["max_u_final"]),
                probe_size=int(cfg["probe_size"]),
                lam_std=float(cfg["lam_std"]),
                lam_bias=float(cfg["lam_bias"]),
                lam_added=float(cfg["lam_added"]),
            ),
            best=dict(
                fitness=float(best["fitness"]),
                view=dict(
                    view_size1=int(best["view"].mask1.sum()),
                    view_size2=int(best["view"].mask2.sum()),
                    view_overlap=float(np.mean(np.logical_and(best["view"].mask1, best["view"].mask2))),
                ),
                policy=best["policy"].__dict__,
            ),
            metrics=dict(
                # distributions used to compute fitness (macroF1 mean/std, probe_drop mean, added mean)
                eval_distributions=detail,
                # test metrics for the final selected best
                test=dict(
                    macroF1=float(f1_test),
                    acc=float(acc_test),
                    probe_drop=float(probe_drop_test),
                    pseudo_added=int(added_test),
                )
            ),
        )

        log_path = os.path.join(cfg["log_dir"], "run_log.jsonl")
        append_jsonl(log_path, log_obj)

    # Optional ablations (macro-F1 on test; logging not repeated to keep logs compact)
    if cfg["run_ablations"]:
        rng = np.random.default_rng(seed + 2025)
        fixed_view = ViewGenome.random(len(columns), rng, frac=0.5, min_features=3)

        # policy-only (evolve B, view fixed)
        best_pol = coevolve_ccssl(
            splits=splits, columns=columns,
            popA_size=1, popB_size=cfg["popB"],
            generations=max(8, cfg["generations"]),
            teams_per_individual=cfg["teams_per_individual"],
            seed=seed + 10,
            eval_seeds=[seed + 111, seed + 113, seed + 117],
            max_u_fit=cfg["max_u_fit"],
            use_probe_penalty=True,
            force_no_disagree=False,
            lam_std=cfg["lam_std"],
            lam_bias=cfg["lam_bias"],
            lam_added=cfg["lam_added"],
            cfg=cfg,
        )
        best_pol["view"] = fixed_view
        f1_pol, _a, _d, _ad = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_T"], splits["y_T"],
            splits["X_P"], splits["y_P"],
            columns, fixed_view, best_pol["policy"],
            rng=np.random.default_rng(seed + 4040),
            max_u_fit=cfg["max_u_final"]
        )
        add_row("Ablation: policy-only", f1_pol, dict())

        # view-only (evolve A, policy fixed)
        fixed_policy = PolicyGenome(
            calibrate=True, tau_start=0.97, tau_end=0.80, max_iters=7,
            max_add_total=250, max_add_per_class=120,
            disagreement_veto=True, class_balance=True,
            veto_min_other_proba=0.5
        )
        best_view = coevolve_ccssl(
            splits=splits, columns=columns,
            popA_size=cfg["popA"], popB_size=1,
            generations=max(8, cfg["generations"]),
            teams_per_individual=cfg["teams_per_individual"],
            seed=seed + 20,
            eval_seeds=[seed + 121, seed + 123, seed + 127],
            max_u_fit=cfg["max_u_fit"],
            use_probe_penalty=True,
            force_no_disagree=False,
            lam_added=cfg["lam_added"],
            lam_bias=cfg["lam_bias"],
            lam_std=cfg["lam_std"],
            cfg=cfg
        )
        best_view["policy"] = fixed_policy
        f1_view, _a, _d, _ad = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_T"], splits["y_T"],
            splits["X_P"], splits["y_P"],
            columns, best_view["view"], fixed_policy,
            rng=np.random.default_rng(seed + 5050),
            max_u_fit=cfg["max_u_final"]
        )
        add_row("Ablation: view-only", f1_view, dict())

        # no probe penalty
        best_nop = coevolve_ccssl(
            splits=splits, columns=columns,
            popA_size=cfg["popA"], popB_size=cfg["popB"],
            generations=cfg["generations"],
            teams_per_individual=cfg["teams_per_individual"],
            seed=seed + 30,
            eval_seeds=[seed + 131, seed + 133, seed + 137],
            max_u_fit=cfg["max_u_fit"],
            use_probe_penalty=False,
            force_no_disagree=False,
            lam_added=cfg["lam_added"],
            lam_bias=cfg["lam_bias"],
            lam_std=cfg["lam_std"],
            cfg=cfg
        )
        f1_nop, _a, _d, _ad = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_T"], splits["y_T"],
            splits["X_P"], splits["y_P"],
            columns, best_nop["view"], best_nop["policy"],
            rng=np.random.default_rng(seed + 6060),
            max_u_fit=cfg["max_u_final"]
        )
        add_row("Ablation: no probe penalty", f1_nop, dict())

        # no disagreement veto
        best_nod = coevolve_ccssl(
            splits=splits, columns=columns,
            popA_size=cfg["popA"], popB_size=cfg["popB"],
            generations=cfg["generations"],
            teams_per_individual=cfg["teams_per_individual"],
            seed=seed + 40,
            eval_seeds=[seed + 141, seed + 143, seed + 147],
            max_u_fit=cfg["max_u_fit"],
            use_probe_penalty=True,
            force_no_disagree=True,
            lam_added=cfg["lam_added"],
            lam_bias=cfg["lam_bias"],
            lam_std=cfg["lam_std"],
            cfg=cfg
        )
        f1_nod, _a, _d, _ad = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_T"], splits["y_T"],
            splits["X_P"], splits["y_P"],
            columns, best_nod["view"], best_nod["policy"],
            rng=np.random.default_rng(seed + 7070),
            max_u_fit=cfg["max_u_final"]
        )
        add_row("Ablation: no disagreement", f1_nod, dict())

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openml_suite", type=str, default="OpenML-CC18")
    ap.add_argument("--dataset_ids", type=int, nargs="*", default=None)
    ap.add_argument("--max_datasets", type=int, default=15)
    ap.add_argument("--max_rows", type=int, default=50000)

    ap.add_argument("--labeled_fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    ap.add_argument("--popA", type=int, default=12)
    ap.add_argument("--popB", type=int, default=12)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--teams_per_individual", type=int, default=2)

    ap.add_argument("--max_u_fit", type=int, default=8000)
    ap.add_argument("--max_u_final", type=int, default=20000)
    ap.add_argument("--probe_size", type=int, default=120)

    ap.add_argument("--label_spreading_max_total", type=int, default=4000)
    ap.add_argument("--run_ablations", action="store_true")

    ap.add_argument("--log_dir", type=str, default=None, help="If set, writes <log_dir>/run_log.jsonl with metric distributions")
    ap.add_argument("--out_csv", type=str, default="results_ccssl_openml.csv")
    ap.add_argument("--seed", type=int, default=0, help="Seed used only to shuffle suite dataset order")


    # --- Operator probabilities & mutation schedules (tunable) ---

    # Pop A (views)
    ap.add_argument("--pcxA", type=float, default=0.8, help="Crossover probability for Pop A (views)")
    ap.add_argument("--pmutA", type=float, default=0.5, help="Mutation probability for Pop A (views)")
    ap.add_argument("--view_flip_E_start", type=float, default=3.0,
                    help="Expected flips per mask at gen=0 (Pop A). p_flip = min(pflip_max, E/d)")
    ap.add_argument("--view_flip_E_end", type=float, default=None,
                    help="Expected flips per mask at last gen (Pop A). If None, uses start.")
    ap.add_argument("--view_pflip_max", type=float, default=0.10,
                    help="Maximum per-bit flip probability for Pop A masks")

    # Pop B (policies)
    ap.add_argument("--pcxB", type=float, default=0.8, help="Crossover probability for Pop B (policies)")
    ap.add_argument("--pmutB", type=float, default=0.5, help="Mutation probability for Pop B (policies)")
    ap.add_argument("--policy_bool_flip_start", type=float, default=0.12,
                    help="Boolean flip prob at gen=0 for Pop B")
    ap.add_argument("--policy_bool_flip_end", type=float, default=None,
                    help="Boolean flip prob at last gen for Pop B. If None, uses start.")

    ap.add_argument("--policy_sigma_tau_start", type=float, default=0.03,
                    help="Gaussian sigma for tau_start mutation")
    ap.add_argument("--policy_sigma_tau_end", type=float, default=0.06,
                    help="Gaussian sigma for tau_end mutation")
    ap.add_argument("--policy_sigma_rho", type=float, default=0.05,
                    help="Gaussian sigma for veto_min_other_proba mutation")

    ap.add_argument("--policy_step_iters", type=int, default=1,
                    help="Max step size for max_iters mutation (uniform in [-step, step])")
    ap.add_argument("--policy_step_add_total", type=int, default=40,
                    help="Max step size for max_add_total mutation")
    ap.add_argument("--policy_step_add_per_class", type=int, default=20,
                    help="Max step size for max_add_per_class mutation")
    
    # Fitness weights
    ap.add_argument("--lam_std", type=float, default=0.20)
    ap.add_argument("--lam_bias", type=float, default=0.70)
    ap.add_argument("--lam_added", type=float, default=0.0005)


    ap.add_argument("--algo", type=str, default="cc", choices=["cc", "ea"],
                    help="cc=two-pop coevolution, ea=single-pop joint genome")
    ap.add_argument("--popEA", type=int, default=None,
                    help="EA population size (default popA+popB)")
    ap.add_argument("--cross_k", type=int, default=10,
                    help="Top-k per side for post-run cross-evaluation")


    args = ap.parse_args()

    cfg = dict(
        popA=args.popA,
        popB=args.popB,
        generations=args.generations,
        teams_per_individual=args.teams_per_individual,
        max_u_fit=args.max_u_fit,
        max_u_final=args.max_u_final,
        probe_size=args.probe_size,
        max_rows=args.max_rows,
        label_spreading_max_total=args.label_spreading_max_total,
        run_ablations=bool(args.run_ablations),
        log_dir=args.log_dir,
        lam_std=args.lam_std,
        lam_bias=args.lam_bias,
        lam_added=args.lam_added,
        algo=args.algo,
        cross_k=int(args.cross_k),
        popEA=(args.popEA if args.popEA is not None else args.popA + args.popB),
        pcxA=args.pcxA,
        pmutA=args.pmutA,
        view_flip_E_start=args.view_flip_E_start,
        view_flip_E_end=(args.view_flip_E_end if args.view_flip_E_end is not None else args.view_flip_E_start),
        view_pflip_max=args.view_pflip_max,
        pcxB=args.pcxB,
        pmutB=args.pmutB,
        policy_bool_flip_start=args.policy_bool_flip_start,
        policy_bool_flip_end=(args.policy_bool_flip_end if args.policy_bool_flip_end is not None else args.policy_bool_flip_start),
        policy_sigma_tau_start=args.policy_sigma_tau_start,
        policy_sigma_tau_end=args.policy_sigma_tau_end,
        policy_sigma_rho=args.policy_sigma_rho,
        policy_step_iters=args.policy_step_iters,
        policy_step_add_total=args.policy_step_add_total,
        policy_step_add_per_class=args.policy_step_add_per_class,
    )


    # Configure OpenML cache early if requested
    configure_openml_cache_from_env()

    if args.dataset_ids is not None and len(args.dataset_ids) > 0:
        dataset_ids = list(args.dataset_ids)
    else:
        dataset_ids = select_openml_datasets_from_suite(
            args.openml_suite, args.max_datasets, args.max_rows, args.seed
        )

    print(f"Datasets to run ({len(dataset_ids)}): {dataset_ids}")
    if os.environ.get("OPENML_CACHE_DIR"):
        print(f"OPENML_CACHE_DIR set to: {os.environ.get('OPENML_CACHE_DIR')}")

    all_rows = []
    for did in dataset_ids:
        try:
            X, y, name = load_openml_dataset(did)
            if np.unique(y).size < 2:
                print(f"Skipping {name}: <2 classes")
                continue
            for frac in args.labeled_fracs:
                for seed in args.seeds:
                    all_rows.append(run_setting(did, name, X, y, frac, seed, cfg))
        except Exception as e:
            print(f"Skipping dataset {did} due to error: {e}")

    if not all_rows:
        raise SystemExit("No results produced (all datasets skipped).")


    # ensure parent directory exists for --out_csv
    out_parent = os.path.dirname(os.path.abspath(args.out_csv))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(args.out_csv, index=False)



    print(f"Wrote: {args.out_csv}")

    summary = (res.dropna(subset=["macroF1_test"])
                 .groupby(["labeled_frac", "method"])["macroF1_test"]
                 .agg(["mean", "std", "count"])
                 .reset_index()
                 .sort_values(["labeled_frac", "mean"], ascending=[True, False]))
    print(summary.to_string(index=False))

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        print(f"Wrote run logs to: {os.path.join(args.log_dir, 'run_log.jsonl')}")


if __name__ == "__main__":
    main()


