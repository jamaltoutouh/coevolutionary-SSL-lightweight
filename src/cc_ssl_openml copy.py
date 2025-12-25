#!/usr/bin/env python3
"""CC-SSL (Simple Variant) for Tabular Data (Mixed Binary + Multiclass) using OpenML.

- No neural networks: Logistic Regression base learners.
- Cooperative coevolution:
  Pop A: evolves two views (feature-subset masks over original columns)
  Pop B: evolves pseudo-label policy (threshold schedule, caps, disagreement veto, calibration flag)
- Fitness: macro-F1 on labeled validation + stability penalty + probe penalty (optional) + mild cost penalty.

Baselines:
- Self-training
- Heuristic co-training (random 50/50 split)
- Label Spreading (auto-skips if too large)

Install:
  pip install openml scikit-learn pandas numpy

Example:
  python cc_ssl_openml.py --openml_suite OpenML-CC18 --max_datasets 15 --max_rows 50000 \
      --labeled_fracs 0.01 0.05 0.10 --seeds 0 1 2 3 4 \
      --popA 12 --popB 12 --generations 12 --teams_per_individual 2 \
      --out_csv results_ccssl_openml.csv

Tip: use --run_ablations to run policy-only/view-only/no-probe/no-disagree variants.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
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


# -----------------------------
# OpenML loading
# -----------------------------
def load_openml_dataset(dataset_id: int) -> Tuple[pd.DataFrame, np.ndarray, str]:
    import openml

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
        y = y_codes.astype(int)
    else:
        y = y_series.to_numpy().astype(int)

    name = f"openml_{dataset_id}_{ds.name}"
    return X, y, name


def select_openml_datasets_from_suite(
    suite_name: str,
    max_datasets: int,
    max_rows: int,
    seed: int
) -> List[int]:
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


def make_lr_pipeline2(
    X_ref: pd.DataFrame,
    cols: List[str],
    calibrate: bool,
    y_for_calib: Optional[np.ndarray]
):
    pre = build_preprocessor_for_columns(X_ref, cols)
    base = Pipeline([
        ("prep", pre),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=400,
            multi_class="auto"
        ))
    ])

    if not calibrate or y_for_calib is None:
        return base

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

import inspect

def make_lr_pipeline(
    X_ref: pd.DataFrame,
    cols: List[str],
    calibrate: bool,
    y_for_calib: Optional[np.ndarray]
):
    pre = build_preprocessor_for_columns(X_ref, cols)

    lr_sig = inspect.signature(LogisticRegression).parameters
    lr_kwargs = dict(solver="lbfgs", max_iter=400)

    # Only set multi_class if available in this sklearn build
    if "multi_class" in lr_sig:
        lr_kwargs["multi_class"] = "auto"

    base = Pipeline([
        ("prep", pre),
        ("clf", LogisticRegression(**lr_kwargs))
    ])

    # (rest of your calibration logic unchanged)
    if not calibrate or y_for_calib is None:
        return base

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
# Variation
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
# CC-SSL co-training (simple)
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
) -> Tuple[float, float, int]:
    d = len(columns)
    cols1 = [columns[i] for i in range(d) if view.mask1[i]]
    cols2 = [columns[i] for i in range(d) if view.mask2[i]]

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
        return 0.0, 1.0, 0

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

        sel1_local = select_pseudolabels(proba1, yhat1, tau, policy.max_add_total, policy.max_add_per_class, policy.class_balance)
        sel2_local = select_pseudolabels(proba2, yhat2, tau, policy.max_add_total, policy.max_add_per_class, policy.class_balance)
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

    try:
        p1 = model1.predict_proba(X_eval[cols1])
        p2 = model2.predict_proba(X_eval[cols2])
        ye = np.argmax((p1 + p2) / 2.0, axis=1)
        f1_eval = macro_f1(y_eval, ye)
    except Exception:
        f1_eval = 0.0

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

    return float(f1_eval), float(probe_drop), int(total_added)


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
    return macro_f1(y_T, y_pred)


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
    f1, _, _ = cotraining_ccssl(X_L, y_L, X_U, X_T, y_T, None, None, columns, view, policy, rng)
    return f1


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

    y_all = np.concatenate([y_L, -np.ones(X_U.shape[0], dtype=int)])
    ls = LabelSpreading(kernel="rbf", gamma=20, max_iter=30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls.fit(X_mat, y_all)
    X_T_mat = pre.transform(X_T[columns])
    y_pred = ls.predict(X_T_mat)
    return macro_f1(y_T, y_pred)


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
    n_L = max(2 * n_classes, int(round(n_pool * labeled_frac)))
    n_L = min(n_L, n_pool - max(30, probe_size + 30))

    perm = rng.permutation(n_pool)
    L_idx = perm[:n_L]
    rest = perm[n_L:]

    n_P = min(probe_size, max(0, rest.size - 30))
    P_idx = rest[:n_P]
    U_idx = rest[n_P:]

    X_L = X_pool.iloc[L_idx].reset_index(drop=True)
    y_L = y_pool[L_idx]
    X_P = X_pool.iloc[P_idx].reset_index(drop=True) if n_P > 0 else None
    y_P = y_pool[P_idx] if n_P > 0 else None
    X_U = X_pool.iloc[U_idx].reset_index(drop=True)

    return dict(X_L=X_L, y_L=y_L, X_U=X_U, X_V=X_V, y_V=y_V, X_P=X_P, y_P=y_P, X_T=X_te, y_T=y_te)


# -----------------------------
# Coevolution
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
) -> Tuple[float, EvalStats]:
    f1s = []
    drops = []
    adds = []
    t0 = time.time()
    for s in eval_seeds:
        rng = np.random.default_rng(s)
        f1, drop, added = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_V"], splits["y_V"],
            splits["X_P"], splits["y_P"],
            columns, view, policy, rng,
            max_u_fit=max_u_fit
        )
        f1s.append(f1); drops.append(drop); adds.append(added)
    elapsed = time.time() - t0
    f1s = np.array(f1s); drops = np.array(drops); adds = np.array(adds)

    stats = EvalStats(
        mean_f1=float(f1s.mean()),
        std_f1=float(f1s.std()),
        mean_probe_drop=float(drops.mean()),
        mean_added=float(adds.mean()),
        mean_seconds=float(elapsed)
    )

    lam_std = 0.30
    lam_bias = 0.80 if use_probe_penalty else 0.0
    lam_added = 0.0008
    fitness = stats.mean_f1 - lam_std * stats.std_f1 - lam_bias * stats.mean_probe_drop - lam_added * stats.mean_added
    return float(fitness), stats


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
    force_no_disagree: bool = False
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
                f, stats = evaluate_team(splits, columns, a, b, eval_seeds, max_u_fit, use_probe_penalty)
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
                f, stats = evaluate_team(splits, columns, a, bb, eval_seeds, max_u_fit, use_probe_penalty)
                vals.append(f)
                if f > best["fitness"]:
                    best = dict(fitness=f, view=a.clone(), policy=bb.clone(), stats=stats)
            B_fit[j] = float(np.mean(vals))

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
def run_setting(
    dataset_id: int,
    dataset_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    labeled_frac: float,
    seed: int,
    cfg: Dict[str, Any]
) -> pd.DataFrame:
    if X.shape[0] > cfg["max_rows"]:
        rng = np.random.default_rng(seed)
        take = rng.choice(X.shape[0], size=cfg["max_rows"], replace=False)
        X = X.iloc[take].reset_index(drop=True)
        y = y[take]

    if np.unique(y).size < 2:
        raise ValueError("Dataset has <2 classes after filtering.")

    splits = make_ssl_splits(X, y, labeled_frac=labeled_frac, seed=seed, probe_size=cfg["probe_size"])
    columns = list(X.columns)

    best = coevolve_ccssl(
        splits=splits,
        columns=columns,
        popA_size=cfg["popA"],
        popB_size=cfg["popB"],
        generations=cfg["generations"],
        teams_per_individual=cfg["teams_per_individual"],
        seed=seed,
        eval_seeds=[seed + 101, seed + 103, seed + 107],
        max_u_fit=cfg["max_u_fit"],
        use_probe_penalty=True,
        force_no_disagree=False
    )

    f1_test, probe_drop, added = cotraining_ccssl(
        splits["X_L"], splits["y_L"], splits["X_U"],
        splits["X_T"], splits["y_T"],
        splits["X_P"], splits["y_P"],
        columns, best["view"], best["policy"],
        rng=np.random.default_rng(seed + 999),
        max_u_fit=cfg["max_u_final"]
    )

    st = self_training_baseline(
        splits["X_L"].copy(), splits["y_L"].copy(),
        splits["X_U"].copy(),
        splits["X_T"], splits["y_T"],
        columns, calibrate=True,
        rng=np.random.default_rng(seed + 555)
    )
    hct = heuristic_cotraining_baseline(
        splits["X_L"], splits["y_L"],
        splits["X_U"],
        splits["X_T"], splits["y_T"],
        columns,
        rng=np.random.default_rng(seed + 777)
    )
    ls = label_spreading_baseline(
        splits["X_L"], splits["y_L"],
        splits["X_U"],
        splits["X_T"], splits["y_T"],
        columns,
        max_total=cfg["label_spreading_max_total"],
        rng=np.random.default_rng(seed + 888)
    )

    rows = []

    def add_row(method: str, f1: float, extra: Dict[str, Any]):
        base = dict(dataset_id=dataset_id, dataset=dataset_name, labeled_frac=labeled_frac, seed=seed,
                    method=method, macroF1_test=float(f1) if f1 == f1 else np.nan)
        base.update(extra)
        rows.append(base)

    add_row("CC-SSL (coevolved)", f1_test, dict(
        probe_drop=float(probe_drop),
        pseudo_added=int(added),
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
    ))
    add_row("Self-training", st, dict())
    add_row("Heuristic co-training", hct, dict())
    if ls is not None:
        add_row("Label Spreading", ls, dict())
    else:
        add_row("Label Spreading (skipped)", np.nan, dict(reason="too_large"))

    # Optional ablations
    if cfg["run_ablations"]:
        rng = np.random.default_rng(seed + 2025)
        fixed_view = ViewGenome.random(len(columns), rng, frac=0.5, min_features=3)

        # policy-only
        best_pol = coevolve_ccssl(
            splits=splits, columns=columns,
            popA_size=1, popB_size=cfg["popB"],
            generations=max(8, cfg["generations"]),
            teams_per_individual=cfg["teams_per_individual"],
            seed=seed + 10,
            eval_seeds=[seed + 111, seed + 113, seed + 117],
            max_u_fit=cfg["max_u_fit"],
            use_probe_penalty=True,
            force_no_disagree=False
        )
        best_pol["view"] = fixed_view
        f1_pol, _, _ = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_T"], splits["y_T"],
            splits["X_P"], splits["y_P"],
            columns, fixed_view, best_pol["policy"],
            rng=np.random.default_rng(seed + 4040),
            max_u_fit=cfg["max_u_final"]
        )
        add_row("Ablation: policy-only", f1_pol, dict())

        # view-only
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
            force_no_disagree=False
        )
        best_view["policy"] = fixed_policy
        f1_view, _, _ = cotraining_ccssl(
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
            force_no_disagree=False
        )
        f1_nop, _, _ = cotraining_ccssl(
            splits["X_L"], splits["y_L"], splits["X_U"],
            splits["X_T"], splits["y_T"],
            splits["X_P"], splits["y_P"],
            columns, best_nop["view"], best_nop["policy"],
            rng=np.random.default_rng(seed + 6060),
            max_u_fit=cfg["max_u_final"]
        )
        add_row("Ablation: no probe penalty", f1_nop, dict())

        # no disagreement
        best_nod = coevolve_ccssl(
            splits=splits, columns=columns,
            popA_size=cfg["popA"], popB_size=cfg["popB"],
            generations=cfg["generations"],
            teams_per_individual=cfg["teams_per_individual"],
            seed=seed + 40,
            eval_seeds=[seed + 141, seed + 143, seed + 147],
            max_u_fit=cfg["max_u_fit"],
            use_probe_penalty=True,
            force_no_disagree=True
        )
        f1_nod, _, _ = cotraining_ccssl(
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

    ap.add_argument("--out_csv", type=str, default="results_ccssl_openml.csv")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    cfg = dict(
        popA=args.popA, popB=args.popB, generations=args.generations,
        teams_per_individual=args.teams_per_individual,
        max_u_fit=args.max_u_fit, max_u_final=args.max_u_final,
        probe_size=args.probe_size, max_rows=args.max_rows,
        label_spreading_max_total=args.label_spreading_max_total,
        run_ablations=bool(args.run_ablations),
    )

    if args.dataset_ids is not None and len(args.dataset_ids) > 0:
        dataset_ids = list(args.dataset_ids)
    else:
        dataset_ids = select_openml_datasets_from_suite(args.openml_suite, args.max_datasets, args.max_rows, args.seed)

    print(f"Datasets to run ({len(dataset_ids)}): {dataset_ids}")

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

    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")

    summary = (res.dropna(subset=["macroF1_test"])
                 .groupby(["labeled_frac", "method"])["macroF1_test"]
                 .agg(["mean", "std", "count"])
                 .reset_index()
                 .sort_values(["labeled_frac", "mean"], ascending=[True, False]))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
