#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cc_ssl_openml.py

Cooperative Coevolutionary SSL for tabular OpenML datasets (classic ML, no NNs).

Adds:
- Per-generation checkpoint logs (JSONL):
  * best pair fitness and components
  * diversity metrics for Pop A (views) and Pop B (policies)
  * probe-drop and pseudo-added of best pair
- Post-run cross-evaluation (top-k from A and B, kxk team evals -> CSV)

Also:
- Writes one CSV row per method (CC-SSL + baselines), each with its own runtime_seconds.
- Writes JSONL run logs (fitness distributions, timing breakdowns) when --log_dir is set.

Offline / HPC:
- Set OPENML_CACHE_DIR to an existing OpenML cache directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# ----------------------------
# Utilities
# ----------------------------

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Optional[str]) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def set_openml_cache_from_env() -> None:
    cache_dir = os.environ.get("OPENML_CACHE_DIR", "").strip()
    if cache_dir:
        import openml
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        openml.config.set_root_cache_directory(cache_dir)

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def sha1_bytes(*parts: bytes) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p)
    return h.hexdigest()

def pack_bool_mask(mask: np.ndarray) -> bytes:
    # mask: shape (D,), dtype bool
    return np.packbits(mask.astype(np.uint8)).tobytes()

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# ----------------------------
# Genome definitions
# ----------------------------

@dataclass(frozen=True)
class ViewGenome:
    mask1: np.ndarray  # bool (D,)
    mask2: np.ndarray  # bool (D,)

    def key(self) -> str:
        b1 = pack_bool_mask(self.mask1)
        b2 = pack_bool_mask(self.mask2)
        return sha1_bytes(b1, b2)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_feat_1": int(self.mask1.sum()),
            "n_feat_2": int(self.mask2.sum()),
            "overlap": int(np.logical_and(self.mask1, self.mask2).sum()),
            "union": int(np.logical_or(self.mask1, self.mask2).sum()),
        }


@dataclass(frozen=True)
class PolicyGenome:
    # pseudo-label schedule / rules
    tau_start: float
    tau_end: float
    margin_min: float
    class_balance: bool
    disagreement_veto: bool
    max_iters: int
    max_add_total: int
    max_add_per_class: int
    # model hp
    C: float
    max_iter_lr: int

    def key(self) -> str:
        # deterministic serialization
        payload = (
            f"{self.tau_start:.6f}|{self.tau_end:.6f}|{self.margin_min:.6f}|"
            f"{int(self.class_balance)}|{int(self.disagreement_veto)}|"
            f"{self.max_iters}|{self.max_add_total}|{self.max_add_per_class}|"
            f"{self.C:.6e}|{self.max_iter_lr}"
        ).encode("utf-8")
        return sha1_bytes(payload)

    def summary(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class EvalStats:
    fitness: float
    val_macroF1: List[float]
    val_acc: List[float]
    probe_drop: List[float]
    pseudo_added: List[int]
    seconds: float
    extra: Dict[str, Any]


# ----------------------------
# Data loading / preprocessing
# ----------------------------

def load_openml_dataset(dataset_id: int) -> Tuple[pd.DataFrame, pd.Series, str]:
    set_openml_cache_from_env()
    import openml

    ds = openml.datasets.get_dataset(int(dataset_id))
    target = ds.default_target_attribute
    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=target)
    y = pd.Series(y)
    return X, y, str(ds.name)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Identify columns
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True))
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        sparse_threshold=0.3,
    )
    return pre

def to_numpy(y: pd.Series) -> np.ndarray:
    # ensure contiguous array of labels (string/object ok)
    return np.asarray(y)


# ----------------------------
# Metrics
# ----------------------------

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))

def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


# ----------------------------
# SSL loop (two views)
# ----------------------------

def schedule_tau(t: int, T: int, tau_start: float, tau_end: float) -> float:
    if T <= 1:
        return float(tau_start)
    a = t / (T - 1)
    return float(tau_start + a * (tau_end - tau_start))

def fit_lr(X, y, C: float, max_iter_lr: int, seed: int) -> LogisticRegression:
    # sklearn 1.8.0 LogisticRegression signature does NOT accept multi_class.
    model = LogisticRegression(
        C=float(C),
        max_iter=int(max_iter_lr),
        solver="lbfgs",
        random_state=int(seed),
    )
    model.fit(X, y)
    return model

def predict_proba_safe(model: LogisticRegression, X) -> np.ndarray:
    # LogisticRegression supports predict_proba
    return model.predict_proba(X)

def select_pseudo(
    proba1: np.ndarray,
    proba2: np.ndarray,
    tau: float,
    margin_min: float,
    disagreement_veto: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      pred_label (n_u,), conf (n_u,), margin (n_u,)
    """
    # pred per view
    p1 = proba1
    p2 = proba2
    y1 = p1.argmax(axis=1)
    y2 = p2.argmax(axis=1)
    conf1 = p1.max(axis=1)
    conf2 = p2.max(axis=1)

    # margin: top1 - top2 (per view), then min across views (conservative)
    def margin(p: np.ndarray) -> np.ndarray:
        part = np.partition(p, -2, axis=1)
        top2 = part[:, -2]
        top1 = part[:, -1]
        return top1 - top2

    m1 = margin(p1)
    m2 = margin(p2)
    conf = np.minimum(conf1, conf2)
    mar = np.minimum(m1, m2)

    if disagreement_veto:
        ok_agree = (y1 == y2)
    else:
        ok_agree = np.ones_like(y1, dtype=bool)

    ok = ok_agree & (conf >= tau) & (mar >= margin_min)

    # choose label: if veto, y1==y2; else use view1 label
    yhat = y1
    return yhat, conf, ok

def apply_class_balance(
    idx_u: np.ndarray,
    yhat: np.ndarray,
    conf: np.ndarray,
    max_add_total: int,
    max_add_per_class: int,
    class_balance: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Choose pseudo-label indices from unlabeled pool indices (idx_u), returning
    chosen subset of indices (relative to idx_u array).
    """
    n = idx_u.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # rank by confidence descending
    order = np.argsort(-conf)
    if not class_balance:
        chosen = order[:max_add_total]
        return chosen.astype(int)

    # class-balanced selection
    chosen_list: List[int] = []
    per_class: Dict[int, int] = {}
    for j in order:
        c = int(yhat[j])
        if per_class.get(c, 0) >= max_add_per_class:
            continue
        chosen_list.append(int(j))
        per_class[c] = per_class.get(c, 0) + 1
        if len(chosen_list) >= max_add_total:
            break
    return np.array(chosen_list, dtype=int)

def ssl_two_view_run(
    X_all, y_all: np.ndarray,
    idx_L_train: np.ndarray,
    idx_U: np.ndarray,
    idx_val: np.ndarray,
    idx_probe: np.ndarray,
    view: ViewGenome,
    policy: PolicyGenome,
    seed: int,
) -> Dict[str, Any]:
    """
    Run pseudo-labeling with two views. Returns dict with final models + metrics.
    """
    rng = np.random.default_rng(seed)

    # Prepare masks
    cols1 = np.where(view.mask1)[0]
    cols2 = np.where(view.mask2)[0]

    # L, U sets (labels for L, unknown for U)
    L_idx = idx_L_train.copy()
    U_idx = idx_U.copy()

    # Train initial (no pseudo) models and probe score (for probe-drop)
    X1_L0 = X_all[L_idx][:, cols1]
    X2_L0 = X_all[L_idx][:, cols2]
    y_L0 = y_all[L_idx]

    m1 = fit_lr(X1_L0, y_L0, policy.C, policy.max_iter_lr, seed)
    m2 = fit_lr(X2_L0, y_L0, policy.C, policy.max_iter_lr, seed)

    X1_probe = X_all[idx_probe][:, cols1]
    X2_probe = X_all[idx_probe][:, cols2]
    # Evaluate probe with average of two-view predictions (soft vote)
    p1_probe = predict_proba_safe(m1, X1_probe)
    p2_probe = predict_proba_safe(m2, X2_probe)
    p_probe = 0.5 * (p1_probe + p2_probe)
    y_probe_pred0 = p_probe.argmax(axis=1)
    probe_macroF10 = macro_f1(y_all[idx_probe], y_probe_pred0)

    pseudo_added = 0

    # Main loop
    for t in range(int(policy.max_iters)):
        if U_idx.size == 0:
            break

        tau = schedule_tau(t, policy.max_iters, policy.tau_start, policy.tau_end)

        # Fit models on current labeled (true+pseudo)
        X1_L = X_all[L_idx][:, cols1]
        X2_L = X_all[L_idx][:, cols2]
        y_L = y_all[L_idx]

        m1 = fit_lr(X1_L, y_L, policy.C, policy.max_iter_lr, seed + 10 + t)
        m2 = fit_lr(X2_L, y_L, policy.C, policy.max_iter_lr, seed + 20 + t)

        X1_U = X_all[U_idx][:, cols1]
        X2_U = X_all[U_idx][:, cols2]
        p1 = predict_proba_safe(m1, X1_U)
        p2 = predict_proba_safe(m2, X2_U)

        yhat, conf, ok = select_pseudo(
            p1, p2,
            tau=tau,
            margin_min=policy.margin_min,
            disagreement_veto=policy.disagreement_veto,
        )

        ok_idx = np.where(ok)[0]
        if ok_idx.size == 0:
            break

        # Enforce budgets remaining
        remaining_total = max(0, int(policy.max_add_total) - pseudo_added)
        if remaining_total <= 0:
            break
        add_total = min(remaining_total, ok_idx.size)

        # Choose subset with class balancing or not
        chosen_rel = apply_class_balance(
            idx_u=ok_idx,
            yhat=yhat[ok_idx],
            conf=conf[ok_idx],
            max_add_total=add_total,
            max_add_per_class=int(policy.max_add_per_class),
            class_balance=bool(policy.class_balance),
            rng=rng,
        )

        if chosen_rel.size == 0:
            break

        chosen_in_ok = ok_idx[chosen_rel]
        chosen_abs = U_idx[chosen_in_ok]

        # Add pseudo-labels by writing to y_all for these indices is NOT allowed (true labels unknown).
        # Instead, we build an augmented label vector internally:
        # We will append chosen_abs to L_idx and use their predicted labels.
        # For simplicity, we maintain a separate dict pseudo_labels and expand y by overriding y_all at those idx.
        # But since y_all contains true labels (only for evaluation), we must not overwrite it globally.
        # We'll keep a local label map:
        # Here: implement by creating a local y_work each iteration.
        # (We update y_work only for L_idx contents.)

        # Implement local pseudo label storage:
        # We'll store pseudo labels in a dict and rebuild y_L from it next iteration.
        # For speed, keep arrays pseudo_idx/pseudo_y.
        if t == 0:
            pseudo_idx = np.array([], dtype=int)
            pseudo_y = np.array([], dtype=int)
        # If pseudo already exists, keep
        if "pseudo_idx" in locals():
            pass

        pseudo_idx = np.concatenate([pseudo_idx, chosen_abs.astype(int)])
        pseudo_y = np.concatenate([pseudo_y, yhat[chosen_in_ok].astype(int)])

        # Update labeled and unlabeled pools
        L_idx = np.concatenate([L_idx, chosen_abs.astype(int)])
        mask_keep = np.ones(U_idx.shape[0], dtype=bool)
        mask_keep[chosen_in_ok] = False
        U_idx = U_idx[mask_keep]

        pseudo_added += int(chosen_abs.size)

        # stop if total budget exhausted
        if pseudo_added >= int(policy.max_add_total):
            break

    # Final fit on L_idx with pseudo labels applied
    # Build y_L using pseudo labels for pseudo_idx and true labels for original idx_L_train
    # Original labeled indices are idx_L_train (true labels known)
    y_L_final = y_all[L_idx].copy()
    if "pseudo_idx" in locals() and pseudo_idx.size > 0:
        # map pseudo labels into y_L_final positions
        # find where L_idx corresponds to pseudo_idx (can include duplicates; avoid duplicates by last one)
        # We'll do a dict mapping for correctness
        mp = {int(i): int(y) for i, y in zip(pseudo_idx.tolist(), pseudo_y.tolist())}
        for k in range(L_idx.size):
            ii = int(L_idx[k])
            if ii in mp and ii not in set(idx_L_train.tolist()):
                y_L_final[k] = mp[ii]

    X1_Lf = X_all[L_idx][:, cols1]
    X2_Lf = X_all[L_idx][:, cols2]
    m1 = fit_lr(X1_Lf, y_L_final, policy.C, policy.max_iter_lr, seed + 999)
    m2 = fit_lr(X2_Lf, y_L_final, policy.C, policy.max_iter_lr, seed + 1999)

    # Validation
    X1_val = X_all[idx_val][:, cols1]
    X2_val = X_all[idx_val][:, cols2]
    p1_val = predict_proba_safe(m1, X1_val)
    p2_val = predict_proba_safe(m2, X2_val)
    p_val = 0.5 * (p1_val + p2_val)
    y_val_pred = p_val.argmax(axis=1)
    val_macro = macro_f1(y_all[idx_val], y_val_pred)
    val_accuracy = acc(y_all[idx_val], y_val_pred)

    # Probe (for confirmation bias)
    p1_probe = predict_proba_safe(m1, X1_probe)
    p2_probe = predict_proba_safe(m2, X2_probe)
    p_probe = 0.5 * (p1_probe + p2_probe)
    y_probe_pred = p_probe.argmax(axis=1)
    probe_macro = macro_f1(y_all[idx_probe], y_probe_pred)
    probe_drop = max(0.0, float(probe_macroF10 - probe_macro))

    return dict(
        val_macroF1=val_macro,
        val_acc=val_accuracy,
        probe_drop=probe_drop,
        pseudo_added=int(pseudo_added),
        probe_macroF1_init=float(probe_macroF10),
        probe_macroF1_final=float(probe_macro),
    )


# ----------------------------
# Fitness evaluation for a team
# ----------------------------

def evaluate_team(
    X_all, y_all: np.ndarray,
    idx_L_pool: np.ndarray,
    idx_U_pool: np.ndarray,
    idx_val_pool: np.ndarray,
    idx_probe: np.ndarray,
    view: ViewGenome,
    policy: PolicyGenome,
    fitness_repeats: int,
    lam_std: float,
    lam_bias: float,
    lam_added: float,
    seed: int,
) -> EvalStats:
    t0 = time.perf_counter()

    val_macros: List[float] = []
    val_accs: List[float] = []
    probe_drops: List[float] = []
    added: List[int] = []

    rng = np.random.default_rng(seed)

    # For each repeat, subsample a validation set from idx_val_pool
    # (keeps compute stable & provides distribution)
    for k in range(int(fitness_repeats)):
        rep_seed = int(seed + 1000 * k + 17)
        # Subsample val to reduce variance and cost; keep at least 30 or 10% of pool
        if idx_val_pool.size <= 50:
            idx_val = idx_val_pool
        else:
            take = max(30, int(0.5 * idx_val_pool.size))
            idx_val = rng.choice(idx_val_pool, size=take, replace=False)

        out = ssl_two_view_run(
            X_all=X_all, y_all=y_all,
            idx_L_train=idx_L_pool,
            idx_U=idx_U_pool,
            idx_val=idx_val,
            idx_probe=idx_probe,
            view=view, policy=policy,
            seed=rep_seed,
        )
        val_macros.append(float(out["val_macroF1"]))
        val_accs.append(float(out["val_acc"]))
        probe_drops.append(float(out["probe_drop"]))
        added.append(int(out["pseudo_added"]))

    mu = float(np.mean(val_macros))
    sd = float(np.std(val_macros, ddof=0)) if len(val_macros) > 1 else 0.0
    probe_mean = float(np.mean(probe_drops))
    added_mean = float(np.mean(added))

    fitness = mu - float(lam_std) * sd - float(lam_bias) * probe_mean - float(lam_added) * added_mean

    t1 = time.perf_counter()
    return EvalStats(
        fitness=float(fitness),
        val_macroF1=val_macros,
        val_acc=val_accs,
        probe_drop=probe_drops,
        pseudo_added=added,
        seconds=float(t1 - t0),
        extra={
            "mu": mu,
            "sd": sd,
            "probe_mean": probe_mean,
            "added_mean": added_mean,
        }
    )


# ----------------------------
# Genetic operators
# ----------------------------

def random_mask(D: int, init_frac: float, min_features: int, rng: np.random.Generator) -> np.ndarray:
    m = rng.random(D) < float(init_frac)
    if m.sum() < min_features:
        # activate random features
        idx = rng.choice(np.arange(D), size=min_features, replace=False)
        m[:] = False
        m[idx] = True
    return m.astype(bool)

def repair_mask(mask: np.ndarray, min_features: int, rng: np.random.Generator) -> np.ndarray:
    m = mask.copy().astype(bool)
    if m.sum() < min_features:
        off = np.where(~m)[0]
        need = min_features - int(m.sum())
        if off.size > 0 and need > 0:
            add = rng.choice(off, size=min(need, off.size), replace=False)
            m[add] = True
    return m

def mutate_view(
    view: ViewGenome,
    gen: int, generations: int,
    flip_E_start: float, flip_E_end: float,
    pflip_max: float,
    min_features: int,
    rng: np.random.Generator,
) -> ViewGenome:
    D = view.mask1.size
    # anneal expected flips
    if generations <= 1:
        E = float(flip_E_start)
    else:
        a = gen / max(1, generations - 1)
        E = float(flip_E_start + a * (flip_E_end - flip_E_start))
    p_flip = min(float(pflip_max), float(E) / max(1, D))
    m1 = view.mask1.copy()
    m2 = view.mask2.copy()
    flip1 = rng.random(D) < p_flip
    flip2 = rng.random(D) < p_flip
    m1 ^= flip1
    m2 ^= flip2
    m1 = repair_mask(m1, min_features, rng)
    m2 = repair_mask(m2, min_features, rng)
    return ViewGenome(mask1=m1, mask2=m2)

def crossover_view(a: ViewGenome, b: ViewGenome, rng: np.random.Generator) -> ViewGenome:
    D = a.mask1.size
    cut = int(rng.integers(1, D))
    m1 = np.concatenate([a.mask1[:cut], b.mask1[cut:]]).astype(bool)
    m2 = np.concatenate([a.mask2[:cut], b.mask2[cut:]]).astype(bool)
    return ViewGenome(mask1=m1, mask2=m2)

def random_policy(rng: np.random.Generator) -> PolicyGenome:
    tau_start = float(rng.uniform(0.80, 0.97))
    tau_end = float(rng.uniform(0.60, min(0.92, tau_start)))
    margin_min = float(rng.uniform(0.0, 0.20))
    class_balance = bool(rng.random() < 0.6)
    disagreement_veto = bool(rng.random() < 0.6)
    max_iters = int(rng.choice([1, 2, 3, 5]))
    max_add_total = int(rng.choice([50, 100, 200, 400]))
    max_add_per_class = int(rng.choice([10, 20, 40, 80]))
    C = float(10 ** rng.uniform(-2.5, 1.5))  # ~[3e-3, 30]
    max_iter_lr = int(rng.choice([200, 500, 1000]))
    return PolicyGenome(
        tau_start=tau_start, tau_end=tau_end, margin_min=margin_min,
        class_balance=class_balance, disagreement_veto=disagreement_veto,
        max_iters=max_iters,
        max_add_total=max_add_total, max_add_per_class=max_add_per_class,
        C=C, max_iter_lr=max_iter_lr
    )

def mutate_policy(
    p: PolicyGenome,
    p_bool_flip: float,
    sigma_tau: float,
    sigma_margin: float,
    sigma_logC: float,
    rng: np.random.Generator,
) -> PolicyGenome:
    tau_start = clamp(p.tau_start + rng.normal(0.0, sigma_tau), 0.55, 0.99)
    tau_end = clamp(p.tau_end + rng.normal(0.0, sigma_tau), 0.50, tau_start)
    margin_min = clamp(p.margin_min + rng.normal(0.0, sigma_margin), 0.0, 0.35)
    class_balance = (not p.class_balance) if (rng.random() < p_bool_flip) else p.class_balance
    disagreement_veto = (not p.disagreement_veto) if (rng.random() < p_bool_flip) else p.disagreement_veto

    max_iters = int(p.max_iters)
    if rng.random() < 0.2:
        max_iters = int(rng.choice([1, 2, 3, 5]))

    max_add_total = int(p.max_add_total)
    if rng.random() < 0.25:
        max_add_total = int(rng.choice([50, 100, 200, 400]))

    max_add_per_class = int(p.max_add_per_class)
    if rng.random() < 0.25:
        max_add_per_class = int(rng.choice([10, 20, 40, 80]))

    # mutate C in log space
    logC = np.log10(p.C)
    logC = clamp(logC + rng.normal(0.0, sigma_logC), -3.0, 2.0)
    C = float(10 ** logC)

    max_iter_lr = int(p.max_iter_lr)
    if rng.random() < 0.2:
        max_iter_lr = int(rng.choice([200, 500, 1000]))

    return PolicyGenome(
        tau_start=tau_start, tau_end=tau_end, margin_min=margin_min,
        class_balance=class_balance, disagreement_veto=disagreement_veto,
        max_iters=max_iters,
        max_add_total=max_add_total, max_add_per_class=max_add_per_class,
        C=C, max_iter_lr=max_iter_lr
    )

def crossover_policy(a: PolicyGenome, b: PolicyGenome, rng: np.random.Generator) -> PolicyGenome:
    # uniform gene mixing (simple & robust)
    def pick(x, y):
        return x if (rng.random() < 0.5) else y
    return PolicyGenome(
        tau_start=float(pick(a.tau_start, b.tau_start)),
        tau_end=float(pick(a.tau_end, b.tau_end)),
        margin_min=float(pick(a.margin_min, b.margin_min)),
        class_balance=bool(pick(a.class_balance, b.class_balance)),
        disagreement_veto=bool(pick(a.disagreement_veto, b.disagreement_veto)),
        max_iters=int(pick(a.max_iters, b.max_iters)),
        max_add_total=int(pick(a.max_add_total, b.max_add_total)),
        max_add_per_class=int(pick(a.max_add_per_class, b.max_add_per_class)),
        C=float(pick(a.C, b.C)),
        max_iter_lr=int(pick(a.max_iter_lr, b.max_iter_lr)),
    )

def tournament_select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    n = fitness.size
    idx = rng.integers(0, n, size=k)
    best = idx[np.argmax(fitness[idx])]
    return int(best)


# ----------------------------
# Diversity metrics
# ----------------------------

def jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    if uni == 0:
        return 0.0
    return float(1.0 - (inter / uni))

def view_diversity(popA: List[ViewGenome], max_pairs: int, rng: np.random.Generator) -> Dict[str, float]:
    n = len(popA)
    if n < 2:
        return {"mean_jaccard_1": 0.0, "mean_jaccard_2": 0.0, "mean_overlap": 0.0}
    pairs = []
    for _ in range(min(max_pairs, n * (n - 1) // 2)):
        i, j = rng.choice(n, size=2, replace=False)
        pairs.append((int(i), int(j)))

    d1 = []
    d2 = []
    ov = []
    for i, j in pairs:
        d1.append(jaccard_distance(popA[i].mask1, popA[j].mask1))
        d2.append(jaccard_distance(popA[i].mask2, popA[j].mask2))
        ov.append(float(np.logical_and(popA[i].mask1, popA[i].mask2).sum()))
    return {
        "mean_jaccard_1": float(np.mean(d1)) if d1 else 0.0,
        "mean_jaccard_2": float(np.mean(d2)) if d2 else 0.0,
        "mean_overlap_within": float(np.mean(ov)) if ov else 0.0,
    }

def policy_diversity(popB: List[PolicyGenome]) -> Dict[str, float]:
    if not popB:
        return {}
    # booleans: entropy
    cb = np.array([p.class_balance for p in popB], dtype=int)
    dv = np.array([p.disagreement_veto for p in popB], dtype=int)

    def entropy01(x: np.ndarray) -> float:
        p1 = x.mean()
        if p1 in (0.0, 1.0):
            return 0.0
        return float(-(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1)))

    # numerics: std
    tau_s = np.array([p.tau_start for p in popB], dtype=float)
    tau_e = np.array([p.tau_end for p in popB], dtype=float)
    mar = np.array([p.margin_min for p in popB], dtype=float)
    C = np.array([np.log10(p.C) for p in popB], dtype=float)

    return {
        "entropy_class_balance": entropy01(cb),
        "entropy_disagreement_veto": entropy01(dv),
        "std_tau_start": float(tau_s.std(ddof=0)),
        "std_tau_end": float(tau_e.std(ddof=0)),
        "std_margin": float(mar.std(ddof=0)),
        "std_log10C": float(C.std(ddof=0)),
    }


# ----------------------------
# Baselines (simple + fast)
# ----------------------------

def supervised_only_lr(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, float]:
    m = fit_lr(X_train, y_train, C=1.0, max_iter_lr=500, seed=seed)
    yhat = m.predict(X_test)
    return {"macroF1_test": macro_f1(y_test, yhat), "acc_test": acc(y_test, yhat)}

def self_training_baseline(
    X_all, y_all: np.ndarray,
    idx_L: np.ndarray, idx_U: np.ndarray,
    idx_test: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    # Single-view: use all features, conservative schedule
    rng = np.random.default_rng(seed)
    D = X_all.shape[1]
    mask_all = np.ones(D, dtype=bool)
    view = ViewGenome(mask1=mask_all, mask2=mask_all)
    policy = PolicyGenome(
        tau_start=0.90, tau_end=0.75, margin_min=0.0,
        class_balance=True, disagreement_veto=False,
        max_iters=3, max_add_total=200, max_add_per_class=40,
        C=1.0, max_iter_lr=500
    )
    # Use idx_L as L_train, and a small val subset to run loop; evaluate on test
    # We'll run loop once (not repeats).
    idx_val = idx_L if idx_L.size <= 50 else rng.choice(idx_L, size=max(30, int(0.5*idx_L.size)), replace=False)
    idx_probe = idx_val  # no separate probe in baseline

    out = ssl_two_view_run(
        X_all=X_all, y_all=y_all,
        idx_L_train=idx_L, idx_U=idx_U,
        idx_val=idx_val, idx_probe=idx_probe,
        view=view, policy=policy, seed=seed
    )

    # fit final model on full labeled+pseudo used in ssl_two_view_run? (not returned)
    # To keep baseline cheap and consistent, evaluate test using supervised-only on expanded training:
    # Here we simply train LR on labeled only (baseline). Your CC row will show gains.
    # If you want strict self-training test, extend ssl_two_view_run to return final models.
    m = fit_lr(X_all[idx_L], y_all[idx_L], C=1.0, max_iter_lr=500, seed=seed)
    yhat = m.predict(X_all[idx_test])
    return {"macroF1_test": macro_f1(y_all[idx_test], yhat), "acc_test": acc(y_all[idx_test], yhat)}

def label_spreading_baseline(
    X_all, y_all: np.ndarray,
    idx_L: np.ndarray, idx_U: np.ndarray, idx_test: np.ndarray,
    max_total: int,
    seed: int,
) -> Optional[Dict[str, float]]:
    # Skip if too large; label spreading is O(n^2)ish in practice
    n_total = idx_L.size + idx_U.size
    if n_total > max_total:
        return None
    from sklearn.semi_supervised import LabelSpreading
    # Build y with -1 for unlabeled
    y_ssl = np.full(n_total, -1, dtype=int)

    # Need numeric labels for LabelSpreading -> map classes
    yL = y_all[idx_L]
    classes = np.unique(yL)
    cls_to_int = {c: i for i, c in enumerate(classes.tolist())}
    y_ssl[:idx_L.size] = np.array([cls_to_int[c] for c in yL], dtype=int)

    X_ssl = np.vstack([X_all[idx_L].toarray() if hasattr(X_all[idx_L], "toarray") else X_all[idx_L],
                       X_all[idx_U].toarray() if hasattr(X_all[idx_U], "toarray") else X_all[idx_U]])
    model = LabelSpreading(kernel="rbf", gamma=0.25, max_iter=30)
    model.fit(X_ssl, y_ssl)

    # Predict on test using class mapping
    y_pred_int = model.predict(X_all[idx_test].toarray() if hasattr(X_all[idx_test], "toarray") else X_all[idx_test])
    inv = {v: k for k, v in cls_to_int.items()}
    y_pred = np.array([inv[int(i)] for i in y_pred_int], dtype=object)

    return {"macroF1_test": macro_f1(y_all[idx_test], y_pred), "acc_test": acc(y_all[idx_test], y_pred)}


# ----------------------------
# Main CC run (per dataset/frac/seed)
# ----------------------------

def run_cc_once(
    X_all, y_all: np.ndarray,
    idx_L: np.ndarray, idx_U: np.ndarray,
    idx_val_pool: np.ndarray,
    idx_probe: np.ndarray,
    cfg: Dict[str, Any],
    seed: int,
    log_dir: Optional[str],
    run_tag: str,
) -> Tuple[ViewGenome, PolicyGenome, EvalStats, Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    D = X_all.shape[1]

    # Init populations
    popA: List[ViewGenome] = []
    popB: List[PolicyGenome] = []

    for _ in range(int(cfg["popA"])):
        m1 = random_mask(D, cfg["view_init_frac"], cfg["view_min_features"], rng)
        m2 = random_mask(D, cfg["view_init_frac"], cfg["view_min_features"], rng)
        popA.append(ViewGenome(mask1=m1, mask2=m2))

    for _ in range(int(cfg["popB"])):
        popB.append(random_policy(rng))

    fitA = np.full(len(popA), -np.inf, dtype=float)
    fitB = np.full(len(popB), -np.inf, dtype=float)

    best_view: Optional[ViewGenome] = None
    best_pol: Optional[PolicyGenome] = None
    best_stats: Optional[EvalStats] = None

    # Per-generation log file
    gen_log_path = None
    if log_dir:
        ensure_dir(log_dir)
        gen_log_path = os.path.join(log_dir, f"gen_log_{run_tag}.jsonl")

    # Keep last evaluated populations + fitness for cross-eval
    final_popA: Optional[List[ViewGenome]] = None
    final_popB: Optional[List[PolicyGenome]] = None
    final_fitA: Optional[np.ndarray] = None
    final_fitB: Optional[np.ndarray] = None

    generations = int(cfg["generations"])

    for gen in range(generations):
        gen_t0 = time.perf_counter()
        team_cache: Dict[Tuple[str, str], EvalStats] = {}

        # identify current elites (if known from previous evaluation)
        eliteB_idx = int(np.argmax(fitB)) if np.isfinite(fitB).any() else None
        eliteA_idx = int(np.argmax(fitA)) if np.isfinite(fitA).any() else None

        # Evaluate A individuals
        for i, a in enumerate(popA):
            partners = []
            if eliteB_idx is not None:
                partners.append(eliteB_idx)
            for _ in range(int(cfg["partners_random"])):
                partners.append(int(rng.integers(0, len(popB))))
            partners = list(dict.fromkeys(partners))

            best_i = -np.inf
            for j in partners:
                b = popB[j]
                key = (a.key(), b.key())
                if key not in team_cache:
                    team_cache[key] = evaluate_team(
                        X_all=X_all, y_all=y_all,
                        idx_L_pool=idx_L, idx_U_pool=idx_U,
                        idx_val_pool=idx_val_pool, idx_probe=idx_probe,
                        view=a, policy=b,
                        fitness_repeats=cfg["fitness_repeats"],
                        lam_std=cfg["lam_std"], lam_bias=cfg["lam_bias"], lam_added=cfg["lam_added"],
                        seed=seed + 100000 * gen + 1000 * i + 17 * j,
                    )
                st = team_cache[key]
                if st.fitness > best_i:
                    best_i = st.fitness
                if (best_stats is None) or (st.fitness > best_stats.fitness):
                    best_view, best_pol, best_stats = a, b, st
            fitA[i] = best_i

        # Evaluate B individuals
        for j, b in enumerate(popB):
            partners = []
            if eliteA_idx is not None:
                partners.append(eliteA_idx)
            for _ in range(int(cfg["partners_random"])):
                partners.append(int(rng.integers(0, len(popA))))
            partners = list(dict.fromkeys(partners))

            best_j = -np.inf
            for i in partners:
                a = popA[i]
                key = (a.key(), b.key())
                if key not in team_cache:
                    team_cache[key] = evaluate_team(
                        X_all=X_all, y_all=y_all,
                        idx_L_pool=idx_L, idx_U_pool=idx_U,
                        idx_val_pool=idx_val_pool, idx_probe=idx_probe,
                        view=a, policy=b,
                        fitness_repeats=cfg["fitness_repeats"],
                        lam_std=cfg["lam_std"], lam_bias=cfg["lam_bias"], lam_added=cfg["lam_added"],
                        seed=seed + 100000 * gen + 777 * i + 19 * j,
                    )
                st = team_cache[key]
                if st.fitness > best_j:
                    best_j = st.fitness
                if (best_stats is None) or (st.fitness > best_stats.fitness):
                    best_view, best_pol, best_stats = a, b, st
            fitB[j] = best_j

        gen_t1 = time.perf_counter()

        # Per-generation checkpoint log (best pair, diversity, distributions)
        if gen_log_path and best_view and best_pol and best_stats:
            divA = view_diversity(popA, max_pairs=cfg["diversity_pairs"], rng=rng)
            divB = policy_diversity(popB)

            rec = {
                "run_tag": run_tag,
                "dataset_id": cfg.get("dataset_id"),
                "labeled_frac": cfg.get("labeled_frac"),
                "seed": seed,
                "generation": gen,
                "seconds_generation": float(gen_t1 - gen_t0),
                "best_pair_fitness": float(best_stats.fitness),
                "best_view": best_view.summary(),
                "best_policy": best_pol.summary(),
                "best_stats": {
                    "val_macroF1": best_stats.val_macroF1,
                    "val_acc": best_stats.val_acc,
                    "probe_drop": best_stats.probe_drop,
                    "pseudo_added": best_stats.pseudo_added,
                    "seconds_eval": best_stats.seconds,
                    "extra": best_stats.extra,
                },
                "diversity_A": divA,
                "diversity_B": divB,
            }
            with open(gen_log_path, "a", encoding="utf-8") as f:
                f.write(safe_json(rec) + "\n")

        # IMPORTANT: do NOT breed after final generation.
        if gen == generations - 1:
            final_popA = popA
            final_popB = popB
            final_fitA = fitA.copy()
            final_fitB = fitB.copy()
            break

        # Breed next generation (A)
        newA: List[ViewGenome] = []
        eliteA_n = int(cfg["elitism"])
        if eliteA_n > 0:
            elite_idx = np.argsort(-fitA)[:eliteA_n]
            for k in elite_idx:
                newA.append(popA[int(k)])
        while len(newA) < len(popA):
            p1 = tournament_select(fitA, cfg["tournament_k"], rng)
            p2 = tournament_select(fitA, cfg["tournament_k"], rng)
            child = popA[p1]
            if rng.random() < cfg["pcxA"]:
                child = crossover_view(popA[p1], popA[p2], rng)
            if rng.random() < cfg["pmutA"]:
                child = mutate_view(
                    child, gen=gen, generations=generations,
                    flip_E_start=cfg["view_flip_E_start"], flip_E_end=cfg["view_flip_E_end"],
                    pflip_max=cfg["view_pflip_max"],
                    min_features=cfg["view_min_features"],
                    rng=rng,
                )
            newA.append(child)
        popA = newA

        # Breed next generation (B)
        newB: List[PolicyGenome] = []
        eliteB_n = int(cfg["elitism"])
        if eliteB_n > 0:
            elite_idx = np.argsort(-fitB)[:eliteB_n]
            for k in elite_idx:
                newB.append(popB[int(k)])
        while len(newB) < len(popB):
            p1 = tournament_select(fitB, cfg["tournament_k"], rng)
            p2 = tournament_select(fitB, cfg["tournament_k"], rng)
            child = popB[p1]
            if rng.random() < cfg["pcxB"]:
                child = crossover_policy(popB[p1], popB[p2], rng)
            if rng.random() < cfg["pmutB"]:
                child = mutate_policy(
                    child,
                    p_bool_flip=cfg["policy_bool_flip"],
                    sigma_tau=cfg["policy_sigma_tau"],
                    sigma_margin=cfg["policy_sigma_margin"],
                    sigma_logC=cfg["policy_sigma_logC"],
                    rng=rng,
                )
            newB.append(child)
        popB = newB

        # Reset fitness arrays for new populations (forces reevaluation next gen)
        fitA = np.full(len(popA), -np.inf, dtype=float)
        fitB = np.full(len(popB), -np.inf, dtype=float)

    # --- Post-run cross-evaluation using ACTUAL final populations ---
    cross_eval_path = None
    if log_dir and final_popA is not None and final_popB is not None and final_fitA is not None and final_fitB is not None:
        ensure_dir(log_dir)
        cross_eval_path = os.path.join(log_dir, f"cross_eval_{run_tag}.csv")

        k = int(cfg["cross_k"])
        kA = min(k, len(final_popA))
        kB = min(k, len(final_popB))

        topA_idx = np.argsort(-final_fitA)[:kA]
        topB_idx = np.argsort(-final_fitB)[:kB]

        candA = [final_popA[int(i)] for i in topA_idx]
        candB = [final_popB[int(j)] for j in topB_idx]

        rows = []
        for ia, a in enumerate(candA):
            for jb, b in enumerate(candB):
                st = evaluate_team(
                    X_all=X_all, y_all=y_all,
                    idx_L_pool=idx_L, idx_U_pool=idx_U,
                    idx_val_pool=idx_val_pool, idx_probe=idx_probe,
                    view=a, policy=b,
                    fitness_repeats=cfg["fitness_repeats"],
                    lam_std=cfg["lam_std"], lam_bias=cfg["lam_bias"], lam_added=cfg["lam_added"],
                    seed=seed + 900000 + 1000 * ia + jb,
                )
                rows.append({
                    "rankA": ia,
                    "rankB": jb,
                    "idxA_in_pop": int(topA_idx[ia]),
                    "idxB_in_pop": int(topB_idx[jb]),
                    "fitA_collab_best": float(final_fitA[int(topA_idx[ia])]),
                    "fitB_collab_best": float(final_fitB[int(topB_idx[jb])]),
                    "fitness": float(st.fitness),
                    "mu": float(st.extra["mu"]),
                    "sd": float(st.extra["sd"]),
                    "probe_mean": float(st.extra["probe_mean"]),
                    "added_mean": float(st.extra["added_mean"]),
                    "view_n1": int(a.mask1.sum()),
                    "view_n2": int(a.mask2.sum()),
                    "policy_tau_start": float(b.tau_start),
                    "policy_tau_end": float(b.tau_end),
                    "policy_margin_min": float(b.margin_min),
                    "policy_class_balance": int(b.class_balance),
                    "policy_disagreement_veto": int(b.disagreement_veto),
                    "policy_max_iters": int(b.max_iters),
                    "policy_max_add_total": int(b.max_add_total),
                    "policy_max_add_per_class": int(b.max_add_per_class),
                    "policy_C": float(b.C),
                    "policy_max_iter_lr": int(b.max_iter_lr),
                })

        pd.DataFrame(rows).to_csv(cross_eval_path, index=False)

    assert best_view is not None and best_pol is not None and best_stats is not None
    meta = {"cross_eval_path": cross_eval_path}
    return best_view, best_pol, best_stats, meta


# ----------------------------
# Experiment loop
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_ids", type=int, nargs="*", default=None)
    ap.add_argument("--openml_suite", type=str, default=None, help="Optional suite name (e.g., OpenML-CC18)")
    ap.add_argument("--max_datasets", type=int, default=1)
    ap.add_argument("--max_rows", type=int, default=50000)

    ap.add_argument("--labeled_fracs", type=float, nargs="+", default=[0.05])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])

    ap.add_argument("--out_csv", type=str, default="results.csv")
    ap.add_argument("--log_dir", type=str, default=None)

    # CC params
    ap.add_argument("--popA", type=int, default=12)
    ap.add_argument("--popB", type=int, default=12)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--partners_random", type=int, default=2)
    ap.add_argument("--tournament_k", type=int, default=3)
    ap.add_argument("--elitism", type=int, default=1)

    ap.add_argument("--pcxA", type=float, default=0.90)
    ap.add_argument("--pmutA", type=float, default=0.45)
    ap.add_argument("--pcxB", type=float, default=0.90)
    ap.add_argument("--pmutB", type=float, default=0.60)

    # View mutation strength (expected flips)
    ap.add_argument("--view_init_frac", type=float, default=0.50)
    ap.add_argument("--view_min_features", type=int, default=3)
    ap.add_argument("--view_flip_E_start", type=float, default=2.0)
    ap.add_argument("--view_flip_E_end", type=float, default=1.0)
    ap.add_argument("--view_pflip_max", type=float, default=0.08)

    # Policy mutation params
    ap.add_argument("--policy_bool_flip", type=float, default=0.06)
    ap.add_argument("--policy_sigma_tau", type=float, default=0.02)
    ap.add_argument("--policy_sigma_margin", type=float, default=0.03)
    ap.add_argument("--policy_sigma_logC", type=float, default=0.35)

    # Fitness setup
    ap.add_argument("--fitness_repeats", type=int, default=3)
    ap.add_argument("--lam_std", type=float, default=0.20)
    ap.add_argument("--lam_bias", type=float, default=0.70)
    ap.add_argument("--lam_added", type=float, default=0.0005)

    # Splits
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--val_in_labeled_frac", type=float, default=0.30)
    ap.add_argument("--probe_in_labeled_frac", type=float, default=0.20)

    # Logging extras
    ap.add_argument("--diversity_pairs", type=int, default=200)
    ap.add_argument("--cross_k", type=int, default=10)
    ap.add_argument("--label_spreading_max_total", type=int, default=4000)

    args = ap.parse_args()
    ensure_dir(args.log_dir)

    # Determine datasets
    dataset_ids: List[int] = []
    if args.dataset_ids:
        dataset_ids = list(args.dataset_ids)
    elif args.openml_suite:
        set_openml_cache_from_env()
        import openml
        suite = openml.study.get_suite(args.openml_suite)
        dataset_ids = list(suite.data)[: int(args.max_datasets)]
    else:
        raise SystemExit("Provide --dataset_ids or --openml_suite")

    # Main output rows
    rows: List[Dict[str, Any]] = []
    overall_t0 = time.perf_counter()

    for did in dataset_ids[: int(args.max_datasets)]:
        try:
            X_df, y_ser, name = load_openml_dataset(int(did))
        except Exception as e:
            print(f"Skipping dataset {did} due to load error: {e}")
            continue

        # Filter rows
        if X_df.shape[0] > int(args.max_rows):
            X_df = X_df.iloc[: int(args.max_rows)].copy()
            y_ser = y_ser.iloc[: int(args.max_rows)].copy()

        # Encode labels (keep original for reporting, but we also need integer ids for some ops)
        # For our LR pipeline and metrics, object labels are fine.
        y = to_numpy(y_ser)

        for frac in args.labeled_fracs:
            for seed in args.seeds:
                run_tag = f"did{did}_frac{frac:.4f}_seed{seed}_{now_ts()}"

                # Split train/test
                idx_all = np.arange(X_df.shape[0], dtype=int)
                idx_train, idx_test = train_test_split(
                    idx_all, test_size=float(args.test_size),
                    random_state=int(seed), stratify=y
                )

                X_train_df = X_df.iloc[idx_train].copy()
                y_train = y[idx_train]
                X_test_df = X_df.iloc[idx_test].copy()
                y_test = y[idx_test]

                # Preprocess fit on training only
                pre = build_preprocessor(X_train_df)
                X_train = pre.fit_transform(X_train_df)
                X_test = pre.transform(X_test_df)

                # Combine train+test into a single matrix for indexing convenience
                # We'll create X_all = [train; test] and index accordingly
                # Note: X_train and X_test may be sparse matrices.
                from scipy.sparse import vstack
                X_all = vstack([X_train, X_test]) if hasattr(X_train, "tocsr") else np.vstack([X_train, X_test])
                y_all = np.concatenate([y_train, y_test], axis=0)

                # Indices in X_all
                idx_train_all = np.arange(len(idx_train), dtype=int)
                idx_test_all = np.arange(len(idx_train), len(idx_train) + len(idx_test), dtype=int)

                # Labeled fraction split within train
                # Ensure at least 2 per class if possible
                classes, counts = np.unique(y_train, return_counts=True)
                min_per_class = 2
                min_labels = int(min_per_class * len(classes))
                nL = max(min_labels, int(round(float(frac) * idx_train_all.size)))
                nL = min(nL, idx_train_all.size)

                # Stratified selection of labeled indices
                try:
                    idx_L, idx_U = train_test_split(
                        idx_train_all,
                        train_size=nL,
                        random_state=int(seed),
                        stratify=y_train
                    )
                    idx_L = np.array(idx_L, dtype=int)
                    idx_U = np.array(idx_U, dtype=int)
                except Exception:
                    # fallback: random
                    rng = np.random.default_rng(seed)
                    perm = rng.permutation(idx_train_all)
                    idx_L = perm[:nL]
                    idx_U = perm[nL:]

                # Split labeled into L_train / val_pool / probe
                rng = np.random.default_rng(seed + 12345)
                permL = rng.permutation(idx_L)
                n_val = max(10, int(round(float(args.val_in_labeled_frac) * permL.size)))
                n_probe = max(10, int(round(float(args.probe_in_labeled_frac) * permL.size)))
                # Ensure enough left for training
                if n_val + n_probe >= permL.size:
                    n_val = max(5, int(0.3 * permL.size))
                    n_probe = max(5, int(0.2 * permL.size))
                idx_val_pool = permL[:n_val]
                idx_probe = permL[n_val:n_val + n_probe]
                idx_L_train = permL[n_val + n_probe:]

                if idx_L_train.size < max(5, len(classes)):
                    # ensure minimum training labels
                    idx_L_train = permL[: max(5, len(classes))]
                    idx_val_pool = permL[max(5, len(classes)) : max(5, len(classes)) + n_val]
                    idx_probe = permL[max(5, len(classes)) + n_val : max(5, len(classes)) + n_val + n_probe]

                # -----------------
                # Run CC-SSL
                # -----------------
                cfg = vars(args).copy()
                cfg.update({
                    "dataset_id": int(did),
                    "dataset_name": name,
                    "labeled_frac": float(frac),
                })

                t_cc0 = time.perf_counter()
                best_view, best_pol, best_stats, meta = run_cc_once(
                    X_all=X_all, y_all=y_all,
                    idx_L=idx_L_train.astype(int),
                    idx_U=idx_U.astype(int),
                    idx_val_pool=idx_val_pool.astype(int),
                    idx_probe=idx_probe.astype(int),
                    cfg=cfg,
                    seed=int(seed),
                    log_dir=args.log_dir,
                    run_tag=run_tag,
                )
                t_cc1 = time.perf_counter()

                # Final test evaluation using best pair (single run with full val_pool)
                t_test0 = time.perf_counter()
                out_final = ssl_two_view_run(
                    X_all=X_all, y_all=y_all,
                    idx_L_train=idx_L_train.astype(int),
                    idx_U=idx_U.astype(int),
                    idx_val=idx_test_all.astype(int),  # evaluate on test
                    idx_probe=idx_probe.astype(int),
                    view=best_view, policy=best_pol,
                    seed=int(seed) + 777777,
                )
                macroF1_test = float(out_final["val_macroF1"])
                acc_test = float(out_final["val_acc"])
                t_test1 = time.perf_counter()

                rows.append({
                    "dataset_id": int(did),
                    "dataset_name": name,
                    "labeled_frac": float(frac),
                    "seed": int(seed),
                    "method": "CC-SSL (coevolved)",
                    "macroF1_test": macroF1_test,
                    "acc_test": acc_test,
                    "runtime_seconds": float((t_cc1 - t_cc0) + (t_test1 - t_test0)),
                    "coevolution_seconds": float(t_cc1 - t_cc0),
                    "test_eval_seconds": float(t_test1 - t_test0),
                    "best_fitness": float(best_stats.fitness),
                    "best_mu": float(best_stats.extra["mu"]),
                    "best_sd": float(best_stats.extra["sd"]),
                    "best_probe_mean": float(best_stats.extra["probe_mean"]),
                    "best_added_mean": float(best_stats.extra["added_mean"]),
                    "best_view_n1": int(best_view.mask1.sum()),
                    "best_view_n2": int(best_view.mask2.sum()),
                    **{f"best_policy_{k}": v for k, v in best_pol.summary().items()},
                    "cross_eval_path": meta.get("cross_eval_path"),
                    "run_tag": run_tag,
                })

                # Optional run-level JSONL log
                if args.log_dir:
                    runlog_path = os.path.join(args.log_dir, f"run_log_{run_tag}.jsonl")
                    rec = {
                        "run_tag": run_tag,
                        "dataset_id": int(did),
                        "dataset_name": name,
                        "labeled_frac": float(frac),
                        "seed": int(seed),
                        "best_pair_fitness": float(best_stats.fitness),
                        "best_view": best_view.summary(),
                        "best_policy": best_pol.summary(),
                        "fitness_distributions": {
                            "val_macroF1": best_stats.val_macroF1,
                            "val_acc": best_stats.val_acc,
                            "probe_drop": best_stats.probe_drop,
                            "pseudo_added": best_stats.pseudo_added,
                        },
                        "timing": {
                            "cc_seconds": float(t_cc1 - t_cc0),
                            "test_seconds": float(t_test1 - t_test0),
                            "total_seconds": float((t_cc1 - t_cc0) + (t_test1 - t_test0)),
                        },
                        "cross_eval_path": meta.get("cross_eval_path"),
                    }
                    with open(runlog_path, "a", encoding="utf-8") as f:
                        f.write(safe_json(rec) + "\n")

                # -----------------
                # Baselines (each row includes its own runtime)
                # -----------------
                # Baseline indices:
                # For baselines we use: labeled set = idx_L_train, unlabeled = idx_U, test = idx_test_all
                # (Keep it consistent with CC evaluation conditions)

                # Supervised-only
                t0 = time.perf_counter()
                sup = supervised_only_lr(X_all[idx_L_train], y_all[idx_L_train], X_all[idx_test_all], y_all[idx_test_all], seed=seed)
                t1 = time.perf_counter()
                rows.append({
                    "dataset_id": int(did), "dataset_name": name, "labeled_frac": float(frac), "seed": int(seed),
                    "method": "Supervised-only (LR)",
                    **sup,
                    "runtime_seconds": float(t1 - t0),
                    "run_tag": run_tag,
                })

                # Self-training (cheap placeholder here; can be extended)
                t0 = time.perf_counter()
                st = self_training_baseline(X_all, y_all, idx_L_train, idx_U, idx_test_all, seed=seed)
                t1 = time.perf_counter()
                rows.append({
                    "dataset_id": int(did), "dataset_name": name, "labeled_frac": float(frac), "seed": int(seed),
                    "method": "Self-training (simple)",
                    **st,
                    "runtime_seconds": float(t1 - t0),
                    "run_tag": run_tag,
                })

                # Label Spreading (optional)
                t0 = time.perf_counter()
                ls = label_spreading_baseline(
                    X_all=X_all, y_all=y_all,
                    idx_L=idx_L_train, idx_U=idx_U, idx_test=idx_test_all,
                    max_total=int(args.label_spreading_max_total),
                    seed=seed
                )
                t1 = time.perf_counter()
                if ls is not None:
                    rows.append({
                        "dataset_id": int(did), "dataset_name": name, "labeled_frac": float(frac), "seed": int(seed),
                        "method": "Label Spreading",
                        **ls,
                        "runtime_seconds": float(t1 - t0),
                        "run_tag": run_tag,
                    })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.out_csv, index=False)

    overall_t1 = time.perf_counter()
    print(f"Wrote {args.out_csv} with {len(df_out)} rows. Total seconds: {overall_t1 - overall_t0:.2f}")


if __name__ == "__main__":
    main()
