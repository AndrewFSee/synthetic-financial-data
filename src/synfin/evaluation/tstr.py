"""Train-on-Synthetic-Test-on-Real (TSTR) benchmark.

Default task: classify whether the *next-step absolute return* is above the
real median (a one-step volatility forecast). Volatility clustering makes this
learnable on real data, so a generator that fails to reproduce volatility
dynamics produces a visible TRTR-TSTR gap. Next-step *direction* is also
available, but it is close to unpredictable, so TRTR and TSTR both sit near
0.5 and the gap says little.

The TRTR-TSTR AUC gap is judged against its own sampling noise: a paired
block bootstrap over the real test windows (both classifiers scored on the
same resample) gives its standard error. The real test set is small (a few
hundred overlapping windows), so a raw gap of 0.04 can be pure noise.

Labels use a threshold taken from the real training windows (a median), so
the benchmark works in any monotone scaling of the return column. A fixed
``> 0`` threshold breaks under min-max scaling, where every return is >= 0.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from synfin.evaluation.statistical_tests import block_bootstrap_indices, is_sliding_windows

logger = logging.getLogger(__name__)

TASKS = ("volatility", "direction")


def _target(windows: np.ndarray, return_idx: int, task: str) -> np.ndarray:
    last = windows[:, -1, return_idx]
    return np.abs(last) if task == "volatility" else last


def _prepare_classification_data(
    windows: np.ndarray,
    return_idx: int,
    task: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Features = all but the last timestep, raw and absolute (flattened).

    The absolute values let a linear classifier use magnitudes (volatility
    clustering), which it cannot compute from signed returns alone.
    Label = last-step target > threshold.
    """
    past = windows[:, :-1, :]
    X = np.concatenate([past, np.abs(past)], axis=-1).reshape(len(windows), -1)
    y = (_target(windows, return_idx, task) > threshold).astype(int)
    return X, y


def tstr_benchmark(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    return_idx: int,
    task: str = "volatility",
    real_test: Optional[np.ndarray] = None,
    test_ratio: float = 0.2,
    random_state: int = 42,
    n_boot: int = 500,
) -> Dict[str, Dict[str, float]]:
    """Run TSTR against a TRTR (train-on-real) baseline on the same real test set.

    Args:
        real_windows: Real training windows, shape (N, seq_len, F).
        synthetic_windows: Synthetic windows, shape (M, seq_len, F).
        return_idx: Index of the log-return feature.
        task: "volatility" (default) or "direction".
        real_test: Held-out real windows. If None, the last ``test_ratio`` of
            ``real_windows`` is used, separated by a one-window gap.
        test_ratio: Test fraction when ``real_test`` is None.
        random_state: Classifier and bootstrap seed.
        n_boot: Paired bootstrap replicates for the AUC gap's standard error.

    Returns:
        Dict with "trtr" and "tstr" metrics (accuracy, f1, auc) and "tstr_gap"
        (accuracy/f1/auc gaps plus ``auc_se`` and ``auc_z`` = gap / se).
    """
    if task not in TASKS:
        raise ValueError(f"Unknown TSTR task {task!r}; use one of {TASKS}.")

    if real_test is None:
        n_test = int(len(real_windows) * test_ratio)
        gap = real_windows.shape[1]
        real_test = real_windows[-n_test:]
        real_windows = real_windows[: len(real_windows) - n_test - gap]

    threshold = float(np.median(_target(real_windows, return_idx, task)))
    X_train_real, y_train_real = _prepare_classification_data(
        real_windows, return_idx, task, threshold
    )
    X_test, y_test = _prepare_classification_data(real_test, return_idx, task, threshold)
    X_synth, y_synth = _prepare_classification_data(synthetic_windows, return_idx, task, threshold)

    results: Dict[str, Dict[str, float]] = {"task": {"name": task, "threshold": threshold}}
    probs: Dict[str, np.ndarray] = {}
    for name, X_train, y_train in [
        ("trtr", X_train_real, y_train_real),
        ("tstr", X_synth, y_synth),
    ]:
        if len(np.unique(y_train)) < 2:
            logger.warning("[TSTR] %s skipped: training labels have a single class.", name.upper())
            results[name] = {"skipped": 1.0}
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        clf = LogisticRegression(C=0.1, max_iter=2000, random_state=random_state)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        probs[name] = y_prob

        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5,
        }
        logger.info(
            "[TSTR] %s — accuracy=%.3f  f1=%.3f  auc=%.3f",
            name.upper(),
            results[name]["accuracy"],
            results[name]["f1"],
            results[name]["auc"],
        )

    if "auc" in results.get("trtr", {}) and "auc" in results.get("tstr", {}):
        results["tstr_gap"] = {
            k: results["trtr"][k] - results["tstr"][k] for k in ["accuracy", "f1", "auc"]
        }
        se = _paired_auc_gap_se(
            y_test, probs["trtr"], probs["tstr"], real_test, n_boot, random_state
        )
        results["tstr_gap"]["auc_se"] = se
        results["tstr_gap"]["auc_z"] = results["tstr_gap"]["auc"] / se if se > 0 else 0.0
    return results


def _paired_auc_gap_se(
    y: np.ndarray,
    p_trtr: np.ndarray,
    p_tstr: np.ndarray,
    test_windows: np.ndarray,
    n_boot: int,
    seed: int,
) -> float:
    """Bootstrap standard error of AUC(trtr) - AUC(tstr) on the same test resamples."""
    rng = np.random.default_rng(seed)
    block = 2 * test_windows.shape[1] if is_sliding_windows(test_windows[:, :, 0]) else 1
    gaps = []
    for _ in range(n_boot):
        idx = block_bootstrap_indices(len(y), block, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        gaps.append(roc_auc_score(y[idx], p_trtr[idx]) - roc_auc_score(y[idx], p_tstr[idx]))
    return float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
