"""Train-on-Synthetic-Test-on-Real (TSTR) benchmark."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _prepare_classification_data(
    windows: np.ndarray,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare binary classification task: predict next-day return direction.

    Uses the window features as input X, and the sign of the last log return
    as the target y (1 = up, 0 = down).

    Args:
        windows: Array of shape (N, seq_len, num_features).
        horizon: Timesteps ahead to predict (default: last step).

    Returns:
        Tuple of (X, y): flattened features and binary labels.
    """
    # Use all but the last timestep as features, last return sign as label
    X = windows[:, :-horizon, :].reshape(len(windows), -1)
    # Assume feature index 5 is LogReturn (after Open, High, Low, Close, Volume)
    log_return_idx = min(5, windows.shape[-1] - 1)
    y = (windows[:, -1, log_return_idx] > 0).astype(int)
    return X, y


def tstr_benchmark(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    classifier: str = "logistic",
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Run TSTR benchmark comparing synthetic vs real training data.

    Trains a classifier on:
      - Real data (TRTR baseline)
      - Synthetic data (TSTR)

    Both are evaluated on the same held-out real test set.

    Args:
        real_windows: Real sequences, shape (N, seq_len, num_features).
        synthetic_windows: Synthetic sequences, shape (M, seq_len, num_features).
        classifier: Classifier type ("logistic" for now).
        test_ratio: Fraction of real data reserved for testing.
        random_state: Random seed.

    Returns:
        Dict with "trtr" and "tstr" sub-dicts containing accuracy, f1, auc.
    """
    X_real, y_real = _prepare_classification_data(real_windows)

    # Train/test split on real data
    n_test = int(len(X_real) * test_ratio)
    X_test, y_test = X_real[-n_test:], y_real[-n_test:]
    X_train_real, y_train_real = X_real[:-n_test], y_real[:-n_test]

    X_synth, y_synth = _prepare_classification_data(synthetic_windows)

    results = {}

    for name, X_train, y_train in [
        ("trtr", X_train_real, y_train_real),
        ("tstr", X_synth, y_synth),
    ]:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_tr, y_train)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5,
        }
        logger.info(
            "[TSTR] %s — accuracy=%.3f  f1=%.3f  auc=%.3f",
            name.upper(), results[name]["accuracy"], results[name]["f1"], results[name]["auc"],
        )

    results["tstr_gap"] = {
        k: results["trtr"][k] - results["tstr"][k] for k in ["accuracy", "f1", "auc"]
    }
    return results
