"""Regression error measures, written out explicitly so they can be unit-tested."""

from __future__ import annotations

import math

import numpy as np


def _as_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape[0]} targets vs {b.shape[0]} predictions")
    if a.size == 0:
        raise ValueError("cannot score an empty set")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("targets and predictions must be finite")
    return a, b


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    a, b = _as_pair(y_true, y_pred)
    return float(math.sqrt(np.mean((a - b) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    a, b = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def r_squared(y_true, y_pred) -> float:
    """Coefficient of determination, 1 - SS_res / SS_tot.

    Returns ``nan`` when the targets are constant (SS_tot is zero).
    """
    a, b = _as_pair(y_true, y_pred)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def score_all(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE and R^2 in one dictionary."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }
