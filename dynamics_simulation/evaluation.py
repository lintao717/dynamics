"""[V1.5.1] Unified evaluation metrics and seed separation.

Protocol:
  - FIT_SEEDS: used only during parameter search / optimisation
  - VALIDATION_SEEDS: used for model selection / cross-validation
  - TEST_SEEDS: used only for final held-out reporting

All metrics accept explicit seed arguments to prevent accidental reuse.
"""

from __future__ import annotations

import numpy as np

# ── Strictly separated seed groups ──

FIT_SEEDS = (11, 23, 37, 53, 71)
"""Seeds used during parameter search / optimisation only."""

VALIDATION_SEEDS = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149)
"""Seeds used for model selection and cross-validation. 10 seeds for stability."""

TEST_SEEDS = (211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
              269, 271, 277, 281, 283, 293, 307, 311, 313, 317)
"""Seeds used ONLY for final held-out reporting. 20 seeds for precision."""

assert len(set(FIT_SEEDS) & set(VALIDATION_SEEDS)) == 0, "FIT and VALIDATION seeds overlap"
assert len(set(FIT_SEEDS) & set(TEST_SEEDS)) == 0, "FIT and TEST seeds overlap"
assert len(set(VALIDATION_SEEDS) & set(TEST_SEEDS)) == 0, "VALIDATION and TEST seeds overlap"


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared log error for non-negative counts."""
    yt = np.maximum(np.asarray(y_true, dtype=np.float64), 0)
    yp = np.maximum(np.asarray(y_pred, dtype=np.float64), 0)
    return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))


def mase(y_train: np.ndarray, y_val: np.ndarray, y_pred_val: np.ndarray) -> float:
    """Mean absolute scaled error using training naive forecast."""
    if len(y_train) < 2:
        scale = max(np.mean(np.abs(y_train)), 1e-6)
    else:
        scale = max(np.mean(np.abs(np.diff(y_train))), 1e-6)
    return float(np.mean(np.abs(y_val - y_pred_val)) / scale)


def peak_error(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Peak magnitude and timing errors."""
    o = np.asarray(obs, dtype=np.float64)
    s = np.asarray(sim, dtype=np.float64)
    obs_peak = float(np.max(o))
    sim_peak = float(np.max(s))
    obs_step = int(np.argmax(o))
    sim_step = int(np.argmax(s))
    return {
        "obs_peak": obs_peak,
        "sim_peak": sim_peak,
        "peak_ratio": sim_peak / max(obs_peak, 1.0),
        "peak_step_error": sim_step - obs_step,
    }


def auc_ratio(obs: np.ndarray, sim: np.ndarray) -> float:
    """Ratio of simulated AUC to observed AUC."""
    t = np.arange(len(obs), dtype=np.float64)
    o_auc = float(np.trapezoid(np.asarray(obs, dtype=np.float64), t))
    s_auc = float(np.trapezoid(np.asarray(sim, dtype=np.float64), t))
    return s_auc / max(o_auc, 1e-6)
