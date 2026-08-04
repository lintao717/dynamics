"""[V1.5.1] Open-loop forecasting baselines.

All baselines are fitted on TRAINING data only, then forecast
into the validation (and optionally test) period.

Baselines provided:
  - zero: predict 0 for all future steps
  - persistence_last: predict last training value for all future steps
  - exponential_decay: a * exp(-lambda * t), fitted on log(y_train) via linear regression
  - pulse_decay: b + a * exp(-lambda * max(t-tp, 0)), tp = argmax(y_train)
"""

from __future__ import annotations

import numpy as np


def _fit_exp_decay(y_train: np.ndarray) -> tuple[float, float]:
    """Fit a * exp(-lambda * t) on training data. Returns (a, lambda)."""
    t = np.arange(len(y_train), dtype=np.float64)
    y_pos = np.maximum(y_train, 1e-6)
    log_y = np.log(y_pos)
    A = np.column_stack([np.ones_like(t), t])
    coeff, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    a = np.exp(coeff[0])
    lam = max(-coeff[1], 1e-6)
    return a, lam


def _fit_pulse_decay(y_train: np.ndarray) -> tuple[float, float, float, int]:
    """Fit b + a * exp(-lambda * max(t-tp, 0)). Returns (b, a, lambda, tp)."""
    t = np.arange(len(y_train), dtype=np.float64)
    tp = int(np.argmax(y_train))
    b = np.mean(y_train[-max(1, len(y_train) // 4):])
    y_adj = np.maximum(y_train - b, 1e-6)
    log_y = np.log(y_adj)
    t_adj = np.maximum(t - tp, 0)
    mask = t >= tp
    if mask.sum() >= 2:
        A = np.column_stack([np.ones_like(t_adj[mask]), t_adj[mask]])
        coeff, _, _, _ = np.linalg.lstsq(A, log_y[mask], rcond=None)
        a = np.exp(coeff[0])
        lam = max(-coeff[1], 1e-6)
    else:
        a = max(y_train) - b
        lam = 0.5
    return b, a, lam, tp


class BaselineForecast:
    """Container for baseline forecasts on a given time horizon."""

    def __init__(self, y_train: np.ndarray, n_total: int):
        """
        Args:
            y_train: Training observations (length = train_end + 1).
            n_total: Total number of time steps (training + validation + test).
        """
        self.y_train = np.asarray(y_train, dtype=np.float64)
        self.n_train = len(y_train)
        self.n_total = n_total
        self.n_future = n_total - self.n_train

        t_all = np.arange(n_total, dtype=np.float64)
        t_future = t_all[self.n_train:]

        # Zero baseline
        self.zero = np.zeros(self.n_future, dtype=np.float64)

        # Persistence: last training value
        self.persistence_last = np.full(self.n_future,
                                        self.y_train[-1] if len(self.y_train) > 0 else 0.0,
                                        dtype=np.float64)

        # Exponential decay
        a_exp, lam_exp = _fit_exp_decay(self.y_train)
        self.exp_decay = a_exp * np.exp(-lam_exp * t_all)
        self.exp_decay_train = self.exp_decay[:self.n_train]
        self.exp_decay_future = self.exp_decay[self.n_train:]

        # Pulse decay
        b_p, a_p, lam_p, tp_p = _fit_pulse_decay(self.y_train)
        self.pulse_decay = b_p + a_p * np.exp(-lam_p * np.maximum(t_all - tp_p, 0))
        self.pulse_decay_train = self.pulse_decay[:self.n_train]
        self.pulse_decay_future = self.pulse_decay[self.n_train:]

    @property
    def best_future(self) -> np.ndarray:
        """Return the best baseline forecast (lowest train RMSLE among non-zero baselines)."""
        from dynamics_simulation.evaluation import rmsle
        candidates = [
            ("persistence", self.persistence_last),
            ("exp_decay", self.exp_decay_future),
            ("pulse_decay", self.pulse_decay_future),
        ]
        best_name, best_pred = min(
            candidates,
            key=lambda x: rmsle(self.y_train, getattr(self, f"{x[0]}_train", None)
                                if x[0] != "persistence"
                                else np.full(self.n_train, self.y_train[-1]))
        )
        return best_pred
