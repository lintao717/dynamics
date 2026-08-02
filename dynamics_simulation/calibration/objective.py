"""Masked multi-target replay loss with chronological split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LossWeights:
    """Per-target loss weights.

    Only ``active_count`` is non-zero by default. The current
    simulator does not produce ``cumulative_users`` or
    ``interaction_count`` in ``simulated_mean``, so those default to 0.

    ``peak_time`` and ``final_size`` are per-segment scalars (train
    peak computed from the training window only; val peak from the
    validation window only). They default to 0 because they are
    secondary diagnostics — not primary calibration targets.

    ``stance`` and ``arousal`` default to 0 because CHECKED has no
    labeled stance/arousal data.
    """

    active_count: float = 1.0
    cumulative_users: float = 0.0
    interaction_count: float = 0.0
    peak_time: float = 0.0
    final_size: float = 0.0
    stance: float = 0.0
    arousal: float = 0.0


@dataclass
class ReplayLoss:
    """Decomposed train/validation loss."""

    train_total: float = 0.0
    val_total: float = 0.0

    train_active_count: float = 0.0
    train_cumulative_users: float = 0.0
    train_interaction_count: float = 0.0
    train_peak_time: float = 0.0
    train_final_size: float = 0.0

    val_active_count: float = 0.0
    val_cumulative_users: float = 0.0
    val_interaction_count: float = 0.0
    val_peak_time: float = 0.0
    val_final_size: float = 0.0


def _nrmse(obs: np.ndarray, sim: np.ndarray, mask: np.ndarray) -> float:
    """Normalized RMSE over masked entries.

    NRMSE = sqrt(mean((obs - sim)^2)) / (obs_max - obs_min + eps)
    """
    valid = mask.astype(bool)
    if not valid.any():
        return 0.0

    obs_valid = obs[valid]
    sim_valid = sim[valid]
    mse = np.mean((obs_valid - sim_valid) ** 2)
    rmse = np.sqrt(mse)

    obs_range = obs_valid.max() - obs_valid.min()
    if obs_range < 1e-8:
        # Constant target: use absolute RMSE
        return float(rmse)

    return float(rmse / obs_range)


def _abs_error(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(obs - sim)))


def _peak_step(arr: np.ndarray) -> int:
    """Step index of maximum value."""
    return int(np.argmax(arr))


def _final_value(arr: np.ndarray) -> float:
    """Last finite value."""
    return float(arr[-1])


def compute_replay_loss(
    observed: dict[str, np.ndarray],
    simulated: dict[str, np.ndarray],
    split,  # TemporalSplit
    weights: LossWeights,
    masks: dict[str, np.ndarray] | None = None,
) -> ReplayLoss:
    """Compute masked train/validation loss.

    Args:
        observed: Dict of observed trajectory arrays.
        simulated: Dict of simulated trajectory arrays.
        split: Chronological train/validation split.
        weights: Per-target loss weights.
        masks: Per-target observation masks (default: all True).

    Returns:
        ReplayLoss with decomposed train/validation components.

    Raises:
        ValueError: If a nonzero-weight target has zero valid observations.
    """
    T = split.total_steps
    train_idx = slice(0, split.train_end_step + 1)
    val_idx = slice(split.train_end_step + 1, T + 1)

    if masks is None:
        masks = {k: np.ones(T + 1, dtype=bool) for k in observed}

    def _score_one(
        name: str, weight: float, fn=_nrmse,
    ) -> tuple[float, float]:
        if weight == 0.0:
            return 0.0, 0.0

        obs = np.asarray(observed[name], dtype=np.float64)
        sim = np.asarray(simulated[name], dtype=np.float64)
        m = masks.get(name, np.ones_like(obs, dtype=bool))

        train_mask = m[train_idx]
        val_mask = m[val_idx]

        train_obs = obs[train_idx]
        val_obs = obs[val_idx]
        train_sim = sim[train_idx]
        val_sim = sim[val_idx]

        # Check for zero valid observations
        if not train_mask.any():
            raise ValueError(
                f"No valid observations for '{name}' in training set"
            )

        train_loss = fn(train_obs, train_sim, train_mask) if train_mask.any() else 0.0
        val_loss = fn(val_obs, val_sim, val_mask) if val_mask.any() else 0.0

        return float(train_loss * weight), float(val_loss * weight)

    loss = ReplayLoss()

    loss.train_active_count, loss.val_active_count = _score_one(
        "active_count", weights.active_count,
    )
    loss.train_cumulative_users, loss.val_cumulative_users = _score_one(
        "cumulative_users", weights.cumulative_users,
    )
    loss.train_interaction_count, loss.val_interaction_count = _score_one(
        "interaction_count", weights.interaction_count,
    )

    # Peak time: per-segment — train peak from train window,
    # val peak from val window. No cross-segment leakage.
    if weights.peak_time > 0:
        obs_arr = np.asarray(observed.get("active_count", np.zeros(1)), dtype=np.float64)
        sim_arr = np.asarray(simulated.get("active_count", np.zeros(1)), dtype=np.float64)

        # Train peak: computed from the training segment only
        train_obs_peak = _peak_step(obs_arr[train_idx])
        train_sim_peak = _peak_step(sim_arr[train_idx])
        train_peak_err = abs(train_obs_peak - train_sim_peak) / max(T, 1)
        loss.train_peak_time = float(train_peak_err * weights.peak_time)

        # Val peak: computed from the validation segment only
        val_obs_peak = _peak_step(obs_arr[val_idx])
        val_sim_peak = _peak_step(sim_arr[val_idx])
        val_peak_err = abs(val_obs_peak - val_sim_peak) / max(len(obs_arr[val_idx]), 1)
        loss.val_peak_time = float(val_peak_err * weights.peak_time)

    # Final size: per-segment — train final = last train-step value,
    # val final = last val-step value.
    if weights.final_size > 0:
        obs_arr = np.asarray(observed.get("active_count", np.zeros(1)), dtype=np.float64)
        sim_arr = np.asarray(simulated.get("active_count", np.zeros(1)), dtype=np.float64)

        # Train final: last value in training segment
        train_obs_final = _final_value(obs_arr[train_idx])
        train_sim_final = _final_value(sim_arr[train_idx])
        train_final_range = max(abs(train_obs_final), abs(train_sim_final), 1.0)
        train_final_err = abs(train_obs_final - train_sim_final) / train_final_range
        loss.train_final_size = float(train_final_err * weights.final_size)

        # Val final: last value in validation segment
        val_obs_final = _final_value(obs_arr[val_idx])
        val_sim_final = _final_value(sim_arr[val_idx])
        val_final_range = max(abs(val_obs_final), abs(val_sim_final), 1.0)
        val_final_err = abs(val_obs_final - val_sim_final) / val_final_range
        loss.val_final_size = float(val_final_err * weights.final_size)

    loss.train_total = (
        loss.train_active_count + loss.train_cumulative_users +
        loss.train_interaction_count + loss.train_peak_time +
        loss.train_final_size
    )
    loss.val_total = (
        loss.val_active_count + loss.val_cumulative_users +
        loss.val_interaction_count + loss.val_peak_time +
        loss.val_final_size
    )

    return loss
