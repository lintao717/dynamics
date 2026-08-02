"""
Stage-1 parameter estimator using scipy.optimize.differential_evolution.

Fixed seed, single-worker, 20 iterations, population size 8.
Objective: train-set masked replay loss across the fixed seed tuple.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from dynamics_simulation.config import ModelParams, default_params
from dynamics_simulation.data.schema import EventCase
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.calibration.split import TemporalSplit
from dynamics_simulation.calibration.parameters import (
    Stage1ParameterSet, apply_parameter_vector,
)
from dynamics_simulation.calibration.objective import (
    compute_replay_loss, LossWeights,
)


@dataclass
class CalibrationResult:
    """Output of one stage-1 calibration run."""

    case_id: str
    best_vector: list[float] | None = None
    best_loss: float = float("inf")
    train_loss: float = float("inf")
    val_loss: float = float("inf")
    success: bool = False
    message: str = ""
    n_iterations: int = 0
    optimizer_settings: dict[str, Any] = field(default_factory=dict)
    parameter_specs: list[dict[str, Any]] = field(default_factory=list)
    base_params_version: str = "v1.1"
    seed_tuple: tuple[int, ...] = ()

    # ── Experiment metadata (for reproducibility) ──
    train_fraction: float = 0.7
    train_end_step: int = 0
    last_data_step: int = 0
    final_step: int = 0
    tail_steps: int = 4
    step_hours: float = 1.0
    network_mode: str = "broadcast"
    git_sha: str = "unknown"
    source_revision: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "case_id": self.case_id,
            "best_vector": self.best_vector,
            "best_loss": self.best_loss,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "success": self.success,
            "message": self.message,
            "n_iterations": self.n_iterations,
            "optimizer_settings": self.optimizer_settings,
            "parameter_specs": self.parameter_specs,
            "base_params_version": self.base_params_version,
            "seed_tuple": list(self.seed_tuple),
            "train_fraction": self.train_fraction,
            "train_end_step": self.train_end_step,
            "last_data_step": self.last_data_step,
            "final_step": self.final_step,
            "tail_steps": self.tail_steps,
            "step_hours": self.step_hours,
            "network_mode": self.network_mode,
            "git_sha": self.git_sha,
            "source_revision": self.source_revision,
        }


def fit_stage1(
    case: EventCase,
    base_params: ModelParams | None = None,
    replay_config: ReplayConfig | None = None,
    split: TemporalSplit | None = None,
    train_fraction: float = 0.7,
) -> CalibrationResult:
    """Fit stage-1 parameters (4 params) to a single event case.

    Uses scipy.optimize.differential_evolution with fixed seed and
    the replay seed tuple for objective evaluation. Only the training
    segment contributes to the loss.

    Args:
        case: Validated EventCase to fit.
        base_params: Base ModelParams (default: default_params()).
        replay_config: Replay configuration (default: broadcast, 5 seeds).
        split: TemporalSplit (default: computed from train_fraction).
        train_fraction: Fraction of steps for training (default 0.7).
            Ignored when *split* is explicitly provided.

    Returns:
        CalibrationResult with best vector, losses, and provenance.
    """
    if base_params is None:
        base_params = default_params()

    if replay_config is None:
        replay_config = ReplayConfig()

    specs = Stage1ParameterSet.to_specs()
    bounds = Stage1ParameterSet.bounds()

    # Use last_data_step (real data only) for the split — NOT final_step
    # which includes artificial tail steps that must not enter the loss.
    from dynamics_simulation.data.timegrid import TimeGrid
    grid = TimeGrid.from_case(
        case,
        step_hours=replay_config.step_hours,
        tail_steps=replay_config.tail_steps,
    )
    T = grid.last_data_step

    # Run one replay to obtain observed trajectory for loss computation
    result_ref = run_replay(case, base_params, replay_config)

    if split is None:
        split = TemporalSplit.by_fraction(
            total_steps=T, train_fraction=train_fraction,
        )
    else:
        # Explicit split must not include artificial tail steps.
        # Tail steps are NOT real data and must not influence
        # train_end_step, the 70/30 ratio, or the effective data
        # points available for calibration.
        if split.total_steps > grid.last_data_step:
            raise ValueError(
                f"Explicit split.total_steps={split.total_steps} exceeds "
                f"last_data_step={grid.last_data_step}. Split includes "
                "artificial tail steps that are not real observations."
            )

    # Pre-build observation dict and masks (only active_count is in simulated_mean)
    obs_dict = {
        "active_count": result_ref.observed.active_count,
    }
    mask_shape = len(result_ref.observed.steps)
    masks = {
        "active_count": result_ref.observed.observation_masks.get(
            "active_count", np.ones(mask_shape, dtype=bool)
        ),
    }
    # Stage 1: only active_count is in simulated_mean.
    # All other metrics default to weight 0 in LossWeights().
    weights = LossWeights()
    # (active_count=1.0 is the only non-zero default)

    iteration_count = [0]  # mutable counter

    def objective(x: np.ndarray) -> float:
        iteration_count[0] += 1
        try:
            params = apply_parameter_vector(base_params, specs, list(x))
        except (ValueError, AttributeError):
            return 1e10  # penalty for invalid params

        result = run_replay(case, params, replay_config)
        sim_dict = result.simulated_mean

        if sim_dict is None or len(sim_dict) == 0:
            return 1e10

        loss = compute_replay_loss(obs_dict, sim_dict, split, weights, masks)
        return float(loss.train_total)

    try:
        from scipy.optimize import differential_evolution

        opt_result = differential_evolution(
            objective,
            bounds,
            seed=20260731,
            workers=1,
            updating="immediate",
            polish=False,
            maxiter=20,
            popsize=8,
        )

        best = list(opt_result.x)

        # Final evaluation with validation loss
        best_params = apply_parameter_vector(base_params, specs, best)
        result_final = run_replay(case, best_params, replay_config)
        sim_final = result_final.simulated_mean
        if sim_final is not None:
            final_loss = compute_replay_loss(
                obs_dict, sim_final, split, weights, masks,
            )
            train_loss = float(final_loss.train_total)
            val_loss = float(final_loss.val_total)
        else:
            train_loss = float("inf")
            val_loss = float("inf")

        # Resolve git sha for provenance
        import subprocess
        git_sha = "unknown"
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass

        return CalibrationResult(
            case_id=case.case_id,
            best_vector=best,
            best_loss=float(opt_result.fun),
            train_loss=train_loss,
            val_loss=val_loss,
            success=opt_result.success,
            message=opt_result.message,
            n_iterations=iteration_count[0],
            optimizer_settings={
                "method": "differential_evolution",
                "seed": 20260731,
                "maxiter": 20,
                "popsize": 8,
                "polish": False,
                "updating": "immediate",
            },
            parameter_specs=[
                {"path": s.path, "low": s.low, "high": s.high}
                for s in specs
            ],
            seed_tuple=replay_config.seeds,
            train_fraction=train_fraction,
            train_end_step=split.train_end_step,
            last_data_step=grid.last_data_step,
            final_step=grid.final_step,
            tail_steps=replay_config.tail_steps,
            step_hours=replay_config.step_hours,
            network_mode=replay_config.network_mode.value,
            git_sha=git_sha,
        )

    except ImportError:
        # scipy not available — return placeholder
        return CalibrationResult(
            case_id=case.case_id,
            success=False,
            message="scipy not available",
        )
