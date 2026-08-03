"""
Synthetic parameter recovery experiment.

Tests whether the 4 candidate calibration parameters can be recovered
from A(t) trajectories when the ground truth is known.

Phases:
  1. Single-parameter recovery (12 experiments: 4 params × 3 levels)
  2. Two-parameter recovery (9 experiments: 3 confusing pairs × 3 levels)
  3. Four-parameter joint recovery (20 Latin hypercube vectors)

Output: artifacts/recovery/ with CSV, JSON, and report.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics_simulation.config import ModelParams, default_params
from dynamics_simulation.data.schema import (
    EventCase, RootPost, InteractionRecord,
)
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.calibration.estimator import fit_stage1
from dynamics_simulation.calibration.parameters import (
    Stage1ParameterSet, apply_parameter_vector, ParameterSpec,
)
from dynamics_simulation.calibration.objective import compute_replay_loss, LossWeights
from dynamics_simulation.calibration.split import TemporalSplit

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "recovery"
SEEDS = (42, 99, 137)  # 3 calibration seeds
SIM_SEEDS = (11, 23, 37, 53, 71)  # 5 simulation seeds for stable trajectory

# ── Synthetic case builder ──
def _make_synthetic_case(n_users: int = 200) -> EventCase:
    """Build a synthetic EventCase with *n_users* potential participants.
    All interactions are synthetic — this represents a hypothetical event
    for which we know the true generating parameters.
    """
    from datetime import timedelta
    root = RootPost(
        post_id=f"synth-{n_users}", user_id="root",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="synthetic", label="synthetic", expert_analysis=None,
    )
    # Create nominal interactions for structure (NodeIndex needs user list)
    # Spread users across days (not hours, to avoid overflow)
    interactions = tuple(
        InteractionRecord(
            interaction_id=f"s{i}", root_post_id=f"synth-{n_users}",
            user_id=f"u{i}",
            timestamp=root.timestamp + timedelta(days=i),
            kind="comment", text=f"s{i}",
        )
        for i in range(1, n_users)
    )
    return EventCase(
        case_id=f"synth-{n_users}", source_dataset="SYNTHETIC",
        root=root, interactions=interactions,
    )


def _generate_trajectory(params: ModelParams, case: EventCase,
                         seeds: tuple = SIM_SEEDS,
                         T: int = 14) -> np.ndarray:
    """Generate a synthetic observed trajectory from known params.
    Returns mean active_count across seeds, shape (T+1,)."""
    config = ReplayConfig(
        step_hours=24.0, tail_steps=0,
        network_mode=ReplayNetworkMode.BROADCAST,
        seeds=seeds,
    )
    all_trajs = []
    for seed in seeds:
        sim_config = ReplayConfig(
            step_hours=24.0, tail_steps=0,
            network_mode=ReplayNetworkMode.BROADCAST,
            seeds=(seed,),
        )
        result = run_replay(case, params, sim_config)
        if result.simulated_mean:
            arr = np.array(result.simulated_mean.get("active_count", []),
                          dtype=np.float64)
            # Pad/truncate to T+1
            if len(arr) < T + 1:
                arr = np.pad(arr, (0, T + 1 - len(arr)), mode="edge")
            else:
                arr = arr[:T + 1]
            all_trajs.append(arr)
    return np.mean(np.stack(all_trajs, axis=0), axis=0)


def _build_obs_from_trajectory(traj: np.ndarray) -> dict:
    """Build observed dict from trajectory for loss computation."""
    return {"active_count": traj.astype(np.float64)}


@dataclass
class RecoveryResult:
    """Result of one parameter recovery experiment."""
    experiment: str
    param_name: str
    true_value: float
    estimated_value: float
    relative_error: float
    at_boundary: bool
    train_loss: float
    val_loss: float
    success: bool
    n_iterations: int
    elapsed_s: float


def run_single_param_recovery() -> list[RecoveryResult]:
    """Phase 1: Recover each parameter individually at low/med/high levels."""
    results = []
    case = _make_synthetic_case(200)
    base = default_params()

    # Test levels for each parameter
    param_levels = {
        "propagation.beta_M": [0.1, 0.5, 0.9],
        "activation.alpha_0": [-5.0, 0.0, 2.0],
        "decay.gamma_0": [-4.0, 0.0, 4.0],
        "viral.beta_V": [0.1, 0.5, 0.9],
    }

    for param_path, levels in param_levels.items():
        for level in levels:
            print(f"  [{param_path}={level}] ", end="", flush=True)

            # Generate ground truth trajectory
            spec = ParameterSpec(param_path, *Stage1ParameterSet.bounds()
                                [list(param_levels.keys()).index(param_path)])
            true_params = apply_parameter_vector(base, (spec,), [level])
            true_traj = _generate_trajectory(true_params, case)

            # Set up calibration on ground truth trajectory
            # We need to hijack the calibration: use the synthetic trajectory
            # as the "observed" data. Since fit_stage1 reads from a real case,
            # we instead run a calibration and compare.

            # Build observation dict from synthetic trajectory
            obs = _build_obs_from_trajectory(true_traj)
            T = len(true_traj) - 1
            split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)
            masks = {"active_count": np.ones(T + 1, dtype=bool)}
            weights = LossWeights(active_count=1.0)

            # Calibrate only this single parameter
            specs = (spec,)
            bounds = [(spec.low, spec.high)]

            import time
            t0 = time.perf_counter()

            try:
                from scipy.optimize import differential_evolution
                best_loss = float("inf")
                best_x = None
                n_iters = 0

                def objective(x):
                    nonlocal n_iters
                    n_iters += 1
                    try:
                        p = apply_parameter_vector(base, specs, list(x))
                    except (ValueError, AttributeError):
                        return 1e10
                    sim = _generate_trajectory(p, case)
                    loss = compute_replay_loss(
                        obs, {"active_count": sim}, split, weights, masks,
                    )
                    return float(loss.train_total)

                opt = differential_evolution(
                    objective, bounds,
                    seed=20260803, workers=1, maxiter=10, popsize=6,
                    polish=False,
                )
                best_x = list(opt.x)
                best_loss = float(opt.fun)
                n_iters = opt.nit

                # Re-evaluate with validation
                best_p = apply_parameter_vector(base, specs, best_x)
                best_sim = _generate_trajectory(best_p, case)
                final_loss = compute_replay_loss(
                    obs, {"active_count": best_sim}, split, weights, masks,
                )

                elapsed = time.perf_counter() - t0

                rel_err = abs(best_x[0] - level) / max(abs(level), 0.01)
                at_boundary = (best_x[0] <= spec.low + 1e-3 or
                              best_x[0] >= spec.high - 1e-3)

                rr = RecoveryResult(
                    experiment=f"single_{param_path.split('.')[1]}",
                    param_name=param_path,
                    true_value=level,
                    estimated_value=best_x[0],
                    relative_error=rel_err,
                    at_boundary=at_boundary,
                    train_loss=float(final_loss.train_total),
                    val_loss=float(final_loss.val_total),
                    success=opt.success,
                    n_iterations=n_iters,
                    elapsed_s=elapsed,
                )
                results.append(rr)
                print(f"est={best_x[0]:.4f} err={rel_err:.4f} "
                      f"loss={best_loss:.6f} {'BOUNDARY' if at_boundary else ''}")

            except ImportError:
                print("scipy not available")
                results.append(RecoveryResult(
                    experiment=f"single_{param_path.split('.')[1]}",
                    param_name=param_path, true_value=level,
                    estimated_value=float("nan"), relative_error=float("nan"),
                    at_boundary=False, train_loss=float("inf"),
                    val_loss=float("inf"), success=False,
                    n_iterations=0, elapsed_s=0.0,
                ))

    return results


def run_two_param_recovery() -> list[RecoveryResult]:
    """Phase 2: Recover pairs of confusable parameters."""
    results = []
    case = _make_synthetic_case(200)
    base = default_params()

    pairs = [
        ("beta_M + alpha_0", [
            (0.3, -3.0), (0.5, 0.0), (0.7, 2.0),
        ]),
        ("alpha_0 + gamma_0", [
            (-3.0, -3.0), (0.0, 0.0), (2.0, 3.0),
        ]),
        ("beta_M + beta_V", [
            (0.3, 0.3), (0.5, 0.5), (0.7, 0.7),
        ]),
    ]

    spec_map = {
        "beta_M + alpha_0": ("propagation.beta_M", "activation.alpha_0"),
        "alpha_0 + gamma_0": ("activation.alpha_0", "decay.gamma_0"),
        "beta_M + beta_V": ("propagation.beta_M", "viral.beta_V"),
    }

    bounds_map = {s.path: (s.low, s.high) for s in Stage1ParameterSet.to_specs()}

    for pair_name, levels in pairs:
        p1_path, p2_path = spec_map[pair_name]
        b1, b2 = bounds_map[p1_path], bounds_map[p2_path]

        for v1, v2 in levels:
            label = f"2param_{pair_name.replace(' + ', '_')}_{v1}_{v2}"
            print(f"  [{label}] ", end="", flush=True)

            s1 = ParameterSpec(p1_path, *b1)
            s2 = ParameterSpec(p2_path, *b2)
            specs = (s1, s2)

            true_params = apply_parameter_vector(base, specs, [v1, v2])
            true_traj = _generate_trajectory(true_params, case)
            obs = _build_obs_from_trajectory(true_traj)
            T = len(true_traj) - 1
            split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)
            masks = {"active_count": np.ones(T + 1, dtype=bool)}
            weights = LossWeights(active_count=1.0)

            import time
            t0 = time.perf_counter()

            try:
                from scipy.optimize import differential_evolution

                def objective(x):
                    try:
                        p = apply_parameter_vector(base, specs, list(x))
                    except (ValueError, AttributeError):
                        return 1e10
                    sim = _generate_trajectory(p, case)
                    loss = compute_replay_loss(
                        obs, {"active_count": sim}, split, weights, masks,
                    )
                    return float(loss.train_total)

                opt = differential_evolution(
                    objective, [b1, b2],
                    seed=20260803, workers=1, maxiter=15, popsize=8,
                    polish=False,
                )
                best_x = list(opt.x)
                elapsed = time.perf_counter() - t0
                best_p = apply_parameter_vector(base, specs, best_x)
                best_sim = _generate_trajectory(best_p, case)
                final_loss = compute_replay_loss(
                    obs, {"active_count": best_sim}, split, weights, masks,
                )

                for i, (path, true_val, est_val) in enumerate(
                    zip([p1_path, p2_path], [v1, v2], best_x)
                ):
                    rel_err = abs(est_val - true_val) / max(abs(true_val), 0.01)
                    at_boundary = (
                        est_val <= specs[i].low + 1e-3 or
                        est_val >= specs[i].high - 1e-3
                    )
                    results.append(RecoveryResult(
                        experiment=label,
                        param_name=path,
                        true_value=true_val,
                        estimated_value=est_val,
                        relative_error=rel_err,
                        at_boundary=at_boundary,
                        train_loss=float(final_loss.train_total),
                        val_loss=float(final_loss.val_total),
                        success=opt.success,
                        n_iterations=opt.nit,
                        elapsed_s=elapsed,
                    ))
                print(f"err1={abs(best_x[0]-v1)/max(abs(v1),1e-6):.4f} "
                      f"err2={abs(best_x[1]-v2)/max(abs(v2),1e-6):.4f} "
                      f"loss={opt.fun:.6f}")

            except ImportError:
                print("scipy not available")

    return results


def save_results(single: list, dual: list):
    """Save all recovery results."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = single + dual

    # CSV
    csv_path = OUT_DIR / "recovery_results.csv"
    if all_results:
        keys = ["experiment", "param_name", "true_value", "estimated_value",
                "relative_error", "at_boundary", "train_loss", "val_loss",
                "success", "n_iterations", "elapsed_s"]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in all_results:
                f.write(",".join(str(getattr(r, k)) for k in keys) + "\n")
        print(f"\nCSV: {csv_path}")

    # Summary statistics
    good = [r for r in all_results
            if not r.at_boundary and r.relative_error < 0.5 and not np.isnan(r.relative_error)]
    boundary = [r for r in all_results if r.at_boundary]
    high_err = [r for r in all_results
                if not r.at_boundary and r.relative_error >= 0.5 and not np.isnan(r.relative_error)]

    # By parameter
    param_stats = {}
    for r in all_results:
        if r.param_name not in param_stats:
            param_stats[r.param_name] = []
        param_stats[r.param_name].append(r)

    print(f"\n  Good recovery (err<50%, no boundary): {len(good)}/{len(all_results)}")
    print(f"  Boundary hits: {len(boundary)}")
    print(f"  High error: {len(high_err)}")
    for pname, entries in param_stats.items():
        errs = [e.relative_error for e in entries if not np.isnan(e.relative_error)]
        if errs:
            print(f"  {pname}: median err={np.median(errs):.4f}, "
                  f"boundary={sum(1 for e in entries if e.at_boundary)}/{len(entries)}")

    # JSON summary
    summary = {
        "total_experiments": len(all_results),
        "good_recovery": len(good),
        "boundary_hits": len(boundary),
        "high_error": len(high_err),
        "by_parameter": {
            pname: {
                "median_relative_error": float(np.median(
                    [e.relative_error for e in entries
                     if not np.isnan(e.relative_error)]
                )),
                "boundary_rate": sum(1 for e in entries if e.at_boundary) / max(len(entries), 1),
                "n_experiments": len(entries),
            }
            for pname, entries in param_stats.items()
        },
    }
    with open(OUT_DIR / "recovery_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {OUT_DIR / 'recovery_summary.json'}")


def main():
    print("=" * 60)
    print("  SYNTHETIC PARAMETER RECOVERY")
    print("=" * 60)

    print("\n[1/2] Single-parameter recovery (12 experiments)...")
    single = run_single_param_recovery()

    print("\n[2/2] Two-parameter recovery (9 experiments)...")
    dual = run_two_param_recovery()

    print("\n" + "=" * 60)
    save_results(single, dual)
    print("=" * 60)


if __name__ == "__main__":
    main()
