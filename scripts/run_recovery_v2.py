"""
V2 synthetic parameter recovery — corrected protocol.

Fixes vs V1:
  - Separate seeds for truth generation, fitting, and holdout validation
  - Range-normalised error (not relative error for near-zero values)
  - Reports optimisation_runs vs parameter_estimates separately
  - Structural mode: multi-seed mean trajectory
  - Practical mode: single-seed trajectory with noise
  - Tracks convergence status and per-parameter boundary flags
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics_simulation.config import ModelParams, default_params
from dynamics_simulation.data.schema import EventCase, RootPost, InteractionRecord
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.calibration.parameters import (
    Stage1ParameterSet, apply_parameter_vector, ParameterSpec,
)
from dynamics_simulation.calibration.objective import compute_replay_loss, LossWeights
from dynamics_simulation.calibration.split import TemporalSplit

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "recovery"

# ── Strictly separated seeds ──
TRUTH_SEEDS = (101, 103, 107, 109, 113)    # generate synthetic truth
FIT_SEEDS = (11, 23, 37, 53, 71)            # loss evaluation during optimisation
HOLDOUT_SEEDS = (211, 223, 227, 229, 233)   # independent validation

SPECS = Stage1ParameterSet.to_specs()
BOUNDS = Stage1ParameterSet.bounds()
SPEC_DICT = {s.path: s for s in SPECS}


def _make_synthetic_case(n_users: int = 200) -> EventCase:
    root = RootPost(
        post_id=f"syn-{n_users}", user_id="root",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="syn", label="syn", expert_analysis=None,
    )
    interactions = tuple(
        InteractionRecord(
            interaction_id=f"s{i}", root_post_id=f"syn-{n_users}",
            user_id=f"u{i}",
            timestamp=root.timestamp + timedelta(days=i),
            kind="comment", text=f"s{i}",
        )
        for i in range(1, n_users)
    )
    return EventCase(case_id=f"syn-{n_users}", source_dataset="SYNTHETIC",
                     root=root, interactions=interactions)


def _generate_trajectory(params: ModelParams, case: EventCase,
                         seeds: tuple, T: int = 14) -> np.ndarray:
    """Generate trajectory with given seeds, return mean across seeds."""
    config = ReplayConfig(
        step_hours=24.0, tail_steps=0,
        network_mode=ReplayNetworkMode.BROADCAST, seeds=seeds,
    )
    result = run_replay(case, params, config)
    if not result.simulated_mean:
        return np.zeros(T + 1)
    arr = np.array(result.simulated_mean.get("active_count", []), dtype=np.float64)
    if len(arr) < T + 1:
        arr = np.pad(arr, (0, T + 1 - len(arr)), mode="edge")
    return arr[:T + 1]


def _range_normalised_error(estimate, truth, spec):
    """Range-normalised error: |est - truth| / (high - low)."""
    denom = spec.high - spec.low
    return abs(estimate - truth) / max(denom, 1e-6)


def _run_single_recovery(param_path, true_val, truth_seeds, fit_seeds,
                         holdout_seeds, case, base, T=14):
    """Recover a single parameter with separate seed sets."""
    spec = SPEC_DICT[param_path]
    specs = (spec,)
    true_params = apply_parameter_vector(base, specs, [true_val])
    truth_traj = _generate_trajectory(true_params, case, truth_seeds, T)
    obs = {"active_count": truth_traj.astype(np.float64)}
    split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)
    masks = {"active_count": np.ones(T + 1, dtype=bool)}
    weights = LossWeights(active_count=1.0)

    t0 = time.perf_counter()
    try:
        from scipy.optimize import differential_evolution
        n_iters = [0]

        def obj(x):
            n_iters[0] += 1
            try:
                p = apply_parameter_vector(base, specs, list(x))
            except (ValueError, AttributeError):
                return 1e10
            sim = _generate_trajectory(p, case, fit_seeds, T)
            loss = compute_replay_loss(obs, {"active_count": sim}, split, weights, masks)
            return float(loss.train_total)

        opt = differential_evolution(
            obj, [(spec.low, spec.high)],
            seed=20260803, workers=1, maxiter=10, popsize=6, polish=False,
        )
        best_x = list(opt.x)
        elapsed = time.perf_counter() - t0

        # Independent holdout evaluation
        best_p = apply_parameter_vector(base, specs, best_x)
        holdout_sim = _generate_trajectory(best_p, case, holdout_seeds, T)
        final_loss = compute_replay_loss(
            obs, {"active_count": holdout_sim}, split, weights, masks,
        )

        at_boundary = (best_x[0] <= spec.low + 1e-3 or best_x[0] >= spec.high - 1e-3)
        return {
            "true_value": true_val,
            "estimated": best_x[0],
            "abs_error": abs(best_x[0] - true_val),
            "range_norm_error": _range_normalised_error(best_x[0], true_val, spec),
            "at_boundary": at_boundary,
            "converged": bool(opt.success),
            "train_loss": float(final_loss.train_total),
            "val_loss": float(final_loss.val_total),
            "holdout_rmsle": float(np.sqrt(np.mean(
                (np.log1p(np.maximum(holdout_sim, 0)) -
                 np.log1p(np.maximum(truth_traj, 0))) ** 2
            ))),
            "n_iterations": n_iters[0],
            "elapsed_s": elapsed,
        }
    except ImportError:
        return {"true_value": true_val, "estimated": float("nan"),
                "abs_error": float("nan"), "range_norm_error": float("nan"),
                "at_boundary": False, "converged": False,
                "train_loss": float("inf"), "val_loss": float("inf"),
                "holdout_rmsle": float("nan"),
                "n_iterations": 0, "elapsed_s": 0.0}


def _run_two_param_recovery(p1_path, p2_path, v1, v2, truth_seeds,
                            fit_seeds, holdout_seeds, case, base, T=14):
    """Recover a parameter pair with separate seed sets."""
    s1, s2 = SPEC_DICT[p1_path], SPEC_DICT[p2_path]
    specs = (s1, s2)
    true_params = apply_parameter_vector(base, specs, [v1, v2])
    truth_traj = _generate_trajectory(true_params, case, truth_seeds, T)
    obs = {"active_count": truth_traj.astype(np.float64)}
    split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)
    masks = {"active_count": np.ones(T + 1, dtype=bool)}
    weights = LossWeights(active_count=1.0)

    t0 = time.perf_counter()
    try:
        from scipy.optimize import differential_evolution

        def obj(x):
            try:
                p = apply_parameter_vector(base, specs, list(x))
            except (ValueError, AttributeError):
                return 1e10
            sim = _generate_trajectory(p, case, fit_seeds, T)
            return float(compute_replay_loss(
                obs, {"active_count": sim}, split, weights, masks,
            ).train_total)

        opt = differential_evolution(
            obj, [(s1.low, s1.high), (s2.low, s2.high)],
            seed=20260803, workers=1, maxiter=15, popsize=8, polish=False,
        )
        best_x = list(opt.x)
        elapsed = time.perf_counter() - t0
        best_p = apply_parameter_vector(base, specs, best_x)
        holdout_sim = _generate_trajectory(best_p, case, holdout_seeds, T)
        final_loss = compute_replay_loss(
            obs, {"active_count": holdout_sim}, split, weights, masks,
        )

        results = []
        for i, (path, true_val, est_val, spec) in enumerate(
            zip([p1_path, p2_path], [v1, v2], best_x, [s1, s2])
        ):
            at_boundary = (est_val <= spec.low + 1e-3 or est_val >= spec.high - 1e-3)
            results.append({
                "experiment": f"{p1_path.split('.')[1]}_{p2_path.split('.')[1]}",
                "param": path,
                "true_value": true_val,
                "estimated": est_val,
                "abs_error": abs(est_val - true_val),
                "range_norm_error": _range_normalised_error(est_val, true_val, spec),
                "at_boundary": at_boundary,
                "converged": bool(opt.success),
                "train_loss": float(final_loss.train_total),
                "val_loss": float(final_loss.val_total),
                "holdout_rmsle": float(np.sqrt(np.mean(
                    (np.log1p(np.maximum(holdout_sim, 0)) -
                     np.log1p(np.maximum(truth_traj, 0))) ** 2
                ))),
                "n_iterations": opt.nit,
                "elapsed_s": elapsed,
            })
        return results
    except ImportError:
        return [{"experiment": "scipy_missing", "param": p,
                 "true_value": v, "estimated": float("nan"),
                 "abs_error": float("nan"), "range_norm_error": float("nan")}
                for p, v in [(p1_path, v1), (p2_path, v2)]]


def main():
    print("=" * 60)
    print("  V2 SYNTHETIC PARAMETER RECOVERY (Corrected Protocol)")
    print(f"  Truth seeds: {TRUTH_SEEDS}")
    print(f"  Fit seeds:   {FIT_SEEDS}")
    print(f"  Holdout:     {HOLDOUT_SEEDS}")
    print("=" * 60)

    case = _make_synthetic_case(200)
    base = default_params()
    all_estimates = []
    run_count = 0

    # ── Single-parameter (12 optimisation runs) ──
    print("\n[1/3] Single-parameter recovery (12 runs)...")
    levels = {
        "propagation.beta_M": [0.2, 0.5, 0.8],
        "activation.alpha_0": [-4.0, 0.0, 1.5],
        "decay.gamma_0": [-3.0, 0.0, 3.0],
        "viral.beta_V": [0.2, 0.5, 0.8],
    }
    for param_path, lvls in levels.items():
        for lvl in lvls:
            run_count += 1
            print(f"  [{param_path}={lvl}] ", end="", flush=True)
            r = _run_single_recovery(param_path, lvl, TRUTH_SEEDS,
                                     FIT_SEEDS, HOLDOUT_SEEDS, case, base)
            r["run_type"] = "single"
            r["param"] = param_path
            r["run_id"] = run_count
            all_estimates.append(r)
            print(f"est={r['estimated']:.4f} err={r['range_norm_error']:.4f} "
                  f"conv={r['converged']} bound={r['at_boundary']}")

    # ── Two-parameter (9 optimisation runs, 18 estimates) ──
    print("\n[2/3] Two-parameter recovery (9 runs)...")
    pairs = [
        ("propagation.beta_M", "activation.alpha_0",
         [(0.3, -3.0), (0.5, 0.0), (0.7, 1.5)]),
        ("activation.alpha_0", "decay.gamma_0",
         [(-3.0, -3.0), (0.0, 0.0), (1.5, 2.5)]),
        ("propagation.beta_M", "viral.beta_V",
         [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]),
    ]
    for p1, p2, lvls in pairs:
        for v1, v2 in lvls:
            run_count += 1
            print(f"  [{p1.split('.')[1]}+{p2.split('.')[1]} {v1},{v2}] ",
                  end="", flush=True)
            results = _run_two_param_recovery(p1, p2, v1, v2, TRUTH_SEEDS,
                                              FIT_SEEDS, HOLDOUT_SEEDS, case, base)
            for r in results:
                r["run_type"] = "dual"
                r["run_id"] = run_count
                all_estimates.append(r)
            e1 = results[0]
            e2 = results[1]
            print(f"e1={e1['range_norm_error']:.4f} e2={e2['range_norm_error']:.4f}")

    # ── Four-parameter pilot (5 runs) ──
    print("\n[3/3] Four-parameter pilot (5 runs)...")
    rng = np.random.default_rng(20260803)
    for i in range(5):
        run_count += 1
        # Latin-hypercube-style sampling within bounds
        x = [rng.uniform(*b) for b in BOUNDS]
        print(f"  [run {i+1}/5] true={[round(v,3) for v in x]} ", end="", flush=True)

        specs = SPECS
        true_params = apply_parameter_vector(base, specs, x)
        truth_traj = _generate_trajectory(true_params, case, TRUTH_SEEDS)
        obs = {"active_count": truth_traj.astype(np.float64)}
        T = len(truth_traj) - 1
        split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)
        masks = {"active_count": np.ones(T + 1, dtype=bool)}
        weights = LossWeights(active_count=1.0)

        t0 = time.perf_counter()
        try:
            from scipy.optimize import differential_evolution

            def obj(v):
                try:
                    p = apply_parameter_vector(base, specs, list(v))
                except (ValueError, AttributeError):
                    return 1e10
                sim = _generate_trajectory(p, case, FIT_SEEDS, T)
                return float(compute_replay_loss(
                    obs, {"active_count": sim}, split, weights, masks,
                ).train_total)

            opt = differential_evolution(
                obj, BOUNDS,
                seed=20260803 + i, workers=1, maxiter=25, popsize=10,
                polish=False,
            )
            best_x = list(opt.x)
            elapsed = time.perf_counter() - t0

            # Holdout evaluation
            best_p = apply_parameter_vector(base, specs, best_x)
            holdout_sim = _generate_trajectory(best_p, case, HOLDOUT_SEEDS, T)
            final_loss = compute_replay_loss(
                obs, {"active_count": holdout_sim}, split, weights, masks,
            )

            for j, (spec, true_val, est_val) in enumerate(zip(SPECS, x, best_x)):
                at_boundary = (est_val <= spec.low + 1e-3 or
                              est_val >= spec.high - 1e-3)
                all_estimates.append({
                    "run_type": "quad",
                    "run_id": run_count,
                    "param": spec.path,
                    "true_value": true_val,
                    "estimated": est_val,
                    "abs_error": abs(est_val - true_val),
                    "range_norm_error": _range_normalised_error(est_val, true_val, spec),
                    "at_boundary": at_boundary,
                    "converged": bool(opt.success),
                    "train_loss": float(final_loss.train_total),
                    "val_loss": float(final_loss.val_total),
                    "holdout_rmsle": float(np.sqrt(np.mean(
                        (np.log1p(np.maximum(holdout_sim, 0)) -
                         np.log1p(np.maximum(truth_traj, 0))) ** 2
                    ))),
                    "n_iterations": opt.nit,
                    "elapsed_s": elapsed,
                })
            errs = [all_estimates[-4 + k]["range_norm_error"] for k in range(4)]
            bounds = [all_estimates[-4 + k]["at_boundary"] for k in range(4)]
            print(f"err={[round(e,4) for e in errs]} "
                  f"bound={sum(bounds)} conv={opt.success}")

        except ImportError:
            for spec, true_val in zip(SPECS, x):
                all_estimates.append({
                    "run_type": "quad", "run_id": run_count,
                    "param": spec.path,
                    "true_value": true_val, "estimated": float("nan"),
                    "abs_error": float("nan"), "range_norm_error": float("nan"),
                    "at_boundary": False, "converged": False,
                    "train_loss": float("inf"), "val_loss": float("inf"),
                    "holdout_rmsle": float("nan"),
                    "n_iterations": 0, "elapsed_s": 0.0,
                })

    # ── Save and report ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_estimates = len(all_estimates)

    # Summary by parameter
    param_groups = {}
    for e in all_estimates:
        p = e["param"]
        if p not in param_groups:
            param_groups[p] = []
        param_groups[p].append(e)

    # CSV
    csv_path = OUT_DIR / "recovery_v2_results.csv"
    keys = [k for k in all_estimates[0].keys() if k not in ("run_type",)]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("run_type," + ",".join(keys) + "\n")
        for e in all_estimates:
            f.write(e["run_type"] + "," + ",".join(str(e.get(k, "")) for k in keys) + "\n")
    print(f"\nCSV: {csv_path} ({n_estimates} parameter estimates from {run_count} runs)")

    # Good recovery criteria: range_norm_error < 0.10
    single_ests = [e for e in all_estimates if e["run_type"] == "single"]
    dual_ests = [e for e in all_estimates if e["run_type"] == "dual"]
    quad_ests = [e for e in all_estimates if e["run_type"] == "quad"]

    good = lambda es: [e for e in es if not e["at_boundary"]
                       and not np.isnan(e["range_norm_error"])
                       and e["range_norm_error"] < 0.10]
    boundary = lambda es: [e for e in es if e["at_boundary"]]

    print(f"\n  Single: {len(good(single_ests))}/{len(single_ests)} good, "
          f"{len(boundary(single_ests))} boundary")
    print(f"  Dual:   {len(good(dual_ests))}/{len(dual_ests)} good, "
          f"{len(boundary(dual_ests))} boundary")
    print(f"  Quad:   {len(good(quad_ests))}/{len(quad_ests)} good, "
          f"{len(boundary(quad_ests))} boundary")

    for pname, entries in param_groups.items():
        errs = [e["range_norm_error"] for e in entries
                if not np.isnan(e["range_norm_error"])]
        bounds_count = sum(1 for e in entries if e["at_boundary"])
        conv_count = sum(1 for e in entries if e.get("converged", False))
        if errs:
            print(f"  {pname}: median_range_err={np.median(errs):.4f} "
                  f"p90={np.percentile(errs, 90):.4f} "
                  f"boundary={bounds_count}/{len(entries)} "
                  f"converged={conv_count}/{len(entries)}")

    # JSON summary
    summary = {
        "protocol": "v2_corrected",
        "truth_seeds": list(TRUTH_SEEDS),
        "fit_seeds": list(FIT_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "optimisation_runs": run_count,
        "parameter_estimates": n_estimates,
        "by_type": {
            "single": {
                "n_runs": 12,
                "n_estimates": len(single_ests),
                "good_recovery": len(good(single_ests)),
                "boundary_hits": len(boundary(single_ests)),
            },
            "dual": {
                "n_runs": 9,
                "n_estimates": len(dual_ests),
                "good_recovery": len(good(dual_ests)),
                "boundary_hits": len(boundary(dual_ests)),
            },
            "quad": {
                "n_runs": 5,
                "n_estimates": len(quad_ests),
                "good_recovery": len(good(quad_ests)),
                "boundary_hits": len(boundary(quad_ests)),
            },
        },
        "by_parameter": {
            pname: {
                "median_range_norm_error": float(np.median(
                    [e["range_norm_error"] for e in entries
                     if not np.isnan(e["range_norm_error"])]
                )),
                "p90_range_norm_error": float(np.percentile(
                    [e["range_norm_error"] for e in entries
                     if not np.isnan(e["range_norm_error"])], 90
                )),
                "boundary_rate": sum(1 for e in entries if e["at_boundary"]) / max(len(entries), 1),
                "convergence_rate": sum(1 for e in entries if e.get("converged", False)) / max(len(entries), 1),
            }
            for pname, entries in param_groups.items()
        },
    }
    with open(OUT_DIR / "recovery_v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {OUT_DIR / 'recovery_v2_summary.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
