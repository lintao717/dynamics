"""[V1.5 S3] Formal 3-Parameter Calibration.

Model: S3 one-shot (D0->A=OFF, D1->A=OFF).
Params: beta_M [0,1], alpha_0 [-6,2], gamma_0 [-6,6].
Optimizer: differential_evolution, maxiter=60, popsize=12, polish=True.
3 optimizer seeds for stability, 5 replay seeds per evaluation.
70/30 temporal split on last_data_step.

Outputs: per-case calibration results, parameter stability report,
         comparison against default params and open-loop baselines.
"""

import json, sys, time
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamics_simulation.config import default_params
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.observations import build_observed_trajectory
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.calibration.parameters import (
    Stage1ParameterSet, apply_parameter_vector, ParameterSpec,
)
from dynamics_simulation.calibration.split import TemporalSplit

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "calibration"
DATA = Path("data/raw/CHECKED/dataset")

SPECS_3 = (
    ParameterSpec("propagation.beta_M", 0.0, 1.0),
    ParameterSpec("activation.alpha_0", -6.0, 2.0),
    ParameterSpec("decay.gamma_0", -6.0, 6.0),
)
BOUNDS_3 = [(s.low, s.high) for s in SPECS_3]

CASES = [
    ("pair_05", "fake", "afc5ba429bc79086f05d6798e4ed978a"),
    ("pair_05", "real", "619f582e27e736ebc4c5def47f954535"),
    ("pair_07", "fake", "b4a497151507853058c8c50b6c6670f5"),
    ("pair_07", "real", "6f2b7e632c64ba5835164abe2f1ab28e"),
]
REPLAY_SEEDS = (11, 23, 37, 53, 71)
OPT_SEEDS = (20260805, 20260817, 20260829)


def _rmsle(a, b):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(b,0))-np.log1p(np.maximum(a,0)))**2)))


def _make_s3(base, values):
    p = apply_parameter_vector(base, SPECS_3, list(values))
    r = p.reactivation
    return replace(p, reactivation=replace(r, r_0_0=-1e9, r_0_1=-1e9))


def calibrate_one(case, obs_train, obs_val, te, n_data, opt_seed):
    """Run differential_evolution for one optimizer seed. Returns (best_params, train_rmsle, val_rmsle, info)."""
    base = default_params()

    def obj(x):
        try:
            p = _make_s3(base, x)
        except ValueError:
            return 1e10
        cfg = ReplayConfig(step_hours=24.0, tail_steps=0,
                          network_mode=ReplayNetworkMode.BROADCAST,
                          seeds=REPLAY_SEEDS, micro_steps=1)
        r = run_replay(case, p, cfg)
        if not r.simulated_mean: return 1e10
        sim = np.array(r.simulated_mean.get("n_A_ts", [0]), dtype=np.float64)
        if len(sim) > n_data: sim = sim[:n_data]
        elif len(sim) < n_data: sim = np.pad(sim, (0, n_data - len(sim)), mode="edge")
        return _rmsle(obs_train, sim[:te+1])

    from scipy.optimize import differential_evolution
    t0 = time.perf_counter()
    opt = differential_evolution(obj, BOUNDS_3, seed=int(opt_seed),
                                 workers=1, maxiter=60, popsize=12, polish=True)
    elapsed = time.perf_counter() - t0

    best_p = _make_s3(base, list(opt.x))
    cfg_val = ReplayConfig(step_hours=24.0, tail_steps=0,
                          network_mode=ReplayNetworkMode.BROADCAST,
                          seeds=REPLAY_SEEDS, micro_steps=1)
    r_val = run_replay(case, best_p, cfg_val)
    sim_val = np.array(r_val.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
    if len(sim_val) > n_data: sim_val = sim_val[:n_data]
    elif len(sim_val) < n_data: sim_val = np.pad(sim_val, (0, n_data - len(sim_val)), mode="edge")
    tr_rmsle = _rmsle(obs_train, sim_val[:te+1])
    vl_rmsle = _rmsle(obs_val, sim_val[te+1:])

    peak_sim = float(np.max(sim_val))
    peak_obs = float(np.max(np.concatenate([obs_train, obs_val])))
    pr = peak_sim / max(peak_obs, 1)

    at_boundary = any(
        x <= lo + 1e-3 or x >= hi - 1e-3
        for x, (lo, hi) in zip(opt.x, BOUNDS_3)
    )

    return {
        "opt_seed": opt_seed,
        "best_x": [float(v) for v in opt.x],
        "beta_M": float(opt.x[0]), "alpha_0": float(opt.x[1]), "gamma_0": float(opt.x[2]),
        "train_rmsle": round(tr_rmsle, 6),
        "val_rmsle": round(vl_rmsle, 6),
        "peak_ratio": round(pr, 4),
        "converged": bool(opt.success),
        "n_iter": opt.nit,
        "at_boundary": at_boundary,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = []

    for pid, label, cid in CASES:
        print(f"\n{'='*60}")
        print(f"  S3 CALIBRATION [{pid} {label}]")
        print(f"{'='*60}")

        path = DATA / f"{label}_news" / f"{cid}.json"
        case = load_checked_case(path)
        idx = NodeIndex.from_case(case)
        grid = TimeGrid.from_case(case, step_hours=24.0, tail_steps=0)
        traj = build_observed_trajectory(case, idx, grid)
        obs_arr = np.array(traj.active_count, dtype=np.float64)
        n_data = grid.last_data_step + 1
        split = TemporalSplit.by_fraction(total_steps=grid.last_data_step, train_fraction=0.7)
        te = split.train_end_step
        obs_train = obs_arr[:te+1]
        obs_val = obs_arr[te+1:]

        # Baselines
        t_tr = np.arange(len(obs_train), dtype=np.float64)
        A = np.column_stack([np.ones_like(t_tr), t_tr])
        c, _, _, _ = np.linalg.lstsq(A, np.log(np.maximum(obs_train,1e-6)), rcond=None)
        bl_exp = np.exp(c[0]) * np.exp(-max(-c[1],1e-6) * np.arange(n_data, dtype=np.float64))
        bl_train = _rmsle(obs_train, bl_exp[:te+1])
        bl_val = _rmsle(obs_val, bl_exp[te+1:])

        # Default params baseline
        base = default_params()
        base_p = _make_s3(base, [base.propagation.beta_M, base.activation.alpha_0, base.decay.gamma_0])
        cfg_d = ReplayConfig(step_hours=24.0, tail_steps=0,
                            network_mode=ReplayNetworkMode.BROADCAST,
                            seeds=REPLAY_SEEDS, micro_steps=1)
        r_d = run_replay(case, base_p, cfg_d)
        sim_d = np.array(r_d.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
        if len(sim_d) > n_data: sim_d = sim_d[:n_data]
        elif len(sim_d) < n_data: sim_d = np.pad(sim_d, (0,n_data-len(sim_d)), mode="edge")
        def_tr = _rmsle(obs_train, sim_d[:te+1])
        def_vl = _rmsle(obs_val, sim_d[te+1:])

        print(f"  N={len(case.user_ids)} steps={n_data} train={te+1} val={n_data-te-1}")
        print(f"  Baseline: train={bl_train:.4f} val={bl_val:.4f}")
        print(f"  Default S3: train={def_tr:.4f} val={def_vl:.4f}")

        # Run 3 calibrations with different optimizer seeds
        calibrations = []
        for oi, opt_seed in enumerate(OPT_SEEDS):
            print(f"  Opt seed {oi+1}/{len(OPT_SEEDS)} ({opt_seed})...", end=" ", flush=True)
            cal = calibrate_one(case, obs_train, obs_val, te, n_data, opt_seed)
            calibrations.append(cal)
            imp = (bl_val - cal["val_rmsle"]) / max(bl_val, 1e-6) * 100
            print(f"val={cal['val_rmsle']:.4f} ({imp:+.0f}% vs bl) "
                  f"beta_M={cal['beta_M']:.3f} alpha_0={cal['alpha_0']:.3f} "
                  f"gamma_0={cal['gamma_0']:.3f} "
                  f"conv={cal['converged']} bound={cal['at_boundary']} "
                  f"{cal['elapsed_s']:.0f}s")

        # Stability check
        beta_Ms = [c["beta_M"] for c in calibrations]
        alpha_0s = [c["alpha_0"] for c in calibrations]
        gamma_0s = [c["gamma_0"] for c in calibrations]
        val_losses = [c["val_rmsle"] for c in calibrations]

        best_cal = min(calibrations, key=lambda c: c["val_rmsle"])
        imp_pct = (bl_val - best_cal["val_rmsle"]) / max(bl_val, 1e-6) * 100

        result = {
            "pid": pid, "label": label,
            "n_agents": len(case.user_ids),
            "n_steps": n_data, "train_steps": te+1, "val_steps": n_data-te-1,
            "bl_train_rmsle": bl_train, "bl_val_rmsle": bl_val,
            "default_train_rmsle": def_tr, "default_val_rmsle": def_vl,
            "best_calibration": best_cal,
            "improvement_vs_baseline_pct": round(imp_pct, 1),
            "calibrations": calibrations,
            "cross_seed_std_beta_M": round(float(np.std(beta_Ms)), 4),
            "cross_seed_std_alpha_0": round(float(np.std(alpha_0s)), 4),
            "cross_seed_std_gamma_0": round(float(np.std(gamma_0s)), 4),
            "cross_seed_cv_val": round(float(np.std(val_losses) / max(np.mean(val_losses), 1e-6)), 4),
        }
        all_results.append(result)

        status = "PASS" if imp_pct > 10 and not best_cal["at_boundary"] else "CHECK"
        print(f"  {status}: improvement={imp_pct:+.1f}% "
              f"cross-seed CV={result['cross_seed_cv_val']:.4f}")

    # Final report
    print(f"\n{'='*60}")
    print(f"  S3 CALIBRATION REPORT")
    print(f"{'='*60}")
    for r in all_results:
        bc = r["best_calibration"]
        imp = r["improvement_vs_baseline_pct"]
        cv = r["cross_seed_cv_val"]
        print(f"  {r['pid']} {r['label']}: val={bc['val_rmsle']:.4f} (vs bl={r['bl_val_rmsle']:.4f}, {imp:+.0f}%) "
              f"beta_M={bc['beta_M']:.3f} alpha_0={bc['alpha_0']:.3f} gamma_0={bc['gamma_0']:.3f} "
              f"cv={cv:.4f} bound={bc['at_boundary']}")

    n_pass = sum(1 for r in all_results
                 if r["improvement_vs_baseline_pct"] > 10
                 and not r["best_calibration"]["at_boundary"]
                 and r["cross_seed_cv_val"] < 0.2)
    print(f"\n  Passed: {n_pass}/{len(all_results)} (improvement>10%, no boundary, cross-seed CV<0.2)")

    if n_pass >= 3:
        print("  >>> S3 CALIBRATION SUCCESSFUL — ready for 20-case extension.")
    elif n_pass >= 2:
        print("  >> Partially successful — review marginal cases before extending.")
    else:
        print("  >> Needs further investigation.")

    with open(OUT / "s3_calibration.json", "w", encoding="utf-8") as f:
        json.dump({"results": all_results,
                   "model": "S3 one-shot (D0/D1 disabled)",
                   "params": "beta_M, alpha_0, gamma_0",
                   "optimizer": "differential_evolution (maxiter=60, popsize=12, polish=True)",
                   "replay_seeds": list(REPLAY_SEEDS),
                   "opt_seeds": list(OPT_SEEDS)},
                  f, indent=2, default=str)
    print(f"Saved: {OUT / 's3_calibration.json'}")


if __name__ == "__main__":
    main()
