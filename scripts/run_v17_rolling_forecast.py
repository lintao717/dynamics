"""[V1.7] Cohort-Conditioned Rolling Forecast.

For each of the 10 development pairs:
  - Cut at 24h, 48h, 72h (where data allows)
  - Predict next 24h, 48h, 72h
  - Use ONE_SHOT structure, 256 Sobol vectors, 5 FIT_SEEDS
  - Evaluate against ForecastTarget (hidden future)
  - Report: future RMSLE, MASE, peak error, baseline comparison

Protocol:
  - Parameters estimated from pre-cutoff history only
  - Future data NEVER used for parameter selection
  - Cohort-conditioned: full participant list known, only history used
  - Outputs marked cohort_conditioned=true, causal_forecast=false
"""

import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamics_simulation.config import default_params, ReactivationMode
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.observations import build_observed_trajectory
from dynamics_simulation.data.forecast import EventHistory, ForecastTarget
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.calibration.parameters import ParameterSpec, apply_parameter_vector
from dynamics_simulation.evaluation import rmsle, mase, peak_error, FIT_SEEDS, VALIDATION_SEEDS
from dynamics_simulation.baseline import BaselineForecast

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "forecast"
DATA = Path("data/raw/CHECKED/dataset")
STEP_H = 24.0

SPECS_3 = (
    ParameterSpec("propagation.beta_M", 0.0, 1.0),
    ParameterSpec("activation.alpha_0", -6.0, 2.0),
    ParameterSpec("decay.gamma_0", -6.0, 6.0),
)
BOUNDS = [(s.low, s.high) for s in SPECS_3]

PAIRS = [
    ("pair_01","26f247cc05fd53e12daa91c98e7feab8","cf5522f51f8be6859840b0b948bb813c"),
    ("pair_02","f983d8f70e265ec265acd9ea93e9b10d","d293a73970751898b5d83f5525a15fcc"),
    ("pair_03","68665bf53973dea5036336fcc5dc9eea","97bdff4a4098a06dc2b2597b514b2df3"),
    ("pair_04","7cfe610e01dafc496143664a1a8cf87d","9a74296ad4c231241c1ac6057c08bbcb"),
    ("pair_05","afc5ba429bc79086f05d6798e4ed978a","619f582e27e736ebc4c5def47f954535"),
    ("pair_06","25d9ed3994c2d5f030b864867facab47","f030f59e7579dcda946ab8a3bd2733c6"),
    ("pair_07","b4a497151507853058c8c50b6c6670f5","6f2b7e632c64ba5835164abe2f1ab28e"),
    ("pair_08","b9801872032a3bee629f6559bbf503ba","3b7e48be19979f9df3a146e2d0277c58"),
    ("pair_09","dfb9f2af5bb9b16ba717b91d6be5fa2f","100536e843cd4e307e1a2865b28b1f05"),
    ("pair_10","3dac58ea8cfef832bde25278a699fd45","22c6a11c223c838fa933b5b4af777fcc"),
]


def _sobol(n, bounds, seed):
    try:
        from scipy.stats.qmc import Sobol
        s = Sobol(d=len(bounds), seed=seed)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        return lo + s.random(n) * (hi - lo)
    except ImportError:
        rng = np.random.default_rng(seed)
        return np.array([[rng.uniform(*b) for b in bounds] for _ in range(n)])


def _fit_params(history, base, vectors):
    """Estimate parameters from pre-cutoff history only."""
    idx = NodeIndex.from_case(history) if hasattr(history, 'user_ids') else None
    # Build a partial EventCase from history
    # Reconstruct case from history for replay compatibility
    case = _history_to_case(history)
    idx = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)
    traj = build_observed_trajectory(case, idx, grid)
    obs = np.array(traj.active_count, dtype=np.float64)
    n = len(obs)
    sp_ct = min(3, max(1, n - 2))  # at least 1 future step for validation
    te = max(0, n - sp_ct - 1)

    if te < 1:  # not enough data
        return None, None

    ot = obs[:te + 1]
    best_train = float("inf")
    best_params = None

    for vec in vectors:
        p = apply_parameter_vector(base, SPECS_3, list(vec))
        try:
            cfg = ReplayConfig(step_hours=STEP_H, tail_steps=0,
                              network_mode=ReplayNetworkMode.BROADCAST,
                              seeds=FIT_SEEDS, micro_steps=1,
                              reactivation_mode=ReactivationMode.ONE_SHOT.value)
            r = run_replay(case, p, cfg)
            if not r.simulated_mean: continue
            s = np.array(r.simulated_mean.get("n_A_ts", [0]), dtype=np.float64)
            if len(s) > n: s = s[:n]
            elif len(s) < n: s = np.concatenate([s, np.full(n - len(s), s[-1])])
            tr = rmsle(ot, s[:len(ot)])
            if tr < best_train:
                best_train = tr
                best_params = vec
        except (ValueError, RuntimeError):
            continue
    return best_params, best_train


def _history_to_case(history):
    """Convert EventHistory back to a minimal EventCase for replay compatibility."""
    from dynamics_simulation.data.schema import EventCase
    return EventCase(
        case_id=history.case_id,
        source_dataset=history.source_dataset,
        root=history.root,
        interactions=history.interactions,
        metadata=history.metadata,
    )


def _forecast(case, params, seeds, target_len):
    """Run forecast with given parameters."""
    p = apply_parameter_vector(default_params(), SPECS_3, list(params))
    cfg = ReplayConfig(step_hours=STEP_H, tail_steps=0,
                      network_mode=ReplayNetworkMode.BROADCAST,
                      seeds=seeds, micro_steps=1,
                      reactivation_mode=ReactivationMode.ONE_SHOT.value)
    r = run_replay(case, p, cfg)
    if not r.simulated_mean: return None
    s = np.array(r.simulated_mean.get("n_A_ts", [0]), dtype=np.float64)
    if len(s) > target_len: s = s[:target_len]
    elif len(s) < target_len: s = np.concatenate([s, np.full(target_len - len(s), s[-1])])
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    vectors = _sobol(256, BOUNDS, 20260806)
    all_forecasts = []
    save_path = OUT / "v17_rolling_forecast.json"

    if save_path.exists():
        all_forecasts = json.loads(save_path.read_text())
        done = {(r["pid"], r["label"], r["cutoff_h"]) for r in all_forecasts}
    else:
        done = set()

    horizons = (1, 2, 3)  # predict 24h, 48h, 72h ahead
    cutoffs_h = [24, 48, 72]  # cutoff at 1 day, 2 days, 3 days

    for pid, fid, rid in PAIRS:
        for lk, cid in [("fake", fid), ("real", rid)]:
            for cutoff_h in cutoffs_h:
                if (pid, lk, cutoff_h) in done:
                    continue

                path = DATA / f"{lk}_news" / f"{cid}.json"
                case = load_checked_case(path)
                idx = NodeIndex.from_case(case)
                grid = TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)

                cutoff_step = cutoff_h // int(STEP_H)
                max_future = grid.last_data_step - cutoff_step
                if max_future < 1:
                    continue  # not enough future data

                try:
                    history = EventHistory.from_case(case, cutoff_step, STEP_H)
                    target = ForecastTarget.from_case(case, cutoff_step, horizons, STEP_H)
                except ValueError:
                    continue

                if (pid, lk, cutoff_h) in done:
                    continue

                print(f"\n[{pid} {lk} cutoff={cutoff_h}h] "
                      f"N={len(case.user_ids)} history_ix={len(history.interactions)} "
                      f"future_users={target.n_future_users}", flush=True)

                # Fit parameters on history only
                best_params, best_train = _fit_params(history, base, vectors)
                if best_params is None:
                    print(f"  SKIP (insufficient history)")
                    continue

                # Forecast
                full_case = _history_to_case(history)
                fc = _forecast(full_case, best_params, VALIDATION_SEEDS,
                               grid.last_data_step + 1)
                if fc is None:
                    print(f"  SKIP (forecast failed)")
                    continue

                # Extract forecast at horizons
                fc_vals = np.array([fc[cutoff_step + h] for h in horizons])
                obs_vals = np.array([target.active_count[i] for i in range(len(horizons))])

                # Baselines (from history only)
                hist_traj = build_observed_trajectory(
                    _history_to_case(history), idx,
                    TimeGrid.from_case(_history_to_case(history), step_hours=STEP_H, tail_steps=0))
                hist_obs = np.array(hist_traj.active_count, dtype=np.float64)
                n_hist = len(hist_obs)
                if n_hist >= 2:
                    bf = BaselineForecast(hist_obs, n_hist + len(horizons))
                    bl_future = bf.best_future[:len(horizons)]
                    bl_rmsle = rmsle(obs_vals, bl_future)
                else:
                    bl_rmsle = 9.0

                fc_rmsle = rmsle(obs_vals, fc_vals)
                pk = peak_error(obs_vals, fc_vals)

                entry = {
                    "pid": pid, "label": lk, "case_id": cid,
                    "cutoff_h": cutoff_h, "cutoff_step": cutoff_step,
                    "horizons": list(horizons),
                    "n_agents": len(case.user_ids),
                    "n_history_ix": len(history.interactions),
                    "n_future_users": target.n_future_users,
                    "obs_future": [float(x) for x in obs_vals],
                    "fc_future": [float(x) for x in fc_vals],
                    "bl_rmsle": round(bl_rmsle, 4),
                    "fc_rmsle": round(fc_rmsle, 4),
                    "peak_ratio": round(pk["peak_ratio"], 4),
                    "peak_step_error": pk["peak_step_error"],
                    "best_params": {
                        "beta_M": float(best_params[0]),
                        "alpha_0": float(best_params[1]),
                        "gamma_0": float(best_params[2]),
                    },
                    "best_train_rmsle": round(best_train, 4),
                    "cohort_conditioned": True,
                    "causal_forecast": False,
                }
                all_forecasts.append(entry)
                save_path.write_text(json.dumps(all_forecasts, indent=2))

                imp = (bl_rmsle - fc_rmsle) / max(bl_rmsle, 1e-6) * 100
                win = "WIN" if fc_rmsle < bl_rmsle else "LOSE"
                print(f"  {win} fc_rmsle={fc_rmsle:.4f} vs bl={bl_rmsle:.4f} ({imp:+.0f}%) "
                      f"obs={[f'{x:.0f}' for x in obs_vals]} fc={[f'{x:.1f}' for x in fc_vals]}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  V1.7 ROLLING FORECAST SUMMARY ({len(all_forecasts)} forecasts)")
    print(f"{'='*60}")
    wins = sum(1 for r in all_forecasts if r["fc_rmsle"] < r["bl_rmsle"])
    print(f"  Model beats baseline: {wins}/{len(all_forecasts)}")
    print(f"  Mean fc_rmsle={np.mean([r['fc_rmsle'] for r in all_forecasts]):.4f}")
    print(f"  Mean bl_rmsle={np.mean([r['bl_rmsle'] for r in all_forecasts]):.4f}")

    for lk in ["fake", "real"]:
        lr = [r for r in all_forecasts if r["label"] == lk]
        w = sum(1 for r in lr if r["fc_rmsle"] < r["bl_rmsle"])
        print(f"  {lk}: {w}/{len(lr)} wins, mean_fc={np.mean([r['fc_rmsle'] for r in lr]):.4f}")

    for ch in cutoffs_h:
        cr = [r for r in all_forecasts if r["cutoff_h"] == ch]
        if cr:
            w = sum(1 for r in cr if r["fc_rmsle"] < r["bl_rmsle"])
            print(f"  cutoff={ch}h: {w}/{len(cr)} wins")

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
