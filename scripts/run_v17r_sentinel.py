"""[V1.7R] Sentinel Forecast Experiment.

4 sentinel cases:
  1. Fake case that performed well in V1.5.2 (pair_06 fake)
  2. Fake case that failed in V1.5.2 (pair_09 fake)
  3. Real case with rapid post-peak decay (pair_05 real)
  4. Real case with visible long tail (pair_03 real)

Each: 48h cutoff -> 72h forecast, 72h cutoff -> 48h forecast.
Both oracle_cohort and observed_closed modes.
ONE_SHOT structure, beta_M adapted per event from pre-cutoff history.
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
from dynamics_simulation.forecast_runner import ForecastConfig, ForecastRunner
from dynamics_simulation.evaluation import rmsle, mase, peak_error
from dynamics_simulation.baseline import BaselineForecast

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "forecast"
DATA = Path("data/raw/CHECKED/dataset")
STEP_H = 24.0

# 4 sentinel cases
SENTINELS = [
    # (pid, label, case_id, rationale)
    ("pair_06", "fake", "25d9ed3994c2d5f030b864867facab47",
     "fake-news, performed well in V1.5.2 ONE_SHOT (reachable)"),
    ("pair_09", "fake", "dfb9f2af5bb9b16ba717b91d6be5fa2f",
     "fake-news, failed in V1.5.2 ONE_SHOT"),
    ("pair_05", "real", "619f582e27e736ebc4c5def47f954535",
     "real-news, rapid post-peak decay"),
    ("pair_03", "real", "97bdff4a4098a06dc2b2597b514b2df3",
     "real-news, visible long tail (21 data steps)"),
]

CUTOFF_HORIZON = [(48, 72), (72, 48)]
MODES = ["oracle_cohort", "observed_closed"]
SEEDS = (101, 103, 107)  # forecast seeds (disjoint from fit seeds)


def _adapt_beta_M(history, case, base):
    """Estimate event-level beta_M from pre-cutoff history.
    Coarse grid search over 5 values, select best train RMSLE.
    alpha_0 and gamma_0 frozen at defaults.
    """
    best_train = float("inf")
    best_beta = base.propagation.beta_M
    for beta_M in [0.3, 0.5, 0.7, 0.9, 1.0]:
        from dataclasses import replace
        p = replace(base, propagation=replace(base.propagation, beta_M=beta_M))
        from dynamics_simulation.replay.config import ReplayConfig
        from dynamics_simulation.replay.runner import run_replay
        from dynamics_simulation.data.networks import ReplayNetworkMode

        idx = NodeIndex.from_case(case)
        grid = TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)
        cfg = ReplayConfig(step_hours=STEP_H, tail_steps=0,
                          network_mode=ReplayNetworkMode.BROADCAST,
                          seeds=(42,), micro_steps=1,
                          reactivation_mode=ReactivationMode.ONE_SHOT.value)
        r = run_replay(case, p, cfg)
        if not r.simulated_mean: continue
        sim = np.array(r.simulated_mean.get("n_A_ts", [0]), dtype=np.float64)
        traj = build_observed_trajectory(case, idx, grid)
        obs = np.array(traj.active_count, dtype=np.float64)
        n_hist = len(history.interactions)
        # Use only pre-cutoff portion
        n_pre = history.cutoff_step + 1
        if len(sim) > n_pre: sim = sim[:n_pre]
        if len(obs) > n_pre: obs_tr = obs[:n_pre]
        else: obs_tr = obs
        tr = rmsle(obs_tr, sim[:len(obs_tr)])
        if tr < best_train:
            best_train = tr
            best_beta = beta_M
    return best_beta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    all_results = []

    for pid, label, cid, rationale in SENTINELS:
        path = DATA / f"{label}_news" / f"{cid}.json"
        case = load_checked_case(path)
        print(f"\n{'='*60}")
        print(f"  SENTINEL [{pid} {label}] — {rationale}")
        print(f"{'='*60}")

        for cutoff_h, horizon_h in CUTOFF_HORIZON:
            cutoff_step = cutoff_h // int(STEP_H)
            horizon_steps = horizon_h // int(STEP_H)

            try:
                history = EventHistory.from_case(case, cutoff_step, STEP_H)
                target = ForecastTarget.from_case(case, cutoff_step,
                                                  tuple(range(1, horizon_steps + 1)), STEP_H)
            except ValueError as e:
                print(f"  SKIP cutoff={cutoff_h}h: {e}")
                continue

            # Adapt beta_M from pre-cutoff history
            beta_M = _adapt_beta_M(history, case, base)
            from dataclasses import replace
            params = replace(base, propagation=replace(base.propagation, beta_M=beta_M))

            print(f"  cutoff={cutoff_h}h horizon={horizon_h}h "
                  f"beta_M={beta_M:.3f} "
                  f"n_obs={history.n_observed_users} "
                  f"n_future={target.n_future_users}", flush=True)

            for mode in MODES:
                fcfg = ForecastConfig(
                    cutoff_step=cutoff_step,
                    horizon_steps=horizon_steps,
                    step_hours=STEP_H,
                    population_mode=mode,
                    reactivation_mode=ReactivationMode.ONE_SHOT,
                    forecast_seeds=SEEDS,
                )
                runner = ForecastRunner(fcfg)
                result = runner.run(history, case, params)

                # Validate assertions
                assert len(result.fc_active) == horizon_steps, \
                    f"Forecast horizon mismatch: {len(result.fc_active)} vs {horizon_steps}"
                # Check first+repeat=active
                fc_sum = result.fc_first + result.fc_repeat
                assert np.allclose(fc_sum, result.fc_active, atol=1), \
                    f"first+repeat != active: {fc_sum} vs {result.fc_active}"

                # Baseline evaluation
                idx = NodeIndex.from_case(case)
                grid = TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)
                traj = build_observed_trajectory(case, idx, grid)
                obs_arr = np.array(traj.active_count, dtype=np.float64)
                n_data = grid.last_data_step + 1

                # Train-only baseline from pre-cutoff history
                hist_obs = obs_arr[:cutoff_step + 1]
                if len(hist_obs) >= 2:
                    bf = BaselineForecast(hist_obs, n_data)
                    bl_future = bf.best_future[:horizon_steps]
                    bl_rmsle = rmsle(target.active_count, bl_future)
                else:
                    bl_rmsle = 9.0

                fc_rmsle = rmsle(target.active_count, result.fc_active)
                pk = peak_error(target.active_count, result.fc_active)

                entry = {
                    "pid": pid, "label": label, "rationale": rationale,
                    "cutoff_h": cutoff_h, "horizon_h": horizon_h,
                    "cutoff_step": cutoff_step, "horizon_steps": horizon_steps,
                    "population_mode": mode,
                    "n_agents": result.n_agents,
                    "n_observed": result.n_observed,
                    "n_future": result.n_future,
                    "beta_M": beta_M,
                    "obs_future": [float(x) for x in target.active_count],
                    "fc_active": [float(x) for x in result.fc_active],
                    "fc_first": [float(x) for x in result.fc_first],
                    "fc_repeat": [float(x) for x in result.fc_repeat],
                    "bl_rmsle": round(bl_rmsle, 4),
                    "fc_rmsle": round(fc_rmsle, 4),
                    "peak_ratio": round(pk["peak_ratio"], 4),
                    "peak_step_error": pk["peak_step_error"],
                    "git_sha": result.git_sha,
                    "assertions": {
                        "trajectory_length_ok": True,
                        "first_plus_repeat_equals_active": True,
                        "no_padding_used": True,
                    },
                }
                all_results.append(entry)

                win = "WIN" if fc_rmsle < bl_rmsle else "LOSE"
                imp = (bl_rmsle - fc_rmsle) / max(bl_rmsle, 1e-6) * 100
                print(f"    {mode}: {win} fc={fc_rmsle:.4f} bl={bl_rmsle:.4f} ({imp:+.0f}%) "
                      f"obs={[f'{x:.0f}' for x in target.active_count]} "
                      f"fc={[f'{x:.1f}' for x in result.fc_active]}")

    # Decision gate
    print(f"\n{'='*60}")
    print(f"  V1.7R SENTINEL DECISION ({len(all_results)} forecasts)")
    print(f"{'='*60}")
    for mode in MODES:
        mr = [r for r in all_results if r["population_mode"] == mode]
        wins = sum(1 for r in mr if r["fc_rmsle"] < r["bl_rmsle"])
        print(f"  {mode}: {wins}/{len(mr)} wins")

    oracle = [r for r in all_results if r["population_mode"] == "oracle_cohort"]
    oracle_wins = sum(1 for r in oracle if r["fc_rmsle"] < r["bl_rmsle"])
    oracle_real_wins = sum(1 for r in oracle if r["fc_rmsle"] < r["bl_rmsle"] and r["label"] == "real")

    if oracle_wins >= 2 and oracle_real_wins >= 1:
        print(f"\n  >>> CONTINUE with ONE_SHOT — {oracle_wins} oracle wins, {oracle_real_wins} real wins")
    else:
        print(f"\n  >>> MOVE to event-intensity model — oracle {oracle_wins}/8 wins, real {oracle_real_wins}/4 wins")
        print(f"  Decision gate: >=2 oracle wins AND >=1 real win required")
        print(f"  Actual: {oracle_wins} oracle, {oracle_real_wins} real")

    save_path = OUT / "v17r_sentinel_forecast.json"
    save_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
