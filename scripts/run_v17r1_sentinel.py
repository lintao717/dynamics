"""[V1.7R.1] Corrected Sentinel Forecast — 4 Oracle cases only.

Fixes vs V1.7R:
  - cutoff_step = (cutoff_h // step_hours) - 1 (48h=step1 means seen 2 windows)
  - Traces saved for full transition diagnostics
  - Behavioral emission includes continuing A-state agents
  - Zero baseline reported separately
"""

import json, sys
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
from dynamics_simulation.evaluation import rmsle, peak_error
from dynamics_simulation.baseline import BaselineForecast
from dataclasses import replace

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "forecast"
DATA = Path("data/raw/CHECKED/dataset")
STEP_H = 24.0

SENTINELS = [
    ("pair_06", "fake", "25d9ed3994c2d5f030b864867facab47", "fake-good-V152"),
    ("pair_09", "fake", "dfb9f2af5bb9b16ba717b91d6be5fa2f", "fake-fail-V152"),
    ("pair_05", "real", "619f582e27e736ebc4c5def47f954535", "real-rapid-decay"),
    ("pair_03", "real", "97bdff4a4098a06dc2b2597b514b2df3", "real-long-tail"),
]

MODES = ["oracle_cohort"]  # Oracle only for V1.7R.1
SEEDS = (101, 103, 107)


def _adapt_beta_M(history, case, base):
    """Grid search beta_M over 5 values, select best train RMSLE on pre-cutoff."""
    best, best_beta = float("inf"), 0.5
    for beta_M in [0.3, 0.5, 0.7, 0.9, 1.0]:
        from dynamics_simulation.replay.config import ReplayConfig
        from dynamics_simulation.replay.runner import run_replay
        from dynamics_simulation.data.networks import ReplayNetworkMode
        p = replace(base, propagation=replace(base.propagation, beta_M=beta_M))
        cfg = ReplayConfig(step_hours=STEP_H, tail_steps=0,
                          network_mode=ReplayNetworkMode.BROADCAST,
                          seeds=(42,), micro_steps=1,
                          reactivation_mode=ReactivationMode.ONE_SHOT.value)
        r = run_replay(case, p, cfg)
        if not r.simulated_mean: continue
        sim = np.array(r.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
        traj = build_observed_trajectory(case, NodeIndex.from_case(case),
            TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0))
        obs = np.array(traj.active_count, dtype=np.float64)
        n_pre = history.cutoff_step + 1
        s = sim[:n_pre] if len(sim) > n_pre else np.pad(sim, (0, n_pre-len(sim)), mode="edge")
        o = obs[:n_pre]
        tr = rmsle(o, s[:len(o)])
        if tr < best: best = tr; best_beta = beta_M
    return best_beta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    all_results = []

    for pid, label, cid, note in SENTINELS:
        path = DATA / f"{label}_news" / f"{cid}.json"
        case = load_checked_case(path)
        print(f"\n{'='*60}\n  [{pid} {label}] {note}\n{'='*60}")

        for cutoff_h, horizon_h in [(48, 72), (72, 48)]:
            # Fix: cutoff_step = windows seen minus 1 (step indices)
            # If cutoff_h=48, we've seen through step 1 (0-indexed)
            # step 0 = [0,24h), step 1 = [24h,48h), so cutoff_step=1
            n_steps_seen = cutoff_h // int(STEP_H)  # how many 24h windows seen
            cutoff_step = n_steps_seen - 1  # step index of last seen window
            horizon_steps = horizon_h // int(STEP_H)

            try:
                history = EventHistory.from_case(case, cutoff_step, STEP_H)
                target = ForecastTarget.from_case(case, cutoff_step,
                                                  tuple(range(1, horizon_steps+1)), STEP_H)
            except ValueError as e:
                print(f"  SKIP c={cutoff_h}h: {e}")
                continue

            beta_M = _adapt_beta_M(history, case, base)
            params = replace(base, propagation=replace(base.propagation, beta_M=beta_M))

            print(f"  cutoff={cutoff_h}h(h_step={cutoff_step}) hz={horizon_h}h "
                  f"beta_M={beta_M:.3f} obs={history.n_observed_users} "
                  f"fut_users={target.n_future_users}", flush=True)

            for mode in MODES:
                fcfg = ForecastConfig(cutoff_step=cutoff_step,
                    horizon_steps=horizon_steps, step_hours=STEP_H,
                    population_mode=mode,
                    reactivation_mode=ReactivationMode.ONE_SHOT,
                    forecast_seeds=SEEDS)
                runner = ForecastRunner(fcfg)
                result = runner.run(history, case, params)

                # Baselines
                traj = build_observed_trajectory(case,
                    NodeIndex.from_case(case),
                    TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0))
                obs_arr = np.array(traj.active_count, dtype=np.float64)
                hist_obs = obs_arr[:cutoff_step+1]
                n_data = len(obs_arr)

                bf = BaselineForecast(hist_obs, n_data) if len(hist_obs) >= 2 else None
                bl_future = bf.best_future[:horizon_steps] if bf else np.zeros(horizon_steps)
                bl_rmsle = rmsle(target.active_count, bl_future) if bf else 9.0
                zero_rmsle = rmsle(target.active_count, np.zeros(horizon_steps))
                fc_rmsle = rmsle(target.active_count, result.fc_active)
                pk = peak_error(target.active_count, result.fc_active)

                # Check if model == zero baseline
                is_zero = np.allclose(result.fc_active, 0, atol=0.01)

                entry = {
                    "pid": pid, "label": label, "note": note,
                    "cutoff_h": cutoff_h, "horizon_h": horizon_h,
                    "cutoff_step": cutoff_step, "horizon_steps": horizon_steps,
                    "population_mode": mode,
                    "beta_M": beta_M,
                    "obs_future": [float(x) for x in target.active_count],
                    "fc_active": [float(x) for x in result.fc_active],
                    "fc_first": [float(x) for x in result.fc_first],
                    "fc_repeat": [float(x) for x in result.fc_repeat],
                    "zero_rmsle": round(zero_rmsle, 4),
                    "bl_rmsle": round(bl_rmsle, 4),
                    "fc_rmsle": round(fc_rmsle, 4),
                    "is_zero": is_zero,
                    "peak_ratio": round(pk["peak_ratio"], 4),
                    "traces": result.traces,
                    "git_sha": result.git_sha,
                }
                all_results.append(entry)

                win = "WIN" if fc_rmsle < bl_rmsle else "LOSE"
                imp = (bl_rmsle - fc_rmsle)/max(bl_rmsle,1e-6)*100
                zero_tag = " [ZERO]" if is_zero else ""
                print(f"    {mode}: {win} fc={fc_rmsle:.4f} bl={bl_rmsle:.4f} "
                      f"zero={zero_rmsle:.4f} ({imp:+.0f}%){zero_tag} "
                      f"obs={[f'{x:.0f}' for x in target.active_count]} "
                      f"fc={[f'{x:.1f}' for x in result.fc_active]} "
                      f"U2E={result.traces['U_to_E'][:6]} "
                      f"E2A={result.traces['E_to_A'][:6]}")

    # Decision
    oracle = [r for r in all_results if r["population_mode"]=="oracle_cohort"]
    non_zero = [r for r in oracle if not r["is_zero"]]
    wins = [r for r in oracle if r["fc_rmsle"] < r["bl_rmsle"] and not r["is_zero"]]
    zero_all = [r for r in oracle if r["is_zero"]]

    print(f"\n{'='*60}")
    print(f"  V1.7R.1 DECISION ({len(oracle)} oracle forecasts)")
    print(f"  Non-zero forecasts: {len(non_zero)}/{len(oracle)}")
    print(f"  Genuine wins (non-zero AND beats bl): {len(wins)}")
    print(f"  Zero forecasts: {len(zero_all)}/{len(oracle)} ({len([r for r in zero_all if r['fc_rmsle']<r['bl_rmsle']])} degenerate wins)")

    for r in oracle:
        print(f"  {r['pid']} {r['label']} c={r['cutoff_h']}: "
              f"U2E={r['traces']['U_to_E'][:4]} E2A={r['traces']['E_to_A'][:4]} "
              f"fc={[f'{x:.1f}' for x in r['fc_active']]} zero={r['is_zero']}")

    save_path = OUT / "v17r1_sentinel_forecast.json"
    save_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
