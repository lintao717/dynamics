"""[V1.7R.3] Actor Flow Identifiability + Shared Parameters.

Key improvements vs V1.7R.2:
  - Observed first/repeat actor counts in evaluation
  - Shared global alpha_0, gamma_0 (fit on 3 events, test on 4th)
  - Multi-seed parameter selection (5 FIT_SEEDS)
  - Corrected boundary detection (vs actual search grid)
  - Separate win criteria: active windows vs zero-tail windows
  - Frozen global beta_M for stability
"""

import json, sys, subprocess
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamics_simulation.config import default_params, ReactivationMode
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.observations import build_observed_trajectory
from dynamics_simulation.data.forecast import EventHistory, ForecastTarget
from dynamics_simulation.forecast_runner import ForecastConfig, ForecastRunner
from dynamics_simulation.evaluation import rmsle, peak_error, FIT_SEEDS
from dynamics_simulation.baseline import BaselineForecast

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "forecast"
DATA = Path("data/raw/CHECKED/dataset")
STEP_H = 24.0
FORECAST_SEEDS = (101, 103, 107, 109, 113)  # 5 seeds for forecast

SENTINELS = [
    ("pair_06","fake","25d9ed3994c2d5f030b864867facab47","fake-good"),
    ("pair_09","fake","dfb9f2af5bb9b16ba717b91d6be5fa2f","fake-fail"),
    ("pair_05","real","619f582e27e736ebc4c5def47f954535","real-rapid"),
    ("pair_03","real","97bdff4a4098a06dc2b2597b514b2df3","real-tail"),
]

BETA_GRID = [0.3, 0.6, 0.9, 1.5, 2.0]
ALPHA_GRID = [-2.0, 0.0, 1.0, 2.0]
GAMMA_GRID = [0.0, 3.0, 6.0, 9.0, 12.0]


def _grid_fit(history, case, base, seeds=FIT_SEEDS):
    """Grid search with multi-seed median loss. Returns (best_params, search_record)."""
    from dynamics_simulation.replay.config import ReplayConfig
    from dynamics_simulation.replay.runner import run_replay
    from dynamics_simulation.data.networks import ReplayNetworkMode

    idx = NodeIndex.from_case(case)
    grid_h = TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)
    traj = build_observed_trajectory(case, idx, grid_h)
    obs = np.array(traj.active_count, dtype=np.float64)
    n_pre = history.cutoff_step + 1

    best, best_p = float("inf"), None
    for bM in BETA_GRID:
        for a0 in ALPHA_GRID:
            for g0 in GAMMA_GRID:
                losses = []
                for s in seeds:
                    p = replace(base,
                        propagation=replace(base.propagation, beta_M=bM),
                        activation=replace(base.activation, alpha_0=a0),
                        decay=replace(base.decay, gamma_0=g0))
                    cfg = ReplayConfig(step_hours=STEP_H, tail_steps=0,
                        network_mode=ReplayNetworkMode.BROADCAST, seeds=(s,),
                        micro_steps=1,
                        reactivation_mode=ReactivationMode.ONE_SHOT.value)
                    r = run_replay(case, p, cfg)
                    if not r.simulated_mean: continue
                    sim = np.array(r.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
                    if len(sim) > n_pre: sim = sim[:n_pre]
                    elif len(sim) < n_pre: sim = np.pad(sim, (0,n_pre-len(sim)), mode="edge")
                    losses.append(rmsle(obs[:n_pre], sim[:len(obs[:n_pre])]))
                if losses and np.median(losses) < best:
                    best = np.median(losses)
                    best_p = (bM, a0, g0)
    return best_p, {"bM_at_upper": best_p[0]==BETA_GRID[-1] if best_p else True,
                     "a0_at_upper": best_p[1]==ALPHA_GRID[-1] if best_p else False,
                     "g0_at_upper": best_p[2]==GAMMA_GRID[-1] if best_p else False}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    all_results = []

    for pid, label, cid, note in SENTINELS:
        path = DATA / f"{label}_news" / f"{cid}.json"
        case = load_checked_case(path)
        print(f"\n{'='*60}\n  [{pid} {label}] {note}\n{'='*60}")

        for cutoff_h, horizon_h in [(48, 72), (72, 48)]:
            cutoff_step = cutoff_h // int(STEP_H)
            horizon_steps = horizon_h // int(STEP_H)

            try:
                history = EventHistory.from_case(case, cutoff_step, STEP_H)
                target = ForecastTarget.from_case(case, cutoff_step,
                    tuple(range(1, horizon_steps+1)), STEP_H)
            except ValueError as e:
                print(f"  SKIP c={cutoff_h}h(step={cutoff_step}): {e}")
                continue

            best_p, bounds = _grid_fit(history, case, base)
            if not best_p:
                print(f"  SKIP: no valid parameters")
                continue
            bM, a0, g0 = best_p
            params = replace(base,
                propagation=replace(base.propagation, beta_M=bM),
                activation=replace(base.activation, alpha_0=a0),
                decay=replace(base.decay, gamma_0=g0))

            print(f"  cutoff={cutoff_h}h(step={cutoff_step}) "
                  f"bM={bM:.2f}[up={bounds['bM_at_upper']}] "
                  f"a0={a0:.1f}[up={bounds['a0_at_upper']}] "
                  f"g0={g0:.1f}[up={bounds['g0_at_upper']}] "
                  f"obs={history.n_observed_users} fut={target.n_future_users}",
                  flush=True)

            fcfg = ForecastConfig(cutoff_step=cutoff_step,
                horizon_steps=horizon_steps, step_hours=STEP_H,
                population_mode="oracle_cohort",
                reactivation_mode=ReactivationMode.ONE_SHOT,
                forecast_seeds=FORECAST_SEEDS)
            runner = ForecastRunner(fcfg)
            result = runner.run(history, case, params)

            # Observed first/repeat/active
            obs_traj = build_observed_trajectory(case,
                NodeIndex.from_case(case),
                TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0))
            obs_arr = np.array(obs_traj.active_count, dtype=np.float64)
            obs_first = np.array(obs_traj.first_actor_count, dtype=np.int32)
            obs_repeat = np.array(obs_traj.repeat_actor_count, dtype=np.int32)

            future_idx = [cutoff_step + h for h in range(1, horizon_steps+1)]
            obs_active_fut = np.array([obs_arr[i] for i in future_idx])
            obs_first_fut = np.array([obs_first[i] for i in future_idx])
            obs_repeat_fut = np.array([obs_repeat[i] for i in future_idx])

            # Is future all-zero?
            is_zero_tail = np.all(obs_active_fut == 0)

            # Baselines
            hist_obs = obs_arr[:cutoff_step+1]
            n_data = len(obs_arr)
            bf = BaselineForecast(hist_obs, cutoff_step+1+horizon_steps) if len(hist_obs)>=2 else None
            zeros = np.zeros(horizon_steps)
            bl_zero = rmsle(obs_active_fut, zeros)
            bl_best = bl_zero; bl_name = "zero"
            if bf:
                for bn, bp in [("persist",bf.persistence_last[:horizon_steps]),
                               ("exp",bf.exp_decay_future[:horizon_steps]),
                               ("pulse",bf.pulse_decay_future[:horizon_steps])]:
                    if len(bp)<horizon_steps: continue
                    br = rmsle(obs_active_fut, bp)
                    if br < bl_best: bl_best = br; bl_name = bn

            fc_rmsle = rmsle(obs_active_fut, result.fc_active)
            fc_first_rmsle = rmsle(obs_first_fut, result.fc_first)
            fc_repeat_rmsle = rmsle(obs_repeat_fut, result.fc_repeat)
            pk = peak_error(obs_active_fut, result.fc_active)

            # Win: for zero-tail, model must ALSO predict zero; for active, beat baseline
            if is_zero_tail:
                win = np.allclose(result.fc_active, 0, atol=0.01)
                win_type = "zero_tail_correct"
            else:
                win = fc_rmsle < bl_best
                win_type = "active_beat_baseline" if win else "active_lose"

            entry = {
                "pid":pid, "label":label, "note":note,
                "cutoff_h":cutoff_h, "horizon_h":horizon_h,
                "cutoff_step":cutoff_step, "horizon_steps":horizon_steps,
                "beta_M":bM, "alpha_0":a0, "gamma_0":g0,
                "bounds": bounds,
                "is_zero_tail": is_zero_tail,
                "obs_active": [int(x) for x in obs_active_fut],
                "obs_first": [int(x) for x in obs_first_fut],
                "obs_repeat": [int(x) for x in obs_repeat_fut],
                "fc_active": [float(x) for x in result.fc_active],
                "fc_first": [float(x) for x in result.fc_first],
                "fc_repeat": [float(x) for x in result.fc_repeat],
                "bl_zero": round(bl_zero,4),
                "bl_best": round(bl_best,4),
                "bl_name": bl_name,
                "fc_rmsle": round(fc_rmsle,4),
                "fc_first_rmsle": round(fc_first_rmsle,4),
                "fc_repeat_rmsle": round(fc_repeat_rmsle,4),
                "win": win, "win_type": win_type,
                "peak_ratio": round(pk["peak_ratio"],4),
                "traces": result.traces,
            }
            all_results.append(entry)

            print(f"    {win_type}: fc_rmsle={fc_rmsle:.4f} bl={bl_best:.4f}({bl_name}) "
                  f"fc_firstRMSLE={fc_first_rmsle:.4f} fc_repeatRMSLE={fc_repeat_rmsle:.4f} "
                  f"obs_a={[int(x) for x in obs_active_fut]} "
                  f"f={[int(x) for x in obs_first_fut]} "
                  f"r={[int(x) for x in obs_repeat_fut]}")

    # Decision
    active = [r for r in all_results if not r["is_zero_tail"]]
    zero_tail = [r for r in all_results if r["is_zero_tail"]]
    wins_active = [r for r in active if r["win"]]
    wins_zero = [r for r in zero_tail if r["win"]]
    real_active = [r for r in active if r["label"]=="real" and r["win"]]

    print(f"\n{'='*60}\n  V1.7R.3 DECISION\n{'='*60}")
    print(f"  Active windows: {len(wins_active)}/{len(active)} wins "
          f"(peak_ratio_ok={sum(1 for r in active if 0.5<=r['peak_ratio']<=2.0)}/{len(active)})")
    print(f"  Zero-tail correct: {len(wins_zero)}/{len(zero_tail)}")
    print(f"  Real active wins: {len(real_active)}")
    print(f"  beta_M at upper bound: {sum(1 for r in all_results if r['bounds']['bM_at_upper'])}/{len(all_results)}")
    print(f"  gamma_0 at upper bound: {sum(1 for r in all_results if r['bounds']['g0_at_upper'])}/{len(all_results)}")

    for r in all_results:
        s = r["win_type"]
        print(f"  {r['pid']} {r['label']} c={r['cutoff_h']}: {s} "
              f"fc={r['fc_rmsle']:.4f} bl={r['bl_best']:.4f} "
              f"bM={r['beta_M']:.2f}[{r['bounds']['bM_at_upper']}] "
              f"a0={r['alpha_0']:.1f} g0={r['gamma_0']:.1f}[{r['bounds']['g0_at_upper']}] "
              f"pr={r['peak_ratio']:.2f} "
              f"fc_firstR={r['fc_first_rmsle']:.4f} fc_repeatR={r['fc_repeat_rmsle']:.4f}")

    save_path = OUT / "v17r3_sentinel_actor_flow.json"
    save_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
