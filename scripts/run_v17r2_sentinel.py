"""[V1.7R.2] Corrected Sentinel — cutoff semantics + emission + calibration.

Fixes:
  - cutoff_step = cutoff_h / step_hours (not -1)
  - Win: must beat ALL baselines including zero
  - User-level behavioral emission via step_observer
  - Joint grid-search over alpha_0, gamma_0 (not just beta_M)
  - Git SHA recorded from committed code
"""

import json, sys, subprocess
from pathlib import Path
import numpy as np
from dataclasses import replace

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

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "forecast"
DATA = Path("data/raw/CHECKED/dataset")
STEP_H = 24.0

# Source git SHA (committed before running)
SRC_SHA = subprocess.check_output(
    ["git","rev-parse","HEAD"], text=True, stderr=subprocess.DEVNULL).strip()

SENTINELS = [
    ("pair_06","fake","25d9ed3994c2d5f030b864867facab47","fake-good-V152"),
    ("pair_09","fake","dfb9f2af5bb9b16ba717b91d6be5fa2f","fake-fail-V152"),
    ("pair_05","real","619f582e27e736ebc4c5def47f954535","real-rapid-decay"),
    ("pair_03","real","97bdff4a4098a06dc2b2597b514b2df3","real-long-tail"),
]
SEEDS = (101, 103, 107)


def _grid_search_params(history, case, base):
    """Joint grid search over beta_M, alpha_0, gamma_0.
    Selects best combination by train RMSLE on pre-cutoff data only.
    """
    from dynamics_simulation.replay.config import ReplayConfig
    from dynamics_simulation.replay.runner import run_replay
    from dynamics_simulation.data.networks import ReplayNetworkMode

    idx = NodeIndex.from_case(case)
    grid_h = TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)
    traj = build_observed_trajectory(case, idx, grid_h)
    obs = np.array(traj.active_count, dtype=np.float64)
    n_pre = history.cutoff_step + 1

    best, best_p = float("inf"), None
    for beta_M in [0.3, 0.5, 0.7, 0.9]:
        for alpha_0 in [-2.0, 0.0, 1.0]:
            for gamma_0 in [0.0, 2.0, 4.0, 5.5]:
                p = replace(base,
                    propagation=replace(base.propagation, beta_M=beta_M),
                    activation=replace(base.activation, alpha_0=alpha_0),
                    decay=replace(base.decay, gamma_0=gamma_0))
                cfg = ReplayConfig(step_hours=STEP_H, tail_steps=0,
                    network_mode=ReplayNetworkMode.BROADCAST, seeds=(42,),
                    micro_steps=1,
                    reactivation_mode=ReactivationMode.ONE_SHOT.value)
                r = run_replay(case, p, cfg)
                if not r.simulated_mean: continue
                s = np.array(r.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
                if len(s) > n_pre: s = s[:n_pre]
                tr = rmsle(obs[:n_pre], s[:len(obs[:n_pre])])
                if tr < best: best = tr; best_p = p
    return best_p if best_p else base


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    all_results = []

    for pid, label, cid, note in SENTINELS:
        path = DATA / f"{label}_news" / f"{cid}.json"
        case = load_checked_case(path)
        print(f"\n{'='*60}\n  [{pid} {label}] {note}\n{'='*60}")

        for cutoff_h, horizon_h in [(48, 72), (72, 48)]:
            # Fix: cutoff_step = windows seen (matching TimeGrid ceil semantics)
            cutoff_step = cutoff_h // int(STEP_H)
            horizon_steps = horizon_h // int(STEP_H)

            try:
                history = EventHistory.from_case(case, cutoff_step, STEP_H)
                target = ForecastTarget.from_case(case, cutoff_step,
                    tuple(range(1, horizon_steps+1)), STEP_H)
            except ValueError as e:
                print(f"  SKIP c={cutoff_h}h(step={cutoff_step}): {e}")
                continue

            params = _grid_search_params(history, case, base)
            bM = params.propagation.beta_M
            a0 = params.activation.alpha_0
            g0 = params.decay.gamma_0
            print(f"  cutoff={cutoff_h}h(step={cutoff_step}) hz={horizon_h}h "
                  f"bM={bM:.3f} a0={a0:.2f} g0={g0:.2f} "
                  f"obs={history.n_observed_users} fut={target.n_future_users}",
                  flush=True)

            fcfg = ForecastConfig(cutoff_step=cutoff_step,
                horizon_steps=horizon_steps, step_hours=STEP_H,
                population_mode="oracle_cohort",
                reactivation_mode=ReactivationMode.ONE_SHOT,
                forecast_seeds=SEEDS)
            runner = ForecastRunner(fcfg)
            result = runner.run(history, case, params)

            # Baselines (all four, including zero)
            obs_arr = np.array(build_observed_trajectory(case,
                NodeIndex.from_case(case),
                TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)
            ).active_count, dtype=np.float64)
            hist_obs = obs_arr[:cutoff_step+1]
            n_data = len(obs_arr)

            bf = BaselineForecast(hist_obs, n_data) if len(hist_obs) >= 2 else None
            zeros = np.zeros(horizon_steps, dtype=np.float64)
            bl_zero = rmsle(target.active_count, zeros)
            bl_best = bl_zero
            bl_name = "zero"
            if bf:
                for bname, bpred in [("persist", bf.persistence_last),
                                      ("exp", bf.exp_decay_future[:horizon_steps]),
                                      ("pulse", bf.pulse_decay_future[:horizon_steps])]:
                    br = rmsle(target.active_count, bpred)
                    if br < bl_best:
                        bl_best = br
                        bl_name = bname

            fc_rmsle = rmsle(target.active_count, result.fc_active)
            pk = peak_error(target.active_count, result.fc_active)
            # Win: must beat ALL baselines (including zero)
            win = fc_rmsle < bl_zero and (bf is None or fc_rmsle < min(
                rmsle(target.active_count, bf.persistence_last),
                rmsle(target.active_count, bf.exp_decay_future[:horizon_steps]),
                rmsle(target.active_count, bf.pulse_decay_future[:horizon_steps])))

            entry = {
                "pid":pid, "label":label, "note":note,
                "cutoff_h":cutoff_h, "horizon_h":horizon_h,
                "cutoff_step":cutoff_step, "horizon_steps":horizon_steps,
                "beta_M":bM, "alpha_0":a0, "gamma_0":g0,
                "obs_future":[float(x) for x in target.active_count],
                "fc_active":[float(x) for x in result.fc_active],
                "fc_first":[float(x) for x in result.fc_first],
                "fc_repeat":[float(x) for x in result.fc_repeat],
                "zero_rmsle":round(bl_zero,4),
                "best_bl_rmsle":round(bl_best,4),
                "best_bl_name":bl_name,
                "fc_rmsle":round(fc_rmsle,4),
                "win":win,
                "peak_ratio":round(pk["peak_ratio"],4),
                "traces":result.traces,
                "source_git_sha":SRC_SHA,
            }
            all_results.append(entry)

            imp = (bl_best - fc_rmsle)/max(bl_best,1e-6)*100
            print(f"    oracle: {'WIN' if win else 'LOSE'} fc={fc_rmsle:.4f} "
                  f"best_bl={bl_best:.4f}({bl_name}) zero={bl_zero:.4f} ({imp:+.0f}%) "
                  f"obs={[f'{x:.0f}' for x in target.active_count]} "
                  f"fc={[f'{x:.1f}' for x in result.fc_active]} "
                  f"E2A={result.traces['E_to_A'][:4]} "
                  f"pr={pk['peak_ratio']:.2f}")

    # Decision
    oracle = [r for r in all_results]
    wins = [r for r in oracle if r["win"]]
    real_wins = [r for r in oracle if r["win"] and r["label"]=="real"]
    peak_ok = [r for r in oracle if 0.5 <= r["peak_ratio"] <= 2.0]
    bM_bound = [r for r in oracle if r["beta_M"] >= 0.95]

    print(f"\n{'='*60}\n  V1.7R.2 DECISION ({len(oracle)} forecasts)\n{'='*60}")
    print(f"  True wins (beats ALL baselines): {len(wins)}/{len(oracle)}")
    print(f"  Real wins: {len(real_wins)}/{len(oracle)//2}")
    print(f"  Peak ratio OK (0.5-2): {len(peak_ok)}/{len(oracle)}")
    print(f"  beta_M near boundary: {len(bM_bound)}/{len(oracle)}")

    for r in oracle:
        s = "WIN" if r["win"] else "LOSE"
        print(f"  {r['pid']} {r['label']} c={r['cutoff_h']}: {s} "
              f"fc={r['fc_rmsle']:.4f} bl={r['best_bl_rmsle']:.4f}({r['best_bl_name']}) "
              f"bM={r['beta_M']:.2f} a0={r['alpha_0']:.1f} g0={r['gamma_0']:.1f} "
              f"pr={r['peak_ratio']:.2f} E2A={r['traces']['E_to_A'][:4]}")

    if len(wins) >= 5 and len(real_wins) >= 2:
        print("\n  >>> PASS — proceed to 10-pair cross-validation")
    else:
        print(f"\n  >>> NOT YET — need >=5 wins AND >=2 real wins, got {len(wins)}/{len(real_wins)}")

    save_path = OUT / "v17r2_sentinel_forecast.json"
    save_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
