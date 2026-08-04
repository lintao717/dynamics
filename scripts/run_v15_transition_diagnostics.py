"""[V1.5 Phase 2] Transition Flow Diagnostics.

Diagnose where the smooth tail comes from:
  - D0->A (delayed first activation)?
  - D1->A (true reactivation)?
  - Continued U->E (new exposures)?
  - A-state duration?

Runs 4 cases with default params, 24h steps, micro_steps=1.
Outputs per-step transition matrices and summary statistics.
"""

import json, sys
from pathlib import Path
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

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "recovery"
DATA = Path("data/raw/CHECKED/dataset")

CASES = [
    ("pair_05", "fake", "afc5ba429bc79086f05d6798e4ed978a"),
    ("pair_05", "real", "619f582e27e736ebc4c5def47f954535"),
    ("pair_07", "fake", "b4a497151507853058c8c50b6c6670f5"),
    ("pair_07", "real", "6f2b7e632c64ba5835164abe2f1ab28e"),
]


def _rmsle(a, b):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(b,0))-np.log1p(np.maximum(a,0)))**2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_diag = []

    for pid, label, cid in CASES:
        print(f"\n=== DIAGNOSTICS [{pid} {label}] ===")
        path = DATA / f"{label}_news" / f"{cid}.json"
        case = load_checked_case(path)
        idx = NodeIndex.from_case(case)
        grid = TimeGrid.from_case(case, step_hours=24.0, tail_steps=0)
        traj = build_observed_trajectory(case, idx, grid)
        obs_arr = np.array(traj.active_count, dtype=np.float64)
        n_data = grid.last_data_step + 1

        print(f"  N={len(case.user_ids)} steps={n_data} obs_peak={obs_arr.max():.0f}")

        # Run replay with detailed transition logging
        config = ReplayConfig(
            step_hours=24.0, tail_steps=0,
            network_mode=ReplayNetworkMode.BROADCAST,
            seeds=(42,), micro_steps=1,
        )
        result = run_replay(case, default_params(), config)

        if not result.simulated_mean:
            print("  NO SIMULATED MEAN — skipping")
            continue

        # Extract per-step transition data from per_seed runs
        sim_na = np.array(result.simulated_mean.get("n_A_ts", [0]), dtype=np.float64)
        sim_ne = np.array(result.simulated_mean.get("n_E_ts", [0]), dtype=np.float64)
        sim_nd = np.array(result.simulated_mean.get("n_D_ts", [0]), dtype=np.float64)

        # Compute derived diagnostics
        # A-state duration proxy: N_A(t) / (E->A inflow at t)
        # But we don't have per-step transitions from the high-level API.
        # Instead, reconstruct from state changes.
        delta_A = np.diff(sim_na)
        A_gains = np.maximum(delta_A, 0)  # positive changes = new A entries
        A_losses = np.maximum(-delta_A, 0)  # negative changes = A exits

        cum_A_gains = A_gains.sum()
        cum_A_days = sim_na.sum()
        avg_A_duration = cum_A_days / max(cum_A_gains, 1)  # avg days in A

        # Share of activations by day
        day1_share = A_gains[0] / max(cum_A_gains, 1) if len(A_gains) > 0 else 0
        day2_share = A_gains[1] / max(cum_A_gains, 1) if len(A_gains) > 1 else 0
        late_share = A_gains[2:].sum() / max(cum_A_gains, 1) if len(A_gains) > 2 else 0

        # U inflow proxy: n_U changes
        nu_ts = np.array(result.simulated_mean.get("n_U_ts", [0]), dtype=np.float64)
        delta_U = -np.diff(nu_ts)  # U decreases = U->E
        U_outflow = np.maximum(delta_U, 0)

        diag = {
            "pid": pid, "label": label,
            "n_agents": len(case.user_ids),
            "n_steps": n_data,
            "obs_peak": float(obs_arr.max()),
            "obs_peak_step": int(obs_arr.argmax()),
            "sim_peak": float(sim_na.max()),
            "sim_peak_step": int(sim_na.argmax()),
            "avg_A_duration_days": float(avg_A_duration),
            "day1_activation_share": float(day1_share),
            "day2_activation_share": float(day2_share),
            "late_activation_share": float(late_share),
            "total_A_gains": float(cum_A_gains),
            "total_A_losses": float(A_losses.sum()),
            "total_U_outflow": float(U_outflow.sum()),
            "U_outflow_by_day": [float(x) for x in U_outflow[:min(10, len(U_outflow))]],
            "A_gains_by_day": [float(x) for x in A_gains[:min(10, len(A_gains))]],
            "A_losses_by_day": [float(x) for x in A_losses[:min(10, len(A_losses))]],
            "n_A_by_day": [float(x) for x in sim_na[:min(10, len(sim_na))]],
        }
        all_diag.append(diag)

        print(f"  avg_A_duration={avg_A_duration:.1f}d day1_act={day1_share:.2%} "
              f"day2_act={day2_share:.2%} late_act={late_share:.2%}")
        print(f"  U_outflow/day: {[f'{x:.0f}' for x in U_outflow[:5]]}")
        print(f"  A_gains/day:   {[f'{x:.0f}' for x in A_gains[:5]]}")
        print(f"  A_losses/day:  {[f'{x:.0f}' for x in A_losses[:5]]}")

    # Summary
    print(f"\n{'='*50}")
    print(f"  TRANSITION DIAGNOSTICS SUMMARY")
    print(f"{'='*50}")
    for d in all_diag:
        print(f"  {d['pid']} {d['label']}: "
              f"avg_A={d['avg_A_duration_days']:.1f}d "
              f"day1={d['day1_activation_share']:.2%} "
              f"day2={d['day2_activation_share']:.2%} "
              f"late={d['late_activation_share']:.2%} "
              f"total_gains={d['total_A_gains']:.0f}")

    # Key question: does late activation come from continued U outflow or A recycling?
    for d in all_diag:
        late_u = sum(d["U_outflow_by_day"][2:]) if len(d["U_outflow_by_day"]) > 2 else 0
        late_a = sum(d["A_gains_by_day"][2:]) if len(d["A_gains_by_day"]) > 2 else 0
        ratio = late_a / max(late_u, 1)
        print(f"  {d['pid']} {d['label']}: "
              f"late_A_gains={late_a:.0f} late_U_outflow={late_u:.0f} "
              f"ratio={ratio:.2f} (if >>1, gains come from D-reactivation not new U->E)")

    with open(OUT / "v15_transition_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(all_diag, f, indent=2)
    print(f"\nSaved: {OUT / 'v15_transition_diagnostics.json'}")


if __name__ == "__main__":
    main()
