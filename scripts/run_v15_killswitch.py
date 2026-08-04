"""[V1.5 Phase 3] D-state Reactivation Kill-Switch Experiment.

First finds a working beta_M level, then tests:
  S0: all mechanisms (baseline at working beta_M)
  S1: D0->A = 0 (no delayed first activation)
  S2: D1->A = 0 (no true reactivation)
  S3: D0->A = 0 AND D1->A = 0 (one-shot activation only)

Uses enhanced beta_M=1.0 (max) + tuned beta_M to get sufficient activations.

Runs on 4 cases, 5 seeds, 24h steps.
"""

import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamics_simulation.config import ModelParams, default_params
from dynamics_simulation.config import PropagationParams, ViralParams
from dynamics_simulation.config import ActivationParams, DecayParams, ReactivationParams
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.observations import build_observed_trajectory
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.calibration.split import TemporalSplit
from dataclasses import replace

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "recovery"
DATA = Path("data/raw/CHECKED/dataset")

CASES = [
    ("pair_05", "fake", "afc5ba429bc79086f05d6798e4ed978a"),
    ("pair_05", "real", "619f582e27e736ebc4c5def47f954535"),
    ("pair_07", "fake", "b4a497151507853058c8c50b6c6670f5"),
    ("pair_07", "real", "6f2b7e632c64ba5835164abe2f1ab28e"),
]


def _make_params(beta_M=1.0, disable_D0=False, disable_D1=False):
    """Build ModelParams with specified beta_M and reactivation settings."""
    p = default_params()
    # Max out broadcast exposure
    p = replace(p, propagation=replace(p.propagation, beta_M=beta_M))
    # Optionally disable D-state reactivation
    if disable_D0 or disable_D1:
        r = p.reactivation
        r0_0 = -1e9 if disable_D0 else r.r_0_0
        r0_1 = -1e9 if disable_D1 else r.r_0_1
        p = replace(p, reactivation=replace(
            p.reactivation, r_0_0=r0_0, r_0_1=r0_1))
    return p


def _rmsle(a, b):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(b, 0)) - np.log1p(np.maximum(a, 0))) ** 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    configs = {
        "S0": {"beta_M": 1.0, "disable_D0": False, "disable_D1": False, "desc": "Max exposure, all mechs"},
        "S1": {"beta_M": 1.0, "disable_D0": True, "disable_D1": False, "desc": "No D0->A"},
        "S2": {"beta_M": 1.0, "disable_D0": False, "disable_D1": True, "desc": "No D1->A"},
        "S3": {"beta_M": 1.0, "disable_D0": True, "disable_D1": True, "desc": "One-shot only (no reactivation)"},
    }

    all_results = []

    for name, cfg in configs.items():
        print(f"\n=== {name}: {cfg['desc']} ===")
        params = _make_params(cfg["beta_M"], cfg["disable_D0"], cfg["disable_D1"])

        for pid, label, cid in CASES:
            print(f"  [{pid} {label}] ", end="", flush=True)
            path = DATA / f"{label}_news" / f"{cid}.json"
            case = load_checked_case(path)
            idx = NodeIndex.from_case(case)
            grid = TimeGrid.from_case(case, step_hours=24.0, tail_steps=0)
            traj = build_observed_trajectory(case, idx, grid)
            obs_arr = np.array(traj.active_count, dtype=np.float64)
            n_data = grid.last_data_step + 1
            split = TemporalSplit.by_fraction(total_steps=grid.last_data_step, train_fraction=0.7)
            te = split.train_end_step
            obs_train = obs_arr[:te + 1]

            # Baseline
            t_tr = np.arange(len(obs_train), dtype=np.float64)
            A = np.column_stack([np.ones_like(t_tr), t_tr])
            c, _, _, _ = np.linalg.lstsq(A, np.log(np.maximum(obs_train, 1e-6)), rcond=None)
            bl = np.exp(c[0]) * np.exp(-max(-c[1], 1e-6) * np.arange(n_data, dtype=np.float64))
            bl_train = _rmsle(obs_train, bl[:te + 1])
            bl_val = _rmsle(obs_arr[te + 1:], bl[te + 1:])

            # Run with 5 seeds
            rcfg = ReplayConfig(step_hours=24.0, tail_steps=0,
                               network_mode=ReplayNetworkMode.BROADCAST,
                               seeds=(11, 23, 37, 53, 71), micro_steps=1)
            result = run_replay(case, params, rcfg)
            if not result.simulated_mean:
                continue

            sm = result.simulated_mean
            sim_na = np.array(sm.get("n_A_ts", [0]), dtype=np.float64)
            if len(sim_na) > n_data: sim_na = sim_na[:n_data]
            elif len(sim_na) < n_data: sim_na = np.pad(sim_na, (0, n_data - len(sim_na)), mode="edge")

            tr_rmsle = _rmsle(obs_train, sim_na[:te + 1])
            vl_rmsle = _rmsle(obs_arr[te + 1:], sim_na[te + 1:])
            pr = float(np.max(sim_na)) / max(float(np.max(obs_arr)), 1)

            # Flow diagnostics
            n_U = np.array(sm.get("n_U_ts", [0]), dtype=np.float64)
            n_E = np.array(sm.get("n_E_ts", [0]), dtype=np.float64)
            dU = -np.diff(n_U) if len(n_U) > 1 else [0]
            U_out = float(np.maximum(np.array(dU), 0).sum())
            total_act = float(np.maximum(np.diff(sim_na), 0).sum())

            reachable = vl_rmsle < bl_val and 0.5 <= pr <= 2.0

            result_entry = {
                "config": name, "pid": pid, "label": label,
                "beta_M": cfg["beta_M"],
                "disable_D0": cfg["disable_D0"], "disable_D1": cfg["disable_D1"],
                "sim_peak": float(np.max(sim_na)),
                "obs_peak": float(np.max(obs_arr)),
                "peak_ratio": round(pr, 4),
                "train_rmsle": round(tr_rmsle, 4),
                "val_rmsle": round(vl_rmsle, 4),
                "bl_val": round(bl_val, 4),
                "U_outflow_total": round(U_out, 1),
                "total_activations": round(total_act, 1),
                "reachable": reachable,
            }
            all_results.append(result_entry)
            status = "OK" if reachable else "XX"
            print(f"{status} peak={sim_na.max():.0f} vs obs={obs_arr.max():.0f} "
                  f"val={vl_rmsle:.4f} vs bl={bl_val:.4f} "
                  f"Uout={U_out:.0f} act={total_act:.0f}")

    # Decision
    print(f"\n{'='*50}")
    print(f"  KILL-SWITCH RESULTS")
    print(f"{'='*50}")
    for name in configs:
        rr = [r for r in all_results if r["config"] == name]
        n = sum(1 for r in rr if r["reachable"])
        print(f"  {name}: {n}/{len(rr)} reachable")
        for r in rr:
            s = "OK" if r["reachable"] else "XX"
            print(f"    {r['pid']} {r['label']}: {s} peak={r['sim_peak']:.0f}/{r['obs_peak']:.0f} "
                  f"val={r['val_rmsle']:.4f} Uout={r['U_outflow_total']:.0f} act={r['total_activations']:.0f}")

    with open(OUT / "v15_killswitch.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUT / 'v15_killswitch.json'}")


if __name__ == "__main__":
    main()
