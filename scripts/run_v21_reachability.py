"""
V2.1 Phase 3: Real-Trajectory Reachability Scan (standalone).

500 LHS vectors × 4 real cases, train-only RMSLE.
Saves incremental results to avoid data loss.
"""

import json, sys, time
from datetime import datetime, timezone, timedelta
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
from dynamics_simulation.calibration.parameters import Stage1ParameterSet, apply_parameter_vector
from dynamics_simulation.calibration.split import TemporalSplit

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "recovery"
DATA = Path("data/raw/CHECKED/dataset")
SPECS = Stage1ParameterSet.to_specs()
BOUNDS = Stage1ParameterSet.bounds()

CASES = [
    ("pair_05", "fake", "afc5ba429bc79086f05d6798e4ed978a"),
    ("pair_05", "real", "619f582e27e736ebc4c5def47f954535"),
    ("pair_07", "fake", "b4a497151507853058c8c50b6c6670f5"),
    ("pair_07", "real", "6f2b7e632c64ba5835164abe2f1ab28e"),
]

def _rmsle(a, b):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(b,0)) - np.log1p(np.maximum(a,0)))**2)))

def _lhs(n, bounds, seed):
    try:
        from scipy.stats.qmc import LatinHypercube
        s = LatinHypercube(d=len(bounds), seed=seed)
        smp = s.random(n=n)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        return lo + smp * (hi - lo)
    except ImportError:
        rng = np.random.default_rng(seed)
        return np.array([[rng.uniform(*b) for b in bounds] for _ in range(n)])

def _sim(params, case, seeds, T):
    cfg = ReplayConfig(step_hours=24.0, tail_steps=0, network_mode=ReplayNetworkMode.BROADCAST, seeds=seeds)
    r = run_replay(case, params, cfg)
    if not r.simulated_mean: return np.zeros(T+1)
    arr = np.array(r.simulated_mean.get("active_count",[]), dtype=np.float64)
    if len(arr) < T+1: arr = np.pad(arr, (0,T+1-len(arr)), mode="edge")
    return arr[:T+1]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    vectors = _lhs(500, BOUNDS, 20260804)

    all_cases = []
    for pid, label, cid in CASES:
        print(f"\n{'='*50}\n  [{pid} {label}]\n{'='*50}")
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

        # Baseline (train-only exp decay)
        t_tr = np.arange(len(obs_train), dtype=np.float64)
        A = np.column_stack([np.ones_like(t_tr), t_tr])
        c,_,_,_ = np.linalg.lstsq(A, np.log(np.maximum(obs_train,1e-6)), rcond=None)
        bl_exp = np.exp(c[0]) * np.exp(-max(-c[1],1e-6) * np.arange(n_data, dtype=np.float64))
        bl_train = _rmsle(obs_train, bl_exp[:te+1])
        bl_val = _rmsle(obs_arr[te+1:], bl_exp[te+1:])

        print(f"  N={len(case.user_ids)} ix={len(case.interactions)} steps={n_data} train={te+1}")
        print(f"  Baseline: trainRMSLE={bl_train:.4f} valRMSLE={bl_val:.4f}")
        print(f"  Obs peak={obs_arr.max():.0f} at step {obs_arr.argmax()}")

        best = float("inf")
        top20 = []
        for vi, vec in enumerate(vectors):
            try:
                p = apply_parameter_vector(base, SPECS, list(vec))
            except ValueError:
                continue
            sim = _sim(p, case, (42,), grid.last_data_step)
            tr_err = _rmsle(obs_train, sim[:te+1])
            vl_err = _rmsle(obs_arr[te+1:], sim[te+1:])
            pr = float(np.max(sim[:n_data])) / max(float(np.max(obs_arr[:n_data])),1)
            ar = float(np.trapezoid(sim[:n_data])) / max(float(np.trapezoid(obs_arr[:n_data])),1e-6)

            entry = {"vi":vi, "params":{s.path:float(vec[i]) for i,s in enumerate(SPECS)},
                     "train_rmsle":float(tr_err), "val_rmsle":float(vl_err),
                     "peak_ratio":float(pr), "auc_ratio":float(ar)}
            if tr_err < best: best = tr_err
            if tr_err < best * 3: top20.append(entry)
            if (vi+1) % 100 == 0:
                print(f"    {vi+1}/500 best={best:.4f}")

        top20.sort(key=lambda x: x["train_rmsle"])
        top20 = top20[:20]

        # Re-evaluate top 20 with 5 seeds
        for e in top20:
            p = apply_parameter_vector(base, SPECS, [e["params"][s.path] for s in SPECS])
            sim5 = _sim(p, case, (11,23,37,53,71), grid.last_data_step)
            e["train_rmsle_5s"] = _rmsle(obs_train, sim5[:te+1])
            e["val_rmsle_5s"] = _rmsle(obs_arr[te+1:], sim5[te+1:])
            e["peak_ratio_5s"] = float(np.max(sim5[:n_data])) / max(float(np.max(obs_arr[:n_data])),1)

        # Check reachability
        reachable = False
        best_e = None
        for e in top20:
            pr = e.get("peak_ratio_5s", e["peak_ratio"])
            ar = e["auc_ratio"]
            vr = e.get("val_rmsle_5s", e["val_rmsle"])
            if vr < bl_val and 0.5 <= pr <= 2.0 and 0.5 <= ar <= 2.0:
                reachable = True
                best_e = e
                break

        result = {"pid":pid, "label":label, "case_id":cid,
                  "bl_train_rmsle":bl_train, "bl_val_rmsle":bl_val,
                  "obs_peak":float(obs_arr.max()), "obs_peak_step":int(obs_arr.argmax()),
                  "best_train_rmsle_500":best,
                  "best_val_rmsle_5s":min(e.get("val_rmsle_5s",1) for e in top20),
                  "reachable":reachable,
                  "top5_vectors":[
                      {"params":e["params"], "train_rmsle_5s":e["train_rmsle_5s"],
                       "val_rmsle_5s":e["val_rmsle_5s"], "peak_ratio_5s":e["peak_ratio_5s"]}
                      for e in top20[:5]
                  ]}
        all_cases.append(result)
        vals = [f"{e['val_rmsle_5s']:.4f}" for e in top20[:5]]
        print(f"    Top5 valRMSLE: {vals}")
        print(f"    REACHABLE: {reachable}  {'YES' if reachable else 'NO'}")
        if best_e:
            print(f"    Best params: {best_e['params']}")

        # Save incrementally
        with open(OUT / "v21_reachability.json", "w", encoding="utf-8") as f:
            json.dump(all_cases, f, indent=2)

    # Decision
    n_r = sum(1 for r in all_cases if r["reachable"])
    print(f"\n{'='*50}")
    print(f"  REACHABILITY: {n_r}/{len(all_cases)} cases reachable")
    for r in all_cases:
        print(f"    {r['pid']} {r['label']}: {'OK' if r['reachable'] else 'XX'} "
              f"bestVal={r['best_val_rmsle_5s']:.4f} vs blVal={r['bl_val_rmsle']:.4f}")
    if n_r >= 2:
        print("  → Proceed to trial calibration")
    else:
        print("  → Model structure cannot generate real trajectories")
        print("  → Check: U→E→A delay, root shock, 24h granularity, observation model")
    print(f"\nSaved: {OUT / 'v21_reachability.json'}")

if __name__ == "__main__":
    main()
