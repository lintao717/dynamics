"""[V1.5 S3] One-Shot Activation Parameter Scan.

Model: D0->A and D1->A DISABLED (r_0_0 = r_0_1 = -1e9).
       Only U->E->A/D, then A->D. No reactivation.

Search: 500 LHS vectors over (beta_M, alpha_0, gamma_0).
        beta_V fixed at default.
Compare: train-only exp-decay baseline RMSLE.
Criteria: val RMSLE < baseline AND 0.5 <= peak_ratio <= 2.0.

4 cases, 5 seeds for top-20 re-evaluation.
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

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "recovery"
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
FIT_SEEDS = (11, 23, 37, 53, 71)


def _rmsle(a, b):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(b,0))-np.log1p(np.maximum(a,0)))**2)))


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


def _make_s3_params(base, values):
    """Apply 3-param vector with D0/D1 disabled."""
    p = apply_parameter_vector(base, SPECS_3, list(values))
    r = p.reactivation
    p = replace(p, reactivation=replace(r, r_0_0=-1e9, r_0_1=-1e9))
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    vectors = _lhs(500, BOUNDS_3, 20260805)
    all_results = []

    print(f"V1.5 S3: One-shot activation, 500 LHS, 3-param")
    print(f"Model: D0->A=OFF, D1->A=OFF (r_0_0=r_0_1=-1e9)")

    for pid, label, cid in CASES:
        print(f"\n=== [{pid} {label}] ===")
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

        # Baseline
        t_tr = np.arange(len(obs_train), dtype=np.float64)
        A = np.column_stack([np.ones_like(t_tr), t_tr])
        c, _, _, _ = np.linalg.lstsq(A, np.log(np.maximum(obs_train,1e-6)), rcond=None)
        bl = np.exp(c[0]) * np.exp(-max(-c[1],1e-6) * np.arange(n_data, dtype=np.float64))
        bl_train = _rmsle(obs_train, bl[:te+1])
        bl_val = _rmsle(obs_arr[te+1:], bl[te+1:])
        print(f"  N={len(case.user_ids)} steps={n_data} peak={obs_arr.max():.0f} bl_val={bl_val:.4f}")

        best = float("inf")
        top = []
        for vi, vec in enumerate(vectors):
            try:
                p = _make_s3_params(base, vec)
            except ValueError:
                continue
            cfg = ReplayConfig(step_hours=24.0, tail_steps=0,
                              network_mode=ReplayNetworkMode.BROADCAST,
                              seeds=(42,), micro_steps=1)
            result = run_replay(case, p, cfg)
            if not result.simulated_mean: continue
            sim_raw = np.array(result.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
            if len(sim_raw) > n_data: sim_raw = sim_raw[:n_data]
            elif len(sim_raw) < n_data: sim_raw = np.pad(sim_raw, (0,n_data-len(sim_raw)), mode="edge")

            tr = _rmsle(obs_train, sim_raw[:te+1])
            vl = _rmsle(obs_arr[te+1:], sim_raw[te+1:])
            pr = float(np.max(sim_raw))/max(float(np.max(obs_arr)),1)

            if tr < best: best = tr
            if tr < best * 3:
                top.append({"vi":vi,
                    "beta_M":float(vec[0]), "alpha_0":float(vec[1]), "gamma_0":float(vec[2]),
                    "train_rmsle":float(tr), "val_rmsle":float(vl), "peak_ratio":float(pr)})
            if (vi+1)%200==0: print(f"    {vi+1}/500 best={best:.4f}",flush=True)

        print(f"  Done. best_train={best:.4f}",flush=True)
        top.sort(key=lambda x:x["train_rmsle"])
        top20 = top[:20]

        # 5-seed re-evaluation
        for e in top20:
            p = _make_s3_params(base, [e["beta_M"], e["alpha_0"], e["gamma_0"]])
            cfg5 = ReplayConfig(step_hours=24.0, tail_steps=0,
                               network_mode=ReplayNetworkMode.BROADCAST,
                               seeds=FIT_SEEDS, micro_steps=1)
            r5 = run_replay(case, p, cfg5)
            if not r5.simulated_mean: continue
            s5 = np.array(r5.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
            if len(s5)>n_data: s5=s5[:n_data]
            elif len(s5)<n_data: s5=np.pad(s5,(0,n_data-len(s5)),mode="edge")
            e["t5"] = _rmsle(obs_train, s5[:te+1])
            e["v5"] = _rmsle(obs_arr[te+1:], s5[te+1:])
            e["pr5"] = float(np.max(s5))/max(float(np.max(obs_arr)),1)

        reachable = False
        for e in top20:
            if e.get("v5",9) < bl_val and 0.5 <= e.get("pr5",0) <= 2.0:
                reachable = True; break

        best_val = min(e.get("v5",9) for e in top20)
        r = {"pid":pid, "label":label, "bl_val":bl_val,
             "best_raw":best, "best_val_5s":best_val,
             "reachable":reachable,
             "top3":[{"beta_M":e["beta_M"],"alpha_0":e["alpha_0"],"gamma_0":e["gamma_0"],
                      "v5":e["v5"],"pr5":e["pr5"]} for e in top20[:3]]}
        all_results.append(r)
        s = "OK" if reachable else "XX"
        vs = [f'{e["v5"]:.4f}' for e in top20[:3]]
        print(f"  {s} best_val={best_val:.4f} vs bl={bl_val:.4f} top3={vs}")
        if r["reachable"]:
            print(f"  >>> REACHABLE! First case to beat baseline.")

    n_r = sum(1 for r in all_results if r["reachable"])
    print(f"\n=== S3 SCAN: {n_r}/{len(all_results)} reachable ===")
    for r in all_results:
        s = "OK" if r["reachable"] else "XX"
        print(f"  {r['pid']} {r['label']}: {s} val={r['best_val_5s']:.4f} vs bl={r['bl_val']:.4f}")
    if n_r >= 2:
        print("  >>> ONE-SHOT MODEL PASSES REACHABILITY! Proceed to calibration.")
    elif n_r >= 1:
        print("  >> Marginal — one case reachable, expand search or adjust peak scaling.")
    else:
        print("  >> Still insufficient — need higher peak with same shape.")

    with open(OUT/"v15_s3_scan.json","w",encoding="utf-8") as f:
        json.dump({"results":all_results,
                   "model":"S3 one-shot (D0->A=OFF, D1->A=OFF)",
                   "params":"beta_M, alpha_0, gamma_0"},f,indent=2)
    print(f"Saved: {OUT/'v15_s3_scan.json'}")


if __name__ == "__main__":
    main()
