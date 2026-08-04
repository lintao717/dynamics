"""[V1.5.1] Formal Mechanism Ablation: 20 cases x 4 structures x 256 Sobol.

Structures:
  FULL: all mechanisms (V1.1 default)
  NO_DELAYED_FIRST: D0->A disabled
  NO_TRUE_REACTIVATION: D1->A disabled
  ONE_SHOT: both D0->A and D1->A disabled

Params: beta_M [0,1], alpha_0 [-6,2], gamma_0 [-6,6]. beta_V fixed.
256 Sobol vectors, single-seed sweep, top-20 with 5 validation seeds.
Paired: same param vectors evaluated on all 4 structures for each case.
"""

import json, sys, time
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamics_simulation.config import default_params, ReactivationMode
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.observations import build_observed_trajectory
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.calibration.parameters import ParameterSpec, apply_parameter_vector
from dynamics_simulation.calibration.split import TemporalSplit
from dynamics_simulation.evaluation import rmsle, FIT_SEEDS, VALIDATION_SEEDS
from dynamics_simulation.baseline import BaselineForecast

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "calibration"
DATA = Path("data/raw/CHECKED/dataset")

SPECS_3 = (
    ParameterSpec("propagation.beta_M", 0.0, 1.0),
    ParameterSpec("activation.alpha_0", -6.0, 2.0),
    ParameterSpec("decay.gamma_0", -6.0, 6.0),
)
BOUNDS = [(s.low, s.high) for s in SPECS_3]

STRUCTURES = [
    (ReactivationMode.FULL, "FULL"),
    (ReactivationMode.NO_DELAYED_FIRST, "NO_DELAYED_FIRST"),
    (ReactivationMode.NO_TRUE_REACTIVATION, "NO_TRUE_REACTIVATION"),
    (ReactivationMode.ONE_SHOT, "ONE_SHOT"),
]

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


def _sobol(n, bounds, seed=20260805):
    try:
        from scipy.stats.qmc import Sobol
        s = Sobol(d=len(bounds), seed=seed)
        smp = s.random(n)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        return lo + smp * (hi - lo)
    except ImportError:
        rng = np.random.default_rng(seed)
        return np.array([[rng.uniform(*b) for b in bounds] for _ in range(n)])


def _make_params(base, values, mode: ReactivationMode):
    p = apply_parameter_vector(base, SPECS_3, list(values))
    if mode == ReactivationMode.NO_DELAYED_FIRST:
        p = replace(p, reactivation=replace(p.reactivation, r_0_0=-20.0))
    elif mode == ReactivationMode.NO_TRUE_REACTIVATION:
        p = replace(p, reactivation=replace(p.reactivation, r_0_1=-20.0))
    elif mode == ReactivationMode.ONE_SHOT:
        p = replace(p, reactivation=replace(p.reactivation, r_0_0=-20.0, r_0_1=-20.0))
    return p


def _sim(params, case, seeds, target_len):
    cfg = ReplayConfig(step_hours=24.0, tail_steps=0,
                      network_mode=ReplayNetworkMode.BROADCAST,
                      seeds=seeds, micro_steps=1)
    r = run_replay(case, params, cfg)
    if not r.simulated_mean: return np.zeros(target_len)
    arr = np.array(r.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
    if len(arr) > target_len:
        arr = arr[:target_len]
    elif len(arr) < target_len:
        # Pad with edge values (tail plateau)
        pad_len = target_len - len(arr)
        arr = np.concatenate([arr, np.full(pad_len, arr[-1])])
    return arr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    vectors = _sobol(256, BOUNDS)
    all_results = []
    save_path = OUT / "v151_mechanism_ablation.json"

    if save_path.exists():
        all_results = json.loads(save_path.read_text())
        done = {(r["pid"],r["label"]) for r in all_results}
    else:
        done = set()

    for pid, fake_id, real_id in PAIRS:
        for lk, cid in [("fake",fake_id),("real",real_id)]:
            if (pid, lk) in done:
                print(f"[{pid} {lk}] SKIP")
                continue

            print(f"\n{'='*50}")
            print(f"  [{pid} {lk}]")
            path = DATA / f"{lk}_news" / f"{cid}.json"
            case = load_checked_case(path)
            idx = NodeIndex.from_case(case)
            grid = TimeGrid.from_case(case, step_hours=24.0, tail_steps=0)
            traj = build_observed_trajectory(case, idx, grid)
            obs_arr = np.array(traj.active_count, dtype=np.float64)
            n = grid.last_data_step + 1
            sp = TemporalSplit.by_fraction(total_steps=grid.last_data_step, train_fraction=0.7)
            te = sp.train_end_step
            ot = obs_arr[:te+1]

            # Baseline
            bf = BaselineForecast(ot, n)
            bl_best_val = rmsle(obs_arr[te+1:], bf.best_future)

            # Ensure target_len for simulation output alignment
            target_len = n

            case_results = {"pid":pid, "label":lk, "case_id":cid,
                          "n_agents":len(case.user_ids), "steps":n, "train":te+1,
                          "bl_val_rmsle":round(bl_best_val,4),
                          "obs_peak":float(obs_arr.max()),
                          "structures":{}}

            for mode, mode_name in STRUCTURES:
                print(f"  {mode_name}: ", end="", flush=True)
                best_train = float("inf")
                top = []
                for vi, vec in enumerate(vectors):
                    try: p = _make_params(base, vec, mode)
                    except ValueError: continue
                    s = _sim(p, case, (42,), target_len)
                    tr = rmsle(ot, s[:te+1])
                    vl = rmsle(obs_arr[te+1:], s[te+1:])
                    pr = float(np.max(s))/max(float(np.max(obs_arr)),1)
                    if tr < best_train: best_train = tr
                    if tr < best_train * 3:
                        top.append({"vi":vi,"train":float(tr),"val":float(vl),
                                     "peak_ratio":float(pr),
                                     "beta_M":float(vec[0]),
                                     "alpha_0":float(vec[1]),
                                     "gamma_0":float(vec[2])})

                top.sort(key=lambda x:x["train"])
                top20 = top[:20]

                # Re-eval with validation seeds
                best_val = float("inf")
                best_peak = 0.0
                for e in top20:
                    p = _make_params(base, [e["beta_M"],e["alpha_0"],e["gamma_0"]], mode)
                    s5 = _sim(p, case, VALIDATION_SEEDS, target_len)
                    vl5 = rmsle(obs_arr[te+1:], s5[te+1:])
                    pr5 = float(np.max(s5))/max(float(np.max(obs_arr)),1)
                    e["val_5s"] = float(vl5)
                    e["peak_ratio_5s"] = float(pr5)
                    if vl5 < best_val: best_val = vl5; best_peak = pr5

                reachable = best_val < bl_best_val and 0.5 <= best_peak <= 2.0
                case_results["structures"][mode_name] = {
                    "best_train_1s": round(best_train,4),
                    "best_val_5s": round(best_val,4),
                    "best_peak_ratio_5s": round(best_peak,4),
                    "reachable": reachable,
                    "top3_params": [
                        {"beta_M":e["beta_M"],"alpha_0":e["alpha_0"],"gamma_0":e["gamma_0"],
                         "val_5s":e["val_5s"],"peak_ratio_5s":e["peak_ratio_5s"]}
                        for e in top20[:3]
                    ],
                }
                status = "OK" if reachable else "XX"
                imp = (bl_best_val-best_val)/max(bl_best_val,1e-6)*100
                print(f"{status} train={best_train:.4f} val={best_val:.4f} "
                      f"({imp:+.0f}%) peak={best_peak:.2f}")

            all_results.append(case_results)
            save_path.write_text(json.dumps(all_results,indent=2))

    # Summary
    print(f"\n{'='*60}")
    print(f"  MECHANISM ABLATION SUMMARY")
    print(f"{'='*60}")
    for mode_name in [s[1] for s in STRUCTURES]:
        n_reach = sum(1 for r in all_results
                     if r["structures"].get(mode_name,{}).get("reachable",False))
        vals = [r["structures"][mode_name]["best_val_5s"]
                for r in all_results if mode_name in r.get("structures",{})]
        if vals:
            print(f"  {mode_name}: {n_reach}/{len(vals)} reachable, "
                  f"mean_val={np.mean(vals):.4f} median_val={np.median(vals):.4f}")

    # By label
    for lk in ["fake","real"]:
        lr = [r for r in all_results if r["label"]==lk]
        print(f"\n  {lk.upper()} ({len(lr)} cases):")
        for mode_name in [s[1] for s in STRUCTURES]:
            n_r = sum(1 for r in lr
                     if r["structures"].get(mode_name,{}).get("reachable",False))
            vals = [r["structures"][mode_name]["best_val_5s"]
                    for r in lr if mode_name in r.get("structures",{})]
            bls = [r["bl_val_rmsle"] for r in lr]
            if vals:
                print(f"    {mode_name}: {n_r}/{len(lr)} reachable, "
                      f"mean_val={np.mean(vals):.4f} vs mean_bl={np.mean(bls):.4f}")

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
