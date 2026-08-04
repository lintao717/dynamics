"""[V1.5.2] Protocol-Hardened Mechanism Ablation.

Fixes vs V1.5.1:
  - Uses ReactivationMode via ReplayConfig (not -20 parameter hack)
  - Parameter search uses 5 FIT_SEEDS (not single seed 42)
  - Single best parameter selected by TRAINING loss only (no validation peek)
  - Final evaluation on VALIDATION_SEEDS
  - Paired delta metrics reported (structure vs FULL)

20 cases x 4 structures x 256 Sobol vectors.
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
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        return lo + s.random(n) * (hi - lo)
    except ImportError:
        rng = np.random.default_rng(seed)
        return np.array([[rng.uniform(*b) for b in bounds] for _ in range(n)])


def _sim(case, base, vec, mode, seeds, target_len):
    p = apply_parameter_vector(base, SPECS_3, list(vec))
    cfg = ReplayConfig(step_hours=24.0, tail_steps=0,
                      network_mode=ReplayNetworkMode.BROADCAST,
                      seeds=seeds, micro_steps=1,
                      reactivation_mode=mode.value)
    r = run_replay(case, p, cfg)
    if not r.simulated_mean: return np.zeros(target_len)
    arr = np.array(r.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
    if len(arr) > target_len: arr = arr[:target_len]
    elif len(arr) < target_len: arr = np.concatenate([arr, np.full(target_len-len(arr), arr[-1])])
    return arr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    vectors = _sobol(256, [(s.low,s.high) for s in SPECS_3])
    all_results = []
    save_path = OUT / "v152_mechanism_ablation.json"

    if save_path.exists():
        all_results = json.loads(save_path.read_text())
        done = {(r["pid"],r["label"]) for r in all_results}
    else:
        done = set()

    for pid, fid, rid in PAIRS:
        for lk, cid in [("fake",fid),("real",rid)]:
            if (pid,lk) in done:
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
            ov = obs_arr[te+1:]

            bf = BaselineForecast(ot, n)
            bl_val = rmsle(ov, bf.best_future)

            cr = {"pid":pid,"label":lk,"case_id":cid,
                  "n_agents":len(case.user_ids),"steps":n,"train":te+1,
                  "bl_val":round(bl_val,4),
                  "obs_peak":float(obs_arr.max())}

            # PHASE 1: Score all vectors with 5 FIT_SEEDS for each structure
            struct_results = {}
            for mode, mode_name in STRUCTURES:
                print(f"  {mode_name}: ", end="", flush=True)
                best_train = float("inf")
                best_params = None

                for vi, vec in enumerate(vectors):
                    try:
                        # Average loss across all 5 fit seeds
                        losses = []
                        for fs in FIT_SEEDS:
                            s = _sim(case, base, vec, mode, (fs,), n)
                            losses.append(rmsle(ot, s[:te+1]))
                        tr = float(np.mean(losses))
                    except ValueError:
                        continue
                    if tr < best_train:
                        best_train = tr
                        best_params = vec

                # PHASE 2: Evaluate ONLY the single best (by train) on validation seeds
                s_val = _sim(case, base, best_params, mode, VALIDATION_SEEDS, n)
                vl = rmsle(ov, s_val[te+1:])
                pr = float(np.max(s_val)) / max(float(np.max(obs_arr)), 1)
                reachable = vl < bl_val and 0.5 <= pr <= 2.0

                struct_results[mode_name] = {
                    "best_train_5s": round(best_train,4),
                    "val_rmsle": round(vl,4),
                    "peak_ratio": round(pr,4),
                    "reachable": reachable,
                    "best_params": {
                        "beta_M": float(best_params[0]),
                        "alpha_0": float(best_params[1]),
                        "gamma_0": float(best_params[2]),
                    },
                }
                imp = (bl_val-vl)/max(bl_val,1e-6)*100
                s = "OK" if reachable else "XX"
                print(f"{s} train={best_train:.4f} val={vl:.4f} ({imp:+.0f}%) pr={pr:.2f}")

            cr["structures"] = struct_results
            all_results.append(cr)
            save_path.write_text(json.dumps(all_results,indent=2))

    # Summary with paired deltas
    print(f"\n{'='*60}")
    print(f"  V1.5.2 MECHANISM ABLATION (protocol-hardened)")
    print(f"{'='*60}")

    for mode_name in [s[1] for s in STRUCTURES]:
        n_r = sum(1 for r in all_results if r["structures"][mode_name]["reachable"])
        vals = [r["structures"][mode_name]["val_rmsle"] for r in all_results]
        full_vals = [r["structures"]["FULL"]["val_rmsle"] for r in all_results]
        deltas = [f - s for f, s in zip(full_vals, vals)]
        med_delta = np.median(deltas)
        n_improved = sum(1 for d in deltas if d > 0)
        # Bootstrap 95% CI
        rng = np.random.default_rng(42)
        boot_meds = [np.median(rng.choice(deltas, size=len(deltas), replace=True))
                      for _ in range(1000)]
        ci_lo = np.percentile(boot_meds, 2.5)
        ci_hi = np.percentile(boot_meds, 97.5)

        print(f"\n  {mode_name}: {n_r}/{len(vals)} reachable")
        print(f"    mean_val={np.mean(vals):.4f} median_val={np.median(vals):.4f}")
        print(f"    delta vs FULL: median={med_delta:.4f} improved={n_improved}/{len(deltas)} "
              f"95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    for lk in ["fake","real"]:
        lr = [r for r in all_results if r["label"]==lk]
        for mode_name in [s[1] for s in STRUCTURES]:
            n_r = sum(1 for r in lr if r["structures"][mode_name]["reachable"])
            print(f"  {lk}: {mode_name} {n_r}/{len(lr)} reachable")

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
