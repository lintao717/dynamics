"""[V1.5 S3] Batch Calibration — 20 Matched Cases.

Model: S3 one-shot (D0/D1 disabled), 3-param (beta_M, alpha_0, gamma_0).
20 cases from checked_pilot_matched_20.yaml.
Single optimizer seed, 5 replay seeds, 70/30 split.
Saves intermediate results for fault tolerance.
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

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "calibration"
DATA = Path("data/raw/CHECKED/dataset")

SPECS_3 = (
    ParameterSpec("propagation.beta_M", 0.0, 1.0),
    ParameterSpec("activation.alpha_0", -6.0, 2.0),
    ParameterSpec("decay.gamma_0", -6.0, 6.0),
)
BOUNDS_3 = [(s.low, s.high) for s in SPECS_3]
REPLAY_SEEDS = (11, 23, 37, 53, 71)

# All 10 pairs from checked_pilot_matched_20.yaml
PAIRS = [
    ("pair_01", "26f247cc05fd53e12daa91c98e7feab8", "cf5522f51f8be6859840b0b948bb813c"),
    ("pair_02", "f983d8f70e265ec265acd9ea93e9b10d", "d293a73970751898b5d83f5525a15fcc"),
    ("pair_03", "68665bf53973dea5036336fcc5dc9eea", "97bdff4a4098a06dc2b2597b514b2df3"),
    ("pair_04", "7cfe610e01dafc496143664a1a8cf87d", "9a74296ad4c231241c1ac6057c08bbcb"),
    ("pair_05", "afc5ba429bc79086f05d6798e4ed978a", "619f582e27e736ebc4c5def47f954535"),
    ("pair_06", "25d9ed3994c2d5f030b864867facab47", "f030f59e7579dcda946ab8a3bd2733c6"),
    ("pair_07", "b4a497151507853058c8c50b6c6670f5", "6f2b7e632c64ba5835164abe2f1ab28e"),
    ("pair_08", "b9801872032a3bee629f6559bbf503ba", "3b7e48be19979f9df3a146e2d0277c58"),
    ("pair_09", "dfb9f2af5bb9b16ba717b91d6be5fa2f", "100536e843cd4e307e1a2865b28b1f05"),
    ("pair_10", "3dac58ea8cfef832bde25278a699fd45", "22c6a11c223c838fa933b5b4af777fcc"),
]


def _rmsle(a, b):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(b,0))-np.log1p(np.maximum(a,0)))**2)))


def _make_s3(base, values):
    p = apply_parameter_vector(base, SPECS_3, list(values))
    r = p.reactivation
    return replace(p, reactivation=replace(r, r_0_0=-1e9, r_0_1=-1e9))


def calibrate(case, obs_train, obs_val, te, n_data):
    base = default_params()
    def obj(x):
        try: p = _make_s3(base, x)
        except ValueError: return 1e10
        cfg = ReplayConfig(step_hours=24.0, tail_steps=0,
                          network_mode=ReplayNetworkMode.BROADCAST,
                          seeds=REPLAY_SEEDS, micro_steps=1)
        r = run_replay(case, p, cfg)
        if not r.simulated_mean: return 1e10
        s = np.array(r.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
        if len(s)>n_data: s=s[:n_data]
        elif len(s)<n_data: s=np.pad(s,(0,n_data-len(s)),mode="edge")
        return _rmsle(obs_train, s[:te+1])

    from scipy.optimize import differential_evolution
    t0 = time.perf_counter()
    opt = differential_evolution(obj, BOUNDS_3, seed=20260805,
                                 workers=1, maxiter=40, popsize=10, polish=True)
    elapsed = time.perf_counter()-t0

    bp = _make_s3(base, list(opt.x))
    cfg = ReplayConfig(step_hours=24.0, tail_steps=0,
                      network_mode=ReplayNetworkMode.BROADCAST,
                      seeds=REPLAY_SEEDS, micro_steps=1)
    rv = run_replay(case, bp, cfg)
    sv = np.array(rv.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
    if len(sv)>n_data: sv=sv[:n_data]
    elif len(sv)<n_data: sv=np.pad(sv,(0,n_data-len(sv)),mode="edge")
    tr = _rmsle(obs_train, sv[:te+1])
    vl = _rmsle(obs_val, sv[te+1:])
    pr = float(np.max(sv))/max(float(np.max(np.concatenate([obs_train,obs_val]))),1)
    at_bound = any(x<=lo+1e-3 or x>=hi-1e-3 for x,(lo,hi) in zip(opt.x, BOUNDS_3))

    return {"beta_M":float(opt.x[0]),"alpha_0":float(opt.x[1]),"gamma_0":float(opt.x[2]),
            "train_rmsle":round(tr,6),"val_rmsle":round(vl,6),
            "peak_ratio":round(pr,4),"converged":bool(opt.success),
            "at_boundary":at_bound,"n_iter":opt.nit,"elapsed_s":round(elapsed,1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    save_path = OUT / "s3_batch_20.json"

    # Load existing progress for fault tolerance
    if save_path.exists():
        results = json.loads(save_path.read_text())
        done_ids = {(r["pid"], r["label"]) for r in results}
    else:
        done_ids = set()

    for pid, fake_id, real_id in PAIRS:
        for lk, cid in [("fake", fake_id), ("real", real_id)]:
            if (pid, lk) in done_ids:
                print(f"[{pid} {lk}] SKIP (already done)")
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

            # Baseline
            tt = np.arange(len(ot), dtype=np.float64)
            A = np.column_stack([np.ones_like(tt), tt])
            c, _, _, _ = np.linalg.lstsq(A, np.log(np.maximum(ot,1e-6)), rcond=None)
            bl = np.exp(c[0])*np.exp(-max(-c[1],1e-6)*np.arange(n,dtype=np.float64))
            bl_val = _rmsle(ov, bl[te+1:])

            print(f"  N={len(case.user_ids)} steps={n} bl_val={bl_val:.4f}", flush=True)
            cal = calibrate(case, ot, ov, te, n)
            imp = (bl_val-cal["val_rmsle"])/max(bl_val,1e-6)*100
            print(f"  val={cal['val_rmsle']:.4f} ({imp:+.0f}%) "
                  f"bM={cal['beta_M']:.3f} a0={cal['alpha_0']:.3f} g0={cal['gamma_0']:.3f} "
                  f"conv={cal['converged']} bound={cal['at_boundary']} {cal['elapsed_s']:.0f}s")

            r = {"pid":pid, "label":lk, "case_id":cid,
                 "n_agents":len(case.user_ids),
                 "steps":n, "train_steps":te+1,
                 "bl_val":bl_val, **cal,
                 "improvement_pct":round(imp,1)}
            results.append(r)

            # Save incrementally
            save_path.write_text(json.dumps(results, indent=2))

    # Summary
    fake = [r for r in results if r["label"]=="fake"]
    real = [r for r in results if r["label"]=="real"]

    def stats(arr): return {"mean":np.mean(arr), "median":np.median(arr),
                            "min":np.min(arr), "max":np.max(arr)}
    print(f"\n{'='*60}")
    print(f"  BATCH CALIBRATION SUMMARY ({len(results)} cases)")
    print(f"{'='*60}")
    for label, group in [("FAKE", fake), ("REAL", real)]:
        imp = [r["improvement_pct"] for r in group]
        vl = [r["val_rmsle"] for r in group]
        bm = [r["beta_M"] for r in group]
        a0 = [r["alpha_0"] for r in group]
        g0 = [r["gamma_0"] for r in group]
        passed = sum(1 for r in group if r["improvement_pct"]>10 and not r["at_boundary"])
        print(f"\n  {label} ({len(group)} cases, {passed} passed):")
        print(f"    val RMSLE:       mean={np.mean(vl):.4f} median={np.median(vl):.4f}")
        print(f"    improvement %:   mean={np.mean(imp):.1f}% median={np.median(imp):.1f}%")
        print(f"    beta_M:          mean={np.mean(bm):.3f} median={np.median(bm):.3f}")
        print(f"    alpha_0:         mean={np.mean(a0):.2f} median={np.median(a0):.2f}")
        print(f"    gamma_0:         mean={np.mean(g0):.2f} median={np.median(g0):.2f}")
        print(f"    boundary hits:   {sum(1 for r in group if r['at_boundary'])}/{len(group)}")
        print(f"    converged:       {sum(1 for r in group if r['converged'])}/{len(group)}")

    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
