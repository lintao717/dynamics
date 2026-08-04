
"""[V1.4] Behavioral Emission Model Reachability Test.

Post-hoc scale correction: Y_hat(t) = p_emit * N_A(t).
p_emit = mean(observed_train) / mean(simulated_train).
Absorbs scale mismatch; dynamics judged on SHAPE only.
"""

import json, sys, time
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
    return float(np.sqrt(np.mean((np.log1p(np.maximum(b, 0)) - np.log1p(np.maximum(a, 0))) ** 2)))


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


def _apply_3p(base, values):
    return apply_parameter_vector(base, SPECS_3, list(values))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    vectors = _lhs(500, BOUNDS_3, 20260804)
    all_results = []

    for pid, label, cid in CASES:
        print(f"\n=== V1.4 EMISSION [{pid} {label}] ===")
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
        obs_mean_train = float(np.mean(obs_train))

        print(f"  N={len(case.user_ids)} ix={len(case.interactions)} steps={n_data}")
        print(f"  Baseline: train={bl_train:.4f} val={bl_val:.4f} peak={obs_arr.max():.0f}")

        best = float("inf")
        top = []
        for vi, vec in enumerate(vectors):
            try:
                p = _apply_3p(base, vec)
            except ValueError:
                continue
            cfg = ReplayConfig(step_hours=24.0, tail_steps=0,
                              network_mode=ReplayNetworkMode.BROADCAST,
                              seeds=(42,), micro_steps=4)
            result = run_replay(case, p, cfg)
            if not result.simulated_mean:
                continue
            sim_raw = np.array(result.simulated_mean.get("n_A_ts", [0]), dtype=np.float64)
            if len(sim_raw) > n_data: sim_raw = sim_raw[:n_data]
            elif len(sim_raw) < n_data: sim_raw = np.pad(sim_raw, (0, n_data - len(sim_raw)), mode="edge")

            p_emit = obs_mean_train / max(np.mean(sim_raw[:te + 1]), 1e-6)
            p_emit = np.clip(p_emit, 0.001, 10.0)
            sim_s = p_emit * sim_raw
            tr = _rmsle(obs_train, sim_s[:te + 1])
            vl = _rmsle(obs_arr[te + 1:], sim_s[te + 1:])
            pr = float(np.max(sim_s[:n_data])) / max(float(np.max(obs_arr[:n_data])), 1)

            if tr < best: best = tr
            if tr < best * 3:
                top.append({"vi": vi,
                    "params": {s.path: float(vec[i]) for i, s in enumerate(SPECS_3)},
                    "p_emit": float(p_emit), "train_rmsle": float(tr),
                    "val_rmsle": float(vl), "peak_ratio": float(pr)})
            if (vi + 1) % 200 == 0:
                print(f"    {vi + 1}/500 best_train={best:.4f}")

        top.sort(key=lambda x: x["train_rmsle"])
        top20 = top[:20]

        # Re-evaluate top-20 with 5 seeds
        for e in top20:
            p = _apply_3p(base, [e["params"][s.path] for s in SPECS_3])
            cfg5 = ReplayConfig(step_hours=24.0, tail_steps=0,
                               network_mode=ReplayNetworkMode.BROADCAST,
                               seeds=FIT_SEEDS, micro_steps=4)
            r5 = run_replay(case, p, cfg5)
            if not r5.simulated_mean: continue
            sim5 = np.array(r5.simulated_mean.get("n_A_ts", [0]), dtype=np.float64)
            if len(sim5) > n_data: sim5 = sim5[:n_data]
            elif len(sim5) < n_data: sim5 = np.pad(sim5, (0, n_data - len(sim5)), mode="edge")
            pe5 = obs_mean_train / max(np.mean(sim5[:te + 1]), 1e-6)
            pe5 = np.clip(pe5, 0.001, 10.0)
            s5s = pe5 * sim5
            e["t5"] = _rmsle(obs_train, s5s[:te + 1])
            e["v5"] = _rmsle(obs_arr[te + 1:], s5s[te + 1:])
            e["pr5"] = float(np.max(s5s[:n_data])) / max(float(np.max(obs_arr[:n_data])), 1)
            e["pe5"] = float(pe5)

        reachable = False
        for e in top20:
            if e.get("v5", 9) < bl_val and 0.5 <= e.get("pr5", 0) <= 2.0:
                reachable = True; break

        result = {"pid": pid, "label": label, "bl_val": bl_val,
                  "best_raw": best, "best_val": min(e.get("v5", 9) for e in top20),
                  "reachable": reachable,
                  "top3": [{"p_emit": e["pe5"], "v5": e["v5"], "pr5": e["pr5"],
                            "params": e["params"]} for e in top20[:3]]}
        all_results.append(result)
        status = "OK" if reachable else "XX"
        vs = [f'{e["v5"]:.4f}' for e in top20[:3]]
        print(f"  {status} best_val={min(e.get('v5',9) for e in top20):.4f} vs bl={bl_val:.4f} top3={vs}")

    n_r = sum(1 for r in all_results if r["reachable"])
    print(f"\n=== V1.4 DECISION: {n_r}/{len(all_results)} reachable ===")
    for r in all_results:
        s = "OK" if r["reachable"] else "XX"
        pe = [f'{e["p_emit"]:.3f}' for e in r["top3"][:1]]
        print(f"  {r['pid']} {r['label']}: {s} val={r['best_val']:.4f} vs bl={r['bl_val']:.4f} p_emit={pe}")
    if n_r >= 2:
        print("  >> EMISSION WORKS - proceed to calibration")
    elif n_r >= 1:
        print("  >> Marginal - helps but needs refinement")
    else:
        print("  >> Still insufficient")

    with open(OUT / "v14_emission.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {OUT / 'v14_emission.json'}")


if __name__ == "__main__":
    main()
