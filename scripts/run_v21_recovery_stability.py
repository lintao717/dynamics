"""
V2.1 Recovery Stability and Real-Trajectory Reachability Test.

Phase 1: Optimizer-stability test
  - 5 truth vectors × 3 optimizer seeds
  - maxiter=50, popsize=12, polish=True
  - Save loss curves, track per-parameter variance across seeds

Phase 2: Practical recovery (single-seed truth)
  - Truth from 1 seed, fit with 5-seed mean
  - Repeat for 5 different truth seeds

Phase 3: Real-trajectory reachability scan
  - 500 true LHS vectors per case
  - 4 cases (pair_05 fake/real, pair_07 fake/real)
  - Train-only RMSLE as selection criterion

Phase 4: 3-param vs 4-param comparison
  - Same truth vectors, beta_V fixed
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics_simulation.config import ModelParams, default_params
from dynamics_simulation.data.schema import EventCase, RootPost, InteractionRecord
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.calibration.parameters import (
    Stage1ParameterSet, apply_parameter_vector, ParameterSpec,
)
from dynamics_simulation.calibration.split import TemporalSplit

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "recovery"
DATA_ROOT = Path("data/raw/CHECKED/dataset")

SPECS = Stage1ParameterSet.to_specs()
BOUNDS = Stage1ParameterSet.bounds()
SPEC_DICT = {s.path: s for s in SPECS}

TRUTH_SEEDS = (101, 103, 107, 109, 113)
FIT_SEEDS = (11, 23, 37, 53, 71)
HOLDOUT_SEEDS = (211, 223, 227, 229, 233)
OPT_SEEDS_3 = (20260803, 20260817, 20260901)

# ── LATIN HYPERCUBE ──

def _lhs_sample(n: int, bounds: list, seed: int) -> np.ndarray:
    """True Latin Hypercube sampling using scipy.stats.qmc."""
    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=len(bounds), seed=seed)
        samples = sampler.random(n=n)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        return lo + samples * (hi - lo)
    except ImportError:
        rng = np.random.default_rng(seed)
        return np.array([[rng.uniform(*b) for b in bounds] for _ in range(n)])


# ── SYNTHETIC CASE ──

def _make_syn_case(n=200):
    root = RootPost(post_id=f"syn-{n}", user_id="root",
                    timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
                    text="syn", label="syn", expert_analysis=None)
    interactions = tuple(
        InteractionRecord(interaction_id=f"s{i}", root_post_id=f"syn-{n}",
                          user_id=f"u{i}",
                          timestamp=root.timestamp + timedelta(days=i),
                          kind="comment", text=f"s{i}")
        for i in range(1, n))
    return EventCase(case_id=f"syn-{n}", source_dataset="SYNTHETIC",
                     root=root, interactions=interactions)


def _rmsle(y_true, y_pred):
    yt = np.maximum(np.asarray(y_true, dtype=np.float64), 0)
    yp = np.maximum(np.asarray(y_pred, dtype=np.float64), 0)
    return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))


def _gen_traj(params, case, seeds, T=14):
    cfg = ReplayConfig(step_hours=24.0, tail_steps=0,
                       network_mode=ReplayNetworkMode.BROADCAST, seeds=seeds)
    r = run_replay(case, params, cfg)
    if not r.simulated_mean:
        return np.zeros(T + 1)
    arr = np.array(r.simulated_mean.get("active_count", []), dtype=np.float64)
    if len(arr) < T + 1:
        arr = np.pad(arr, (0, T + 1 - len(arr)), mode="edge")
    return arr[:T + 1]


def _gen_real_obs(case_id, label):
    """Load real CHECKED case and return observed active_count array."""
    path = DATA_ROOT / f"{label}_news" / f"{case_id}.json"
    case = load_checked_case(path)
    from dynamics_simulation.data.indexing import NodeIndex
    from dynamics_simulation.data.timegrid import TimeGrid
    from dynamics_simulation.data.observations import build_observed_trajectory
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=24.0, tail_steps=0)
    traj = build_observed_trajectory(case, index, grid)
    return np.array(traj.active_count, dtype=np.float64), grid.last_data_step, case


# ── PHASE 1: OPTIMIZER STABILITY ──

def run_optimizer_stability():
    """5 truth vectors × 3 opt seeds, save per-generation loss."""
    print("=" * 60)
    print("  V2.1 PHASE 1: OPTIMIZER STABILITY")
    print("=" * 60)

    case = _make_syn_case(200)
    base = default_params()
    rng = np.random.default_rng(20260803)

    # Generate 5 truth vectors via LHS
    truth_vectors = _lhs_sample(5, BOUNDS, 20260803)
    results = []

    for vi, true_x in enumerate(truth_vectors):
        true_params = apply_parameter_vector(base, SPECS, list(true_x))
        true_traj = _gen_traj(true_params, case, TRUTH_SEEDS)
        obs = {"active_count": true_traj.astype(np.float64)}
        T = len(true_traj) - 1
        split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)
        masks = {"active_count": np.ones(T + 1, dtype=bool)}

        for oi, opt_seed in enumerate(OPT_SEEDS_3):
            label = f"v{vi+1}_opt{oi+1}"
            print(f"  [{label}] true={[round(v,3) for v in true_x]}", end=" ", flush=True)

            t0 = time.perf_counter()
            try:
                from scipy.optimize import differential_evolution

                losses = []
                def obj(x):
                    try:
                        p = apply_parameter_vector(base, SPECS, list(x))
                    except (ValueError, AttributeError):
                        return 1e10
                    sim = _gen_traj(p, case, FIT_SEEDS, T)
                    loss = float(np.sqrt(np.mean(
                        (np.log1p(np.maximum(sim, 0)) -
                         np.log1p(np.maximum(true_traj, 0))) ** 2
                    )))
                    losses.append(loss)
                    return loss

                opt = differential_evolution(
                    obj, BOUNDS, seed=int(opt_seed),
                    workers=1, maxiter=50, popsize=12, polish=True,
                )
                best_x = list(opt.x)
                elapsed = time.perf_counter() - t0
                best_p = apply_parameter_vector(base, SPECS, best_x)
                holdout_sim = _gen_traj(best_p, case, HOLDOUT_SEEDS, T)
                holdout_rmsle = _rmsle(true_traj, holdout_sim)

                for j, (spec, tv, ev) in enumerate(zip(SPECS, true_x, best_x)):
                    range_err = abs(ev - tv) / (spec.high - spec.low)
                    at_bound = (ev <= spec.low + 1e-3 or ev >= spec.high - 1e-3)
                    results.append({
                        "phase": "stability", "truth_id": vi + 1,
                        "opt_seed": opt_seed,
                        "param": spec.path,
                        "true_val": float(tv),
                        "estimated": float(ev),
                        "range_norm_error": float(range_err),
                        "at_boundary": at_bound,
                        "converged": bool(opt.success),
                        "final_loss": float(opt.fun),
                        "holdout_rmsle": float(holdout_rmsle),
                        "n_iterations": opt.nit,
                        "n_evaluations": len(losses),
                        "final_10_loss_std": float(np.std(losses[-10:])) if len(losses) >= 10 else float("nan"),
                        "elapsed_s": elapsed,
                    })
                print(f"err=[{', '.join(f'{abs(best_x[j]-true_x[j])/(spec.high-spec.low):.4f}' for j, spec in enumerate(SPECS))}] "
                      f"holdoutRMSLE={holdout_rmsle:.4f} conv={opt.success} "
                      f"finalStd={np.std(losses[-10:]):.6f}" if len(losses) >= 10 else f"holdoutRMSLE={holdout_rmsle:.4f}")
            except ImportError:
                print("scipy missing")

    # Per-vector summary
    print("\n  Per-vector stability:")
    for vi in range(5):
        vr = [r for r in results if r["truth_id"] == vi + 1]
        errs = {}
        for pname in [s.path for s in SPECS]:
            p_ests = [r["estimated"] for r in vr if r["param"] == pname]
            p_true = vr[0]["true_val"] if vr else 0
            if len(p_ests) > 1:
                spec = SPEC_DICT[pname]
                cross_std = np.std(p_ests) / (spec.high - spec.low)
                errs[pname] = cross_std
        print(f"    v{vi+1}: cross-seed param std = {{{', '.join(f'{k}: {v:.4f}' for k,v in errs.items())}}}")

    return results


# ── PHASE 2: PRACTICAL RECOVERY ──

def run_practical_recovery():
    """Single-seed truth, multi-seed fit, 5 repetitions."""
    print("\n" + "=" * 60)
    print("  V2.1 PHASE 2: PRACTICAL RECOVERY (single-seed truth)")
    print("=" * 60)

    case = _make_syn_case(200)
    base = default_params()
    rng = np.random.default_rng(20260804)
    truth_vectors = [rng.uniform(*b) for b in BOUNDS]

    results = []
    for rep, truth_seed in enumerate([101, 103, 107, 109, 113]):
        true_params = apply_parameter_vector(base, SPECS, truth_vectors)
        # Single-seed truth
        true_traj = _gen_traj(true_params, case, (truth_seed,))
        obs = {"active_count": true_traj.astype(np.float64)}
        T = len(true_traj) - 1
        split = TemporalSplit.by_fraction(total_steps=T, train_fraction=0.7)
        masks = {"active_count": np.ones(T + 1, dtype=bool)}

        print(f"  [rep {rep+1}/5] truth_seed={truth_seed}", end=" ", flush=True)

        t0 = time.perf_counter()
        try:
            from scipy.optimize import differential_evolution

            def obj(x):
                try:
                    p = apply_parameter_vector(base, SPECS, list(x))
                except (ValueError, AttributeError):
                    return 1e10
                sim = _gen_traj(p, case, FIT_SEEDS, T)
                return _rmsle(true_traj, sim)

            opt = differential_evolution(
                obj, BOUNDS, seed=20260804 + rep,
                workers=1, maxiter=40, popsize=10, polish=False,
            )
            best_x = list(opt.x)
            elapsed = time.perf_counter() - t0
            best_p = apply_parameter_vector(base, SPECS, best_x)
            holdout_sim = _gen_traj(best_p, case, HOLDOUT_SEEDS, T)

            for j, (spec, tv, ev) in enumerate(zip(SPECS, truth_vectors, best_x)):
                range_err = abs(ev - tv) / (spec.high - spec.low)
                results.append({
                    "phase": "practical",
                    "rep": rep + 1,
                    "truth_seed": truth_seed,
                    "param": spec.path,
                    "true_val": tv,
                    "estimated": float(ev),
                    "range_norm_error": float(range_err),
                    "holdout_rmsle": float(_rmsle(true_traj, holdout_sim)),
                    "converged": bool(opt.success),
                })
            print(f"err=[{', '.join(f'{abs(best_x[j]-truth_vectors[j])/(SPECS[j].high-SPECS[j].low):.4f}' for j in range(4))}] "
                  f"holdoutRMSLE={_rmsle(true_traj, holdout_sim):.4f}")
        except ImportError:
            print("scipy missing")

    return results


# ── PHASE 3: REACHABILITY SCAN ──

def run_reachability_scan():
    """500 LHS vectors × 4 real cases, train-only RMSLE."""
    print("\n" + "=" * 60)
    print("  V2.1 PHASE 3: REAL-TRAJECTORY REACHABILITY")
    print("=" * 60)

    cases_to_test = [
        ("pair_05", "fake", "afc5ba429bc79086f05d6798e4ed978a"),
        ("pair_05", "real", "619f582e27e736ebc4c5def47f954535"),
        ("pair_07", "fake", "b4a497151507853058c8c50b6c6670f5"),
        ("pair_07", "real", "6f2b7e632c64ba5835164abe2f1ab28e"),
    ]

    base = default_params()
    vectors = _lhs_sample(500, BOUNDS, 20260804)

    all_results = []

    for pid, label, case_id in cases_to_test:
        print(f"\n  [{pid} {label}] Loading real case...", end=" ", flush=True)
        obs_arr, last_data, case = _gen_real_obs(case_id, label)
        n_data = last_data + 1
        split = TemporalSplit.by_fraction(total_steps=last_data, train_fraction=0.7)
        train_end = split.train_end_step
        obs_train = obs_arr[:train_end + 1]

        # Best open-loop baseline (train-only exp decay)
        t_train = np.arange(len(obs_train), dtype=np.float64)
        y_pos = np.maximum(obs_train, 1e-6)
        A = np.column_stack([np.ones_like(t_train), t_train])
        coeff, _, _, _ = np.linalg.lstsq(A, np.log(y_pos), rcond=None)
        bl_a = np.exp(coeff[0])
        bl_lam = max(-coeff[1], 1e-6)
        t_full = np.arange(n_data, dtype=np.float64)
        bl_exp = bl_a * np.exp(-bl_lam * t_full)
        bl_train_rmsle = _rmsle(obs_train, bl_exp[:train_end + 1])
        bl_val_rmsle = _rmsle(obs_arr[train_end + 1:], bl_exp[train_end + 1:])

        print(f"N={len(case.user_ids)} ix={len(case.interactions)} "
              f"train_steps={train_end+1} bl_trainRMSLE={bl_train_rmsle:.4f}")

        best_train_rmsle = float("inf")
        best_vec = None
        top20 = []

        for vi, vec in enumerate(vectors):
            try:
                params = apply_parameter_vector(base, SPECS, list(vec))
            except ValueError:
                continue
            sim = _gen_traj(params, case, (42,), T=last_data)
            train_rmsle = _rmsle(obs_train, sim[:train_end + 1])
            val_rmsle = _rmsle(obs_arr[train_end + 1:], sim[train_end + 1:])
            peak_ratio = float(np.max(sim[:n_data])) / max(float(np.max(obs_arr[:n_data])), 1)
            auc_ratio = float(np.trapezoid(sim[:n_data])) / max(float(np.trapezoid(obs_arr[:n_data])), 1e-6)

            entry = {
                "pid": pid, "label": label,
                "vec_idx": vi,
                "params": {s.path: float(vec[i]) for i, s in enumerate(SPECS)},
                "train_rmsle": float(train_rmsle),
                "val_rmsle": float(val_rmsle),
                "peak_ratio": float(peak_ratio),
                "auc_ratio": float(auc_ratio),
            }

            if train_rmsle < best_train_rmsle:
                best_train_rmsle = train_rmsle
                best_vec = entry

            if train_rmsle < best_train_rmsle * 2:  # keep top candidates
                top20.append(entry)

            if (vi + 1) % 100 == 0:
                print(f"    {vi+1}/500 best_trainRMSLE={best_train_rmsle:.4f}")

        # Sort and keep top 20
        top20.sort(key=lambda x: x["train_rmsle"])
        top20 = top20[:20]

        # Re-evaluate top 20 with 5 seeds
        for e in top20:
            p = apply_parameter_vector(base, SPECS, [e["params"][s.path] for s in SPECS])
            sim5 = _gen_traj(p, case, FIT_SEEDS, T=last_data)
            e["train_rmsle_5seed"] = float(_rmsle(obs_train, sim5[:train_end + 1]))
            e["val_rmsle_5seed"] = float(_rmsle(obs_arr[train_end + 1:], sim5[train_end + 1:]))
            e["peak_ratio_5seed"] = float(np.max(sim5[:n_data])) / max(float(np.max(obs_arr[:n_data])), 1)

        print(f"    Top-20 5seed: best trainRMSLE={min(e['train_rmsle_5seed'] for e in top20):.4f} "
              f"best valRMSLE={min(e['val_rmsle_5seed'] for e in top20):.4f} "
              f"baseline valRMSLE={bl_val_rmsle:.4f}")

        # Check reachability criteria
        reachable = False
        for e in top20:
            pr = e.get("peak_ratio_5seed", e["peak_ratio"])
            ar = e["auc_ratio"]
            if (e.get("val_rmsle_5seed", e["val_rmsle"]) < bl_val_rmsle and
                0.5 <= pr <= 2.0 and 0.5 <= ar <= 2.0):
                reachable = True
                break

        all_results.append({
            "pid": pid, "label": label,
            "bl_train_rmsle": bl_train_rmsle,
            "bl_val_rmsle": bl_val_rmsle,
            "best_train_rmsle": best_train_rmsle,
            "best_val_rmsle": min(e.get("val_rmsle_5seed", e["val_rmsle"]) for e in top20),
            "reachable": reachable,
            "top20": top20[:5],  # save top 5 vectors
        })
        print(f"    REACHABLE: {reachable}")

    return all_results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1
    stability = run_optimizer_stability()

    # Phase 2
    practical = run_practical_recovery()

    # Phase 3
    reachability = run_reachability_scan()

    # ── DECISION REPORT ──
    print("\n" + "=" * 60)
    print("  V2.1 DECISION REPORT")
    print("=" * 60)

    # Stability summary
    st = [r for r in stability if r["phase"] == "stability"]
    n_stability_runs = len(set((r["truth_id"], r["opt_seed"]) for r in st))
    n_all_good = 0
    for vi in range(5):
        vr = [r for r in st if r["truth_id"] == vi + 1]
        all_good = all(
            any(r2["range_norm_error"] < 0.10 for r2 in vr if r2["param"] == s.path)
            for s in SPECS
        )
        if all_good:
            n_all_good += 1
    print(f"\n  Stability: {n_all_good}/5 vectors have all 4 params <0.10")
    for j, spec in enumerate(SPECS):
        vals = [r["range_norm_error"] for r in st if r["param"] == spec.path]
        opt_stds = []
        for vi in range(5):
            ests = [r["estimated"] for r in st if r["truth_id"] == vi + 1 and r["param"] == spec.path]
            if len(ests) > 1:
                opt_stds.append(np.std(ests) / (spec.high - spec.low))
        print(f"    {spec.path}: median_err={np.median(vals):.4f} p90={np.percentile(vals, 90):.4f} "
              f"cross_seed_std={np.mean(opt_stds):.4f}")

    # Practical summary
    pr = [r for r in practical if r["phase"] == "practical"]
    print(f"\n  Practical recovery ({len(set(r['rep'] for r in pr))} reps):")
    for spec in SPECS:
        vals = [r["range_norm_error"] for r in pr if r["param"] == spec.path]
        print(f"    {spec.path}: median_err={np.median(vals):.4f} p90={np.percentile(vals, 90):.4f}")

    # Reachability summary
    print(f"\n  Real-trajectory reachability:")
    n_reachable = sum(1 for r in reachability if r["reachable"])
    for r in reachability:
        print(f"    {r['pid']} {r['label']}: reachable={r['reachable']} "
              f"bestValRMSLE={r['best_val_rmsle']:.4f} vs blValRMSLE={r['bl_val_rmsle']:.4f}")
    print(f"    {n_reachable}/{len(reachability)} cases reachable")

    # Decision
    print("\n  DECISION:")
    stability_ok = n_stability_runs >= 10
    practical_ok = np.median([np.median([e["range_norm_error"] for e in pr if e["param"] == s.path]) for s in SPECS]) < 0.15 if pr else False
    reachable_ok = n_reachable >= 2

    if stability_ok and reachable_ok:
        print("  ✓ Proceed to trial calibration (2 pairs, 4 cases)")
        if np.percentile([e["range_norm_error"] for e in pr if e["param"] == "viral.beta_V"], 90) > 0.30:
            print("  ⚠ beta_V P90 > 0.30 — consider 3-param model for calibration")
    elif stability_ok and not reachable_ok:
        print("  ✗ Real trajectories NOT reachable with current model structure")
        print("  → Investigate U→E→A scheduling, root-post shock, or observation model")
    else:
        print("  ⚠ Marginal — re-evaluate after further experiments")

    # Save results
    all_data = {
        "stability": stability,
        "practical": practical,
        "reachability": [{
            k: v for k, v in r.items() if k != "top20"
        } for r in reachability],
        "reachability_top5": [{
            "pid": r["pid"], "label": r["label"],
            "top5": r.get("top20", [])[:5],
        } for r in reachability],
    }
    with open(OUT_DIR / "v21_decision_report.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\n  Report: {OUT_DIR / 'v21_decision_report.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
