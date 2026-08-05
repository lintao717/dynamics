"""[V1.7R.3] 5-Fold Cross-Validation by Pair.

Folds: [1,2] [3,4] [5,6] [7,8] [9,10]
Train: estimate shared alpha_0, gamma_0 on 8 training pairs.
Test: per-event beta_M, evaluate forecast on 2 held-out pairs.
ONE_SHOT, oracle_cohort, 48h->72h + 72h->48h.
"""

import json, subprocess, sys
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamics_simulation.config import default_params, ReactivationMode
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.observations import build_observed_trajectory
from dynamics_simulation.data.forecast import EventHistory, ForecastTarget
from dynamics_simulation.forecast_runner import ForecastConfig, ForecastRunner
from dynamics_simulation.evaluation import rmsle
from dynamics_simulation.baseline import BaselineForecast

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "forecast"
DATA = Path("data/raw/CHECKED/dataset")
STEP_H = 24.0
FORECAST_SEEDS = (101, 103, 107, 109, 113)

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
FOLDS = [(0,1),(2,3),(4,5),(6,7),(8,9)]

BETA_GRID = [0.3, 0.6, 0.9, 1.5, 2.0]
ALPHA_GRID = [-2.0, 0.0, 1.0, 2.0]
GAMMA_GRID = [0.0, 3.0, 6.0, 9.0, 12.0]


def _fit_event(history, case, base, bM, a0, g0):
    """Compute train loss for one (case, cutoff) with given params."""
    from dynamics_simulation.replay.config import ReplayConfig
    from dynamics_simulation.replay.runner import run_replay
    from dynamics_simulation.data.networks import ReplayNetworkMode
    p = replace(base,
        propagation=replace(base.propagation, beta_M=bM),
        activation=replace(base.activation, alpha_0=a0),
        decay=replace(base.decay, gamma_0=g0))
    losses = []
    for s in [11, 23, 37, 53, 71]:  # multi-seed
        cfg = ReplayConfig(step_hours=STEP_H, tail_steps=0,
            network_mode=ReplayNetworkMode.BROADCAST, seeds=(s,),
            micro_steps=1, reactivation_mode=ReactivationMode.ONE_SHOT.value)
        r = run_replay(case, p, cfg)
        if not r.simulated_mean: continue
        traj = build_observed_trajectory(case, NodeIndex.from_case(case),
            TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0))
        obs = np.array(traj.active_count, dtype=np.float64)
        n_pre = history.cutoff_step + 1
        sim = np.array(r.simulated_mean.get("n_A_ts",[0]), dtype=np.float64)
        if len(sim) > n_pre: sim = sim[:n_pre]
        elif len(sim) < n_pre: sim = np.pad(sim, (0,n_pre-len(sim)), mode="edge")
        losses.append(rmsle(obs[:n_pre], sim[:len(obs[:n_pre])]))
    return np.median(losses) if losses else 9.0


def _eval_forecast(history, case, bM, a0, g0, cutoff_step, horizon_steps,
                   target, is_zero_tail):
    """Run forecast and return metrics."""
    base = default_params()
    params = replace(base,
        propagation=replace(base.propagation, beta_M=bM),
        activation=replace(base.activation, alpha_0=a0),
        decay=replace(base.decay, gamma_0=g0))
    fcfg = ForecastConfig(cutoff_step=cutoff_step, horizon_steps=horizon_steps,
        step_hours=STEP_H, population_mode="oracle_cohort",
        reactivation_mode=ReactivationMode.ONE_SHOT, forecast_seeds=FORECAST_SEEDS)
    result = ForecastRunner(fcfg).run(history, case, params)

    traj = build_observed_trajectory(case, NodeIndex.from_case(case),
        TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0))
    obs_arr = np.array(traj.active_count, dtype=np.float64)
    hist_obs = obs_arr[:cutoff_step+1]
    n_data = len(obs_arr)
    bf = BaselineForecast(hist_obs, cutoff_step+1+horizon_steps) if len(hist_obs)>=2 else None
    obs_fut = np.array([obs_arr[cutoff_step+h] for h in range(1, horizon_steps+1)])
    bl_zero = rmsle(obs_fut, np.zeros(horizon_steps))
    bl_best = bl_zero
    if bf:
        for bp in [bf.persistence_last[:horizon_steps],
                   bf.exp_decay_future[:horizon_steps],
                   bf.pulse_decay_future[:horizon_steps]]:
            if len(bp) < horizon_steps: continue
            br = rmsle(obs_fut, bp)
            if br < bl_best: bl_best = br
    fc_rmsle = rmsle(obs_fut, result.fc_active)
    pr = float(np.max(result.fc_active))/max(float(np.max(obs_fut)),1)
    if is_zero_tail:
        win = np.allclose(result.fc_active, 0, atol=0.01)
    else:
        win = fc_rmsle < bl_best
    return {"fc_rmsle": round(fc_rmsle,4), "bl_best": round(bl_best,4),
            "win": win, "peak_ratio": round(pr,4), "is_zero_tail": is_zero_tail}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    all_folds = []

    for fold_idx, (fi, fj) in enumerate(FOLDS):
        test_pair_indices = [fi, fj]
        train_pair_indices = [i for i in range(10) if i not in test_pair_indices]
        test_pairs = [PAIRS[i] for i in test_pair_indices]
        train_pairs = [PAIRS[i] for i in train_pair_indices]

        print(f"\n{'='*60}\n  FOLD {fold_idx+1}/5: test={[p[0] for p in test_pairs]}")
        print(f"{'='*60}")

        # ── Train shared a0, g0 on all training events ──
        best_loss = float("inf")
        best_a0, best_g0 = 0.0, 0.0
        for a0 in ALPHA_GRID:
            for g0 in GAMMA_GRID:
                fold_losses = []
                for pid, fid, rid in train_pairs:
                    for lk, cid in [("fake",fid),("real",rid)]:
                        try:
                            case = load_checked_case(DATA/f"{lk}_news/{cid}.json")
                        except Exception: continue
                        for cutoff_h in [48, 72]:
                            cs = cutoff_h//int(STEP_H)
                            try: history = EventHistory.from_case(case, cs, STEP_H)
                            except ValueError: continue
                            # Per-event beta_M search
                            best_bm, best_el = 0.5, float("inf")
                            for bM in BETA_GRID:
                                el = _fit_event(history, case, base, bM, a0, g0)
                                if el < best_el: best_el = el; best_bm = bM
                            fold_losses.append(best_el)
                if fold_losses:
                    med_loss = np.median(fold_losses)
                    if med_loss < best_loss: best_loss = med_loss; best_a0 = a0; best_g0 = g0

        print(f"  Trained: a0={best_a0:.1f} g0={best_g0:.1f} loss={best_loss:.4f}")

        # ── Evaluate on test pairs ──
        fold_results = {"fold": fold_idx+1, "test_pairs": [p[0] for p in test_pairs],
                        "best_a0": best_a0, "best_g0": best_g0, "forecasts": []}
        for pid, fid, rid in test_pairs:
            for lk, cid in [("fake",fid),("real",rid)]:
                try: case = load_checked_case(DATA/f"{lk}_news/{cid}.json")
                except Exception: continue
                for cutoff_h in [48, 72]:
                    cs = cutoff_h//int(STEP_H); hs = (72 if cutoff_h==48 else 48)//int(STEP_H)
                    try:
                        history = EventHistory.from_case(case, cs, STEP_H)
                        target = ForecastTarget.from_case(case, cs, tuple(range(1,hs+1)), STEP_H)
                    except ValueError: continue
                    # Per-event beta_M
                    best_bm = 0.5; best_el = float("inf")
                    for bM in BETA_GRID:
                        el = _fit_event(history, case, base, bM, best_a0, best_g0)
                        if el < best_el: best_el = el; best_bm = bM
                    traj = build_observed_trajectory(case, NodeIndex.from_case(case),
                        TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0))
                    obs_fut = np.array([traj.active_count[cs+h] for h in range(1,hs+1)])
                    is_zt = np.all(obs_fut == 0)
                    ev = _eval_forecast(history, case, best_bm, best_a0, best_g0,
                                        cs, hs, target, is_zt)
                    ev.update({"pid":pid, "label":lk, "cutoff_h":cutoff_h,
                               "bM":best_bm, "a0":best_a0, "g0":best_g0})
                    fold_results["forecasts"].append(ev)
                    print(f"    {pid} {lk} c={cutoff_h}: "
                          f"{'WIN' if ev['win'] else 'LOSE'} "
                          f"fc={ev['fc_rmsle']:.4f} bl={ev['bl_best']:.4f} "
                          f"bM={best_bm:.2f} pr={ev['peak_ratio']:.2f}")
        all_folds.append(fold_results)

    # Summary
    print(f"\n{'='*60}\n  CROSS-VALIDATION SUMMARY\n{'='*60}")
    total_wins = 0; total_active = 0
    for fold in all_folds:
        fw = [f for f in fold["forecasts"]]
        wins = sum(1 for f in fw if f["win"])
        active = [f for f in fw if not f["is_zero_tail"]]
        aw = sum(1 for f in active if f["win"])
        print(f"  Fold {fold['fold']}: {wins}/{len(fw)} total, {aw}/{len(active)} active wins "
              f"a0={fold['best_a0']:.1f} g0={fold['best_g0']:.1f}")
        total_wins += wins; total_active += len(active)
    print(f"  Overall: {total_wins}/{sum(len(f['forecasts']) for f in all_folds)} total, "
          f"{sum(1 for f in all_folds for ff in f['forecasts'] if ff['win'] and not ff['is_zero_tail'])}"
          f"/{total_active} active wins")

    save_path = OUT / "v17r3_crossval.json"
    save_path.write_text(json.dumps(all_folds, indent=2, default=str))
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
