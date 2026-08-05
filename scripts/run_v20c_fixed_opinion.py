"""[V2.0-C Fixed] Opinion calibration with corrected metrics.

Fixes:
  - Use o_hat_mean (public) vs stance_mean (observed public)
  - Mask empty windows (no observations -> excluded from loss)
  - Correct correlation loss (penalises negative corr)
  - Use o_polarization_ts for polarization comparison
  - Export full results to JSON
"""

import json, sys
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamics_simulation.config import default_params
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.semantics.annotator import SemanticAnnotator
from dynamics_simulation.semantics.aggregation import aggregate_window

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "opinion"
DATA = Path("data/raw/CHECKED/dataset")
STEP_H = 24.0
CASE_ID = "25d9ed3994c2d5f030b864867facab47"
CASE_LABEL = "fake"


def _extract_observed(case, grid):
    annotator = SemanticAnnotator()
    by_step = {}
    for ix in case.interactions:
        step = grid.step_of(ix.timestamp)
        sig = annotator.annotate(ix.interaction_id, ix.user_id, step, ix.text)
        by_step.setdefault(step, []).append(sig)
    T = grid.last_data_step + 1
    return {step: aggregate_window(by_step.get(step, []), step) for step in range(T)}


def _simulate(case, params):
    cfg = ReplayConfig(step_hours=STEP_H, tail_steps=0,
                       network_mode=ReplayNetworkMode.CUMULATIVE_INTERACTION,
                       seeds=(42,), micro_steps=1,
                       reactivation_mode="one_shot")
    result = run_replay(case, params, cfg)
    if not result.simulated_mean: return None
    sm = result.simulated_mean
    T = len(sm.get("o_mean_ts", []))
    return {
        # Public expression (comparable to observed stance from comments)
        "o_hat_mean": np.array(sm.get("o_hat_mean_ts", [0]))[:T],
        "o_hat_std": np.array(sm.get("o_hat_std_ts", [0]))[:T],
        # Private opinion (diagnostic only)
        "o_mean": np.array(sm.get("o_mean_ts", [0]))[:T],
        "o_std": np.array(sm.get("o_std_ts", [0]))[:T],
        "o_pol": np.array(sm.get("o_polarization_ts", [0]))[:T],
        "public_bias": np.array(sm.get("public_bias_ts", [0]))[:T],
    }


def _score(obs, sim):
    """Score simulation against observed, only on valid (non-empty) windows."""
    T = max(len(obs), 0)
    if sim is None or T == 0: return 9.0, {}

    valid = np.array([obs[s].n_interactions > 0 for s in range(min(T, len(obs)))])
    if len(valid) < T: valid = np.pad(valid, (0, T - len(valid)), constant_values=False)
    if not valid.any(): return 9.0, {}

    # Public stance: compare o_hat_mean (public expression) vs stance_mean (observed)
    obs_stance = np.array([obs[s].stance_mean for s in range(T)])
    sim_stance = sim["o_hat_mean"][:T]
    stance_mae = np.mean(np.abs(obs_stance[valid] - sim_stance[valid]))

    # Correlation: penalise negative correlation (don't use abs)
    if np.std(sim_stance[valid]) > 1e-6 and np.std(obs_stance[valid]) > 1e-6:
        corr = np.corrcoef(sim_stance[valid], obs_stance[valid])[0, 1]
        corr_loss = (1.0 - corr) / 2.0  # corr=1->0, corr=0->0.5, corr=-1->1
    else:
        corr_loss = 0.5

    # Polarization
    obs_pol = np.array([obs[s].polarization for s in range(T)])
    sim_pol = sim["o_pol"][:T]
    pol_mae = np.mean(np.abs(obs_pol[valid] - sim_pol[valid]))

    # Support/oppose ratio JS distance
    obs_sup = np.array([obs[s].support_ratio for s in range(T)])
    sim_arr = sim["o_hat_mean"][:T]
    sim_sup = np.clip((sim_arr + 1) / 2, 0, 1)  # proxy: stance [-1,1] -> [0,1]
    js = 0.5 * np.mean((obs_sup[valid] - sim_sup[valid]) ** 2)

    total = float(0.3 * stance_mae + 0.3 * corr_loss + 0.2 * pol_mae + 0.2 * js)
    details = {"stance_mae": round(stance_mae, 4), "corr": round(corr if 'corr' in dir() else 0, 4),
               "corr_loss": round(corr_loss, 4), "pol_mae": round(pol_mae, 4),
               "js_dist": round(js, 4), "n_valid": int(valid.sum())}
    return total, details


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    case = load_checked_case(DATA / f"{CASE_LABEL}_news" / f"{CASE_ID}.json")
    grid = TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)
    obs = _extract_observed(case, grid)
    print(f"Case: {CASE_ID}, {len(case.interactions)} ix, {grid.last_data_step+1} steps")

    results = []
    best, best_params = float("inf"), None
    for mu in [0.05, 0.1, 0.2, 0.3]:
        for zeta in [0.2, 0.4, 0.6, 0.8]:
            for lam_c in [0.1, 0.3, 0.5]:
                for lam_h in [0.05, 0.1, 0.2]:
                    for lam_spiral in [0.1, 0.3]:
                        p = replace(base, opinion=replace(base.opinion,
                            mu_mean=mu, zeta_mean=zeta,
                            lambda_c=lam_c, lambda_h=lam_h,
                            lambda_spiral=lam_spiral))
                        sim = _simulate(case, p)
                        score, detail = _score(obs, sim)
                        entry = {"mu": mu, "zeta": zeta, "lam_c": lam_c,
                                 "lam_h": lam_h, "lam_spiral": lam_spiral,
                                 "score": round(score, 4), **detail}
                        results.append(entry)
                        if score < best: best = score; best_params = entry

    results.sort(key=lambda x: x["score"])
    print(f"\nBest: {best_params}")
    print(f"Top 5:")
    for r in results[:5]: print(f"  {r}")

    # Full comparison at best
    best_p = replace(base, opinion=replace(base.opinion,
        mu_mean=best_params["mu"], zeta_mean=best_params["zeta"],
        lambda_c=best_params["lam_c"], lambda_h=best_params["lam_h"],
        lambda_spiral=best_params["lam_spiral"]))
    best_sim = _simulate(case, best_p)

    T = grid.last_data_step + 1
    for step in range(T):
        if obs[step].n_interactions == 0: continue
        o = obs[step]
        oh = best_sim["o_hat_mean"][step] if step < len(best_sim["o_hat_mean"]) else 0
        ostd = best_sim["o_hat_std"][step] if step < len(best_sim["o_hat_std"]) else 0
        os = best_sim["o_std"][step] if step < len(best_sim["o_std"]) else 0
        print(f"  Step {step}: obs_stance={o.stance_mean:.3f} sim_o_hat={oh:.3f} "
              f"obs_pol={o.polarization:.3f} sim_pol={best_sim['o_pol'][step]:.3f} "
              f"priv_std={os:.3f} o_hat_std={ostd:.3f}")

    save_path = OUT / "v20c_fixed_opinion.json"
    save_path.write_text(json.dumps({"best": best_params, "results": results[:10]}, indent=2))
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
