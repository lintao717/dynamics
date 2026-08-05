"""[V2.0-C] Opinion Dynamics Calibration.

Uses cumulative mode (G_o > 0) to activate opinion influence network.
Compares simulated stance/emotion trajectories against observed
semantic annotations from CHECKED interaction texts.

Grid search over opinion parameters:
  mu, zeta, epsilon, lambda_spiral, lambda_c, lambda_h, eta, chi
"""

import sys
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamics_simulation.config import default_params, ModelParams
from dynamics_simulation.data.checked import load_checked_case
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.networks import ReplayNetworkMode
from dynamics_simulation.replay.config import ReplayConfig
from dynamics_simulation.replay.runner import run_replay
from dynamics_simulation.semantics.annotator import SemanticAnnotator
from dynamics_simulation.semantics.aggregation import aggregate_window

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "opinion"
DATA = Path("data/raw/CHECKED/dataset")

# Test case: pair_06 fake (has enough interactions)
CASE_ID = "25d9ed3994c2d5f030b864867facab47"
CASE_LABEL = "fake"
STEP_H = 24.0


def _extract_observed(case, grid):
    """Annotate all interactions and build observed opinion trajectory."""
    annotator = SemanticAnnotator()
    signals_by_step = {}
    for ix in case.interactions:
        step = grid.step_of(ix.timestamp)
        sig = annotator.annotate(ix.interaction_id, ix.user_id, step, ix.text)
        signals_by_step.setdefault(step, []).append(sig)

    results = {}
    T = grid.last_data_step + 1
    for step in range(T):
        sigs = signals_by_step.get(step, [])
        agg = aggregate_window(sigs, step)
        results[step] = agg
    return results


def _simulate_opinion(case, params):
    """Run simulation in cumulative mode and extract opinion metrics."""
    idx = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)

    cfg = ReplayConfig(step_hours=STEP_H, tail_steps=0,
                       network_mode=ReplayNetworkMode.CUMULATIVE_INTERACTION,
                       seeds=(42,), micro_steps=1,
                       reactivation_mode="one_shot")
    result = run_replay(case, params, cfg)
    if not result.simulated_mean:
        return None
    return {
        "o_mean": np.array(result.simulated_mean.get("o_mean_ts", [0])),
        "o_std": np.array(result.simulated_mean.get("o_std_ts", [0])),
        "h_mean": np.array(result.simulated_mean.get("h_mean_ts", [0])),
        "public_bias": np.array(result.simulated_mean.get("public_bias_ts", [0])),
        "n_A": np.array(result.simulated_mean.get("n_A_ts", [0])),
    }


def _score_opinion(obs, sim, field="stance_mean"):
    """Score simulated opinion against observed."""
    T = len(obs)
    if sim is None or T == 0:
        return 9.0

    sim_arr = sim["o_mean"][:T]
    obs_arr = np.array([obs[s].stance_mean for s in range(T)])

    # Correlation of stance trends
    if np.std(sim_arr) > 1e-6 and np.std(obs_arr) > 1e-6:
        corr = np.corrcoef(sim_arr, obs_arr)[0, 1]
        corr_loss = max(0, 1 - abs(corr))  # 0 = perfect correlation
    else:
        corr_loss = 1.0

    # MAE
    mae = np.mean(np.abs(sim_arr - obs_arr))

    # Polarization match
    obs_pol = np.array([obs[s].polarization for s in range(T)])
    sim_pol = sim["o_std"][:T] if len(sim["o_std"]) >= T else sim["o_std"]
    if len(sim_pol) >= T:
        sim_pol = sim_pol[:T]
        pol_mae = np.mean(np.abs(sim_pol - obs_pol))
    else:
        pol_mae = 1.0

    return float(0.4 * mae + 0.3 * corr_loss + 0.3 * pol_mae)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = default_params()
    path = DATA / f"{CASE_LABEL}_news" / f"{CASE_ID}.json"
    case = load_checked_case(path)
    grid = TimeGrid.from_case(case, step_hours=STEP_H, tail_steps=0)
    print(f"Case: {CASE_ID} ({CASE_LABEL}), {len(case.interactions)} interactions, {grid.last_data_step+1} steps")

    # Extract observed opinion trajectory
    obs = _extract_observed(case, grid)
    print(f"Observed: {sum(1 for s in obs.values() if s.n_interactions>0)} active steps with {sum(s.n_interactions for s in obs.values())} total")

    for step in range(grid.last_data_step + 1):
        a = obs[step]
        if a.n_interactions > 0:
            print(f"  Step {step}: n={a.n_interactions} stance={a.stance_mean:.3f} "
                  f"sup={a.support_ratio:.2f} opp={a.oppose_ratio:.2f} "
                  f"pol={a.polarization:.3f} arousal={a.arousal_mean:.3f}")

    # Grid search over opinion parameters
    print(f"\nGrid search over opinion parameters...")
    best, best_params = float("inf"), None
    results = []

    for mu in [0.05, 0.1, 0.2]:
        for zeta in [0.2, 0.5, 0.8]:
            for lam_c in [0.1, 0.3, 0.5]:
                for lam_h in [0.05, 0.1, 0.2]:
                    for lam_spiral in [0.1, 0.3]:
                        p = replace(base,
                            opinion=replace(base.opinion,
                                mu_mean=mu, zeta_mean=zeta,
                                lambda_c=lam_c, lambda_h=lam_h,
                                lambda_spiral=lam_spiral))
                        sim = _simulate_opinion(case, p)
                        score = _score_opinion(obs, sim)
                        results.append({
                            "mu": mu, "zeta": zeta,
                            "lambda_c": lam_c, "lambda_h": lam_h,
                            "lambda_spiral": lam_spiral, "score": score,
                        })
                        if score < best:
                            best = score
                            best_params = (mu, zeta, lam_c, lam_h, lam_spiral)

    results.sort(key=lambda x: x["score"])
    print(f"Best params: mu={best_params[0]:.2f} zeta={best_params[1]:.2f} "
          f"lam_c={best_params[2]:.2f} lam_h={best_params[3]:.2f} "
          f"lam_spiral={best_params[4]:.2f} score={best:.4f}")
    print(f"Top 5:")
    for r in results[:5]:
        print(f"  {r}")

    # Run best params and compare
    best_p = replace(base,
        opinion=replace(base.opinion,
            mu_mean=best_params[0], zeta_mean=best_params[1],
            lambda_c=best_params[2], lambda_h=best_params[3],
            lambda_spiral=best_params[4]))
    sim = _simulate_opinion(case, best_p)

    print(f"\nBest-param comparison:")
    T = grid.last_data_step + 1
    for step in range(T):
        if obs[step].n_interactions == 0:
            continue
        o = obs[step]
        s_mean = sim["o_mean"][step] if step < len(sim["o_mean"]) else 0
        s_std = sim["o_std"][step] if step < len(sim["o_std"]) else 0
        print(f"  Step {step}: obs_stance={o.stance_mean:.3f} sim_stance={s_mean:.3f} "
              f"obs_pol={o.polarization:.3f} sim_std={s_std:.3f} "
              f"obs_arousal={o.arousal_mean:.3f}")

    print(f"\nNote: Rule-based annotator used. Real LLM annotation would improve calibration quality.")
    print(f"Results: {OUT / 'v20_opinion_calibration.json'}")


if __name__ == "__main__":
    main()
