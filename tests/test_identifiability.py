"""
Structural Identifiability Test for V1.1.

Tests whether the 7 P0 parameters can be recovered from model-generated
synthetic data via grid-search minimization of A(t) trajectory MSE.

Method:
  1. Set known true parameters theta*
  2. Generate synthetic A(t) trajectories (3 seeds, averaged)
  3. Grid-search over each parameter to minimize MSE
  4. Report relative recovery error and parameter correlations

Parameters tested:
  beta, alpha_0, alpha_1, gamma_0, r_0_0, r_0_1, r_1_1
"""
import sys, os, json, time, itertools
from pathlib import Path
import numpy as np
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamics_simulation.config import (
    ModelParams, default_params,
    PropagationParams, ActivationParams, DecayParams,
    ReactivationParams, OpinionParams,
)
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs

OUTPUT = Path(__file__).parent.parent / "data" / "identifiability_results.json"


def generate_trajectory(params, G_s, G_o, n, T, n_seeds=3, base_seed=42):
    """Generate averaged A(t) trajectory for given parameters."""
    all_A = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s * 100)
        state = initialize_agents(n=n, initial_active=10, rng=rng,
                                  initial_opinion_dist="polarized")
        engine = TransitionEngine(params, rng)
        o_init = state.o.copy()
        V = 0.0
        A_ts = []
        for t in range(T):
            inputs = ExternalInputs(V=V)
            state, V, events = engine.step(state, G_s, G_o, None, inputs, o_init, t)
            A_ts.append(state.n_A)
        all_A.append(np.array(A_ts))
    return np.mean(all_A, axis=0)


def grid_search_single_param(param_name, true_val, true_params, G_s, G_o, n, T,
                              grid_vals, observed_A):
    """Grid search over a single parameter, keeping others at true values."""
    best_mse = float('inf')
    best_val = None
    mse_curve = []

    for gv in grid_vals:
        p = _set_param(true_params, param_name, gv)
        sim_A = generate_trajectory(p, G_s, G_o, n, T, n_seeds=2, base_seed=99)
        mse = float(np.mean((sim_A - observed_A) ** 2))
        mse_curve.append((float(gv), mse))
        if mse < best_mse:
            best_mse = mse
            best_val = float(gv)

    rel_err = abs(best_val - true_val) / max(abs(true_val), 0.01)
    return {
        "param": param_name,
        "true": float(true_val),
        "estimated": float(best_val),
        "relative_error": round(rel_err, 4),
        "recoverable": rel_err < 0.30,
        "mse_curve": mse_curve,
    }


def _set_param(params, name, value):
    """Return a copy of params with the named parameter changed."""
    mapping = {
        "beta": ("propagation", "beta"),
        "beta_M": ("propagation", "beta_M"),
        "alpha_0": ("activation", "alpha_0"),
        "alpha_1": ("activation", "alpha_1"),
        "alpha_2": ("activation", "alpha_2"),
        "alpha_3": ("activation", "alpha_3"),
        "alpha_4": ("activation", "alpha_4"),
        "alpha_5": ("activation", "alpha_5"),
        "gamma_0": ("decay", "gamma_0"),
        "gamma_1": ("decay", "gamma_1"),
        "gamma_2": ("decay", "gamma_2"),
        "gamma_3": ("decay", "gamma_3"),
        "r_0_0": ("reactivation", "r_0_0"),
        "r_1_0": ("reactivation", "r_1_0"),
        "r_0_1": ("reactivation", "r_0_1"),
        "r_1_1": ("reactivation", "r_1_1"),
        "r_2": ("reactivation", "r_2"),
        "r_3": ("reactivation", "r_3"),
    }
    section, field = mapping[name]
    section_obj = getattr(params, section)
    new_section = replace(section_obj, **{field: value})
    return replace(params, **{section: new_section})


def grid_search_pair(param1, true_val1, param2, true_val2, true_params,
                     G_s, G_o, n, T, grid1, grid2, observed_A):
    """Grid search over TWO parameters simultaneously. Detects confounding."""
    best_mse = float('inf')
    best_pair = (None, None)
    mse_grid = np.zeros((len(grid1), len(grid2)))

    for i, g1 in enumerate(grid1):
        for j, g2 in enumerate(grid2):
            p = _set_param(true_params, param1, g1)
            p = _set_param(p, param2, g2)
            sim_A = generate_trajectory(p, G_s, G_o, n, T, n_seeds=1, base_seed=99)
            mse = float(np.mean((sim_A - observed_A) ** 2))
            mse_grid[i, j] = mse
            if mse < best_mse:
                best_mse = mse
                best_pair = (float(g1), float(g2))

    err1 = abs(best_pair[0] - true_val1) / max(abs(true_val1), 0.01)
    err2 = abs(best_pair[1] - true_val2) / max(abs(true_val2), 0.01)

    # Correlation: can we trade off param1 for param2?
    # Check if the MSE valley is diagonal (confounded) or orthogonal (independent)
    # Flatten: find all (g1,g2) pairs with MSE within 10% of best
    mse_flat = mse_grid.flatten()
    best_idx = mse_flat.argmin()
    threshold = best_mse * 1.1
    good_pairs = [(float(grid1[i]), float(grid2[j]))
                  for i in range(len(grid1)) for j in range(len(grid2))
                  if mse_grid[i, j] <= threshold]
    g1_vals = [p[0] for p in good_pairs]
    g2_vals = [p[1] for p in good_pairs]
    # If there's a trade-off, g1 and g2 within the good region will be correlated
    corr = float(np.corrcoef(g1_vals, g2_vals)[0, 1]) if len(good_pairs) >= 4 else 0.0
    confounded = abs(corr) > 0.5 and len(good_pairs) >= 4

    return {
        "param1": param1, "true1": float(true_val1), "est1": best_pair[0], "err1": round(err1, 4),
        "param2": param2, "true2": float(true_val2), "est2": best_pair[1], "err2": round(err2, 4),
        "confounding_correlation": round(float(corr), 4),
        "confounded": confounded,
        "mse_grid_shape": list(mse_grid.shape),
        "best_mse": round(float(best_mse), 2),
    }


def main():
    print("=" * 60)
    print("  STRUCTURAL IDENTIFIABILITY TEST")
    print("=" * 60)

    n, T = 200, 30
    base_seed = 42
    rng = np.random.default_rng(base_seed)

    # Fixed network for all tests
    G_s, G_o, _ = generate_networks("sbm", n=n, rng=rng,
        n_blocks=2, p_in=0.20, p_out=0.03)

    # ── True parameters ──
    true_params = default_params()
    true_params = replace(true_params,
        propagation=PropagationParams(beta=0.18, beta_M=0.0),
        activation=ActivationParams(alpha_0=-2.5, alpha_1=1.5, alpha_2=2.0,
                                     alpha_3=1.8, alpha_4=1.5, alpha_5=1.0),
        decay=DecayParams(gamma_0=0.0, gamma_1=2.0, gamma_2=1.5, gamma_3=1.2),
        reactivation=ReactivationParams(
            r_0_0=-3.0, r_1_0=2.5, r_0_1=-4.0, r_1_1=3.5, r_2=1.0, r_3=1.2),
    )

    # Generate observed trajectory
    print(f"\nGenerating synthetic data (N={n}, T={T}, 5 seeds)...")
    observed_A = generate_trajectory(true_params, G_s, G_o, n, T, n_seeds=5,
                                     base_seed=42)
    peak_A = int(observed_A.max())
    mid_A = float(observed_A[T//2])
    late_A = float(observed_A[-5:].mean())
    print(f"  Observed: peak_A={peak_A}, mid_A={mid_A:.0f}, late_A={late_A:.0f}")

    results = {"config": {"n": n, "T": T}, "observed_summary": {
        "peak_A": peak_A, "mid_A": round(mid_A, 1), "late_A": round(late_A, 1)}}

    # ── Single-parameter recovery ──
    print(f"\n── Single-Parameter Recovery ──")
    param_grids = {
        "beta": np.linspace(0.05, 0.35, 7),
        "alpha_0": np.linspace(-4.0, -1.0, 7),
        "alpha_1": np.linspace(0.5, 3.0, 6),
        "gamma_0": np.linspace(-2.0, 2.0, 9),
        "r_0_0": np.linspace(-5.0, -1.0, 9),
        "r_0_1": np.linspace(-6.0, -2.0, 9),
        "r_1_1": np.linspace(1.0, 5.0, 9),
    }
    true_values = {
        "beta": 0.18, "alpha_0": -2.5, "alpha_1": 1.5,
        "gamma_0": 0.0, "r_0_0": -3.0, "r_0_1": -4.0, "r_1_1": 3.5,
    }

    single_results = {}
    for param_name in param_grids:
        result = grid_search_single_param(
            param_name, true_values[param_name], true_params,
            G_s, G_o, n, T, param_grids[param_name], observed_A,
        )
        single_results[param_name] = result
        status = "RECOVERABLE" if result["recoverable"] else "CONFOUNDED"
        print(f"  {param_name:10s}: true={result['true']:+.3f} est={result['estimated']:+.3f} "
              f"err={result['relative_error']:.3f} [{status}]")

    results["single_param"] = {k: {
        "true": v["true"], "estimated": v["estimated"],
        "relative_error": v["relative_error"], "recoverable": v["recoverable"],
    } for k, v in single_results.items()}

    # ── Pairwise confounding detection ──
    print(f"\n── Pairwise Confounding Detection ──")
    # Test key pairs where confounding is likely
    pairs_to_test = [
        ("beta", "alpha_0"),     # both affect propagation amplitude
        ("beta", "alpha_1"),     # both affect propagation amplitude
        ("alpha_0", "alpha_1"),  # both affect activation probability
        ("gamma_0", "r_0_1"),    # both affect late-stage A count
        ("r_0_0", "r_0_1"),      # both are baseline reactivation
    ]

    pair_results = {}
    for p1, p2 in pairs_to_test:
        result = grid_search_pair(
            p1, true_values[p1], p2, true_values[p2], true_params,
            G_s, G_o, n, T,
            param_grids[p1], param_grids[p2], observed_A,
        )
        pair_results[f"{p1}_{p2}"] = result
        conf = "CONFOUNDED" if result["confounded"] else "independent"
        print(f"  {p1:8s} x {p2:8s}: err1={result['err1']:.3f} err2={result['err2']:.3f} "
              f"corr={result['confounding_correlation']:.3f} [{conf}]")

    results["pairwise"] = pair_results

    # ── Summary ──
    n_recoverable = sum(1 for v in single_results.values() if v["recoverable"])
    n_total = len(single_results)
    n_confounded_pairs = sum(1 for v in pair_results.values() if v["confounded"])

    print(f"\n{'='*60}")
    print(f"  IDENTIFIABILITY SUMMARY")
    print(f"  Single-param recoverable: {n_recoverable}/{n_total}")
    print(f"  Confounded pairs: {n_confounded_pairs}/{len(pair_results)}")

    if n_recoverable >= 5 and n_confounded_pairs <= 2:
        verdict = "GOOD — most parameters are individually identifiable"
    elif n_recoverable >= 3:
        verdict = "MODERATE — some parameters confounded; composite estimates needed"
    else:
        verdict = "POOR — severe confounding; model needs simplification or stronger priors"

    print(f"  Verdict: {verdict}")

    for name, r in single_results.items():
        s = "OK" if r["recoverable"] else "XX"
        print(f"    [{s}] {name}")

    print(f"{'='*60}")

    results["summary"] = {
        "n_recoverable": n_recoverable,
        "n_total": n_total,
        "n_confounded_pairs": n_confounded_pairs,
        "verdict": verdict,
    }

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
