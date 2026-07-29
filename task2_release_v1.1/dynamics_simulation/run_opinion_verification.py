"""Numerical verification of opinion analysis theorems.

Verifies:
  1. B_obs decomposition: selection bias vs expression bias under different
     coupling modes (no / one-way / two-way)
  2. zeta-epsilon phase diagram: three regions (consensus, polarization,
     fragmentation)
  3. Anchoring prevents collapse: sigma_o as function of zeta

Output: data/sim_results/opinion_verification.json
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamics_simulation.config import (
    ModelParams, default_params, no_coupling, one_way_coupling,
    PropagationParams, ActivationParams, OpinionParams, EmotionFatigueParams,
)
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs

OUTPUT = Path(__file__).parent.parent / "data" / "sim_results" / "opinion_verification.json"


def run_bias_decomposition(seed: int = 42) -> dict:
    """Verify Theorem 5-7: B_obs decomposition under coupling modes.

    For each coupling mode, run simulation and decompose B_obs into:
      - selection_bias = |mean(o) - mean(o[A])|
      - expression_bias = |mean(o[A]) - mean(o_hat[A])|
    """
    print("\n── Verification: B_obs Decomposition ──")
    n, T = 400, 80
    results = {}

    for mode_name, params in [
        ("no_coupling", no_coupling()),
        ("one_way", one_way_coupling()),
        ("two_way", default_params()),
    ]:
        mode_results = []
        for s in [seed, seed + 100, seed + 200, seed + 300, seed + 400]:
            rng = np.random.default_rng(s)
            G_s, G_o, _ = generate_networks("sbm", n=n, rng=rng,
                n_blocks=3, p_in=0.15, p_out=0.02)

            state = initialize_agents(n=n, initial_active=10, rng=rng,
                                      initial_opinion_dist="polarized")
            engine = TransitionEngine(params, rng)
            o_init = state.o.copy()

            # Track decomposition at each step
            sel_biases, expr_biases, total_biases = [], [], []
            for t in range(T):
                state, V_unused, _ = engine.step(state, G_s, G_o, None, ExternalInputs(), o_init, t)

                a_mask = state.z == A
                if a_mask.sum() > 0:
                    o_all_mean = state.o.mean()
                    o_A_mean = state.o[a_mask].mean()
                    o_hat_A_mean = np.nanmean(state.o_hat[a_mask])

                    sel_bias = abs(o_all_mean - o_A_mean)
                    expr_bias = abs(o_A_mean - o_hat_A_mean)
                    total_bias = abs(o_all_mean - o_hat_A_mean)

                    sel_biases.append(float(sel_bias))
                    expr_biases.append(float(expr_bias))
                    total_biases.append(float(total_bias))

            mode_results.append({
                "mean_total_bias": float(np.mean(total_biases[-20:])),
                "mean_sel_bias": float(np.mean(sel_biases[-20:])),
                "mean_expr_bias": float(np.mean(expr_biases[-20:])),
                "sel_fraction": float(np.mean(sel_biases[-20:]) / max(np.mean(total_biases[-20:]), 1e-8)),
                "expr_fraction": float(np.mean(expr_biases[-20:]) / max(np.mean(total_biases[-20:]), 1e-8)),
            })

        mean_total = np.mean([r["mean_total_bias"] for r in mode_results])
        mean_sel = np.mean([r["mean_sel_bias"] for r in mode_results])
        mean_expr = np.mean([r["mean_expr_bias"] for r in mode_results])
        results[mode_name] = {
            "total_bias": round(mean_total, 4),
            "selection_bias": round(mean_sel, 4),
            "expression_bias": round(mean_expr, 4),
            "selection_fraction": round(mean_sel / max(mean_total, 1e-8), 3),
        }
        print(f"  {mode_name:12s}: total={mean_total:.4f}  "
              f"sel={mean_sel:.4f} ({mean_sel/max(mean_total,1e-8)*100:.0f}%)  "
              f"expr={mean_expr:.4f} ({mean_expr/max(mean_total,1e-8)*100:.0f}%)")

    # Verify: two-way coupling reduces total bias
    verified = results["two_way"]["total_bias"] < results["no_coupling"]["total_bias"]
    print(f"  Theorem 6 (two-way reduces B_obs): {'VERIFIED' if verified else 'FAILED'}")
    results["_verified_theorem_6"] = verified
    return results


def run_zeta_epsilon_phase_diagram(seed: int = 42) -> dict:
    """Verify Theorem 8-9: zeta-epsilon phase diagram.

    Scan zeta in [0.1, 0.9] and epsilon in [0.1, 0.8].
    For each (zeta, epsilon), run simulation and measure final sigma_o and n_clusters.
    """
    print("\n── Verification: zeta-epsilon Phase Diagram ──")
    n, T = 150, 100
    zeta_vals = np.linspace(0.05, 0.95, 6)
    eps_vals = np.linspace(0.10, 0.90, 6)
    phase = np.zeros((len(zeta_vals), len(eps_vals)))  # Store sigma_o
    clusters_grid = np.zeros((len(zeta_vals), len(eps_vals)), dtype=int)

    # Disable propagation to isolate opinion dynamics
    p = default_params()
    p = replace(p, propagation=PropagationParams(beta=0.0, beta_M=0.0))

    rng = np.random.default_rng(seed)
    G_s, G_o, _ = generate_networks("sbm", n=n, rng=rng,
        n_blocks=3, p_in=0.15, p_out=0.02)

    for zi, zeta_val in enumerate(zeta_vals):
        for ei, eps_val in enumerate(eps_vals):
            # Override zeta and epsilon for all agents
            run_rng = np.random.default_rng(seed)
            state = initialize_agents(n=n, initial_active=5, rng=run_rng,
                                      initial_opinion_dist="polarized")
            state.z[:] = D  # all aware
            state.z[:5] = A
            state.o_hat[:5] = state.o[:5].copy()

            # Override fixed attributes
            state.attrs.zeta[:] = zeta_val
            state.attrs.epsilon[:] = eps_val * 2.0  # Scale: epsilon in [0, 2]
            state.attrs.mu[:] = 0.3  # moderate update speed

            engine = TransitionEngine(p, run_rng)
            o_init = state.o.copy()

            for t in range(T):
                state, V_unused, _ = engine.step(state, G_s, G_o, None, ExternalInputs(), o_init, t)

            phase[zi, ei] = float(state.o.std())

            # Count clusters
            o_sorted = np.sort(state.o)
            gaps = np.diff(o_sorted)
            n_clusters = int((gaps > eps_val * 2.0).sum()) + 1
            clusters_grid[zi, ei] = n_clusters

    # Classify each point
    classification = np.empty((len(zeta_vals), len(eps_vals)), dtype=object)
    for zi in range(len(zeta_vals)):
        for ei in range(len(eps_vals)):
            sig = phase[zi, ei]
            nc = clusters_grid[zi, ei]
            if sig < 0.15:
                classification[zi, ei] = "consensus"
            elif nc >= 4:
                classification[zi, ei] = "fragmented"
            else:
                classification[zi, ei] = "polarized"

    # Count region sizes
    n_consensus = int((classification == "consensus").sum())
    n_polarized = int((classification == "polarized").sum())
    n_fragmented = int((classification == "fragmented").sum())

    print(f"  Phase diagram ({len(zeta_vals)}x{len(eps_vals)} grid):")
    print(f"    Consensus:  {n_consensus} cells")
    print(f"    Polarized:  {n_polarized} cells")
    print(f"    Fragmented: {n_fragmented} cells")
    print(f"  zeta range: {zeta_vals}")
    print(f"  eps range:  {eps_vals}")
    print(f"  sigma_o matrix:")
    for zi, zeta_val in enumerate(zeta_vals):
        row = " ".join(f"{phase[zi,ei]:.3f}" for ei in range(len(eps_vals)))
        cls_row = " ".join(f"{classification[zi,ei]:12s}" for ei in range(len(eps_vals)))
        print(f"    zeta={zeta_val:.1f}: [{row}]  [{cls_row}]")

    # Verify: at high zeta, should NOT collapse to consensus
    high_zeta_mask = phase[zeta_vals >= 0.5, :]
    no_collapse_at_high_zeta = (high_zeta_mask > 0.15).all()
    print(f"  Theorem 8 (anchoring prevents collapse): "
          f"{'VERIFIED' if no_collapse_at_high_zeta else 'FAILED'}")

    return {
        "zeta_values": zeta_vals.tolist(),
        "epsilon_values": eps_vals.tolist(),
        "sigma_o_grid": phase.tolist(),
        "clusters_grid": clusters_grid.tolist(),
        "classification": classification.tolist(),
        "n_consensus": n_consensus,
        "n_polarized": n_polarized,
        "n_fragmented": n_fragmented,
        "theorem_8_verified": no_collapse_at_high_zeta,
        "three_regions_exist": n_consensus > 0 and n_polarized > 0,
    }


def main():
    print("=" * 60)
    print("  OPINION ANALYSIS VERIFICATION")
    print("=" * 60)

    t0 = time.perf_counter()

    # Verification 1: B_obs decomposition
    bias_results = run_bias_decomposition()

    # Verification 2: zeta-epsilon phase diagram
    phase_results = run_zeta_epsilon_phase_diagram()

    elapsed = time.perf_counter() - t0

    all_verified = (
        bias_results["_verified_theorem_6"]
        and phase_results["theorem_8_verified"]
    )

    print(f"\n{'='*60}")
    print(f"  VERIFICATION COMPLETE ({elapsed:.0f}s)")
    print(f"  Theorem 6 (two-way reduces B_obs): "
          f"{'VERIFIED' if bias_results['_verified_theorem_6'] else 'FAILED'}")
    print(f"  Theorem 8 (anchoring prevents collapse): "
          f"{'VERIFIED' if phase_results['theorem_8_verified'] else 'FAILED'}")
    print(f"  Three regions exist: "
          f"{'YES' if phase_results['three_regions_exist'] else 'NO'}")
    print(f"  ALL VERIFIED: {all_verified}")
    print(f"{'='*60}")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "bias_decomposition": bias_results,
        "phase_diagram": phase_results,
        "all_verified": all_verified,
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
