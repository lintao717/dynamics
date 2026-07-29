"""
Within-community heterogeneity + silence spiral experiment.

Tests whether the silence spiral mechanism activates when a community
contains a substantial minority faction. The key manipulation:

  - Take a 3-block SBM network
  - In Block 0 (largest block): inject 35% minority-opinion agents (o < 0)
    while the block majority (65%) holds o > 0
  - Block 1: all majority (o > 0)
  - Block 2: all minority (o < 0)

  Compare lambda_spiral = 0 vs 0.5 vs 0.85:
    - Minority expression rate in Block 0
    - Public opinion bias B_obs in Block 0
    - Gamma_i trajectory for minority agents

Usage:
    python -m dynamics_simulation.experiments.heterogeneity_spiral
"""

import sys, os, json, time
from pathlib import Path
from dataclasses import replace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dynamics_simulation.config import (
    ModelParams, default_params,
    PropagationParams, ActivationParams, OpinionParams,
)
from dynamics_simulation.networks import generate_networks, stochastic_block
from dynamics_simulation.agents import initialize_agents, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs
from dynamics_simulation.simulation import SimulationConfig, SimulationRunner


OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "sim_results"
SEEDS = [42, 123, 789, 555, 888]


def build_heterogeneous_network(
    n: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """Build SBM network with engineered opinion heterogeneity.

    Block 0 (n=150): 65% pro (o>0), 35% anti (o<0) — mixed community
    Block 1 (n=80): 100% pro (o>0) — echo chamber
    Block 2 (n=70): 100% anti (o<0) — echo chamber

    Returns:
        G_s, G_o, communities, opinion_init, block_labels
    """
    rng = np.random.default_rng(seed)
    block_sizes = [150, 80, 70]

    G_s, G_o, communities = stochastic_block(
        n=n, n_blocks=3, p_in=0.18, p_out=0.02,
        block_sizes=block_sizes, directed=True, rng=rng,
    )

    # Build opinion initialization
    o_init = np.zeros(n)

    # Block 0: mixed (65% pro, 35% anti)
    b0 = communities[0]
    n_b0 = len(b0)
    n_pro = int(n_b0 * 0.65)
    n_anti = n_b0 - n_pro
    b0_pro = rng.choice(b0, size=n_pro, replace=False)
    b0_anti = np.array([i for i in b0 if i not in b0_pro])

    # KEY: use extreme opinions to create clear Gamma gap
    o_init[b0_pro] = rng.normal(0.80, 0.10, size=n_pro)
    o_init[b0_anti] = rng.normal(-0.80, 0.10, size=n_anti)

    # Block 1: all pro
    b1 = communities[1]
    o_init[b1] = rng.normal(0.80, 0.10, size=len(b1))

    # Block 2: all anti
    b2 = communities[2]
    o_init[b2] = rng.normal(-0.80, 0.10, size=len(b2))

    o_init = np.clip(o_init, -1.0, 1.0)

    # Block labels for analysis
    block_labels = np.zeros(n, dtype=np.int32)
    for bid, nodes in communities.items():
        for node in nodes:
            block_labels[node] = bid

    return G_s, G_o, communities, o_init, block_labels, b0_anti, b0_pro


def run_heterogeneity_experiment():
    """Run the within-community heterogeneity experiment.

    KEY INSIGHT from v1: silence spiral only activates when:
    1. Minority agents perceive low Gamma_i (< 0.5) — needs ACTIVE majority
    2. The local climate is well-established — needs many A agents expressing
    3. Minority agents are aware (D state) — so they can decide D→A

    Fix: Start with 60 active agents from the PRO faction only.
    This creates an immediate pro-majority climate that anti-minority
    agents can perceive, triggering the silence spiral mechanism.
    """
    ensure_dirs()

    print("=" * 60)
    print("  SILENCE SPIRAL (FIXED: HIGH INITIAL ACTIVITY)")
    print("=" * 60)

    n, T = 300, 40
    n_initial_active = 80  # 80 pro agents → strong majority climate
    all_results = []

    for lam_spiral, label in [(0.0, "none"), (0.50, "moderate"), (0.85, "strong")]:
        p = default_params()
        p = replace(p, opinion=OpinionParams(lambda_spiral=lam_spiral))

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            G_s, G_o, communities, o_init, block_labels, b0_anti, b0_pro = \
                build_heterogeneous_network(n=n, seed=seed)

            state = initialize_agents(n=n, initial_active=0, rng=rng,
                                      initial_opinion_dist="uniform")
            # Override opinions
            state.o[:] = o_init

            # KEY FIX: Activate PRO agents in Block 0 to establish majority climate
            # Pick n_initial_active agents from the pro faction
            active_pro = rng.choice(b0_pro, size=min(n_initial_active, len(b0_pro)),
                                     replace=False)
            state.z[active_pro] = A
            state.o_hat[active_pro] = o_init[active_pro]
            # Give active agents some initial arousal
            state.h[active_pro] = rng.uniform(0.6, 0.9, size=len(active_pro))

            engine = TransitionEngine(p, rng)
            o_initial_arr = state.o.copy()

            minority_express_frac = []
            minority_gamma = []
            public_bias_b0 = []

            for t in range(T):
                inputs = ExternalInputs()
                state, V_unused, _ = engine.step(state, G_s, G_o, None, inputs, o_initial_arr, t)


                # Per-agent Gamma for minority agents
                b0_nodes = communities[0]
                a_b0 = state.z[b0_nodes] == A
                if a_b0.sum() > 0:
                    climate_b0 = np.nanmean(state.o_hat[b0_nodes][a_b0])
                    gamma_anti = 1.0 - np.abs(state.o[b0_anti] - climate_b0) / 2.0
                    gamma_anti = np.clip(gamma_anti, 0.0, 1.0)
                    minority_gamma.append(float(gamma_anti.mean()))
                else:
                    minority_gamma.append(0.5)

                # Minority expression rate (among aware minority agents)
                anti_A = (state.z[b0_anti] == A).sum()
                anti_D = (state.z[b0_anti] == D).sum()
                anti_aware = anti_A + anti_D
                minority_express_frac.append(
                    anti_A / max(anti_aware, 1)
                )

                # Public bias in Block 0
                if a_b0.sum() > 0:
                    o_hat_b0 = np.nanmean(state.o_hat[b0_nodes][a_b0])
                    bias = abs(state.o[b0_nodes].mean() - o_hat_b0)
                else:
                    bias = 0.0
                public_bias_b0.append(float(bias))

            # Summary metrics
            result = {
                "lambda_spiral": lam_spiral,
                "label": label,
                "seed": seed,
                "peak_A": int(state.n_A),
                "final_minority_express_frac": minority_express_frac[-1],
                "mean_minority_express_frac": float(np.mean(minority_express_frac[-30:])),
                "final_gamma_minority": minority_gamma[-1],
                "mean_gamma_minority": float(np.mean(minority_gamma[-30:])),
                "final_public_bias_b0": public_bias_b0[-1],
                "mean_public_bias_b0": float(np.mean(public_bias_b0[-30:])),
                "minority_gamma_t": minority_gamma,
                "minority_express_t": minority_express_frac,
            }
            all_results.append(result)

            print(f"  lambda={lam_spiral:.2f} seed={seed}: "
                  f"min_expr={result['mean_minority_express_frac']:.3f}, "
                  f"gamma={result['mean_gamma_minority']:.3f}, "
                  f"bias={result['mean_public_bias_b0']:.3f}")

    # ── Aggregate by lambda_spiral ──
    print(f"\n{'─'*40}")
    print("  AGGREGATE RESULTS (mean ± std across seeds):")
    print(f"  {'lambda':<8} {'expr_rate':<12} {'Gamma':<12} {'bias':<12} {'minority_A':<12}")
    for lam in [0.0, 0.50, 0.85]:
        lam_results = [r for r in all_results if r["lambda_spiral"] == lam]
        expr = [r["mean_minority_express_frac"] for r in lam_results]
        gamma = [r["mean_gamma_minority"] for r in lam_results]
        bias = [r["mean_public_bias_b0"] for r in lam_results]
        # Early-phase metrics (t=0-15)
        expr_early = [r["minority_express_t"][5] if len(r["minority_express_t"]) > 5 else 0
                      for r in lam_results]
        gamma_early = [r["minority_gamma_t"][5] if len(r["minority_gamma_t"]) > 5 else 0
                       for r in lam_results]
        print(f"  {lam:<8.2f} {np.mean(expr):.3f}±{np.std(expr):.3f}   "
              f"{np.mean(gamma):.3f}±{np.std(gamma):.3f}   "
              f"{np.mean(bias):.3f}±{np.std(bias):.3f}   "
              f"—")
        print(f"  {'':8} early(t=5): expr={np.mean(expr_early):.3f} gamma={np.mean(gamma_early):.3f}")

    # ── Statistical test ──
    expr_0 = [r["mean_minority_express_frac"] for r in all_results if r["lambda_spiral"] == 0.0]
    expr_085 = [r["mean_minority_express_frac"] for r in all_results if r["lambda_spiral"] == 0.85]
    effect = np.mean(expr_0) - np.mean(expr_085)
    print(f"\n  lambda_spiral effect on minority expression: {effect:.4f} "
          f"({'DETECTED' if effect > 0.01 else 'WEAK'} )")

    # ── Deterministic check from model equations ──
    print(f"\n{'─'*40}")
    print("  DETERMINISTIC CHECK (from equations, no sim noise):")
    o_pro, o_anti = 0.80, -0.80
    climate = 0.80  # pure pro climate
    gamma_anti = 1.0 - abs(o_anti - climate) / 2.0
    print(f"  o_anti={o_anti}, climate={climate} -> Gamma_anti={gamma_anti:.3f}")
    if gamma_anti < 0.5:
        a0, a1, a2, a3, a4, a5 = -2.5, 1.5, 2.0, 1.8, 1.5, 1.0
        h, f, c = 0.5, 0.0, 0.3
        logit = a0 + a1*abs(o_anti) + a2*h + a3*gamma_anti - a4*f - a5*c
        p_base = 1.0 / (1.0 + np.exp(-logit))
        for lam in [0.0, 0.5, 0.85]:
            penalty = 1.0 - lam * (0.5 - gamma_anti)
            p_final = p_base * penalty
            print(f"  lambda={lam:.2f}: P_base={p_base:.4f} penalty={penalty:.3f} "
                  f"P_final={p_final:.4f} (reduction={(1-penalty)*100:.1f}%)")
    else:
        print(f"  Gamma_anti >= 0.5 -> spiral NOT triggered")

    # ── Save ──
    output_path = OUTPUT_DIR / "experiment_heterogeneity_spiral.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Within-community heterogeneity + silence spiral",
            "n": n, "T": T, "seeds": SEEDS,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved: {output_path}")

    return all_results


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    run_heterogeneity_experiment()
