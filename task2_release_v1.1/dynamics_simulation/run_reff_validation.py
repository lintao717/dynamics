"""Numerical verification of R_eff theorems.

Validates the four theorems from docs/reff_derivation.md through
controlled numerical experiments.
"""
import sys, os, json
from pathlib import Path
import numpy as np
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamics_simulation.config import (
    default_params, PropagationParams, ActivationParams, OpinionParams,
)
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents
from dynamics_simulation.reff import compute_reff, analyze_reff_across_opinion_gradient

OUTPUT = Path(__file__).parent.parent / "data" / "sim_results" / "reff_validation.json"


def main():
    print("=" * 60)
    print("  R_eff THEOREM VERIFICATION")
    print("=" * 60)

    seed = 42
    n = 200
    rng = np.random.default_rng(seed)

    # Generate one SBM network for all tests
    G_s, G_o, communities = generate_networks(
        "sbm", n=n, rng=rng, n_blocks=3, p_in=0.15, p_out=0.02,
    )

    # Baseline state: all agents in D (aware), opinions set
    state = initialize_agents(n=n, initial_active=5, rng=rng,
                              initial_opinion_dist="polarized")
    state.z[:] = 0  # All in U (susceptible)
    state.z[:5] = 2  # 5 active for climate
    state.m[:5] = 1
    state.o_hat[:5] = state.o[:5].copy()

    base_params = default_params()

    results = {}

    # ── Verification 1: R_eff as function of |o| ──
    print("\n── Verification 1: dR_eff/d||o|| > 0 ──")
    gradient = analyze_reff_across_opinion_gradient(
        G_s, G_o, state, base_params,
        opinion_range=(0.0, 0.8), n_points=7,
    )
    print(f"  |o| range: {gradient['opinion_values']}")
    print(f"  R_eff:     {[f'{r:.3f}' for r in gradient['R_eff_values']]}")
    print(f"  Slope: {gradient['slope']:.4f} "
          f"({'POSITIVE (verified)' if gradient['slope'] > 0.01 else 'FLAT/NEGATIVE (fails)'})")
    results["verification_1"] = {
        "theorem": "dR_eff/d||o|| > 0",
        "slope": gradient["slope"],
        "verified": gradient["verified"],
    }

    # ── Verification 2: dR_eff/dalpha_1 > 0 ──
    print("\n── Verification 2: dR_eff/dalpha_1 > 0 ──")
    result = compute_reff(state, G_s, G_o, base_params, compute_partials=True)
    dReff_da1 = result.partial_derivatives["dReff_dalpha1"]
    print(f"  R_eff = {result.R_eff:.4f}")
    print(f"  dR_eff/dalpha_1 = {dReff_da1:.4f} "
          f"({'POSITIVE [OK]' if dReff_da1 > 0.01 else 'NEGATIVE/ZERO [FAIL]'})")
    results["verification_2"] = {
        "theorem": "dR_eff/dalpha_1 > 0",
        "derivative": dReff_da1,
        "verified": dReff_da1 > 0.01,
    }

    # ── Verification 3: dR_eff/dlambda_spiral <= 0 ──
    print("\n── Verification 3: dR_eff/dlambda_spiral <= 0 ──")
    # Create minority condition: set a subpopulation with anti opinions
    state_minority = state.copy()
    minority_idx = np.arange(50, 100)
    state_minority.o[minority_idx] = -0.7
    state_minority.z[minority_idx] = 0  # U state (susceptible)
    active_idx = np.arange(5)
    state_minority.o[active_idx] = 0.7
    state_minority.o_hat[active_idx] = 0.7
    state_minority.z[active_idx] = 2  # A
    state_minority.m[active_idx] = 1

    p_no_spiral = replace(base_params, opinion=OpinionParams(lambda_spiral=0.0))
    p_strong_spiral = replace(base_params, opinion=OpinionParams(lambda_spiral=0.85))

    r_no = compute_reff(state_minority, G_s, G_o, p_no_spiral, compute_partials=False)
    r_strong = compute_reff(state_minority, G_s, G_o, p_strong_spiral, compute_partials=False)

    diff = r_no.R_eff - r_strong.R_eff
    print(f"  R_eff(lambda=0)    = {r_no.R_eff:.4f}")
    print(f"  R_eff(lambda=0.85) = {r_strong.R_eff:.4f}")
    print(f"  Delta = {diff:.4f} "
          f"({'NON-NEGATIVE [OK] (spiral suppresses)' if diff >= -1e-6 else 'NEGATIVE [FAIL] (spiral increases)'})")
    results["verification_3"] = {
        "theorem": "dR_eff/dlambda_spiral <= 0",
        "R_eff_no_spiral": r_no.R_eff,
        "R_eff_strong_spiral": r_strong.R_eff,
        "delta": diff,
        "verified": diff >= -1e-6,
    }

    # ── Verification 4: Network structure effect ──
    print("\n── Verification 4: R_eff depends on network structure ──")
    net_results = {}
    for net_type, kwargs in [
        ("er", {"p": 0.05}),
        ("ba", {"m": 5}),
        ("ws", {"k": 10, "p_rewire": 0.10}),
        ("sbm", {"n_blocks": 3, "p_in": 0.15, "p_out": 0.02}),
    ]:
        net_rng = np.random.default_rng(seed)
        Gs, Go, comms = generate_networks(net_type, n=n, rng=net_rng, **kwargs)
        # Compute k_eff
        k_out = Gs.sum(axis=0)
        k_in = Gs.sum(axis=1)
        k_eff = float((k_out * k_in).sum() / max(k_out.sum(), 1e-8))
        result = compute_reff(state, Gs, Go, base_params, compute_partials=False)
        net_results[net_type] = {
            "k_eff": round(k_eff, 2),
            "R_eff": round(result.R_eff, 4),
            "R_mf": round(result.components["R_mf_approx"], 4),
        }
        print(f"  {net_type:5s}: k_eff={k_eff:6.2f}, R_eff={result.R_eff:.4f}, "
              f"R_mf={result.components['R_mf_approx']:.4f}")
    results["verification_4"] = {
        "theorem": "SBM R_eff > ER R_eff under same beta",
        "networks": net_results,
        "verified": net_results["sbm"]["R_eff"] > net_results["er"]["R_eff"],
    }

    # ── Summary ──
    all_verified = all(
        results[k].get("verified", False)
        for k in ["verification_1", "verification_2", "verification_3", "verification_4"]
    )
    print(f"\n{'='*60}")
    print(f"  ALL THEOREMS VERIFIED: {all_verified}")
    for k, v in results.items():
        status = "[OK]" if v.get("verified", False) else "[FAIL]"
        print(f"  [{status}] {v['theorem']}")
    print(f"{'='*60}")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
