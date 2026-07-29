"""
Dedicated silence spiral experiment.

Design:
  - 2-block SBM. Block 0: 65% majority (o>0), 35% minority (o<0).
  - ALL majority agents start ACTIVE → strong pro climate from t=0.
  - Minority agents start in DORMANT → can decide D→A each step.
  - Scan lambda_spiral in [0.0, 1.0], 30 seeds each.
  - Climate visibility guaranteed: v_i >> V_MIN=0.10 for minority agents.

Metrics:
  - Minority expression rate (fraction of minority that is A)
  - Selection bias (o_bar_A - o_bar)
  - Expression bias (o_hat_bar_A - o_bar_A)
  - B_obs
  - Mean Gamma_i for minority agents
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
from dataclasses import replace
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dynamics_simulation.config import (
    default_params, OpinionParams,
)
from dynamics_simulation.networks import stochastic_block
from dynamics_simulation.agents import initialize_agents, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs

OUTPUT = Path(__file__).parent.parent.parent / "data" / "sim_results" / "silence_spiral_dedicated.json"


def build_setup(n=200, majority_frac=0.65, seed=42):
    """Build majority-minority setup with controlled climate.

    Block 0 (n=200): 65% majority (o~+0.80), 35% minority (o~-0.80).
    All majority agents start ACTIVE, all minority agents start DORMANT.
    This guarantees strong pro-climate (climate ~ +0.8) from t=0,
    creating a hostile climate for minority agents (Gamma ~ 0.2).
    """
    rng = np.random.default_rng(seed)

    # Single-block dense network for maximum climate exposure
    G_s, G_o, communities = stochastic_block(
        n=n, n_blocks=1, p_in=0.25, p_out=0.0,
        directed=True, rng=rng,
    )

    # Build opinion distribution
    n_maj = int(n * majority_frac)
    n_min = n - n_maj
    all_idx = list(range(n))
    rng.shuffle(all_idx)
    maj_idx = np.array(all_idx[:n_maj])
    min_idx = np.array(all_idx[n_maj:])

    o_init = np.zeros(n)
    o_init[maj_idx] = np.clip(rng.normal(0.80, 0.08, size=n_maj), 0.5, 1.0)
    o_init[min_idx] = np.clip(rng.normal(-0.80, 0.08, size=n_min), -1.0, -0.5)

    # Initialize: majority active, minority dormant
    state = initialize_agents(n=n, initial_active=0, rng=rng,
                              initial_opinion_dist="uniform")
    state.o[:] = o_init
    state.z[maj_idx] = A
    state.m[maj_idx] = 1
    state.o_hat[maj_idx] = o_init[maj_idx]
    state.h[maj_idx] = rng.uniform(0.5, 0.8, size=n_maj)
    state.z[min_idx] = D  # aware, can reactivate
    state.m[min_idx] = 0  # never expressed

    return G_s, G_o, communities, state, maj_idx, min_idx


def run_one(params, lam, n, T, n_seeds=30, base_seed=42):
    """Run n_seeds simulations and collect minority expression metrics."""
    p = replace(params, opinion=OpinionParams(
        lambda_spiral=lam, climate_visibility_threshold=0.02))

    metrics = defaultdict(list)
    for s in range(n_seeds):
        seed = base_seed + s * 100
        G_s, G_o, comms, state, maj_idx, min_idx = build_setup(
            n=n, majority_frac=0.65, seed=seed)
        engine = TransitionEngine(p, np.random.default_rng(seed))
        o_init = state.o.copy()
        V = 0.0

        for t in range(T):
            state, V, _ = engine.step(state, G_s, G_o, None,
                                       ExternalInputs(V=V), o_init, t)

        # Final-state metrics
        min_A = (state.z[min_idx] == A).sum()
        min_D = (state.z[min_idx] == D).sum()
        min_aware = min_A + min_D
        maj_A = (state.z[maj_idx] == A).sum()

        metrics["minority_expr_rate"].append(min_A / max(min_aware, 1))
        metrics["majority_expr_rate"].append(maj_A / len(maj_idx))
        metrics["o_mean"].append(float(state.o.mean()))
        metrics["o_hat_A_mean"].append(
            float(np.nanmean(state.o_hat[state.z == A]))
            if (state.z == A).sum() > 0 else float('nan'))

        # Gamma for minority
        expressed_mask = (state.z == A) & np.isfinite(state.o_hat)
        if expressed_mask.any():
            expressed = expressed_mask.astype(np.float64)
            v_i = G_o.dot(expressed)
            weighted = G_o.dot(np.where(expressed_mask, state.o_hat, 0.0))
            climate = np.divide(weighted, v_i, out=np.zeros(n), where=v_i > 1e-8)
            gamma_min = 1.0 - np.abs(state.o[min_idx] - climate[min_idx]) / 2.0
            metrics["gamma_minority"].append(float(np.clip(gamma_min, 0, 1).mean()))
            metrics["v_i_minority"].append(float(v_i[min_idx].mean()))
        else:
            metrics["gamma_minority"].append(0.5)
            metrics["v_i_minority"].append(0.0)

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


def main():
    print("=" * 60)
    print("  DEDICATED SILENCE SPIRAL EXPERIMENT")
    print("  Design: 65% majority (A, o~+0.8) + 35% minority (D, o~-0.8)")
    print("=" * 60)

    n, T, n_seeds = 200, 40, 30
    from dynamics_simulation.config import DecayParams
    params = default_params()
    # Disable decay: keep majority permanently active to maintain climate
    params = replace(params, decay=DecayParams(
        gamma_0=-10.0, gamma_1=0.0, gamma_2=0.0, gamma_3=0.0))
    lambda_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}

    print(f"\n  N={n}, T={T}, seeds={n_seeds}")
    print(f"  lambda  minority_expr  majority_expr  gamma_min  v_i_min  sel_bias")
    print(f"  {'─'*65}")

    for lam in lambda_vals:
        r = run_one(params, lam, n, T, n_seeds)
        results[f"lambda_{lam:.1f}"] = {k: {"mean": v[0], "std": v[1]} for k, v in r.items()}

        # Compute selection bias: o_bar_A - o_bar
        sel_bias = r["o_hat_A_mean"][0] - r["o_mean"][0] if not np.isnan(r["o_hat_A_mean"][0]) else 0
        print(f"  {lam:.1f}     {r['minority_expr_rate'][0]:.4f}±{r['minority_expr_rate'][1]:.4f}   "
              f"{r['majority_expr_rate'][0]:.4f}±{r['majority_expr_rate'][1]:.4f}   "
              f"{r['gamma_minority'][0]:.3f}±{r['gamma_minority'][1]:.3f}   "
              f"{r['v_i_minority'][0]:.2f}±{r['v_i_minority'][1]:.2f}   "
              f"{sel_bias:+.4f}")

    # ── Effect size ──
    expr_0 = np.array([v["minority_expr_rate"]["mean"] for v in results.values()])
    lam_arr = np.array(lambda_vals)
    slope, _ = np.polyfit(lam_arr, expr_0, 1)

    print(f"\n  lambda_spiral -> minority expression slope: {slope:.4f}")
    if slope < -0.01:
        print(f"  Spiral EFFECT DETECTED: minority expression decreases with lambda")
    elif abs(slope) < 0.01:
        print(f"  Spiral effect WEAK/ABSENT")
    else:
        print(f"  Unexpected: minority expression INCREASES with lambda")

    # ── Save ──
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({
            "design": "65% majority (A, o~+0.8), 35% minority (D, o~-0.8)",
            "n": n, "T": T, "n_seeds": n_seeds,
            "lambda_vals": lambda_vals,
            "slope": round(float(slope), 4),
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
