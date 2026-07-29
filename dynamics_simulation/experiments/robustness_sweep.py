"""
Week 2: Robustness + Equal-Degree Network Comparison.

1. Multi-seed robustness: 30 seeds, default params, report CI
2. Equal-degree comparison: ER/BA/WS/SBM with matched average degree
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
from dataclasses import replace
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dynamics_simulation.config import default_params, PropagationParams
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs

OUTPUT = Path(__file__).parent.parent.parent / "data" / "sim_results" / "robustness_sweep.json"


def run_seed(params, G_s, G_o, n, T, seed):
    """Run one simulation and return key metrics."""
    rng = np.random.default_rng(seed)
    state = initialize_agents(n=n, initial_active=10, rng=rng,
                              initial_opinion_dist="polarized")
    engine = TransitionEngine(params, rng)
    o_init = state.o.copy()
    V = 0.0
    A_ts = []
    for t in range(T):
        state, V, _ = engine.step(state, G_s, G_o, None,
                                   ExternalInputs(V=V), o_init, t)
        A_ts.append(state.n_A)
    A = np.array(A_ts)
    return {
        "peak_A": int(A.max()), "peak_t": int(A.argmax()),
        "final_A": int(A[-1]), "mean_A": float(A.mean()),
        "total_activations": int(A.sum()),
        "final_o_std": float(state.o.std()),
        "final_h_mean": float(state.h.mean()),
    }


def multi_seed_robustness(n=300, T=60, n_seeds=30, base_seed=42):
    """Run 30 seeds with default SBM, report mean + 95% CI."""
    print("\n── Multi-Seed Robustness (30 seeds) ──")
    params = default_params()
    rng = np.random.default_rng(base_seed)
    G_s, G_o, _ = generate_networks("sbm", n=n, rng=rng,
        n_blocks=3, p_in=0.15, p_out=0.02)
    # Same network for all seeds (only transition RNG varies)

    all_metrics = defaultdict(list)
    for s in range(n_seeds):
        m = run_seed(params, G_s, G_o, n, T, base_seed + s * 100)
        for k, v in m.items():
            all_metrics[k].append(v)

    results = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        ci95 = 1.96 * std / np.sqrt(n_seeds)
        results[k] = {"mean": round(mean, 2), "std": round(std, 2),
                       "ci95": round(ci95, 2), "cv": round(std / max(abs(mean), 1e-8), 3)}
        print(f"  {k:20s}: {mean:8.2f} ± {ci95:.2f} (CV={std/max(abs(mean),1e-8):.3f})")

    return results


def equal_degree_comparison(n=300, target_k=8, T=60, n_seeds=10, base_seed=42):
    """Compare ER/BA/WS/SBM with matched average degree."""
    print("\n── Equal-Degree Network Comparison ──")
    params = default_params()

    configs = {
        "ER": ("er", {"p": target_k / (n - 1)}),
        "BA": ("ba", {"m": max(2, target_k // 2)}),
        "WS": ("ws", {"k": max(4, target_k), "p_rewire": 0.10}),
        "SBM": ("sbm", {"n_blocks": 3, "p_in": 0.06, "p_out": 0.005}),
    }

    results = {}
    for name, (net_type, kwargs) in configs.items():
        net_rng = np.random.default_rng(base_seed)
        G_s, G_o, _ = generate_networks(net_type, n=n, rng=net_rng, **kwargs)
        k_actual = float(G_s.sum(axis=0).mean())
        all_peak = []
        for s in range(n_seeds):
            m = run_seed(params, G_s, G_o, n, T, base_seed + s * 100)
            all_peak.append(m["peak_A"])

        arr = np.array(all_peak)
        results[name] = {
            "k_target": target_k, "k_actual": round(k_actual, 1),
            "peak_A_mean": round(float(arr.mean()), 1),
            "peak_A_std": round(float(arr.std(ddof=1)), 1),
        }
        print(f"  {name:5s}: k={k_actual:.1f} peak_A={arr.mean():.0f}±{arr.std(ddof=1):.0f}")

    # Check: SBM should have higher peak than ER at same degree
    sbm_peak = results["SBM"]["peak_A_mean"]
    er_peak = results["ER"]["peak_A_mean"]
    print(f"\n  SBM/ER peak ratio: {sbm_peak/max(er_peak,1):.1f}x")
    print(f"  Community structure amplification: "
          f"{'CONFIRMED' if sbm_peak > er_peak * 1.2 else 'WEAK/ABSENT'}")

    return results


def main():
    print("=" * 60)
    print("  ROBUSTNESS + EQUAL-DEGREE COMPARISON")
    print("=" * 60)

    robustness = multi_seed_robustness()
    equal_deg = equal_degree_comparison()

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = {"robustness_30seeds": robustness, "equal_degree": equal_deg}
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
