"""
Verify PVA mechanism: compare cascade size DISTRIBUTIONS with/without PVA.

Key test: PVA should produce heavier-tailed cascade size distributions
(more medium-large cascades, fewer tiny cascades) compared to network-only.

The observed cascade data has sizes: mean=147, median=81, range=[1, 508]
with a heavy-tailed distribution (a few very large cascades).
"""
import sys, os, json
from pathlib import Path
import numpy as np
from dataclasses import replace
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamics_simulation.config import (
    ModelParams, default_params,
    PropagationParams, ViralParams,
)
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs

OUTPUT = Path(__file__).parent.parent / "data" / "sim_results" / "pva_verification.json"


def run_cascade_experiment(params, n, T, n_seeds=20, base_seed=42):
    """Run many short simulations, each starting from ONE initial post.
    Measure cascade size = total new A agents created.

    This mimics the empirical cascade data: one root post generates
    a cascade of reposts.
    """
    cascade_sizes = []

    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s * 100)
        G_s, G_o, _ = generate_networks("sbm", n=n, rng=rng,
            n_blocks=3, p_in=0.15, p_out=0.02)

        # Start with ONE active agent
        state = initialize_agents(n=n, initial_active=1, rng=rng,
                                  initial_opinion_dist="polarized")
        engine = TransitionEngine(params, rng)
        o_init = state.o.copy()
        V = 0.0

        # Count total new activations (cascade size) in first 10 steps
        initial_agent = int(np.where(state.z == A)[0][0])
        cascade_count = 0

        for t in range(T):
            inputs = ExternalInputs(V=V)
            state, V = engine.step(state, G_s, G_o, None, inputs, o_init, t)
            # Count all A agents except the initial seed
            n_A = state.n_A
            cascade_count = max(cascade_count, n_A - 1)  # exclude seed

        cascade_sizes.append(cascade_count)

    return np.array(cascade_sizes)


def main():
    print("=" * 60)
    print("  PVA CASCADE SIZE DISTRIBUTION VERIFICATION")
    print("=" * 60)

    n, T, n_seeds = 300, 30, 50
    base_seed = 42

    # Three configurations
    configs = {
        "network_only": replace(default_params(),
            viral=ViralParams(beta_V=0.0, delta_V=0.0, eta_V=0.0),
        ),
        "pva_moderate": replace(default_params(),
            viral=ViralParams(beta_V=0.01, delta_V=0.40, eta_V=0.05),
        ),
        "pva_strong": replace(default_params(),
            viral=ViralParams(beta_V=0.03, delta_V=0.40, eta_V=0.10),
        ),
    }

    results = {}
    for name, params in configs.items():
        print(f"\n  {name}...")
        sizes = run_cascade_experiment(params, n, T, n_seeds, base_seed)
        results[name] = {
            "mean": float(np.mean(sizes)),
            "median": float(np.median(sizes)),
            "std": float(np.std(sizes)),
            "min": int(np.min(sizes)),
            "max": int(np.max(sizes)),
            "skewness": float((np.mean(sizes) - np.median(sizes)) / max(np.std(sizes), 1)),
            "pct_gt_10": float((sizes > 10).mean()),
            "pct_gt_50": float((sizes > 50).mean()),
            "pct_gt_100": float((sizes > 100).mean()),
            "sizes": sizes.tolist(),
        }
        print(f"    mean={results[name]['mean']:.0f} median={results[name]['median']:.0f} "
              f"max={results[name]['max']} >10:{results[name]['pct_gt_10']:.0%} "
              f">50:{results[name]['pct_gt_50']:.0%} >100:{results[name]['pct_gt_100']:.0%}")

    # ── Comparison ──
    net = results["network_only"]
    pva = results["pva_moderate"]
    pva_s = results["pva_strong"]

    mean_amplification = pva["mean"] / max(net["mean"], 1)
    strong_amplification = pva_s["mean"] / max(net["mean"], 1)
    tail_amplification = pva["pct_gt_50"] / max(net["pct_gt_50"], 0.01)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"  Mean amplification (moderate PVA): {mean_amplification:.1f}x")
    print(f"  Mean amplification (strong PVA):   {strong_amplification:.1f}x")
    print(f"  Tail amplification (>50):          {tail_amplification:.1f}x")
    print(f"  PVA produces heavier-tailed distribution: "
          f"{'YES' if pva['skewness'] > net['skewness'] else 'NO'}")

    # Observed cascade data for reference
    import json as _json
    with open("d:/舆情分析/Tibet_data_collector/data/cascade/repost_edges.jsonl", encoding='utf-8') as f:
        edges = [_json.loads(line) for line in f]
    obs_sizes = list(Counter(e['root_post_id'] for e in edges).values())
    obs = {
        "mean": float(np.mean(obs_sizes)),
        "median": float(np.median(obs_sizes)),
        "max": int(np.max(obs_sizes)),
        "min": int(np.min(obs_sizes)),
        "pct_gt_50": float((np.array(obs_sizes) > 50).mean()),
        "pct_gt_100": float((np.array(obs_sizes) > 100).mean()),
    }
    print(f"\n  Observed cascade stats (N~7000, 45 cascades):")
    print(f"    mean={obs['mean']:.0f} median={obs['median']:.0f} "
          f"max={obs['max']} >50:{obs['pct_gt_50']:.0%} >100:{obs['pct_gt_100']:.0%}")

    results["observed"] = obs

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
