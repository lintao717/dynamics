"""
Long-term polarization experiment (T=500).

Tracks opinion dynamics over an extended time horizon to observe:
  1. Opinion convergence vs. persistent polarization
  2. Number of opinion clusters over time
  3. Public-private opinion gap (B_obs) trajectory
  4. Whether the system reaches a stable equilibrium

Key question: Does the model avoid degenerate states (total consensus,
  permanent fragmentation) under default parameters?

Usage:
    python -m dynamics_simulation.experiments.longterm_polarization
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
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs
from dynamics_simulation.simulation import SimulationConfig, SimulationRunner


OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "sim_results"
SEEDS = [42, 123, 789]


def estimate_clusters(o: np.ndarray, eps: float = 0.35) -> int:
    """Estimate number of opinion clusters using gap-based method."""
    if len(o) < 2:
        return 1
    o_sorted = np.sort(o)
    gaps = np.diff(o_sorted)
    return int((gaps > eps).sum()) + 1


def run_longterm_experiment():
    """Run T=500 simulation and track opinion evolution."""
    ensure_dirs()

    print("=" * 60)
    print("  LONG-TERM POLARIZATION EXPERIMENT (T=500)")
    print("=" * 60)

    n, T = 500, 500
    all_results = []

    for seed in SEEDS:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Default bidirectional coupling
        p = default_params()

        cfg = SimulationConfig(
            n_agents=n, initial_active=10, T=T,
            network_type="sbm",
            network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
            params=p, seed=seed, verbose=False,
            snapshot_interval=10,
        )
        runner = SimulationRunner(cfg)
        metrics = runner.run()

        # ── Collect opinion snapshots from the simulation ──
        # We need to re-run with opinion tracking (metrics only records means)
        # Let's do a manual run for detailed tracking
        rng2 = np.random.default_rng(seed)
        G_s, G_o, communities = generate_networks("sbm", n=n, rng=rng2,
            n_blocks=3, p_in=0.15, p_out=0.02)
        state = initialize_agents(n=n, initial_active=10, rng=rng2,
                                  initial_opinion_dist="polarized")
        engine = TransitionEngine(p, rng2)
        o_initial = state.o.copy()

        # Track every 25 steps
        track_interval = 25
        opinion_snapshots = {
            "steps": [],
            "o_std": [],
            "o_polarization": [],
            "n_clusters": [],
            "h_mean": [],
            "f_mean": [],
            "n_A": [],
            "n_D": [],
            "n_U": [],
        }

        for t in range(T):
            inputs = ExternalInputs(
                staleness=t / max(T, 1),
                novelty=np.exp(-t / 80.0),  # Slowly decaying novelty
            )
            state, V_unused, _ = engine.step(state, G_s, G_o, None, inputs, o_initial, t)


            if t % track_interval == 0 or t == T - 1:
                o_abs = np.abs(state.o)
                pol = 1.0 - (o_abs.std() / max(state.o.std(), 1e-8))
                opinion_snapshots["steps"].append(t)
                opinion_snapshots["o_std"].append(float(state.o.std()))
                opinion_snapshots["o_polarization"].append(float(max(0, pol)))
                opinion_snapshots["n_clusters"].append(
                    estimate_clusters(state.o, eps=p.opinion.epsilon_mean))
                opinion_snapshots["h_mean"].append(float(state.h.mean()))
                opinion_snapshots["f_mean"].append(float(state.f.mean()))
                opinion_snapshots["n_A"].append(int(state.n_A))
                opinion_snapshots["n_D"].append(int(state.n_D))
                opinion_snapshots["n_U"].append(int(state.n_U))

        elapsed = time.perf_counter() - t0

        # ── Key diagnostics ──
        o_std_first = opinion_snapshots["o_std"][0]
        o_std_last = opinion_snapshots["o_std"][-1]
        o_std_min = min(opinion_snapshots["o_std"])
        clusters_first = opinion_snapshots["n_clusters"][0]
        clusters_last = opinion_snapshots["n_clusters"][-1]
        clusters_min = min(opinion_snapshots["n_clusters"])
        pol_first = opinion_snapshots["o_polarization"][0]
        pol_last = opinion_snapshots["o_polarization"][-1]

        # Convergence metric: did opinions collapse?
        collapsed = o_std_last < 0.10
        fragmented = clusters_last >= 4

        result = {
            "seed": seed,
            "o_std_first": round(o_std_first, 4),
            "o_std_last": round(o_std_last, 4),
            "o_std_min": round(o_std_min, 4),
            "o_std_change": round(o_std_last - o_std_first, 4),
            "clusters_first": clusters_first,
            "clusters_last": clusters_last,
            "clusters_min": clusters_min,
            "pol_first": round(pol_first, 4),
            "pol_last": round(pol_last, 4),
            "final_n_A": opinion_snapshots["n_A"][-1],
            "final_n_D": opinion_snapshots["n_D"][-1],
            "final_n_U": opinion_snapshots["n_U"][-1],
            "final_h_mean": round(opinion_snapshots["h_mean"][-1], 4),
            "final_f_mean": round(opinion_snapshots["f_mean"][-1], 4),
            "collapsed": collapsed,
            "fragmented": fragmented,
            "elapsed_s": round(elapsed, 1),
            "opinion_snapshots": opinion_snapshots,
        }
        all_results.append(result)

        verdict = ("COLLAPSED" if collapsed else
                   "FRAGMENTED" if fragmented else
                   "STABLE POLARIZED")
        print(f"  seed={seed}: σ_o: {o_std_first:.3f}→{o_std_last:.3f}, "
              f"clusters: {clusters_first}→{clusters_last}, "
              f"pol: {pol_first:.3f}→{pol_last:.3f}, "
              f"A={opinion_snapshots['n_A'][-1]}, "
              f"D={opinion_snapshots['n_D'][-1]}, "
              f"U={opinion_snapshots['n_U'][-1]} "
              f"[{verdict}] ({elapsed:.0f}s)")

    # ── Summary ──
    o_std_changes = [r["o_std_change"] for r in all_results]
    n_collapsed = sum(1 for r in all_results if r["collapsed"])
    n_fragmented = sum(1 for r in all_results if r["fragmented"])
    n_stable = len(all_results) - n_collapsed - n_fragmented

    print(f"\n{'─'*40}")
    print(f"  SUMMARY: ")
    print(f"    Collapsed (σ<0.1): {n_collapsed}/{len(all_results)}")
    print(f"    Fragmented (≥4 clusters): {n_fragmented}/{len(all_results)}")
    print(f"    Stable polarized: {n_stable}/{len(all_results)}")
    print(f"    Mean σ_o change: {np.mean(o_std_changes):.4f}")

    if n_stable >= len(all_results) * 0.5:
        print(f"    VERDICT: Model avoids degenerate states — opinions remain")
        print(f"             polarized without collapsing or fragmenting.")
    elif n_collapsed > 0:
        print(f"    WARNING: Opinions collapsed in {n_collapsed} runs.")
        print(f"             Consider increasing ζ (anchoring) or ε (confidence threshold).")
    else:
        print(f"    WARNING: Opinions fragmented in {n_fragmented} runs.")

    # ── Save ──
    output_path = OUTPUT_DIR / "experiment_longterm_polarization.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Long-term polarization (T=500)",
            "n": n, "T": T, "seeds": SEEDS,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved: {output_path}")

    return all_results


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    run_longterm_experiment()
