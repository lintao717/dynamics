"""
Run all 6 verification experiments for the integrated dynamics model.

Usage:
    python -m dynamics_simulation.experiments.run_all

Output:
    data/sim_results/experiment_*.json    — Raw metrics
    data/sim_results/figures/             — Plots (PNG)

Each experiment runs with 3 random seeds for reproducibility assessment.
"""

from __future__ import annotations

import sys, os, json, time
from pathlib import Path
from typing import Callable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from dataclasses import replace

from dynamics_simulation.config import (
    ModelParams, default_params, no_coupling, one_way_coupling,
    no_silence_spiral, strong_silence_spiral,
    high_propagation, low_propagation,
    PropagationParams, ActivationParams, DecayParams,
    ReactivationParams, OpinionParams, EmotionFatigueParams,
)
from dynamics_simulation.simulation import SimulationConfig, SimulationRunner
from dynamics_simulation.transitions import ExternalInputs
from dynamics_simulation.metrics import SimulationMetrics

# ── Output paths ──
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "sim_results"
FIGURE_DIR = OUTPUT_DIR / "figures"
SEEDS = [42, 123, 789]  # Multiple seeds for robustness


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# External input functions for shock experiments
# ═══════════════════════════════════════════════════════════════

def shock_at_step_40(n: int, t: int, T: int) -> ExternalInputs:
    """Inject a strong external shock at t=40."""
    return ExternalInputs(
        shock=0.85 if t == 40 else (0.15 if t == 41 else 0.0),
        novelty=0.6 if t == 40 else 0.0,
        staleness=t / max(T, 1),
    )


def double_shock(n: int, t: int, T: int) -> ExternalInputs:
    """Two shocks: one at t=25, another at t=55."""
    shock = 0.0
    novelty = 0.0
    if t == 25:
        shock = 0.7
        novelty = 0.5
    elif t == 55:
        shock = 0.9
        novelty = 0.7
    return ExternalInputs(
        shock=shock,
        novelty=novelty,
        staleness=t / max(T, 1),
    )


def gradual_novelty(n: int, t: int, T: int) -> ExternalInputs:
    """Novelty that decays exponentially from t=0."""
    novelty = np.exp(-t / 15.0)
    return ExternalInputs(
        shock=0.0,
        novelty=novelty,
        staleness=t / max(T, 1),
    )


# ═══════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════


def run_experiment(
    name: str,
    configs: list[SimulationConfig],
    description: str = "",
) -> dict:
    """Run an experiment with multiple configs and seeds, save results.

    Args:
        name: Experiment name (used for output files).
        configs: List of SimulationConfig objects (one per condition).
        description: Human-readable description.

    Returns:
        Dict with experiment results.
    """
    ensure_dirs()
    all_results = []

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    if description:
        print(f"  {description}")
    print(f"{'='*60}")

    for ci, cfg in enumerate(configs):
        cfg_label = cfg.params.__class__.__name__ if hasattr(cfg.params, '__class__') else f"cond_{ci}"
        for seed in SEEDS:
            cfg.seed = seed
            cfg.verbose = False  # Quiet mode for batch runs

            t0 = time.perf_counter()
            runner = SimulationRunner(cfg)
            metrics = runner.run()
            elapsed = time.perf_counter() - t0

            result = {
                "experiment": name,
                "condition": ci,
                "seed": seed,
                "peak_A": metrics.peak_A,
                "peak_A_step": metrics.peak_A_step,
                "final_o_std": metrics.final_opinion_std,
                "max_public_bias": metrics.max_public_bias,
                "total_activations": metrics.total_activations,
                "total_reactivations": metrics.total_reactivations,
                "elapsed_s": round(elapsed, 2),
            }

            # Include time series for first seed only (keep output manageable)
            if seed == SEEDS[0]:
                result["time_series"] = {
                    "steps": metrics.steps.tolist(),
                    "n_A": metrics.n_A_ts.tolist(),
                    "n_D": metrics.n_D_ts.tolist(),
                    "o_mean": metrics.o_mean_ts.tolist(),
                    "o_std": metrics.o_std_ts.tolist(),
                    "h_mean": metrics.h_mean_ts.tolist(),
                    "f_mean": metrics.f_mean_ts.tolist(),
                    "public_bias": metrics.public_bias_ts.tolist(),
                }

            all_results.append(result)

            print(f"  [{cfg_label}] seed={seed}: peak_A={metrics.peak_A} "
                  f"@t={metrics.peak_A_step}, B_obs={metrics.max_public_bias:.3f}, "
                  f"react={metrics.total_reactivations} ({elapsed:.1f}s)")

    # Save results
    output_path = OUTPUT_DIR / f"experiment_{name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": name,
            "description": description,
            "seeds": SEEDS,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  Saved: {output_path}")

    return {"name": name, "results": all_results}


# ═══════════════════════════════════════════════════════════════
# Experiment 1: Basic U-E-A-D propagation (no opinion coupling)
# ═══════════════════════════════════════════════════════════════

def experiment_1_basic_propagation():
    """Verify that the basic U→E→A/D dynamics work without opinion coupling.

    Tests across 4 network types. All use no_coupling params so that
    propagation is independent of opinion state.
    """
    configs = []
    for net_type, net_kwargs in [
        ("er", {"p": 0.05}),
        ("ba", {"m": 5}),
        ("ws", {"k": 10, "p_rewire": 0.10}),
        ("sbm", {"n_blocks": 3, "p_in": 0.15, "p_out": 0.02}),
    ]:
        configs.append(SimulationConfig(
            n_agents=500,
            initial_active=10,
            T=100,
            network_type=net_type,
            network_kwargs=net_kwargs,
            params=no_coupling(),
            verbose=False,
        ))
    return run_experiment(
        "1_basic_propagation",
        configs,
        "Basic U-E-A-D propagation across 4 network types (opinion coupling OFF)",
    )


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Opinion influence on propagation
# ═══════════════════════════════════════════════════════════════

def experiment_2_opinion_influence():
    """Test whether opinion extremity (α_1) changes propagation peaks.

    Compares: α_1=0 (no opinion effect) vs α_1=1.5 (moderate) vs α_1=3.0 (strong).
    All on SBM networks.
    """
    configs = []
    for alpha_1, label in [(0.0, "none"), (1.5, "moderate"), (3.0, "strong")]:
        p = replace(default_params(),
            activation=ActivationParams(alpha_1=alpha_1)
        )
        configs.append(SimulationConfig(
            n_agents=500, initial_active=10, T=100,
            network_type="sbm",
            network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
            params=p, verbose=False,
        ))
    return run_experiment(
        "2_opinion_influence",
        configs,
        "Effect of opinion extremity (α_1) on propagation: none vs moderate vs strong",
    )


# ═══════════════════════════════════════════════════════════════
# Experiment 3: Silence spiral
# ═══════════════════════════════════════════════════════════════

def experiment_3_silence_spiral():
    """Compare λ_spiral=0 vs λ_spiral=0.5 vs λ_spiral=0.85.

    Measures: public opinion bias B_obs, proportion of minority expressing.
    """
    configs = []
    for lam, label in [(0.0, "none"), (0.50, "moderate"), (0.85, "strong")]:
        p = replace(default_params(),
            opinion=OpinionParams(lambda_spiral=lam)
        )
        configs.append(SimulationConfig(
            n_agents=500, initial_active=10, T=100,
            network_type="sbm",
            network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
            params=p, verbose=False,
        ))
    return run_experiment(
        "3_silence_spiral",
        configs,
        "Silence spiral strength: λ_spiral = 0 vs 0.5 vs 0.85",
    )


# ═══════════════════════════════════════════════════════════════
# Experiment 4: Shock reactivation
# ═══════════════════════════════════════════════════════════════

def experiment_4_shock_reactivation():
    """Inject a shock at t=40 and verify D→A reactivation produces a second peak.

    Compares: no shock vs single shock at t=40 vs double shock at t=25,55.
    """
    configs = []
    p = default_params()

    # No shock (baseline)
    configs.append(SimulationConfig(
        n_agents=500, initial_active=10, T=100,
        network_type="sbm",
        network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
        params=p, verbose=False,
    ))

    # Single shock at t=40
    configs.append(SimulationConfig(
        n_agents=500, initial_active=10, T=100,
        network_type="sbm",
        network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
        params=p, verbose=False,
        input_fn=shock_at_step_40,
    ))

    # Double shock
    configs.append(SimulationConfig(
        n_agents=500, initial_active=10, T=100,
        network_type="sbm",
        network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
        params=p, verbose=False,
        input_fn=double_shock,
    ))

    return run_experiment(
        "4_shock_reactivation",
        configs,
        "D→A reactivation: no shock vs single shock (t=40) vs double shock (t=25,55)",
    )


# ═══════════════════════════════════════════════════════════════
# Experiment 5: Bidirectional coupling comparison
# ═══════════════════════════════════════════════════════════════

def experiment_5_bidirectional_coupling():
    """Compare three coupling modes:
    - no_coupling: propagation ⊥ opinion (independent)
    - one_way: opinion → propagation but not propagation → opinion
    - two_way (default): full bidirectional coupling

    Key metrics: opinion clustering, public bias, propagation dynamics differences.
    """
    configs = []
    for p, label in [
        (no_coupling(), "no_coupling"),
        (one_way_coupling(), "one_way"),
        (default_params(), "two_way"),
    ]:
        configs.append(SimulationConfig(
            n_agents=500, initial_active=10, T=100,
            network_type="sbm",
            network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
            params=p, verbose=False,
        ))
    return run_experiment(
        "5_bidirectional_coupling",
        configs,
        "Coupling modes: none vs one-way (opinion→prop) vs two-way (default)",
    )


# ═══════════════════════════════════════════════════════════════
# Experiment 6: Parameter sensitivity
# ═══════════════════════════════════════════════════════════════

def experiment_6_parameter_sensitivity():
    """Sweep key parameters to assess model sensitivity.

    Dimensions:
      - Network: p_in ∈ {0.05, 0.15, 0.30} (within-block connectivity)
      - Network: p_out ∈ {0.01, 0.02, 0.05} (cross-block connectivity)
      - Propagation: β ∈ {0.05, 0.12, 0.25}
      - Opinion: λ_spiral ∈ {0.0, 0.5}
      - Expression cost: α_5 ∈ {0.5, 1.0, 2.0}

    Uses a reduced set of combinations to keep runtime manageable.
    """
    configs = []

    # Baseline
    configs.append(SimulationConfig(
        n_agents=500, initial_active=10, T=80,
        network_type="sbm",
        network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
        params=default_params(), verbose=False,
    ))

    # Vary p_in (within-block density)
    for p_in in [0.05, 0.30]:
        configs.append(SimulationConfig(
            n_agents=500, initial_active=10, T=80,
            network_type="sbm",
            network_kwargs={"n_blocks": 3, "p_in": p_in, "p_out": 0.02},
            params=default_params(), verbose=False,
        ))

    # Vary p_out (cross-block density)
    for p_out in [0.01, 0.05]:
        configs.append(SimulationConfig(
            n_agents=500, initial_active=10, T=80,
            network_type="sbm",
            network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": p_out},
            params=default_params(), verbose=False,
        ))

    # Vary β
    for beta in [0.05, 0.25]:
        p = replace(default_params(),
            propagation=PropagationParams(beta=beta)
        )
        configs.append(SimulationConfig(
            n_agents=500, initial_active=10, T=80,
            network_type="sbm",
            network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
            params=p, verbose=False,
        ))

    # Vary λ_spiral
    for lam in [0.0, 0.85]:
        p = replace(default_params(),
            opinion=OpinionParams(lambda_spiral=lam)
        )
        configs.append(SimulationConfig(
            n_agents=500, initial_active=10, T=80,
            network_type="sbm",
            network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
            params=p, verbose=False,
        ))

    return run_experiment(
        "6_parameter_sensitivity",
        configs,
        "Parameter sensitivity: p_in, p_out, β, λ_spiral",
    )


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

ALL_EXPERIMENTS = [
    ("1_basic_propagation", experiment_1_basic_propagation),
    ("2_opinion_influence", experiment_2_opinion_influence),
    ("3_silence_spiral", experiment_3_silence_spiral),
    ("4_shock_reactivation", experiment_4_shock_reactivation),
    ("5_bidirectional_coupling", experiment_5_bidirectional_coupling),
    ("6_parameter_sensitivity", experiment_6_parameter_sensitivity),
]


def run_all(experiments: list[str] | None = None):
    """Run all or selected experiments.

    Args:
        experiments: List of experiment names to run. None = run all.
    """
    ensure_dirs()

    print("=" * 60)
    print("  DYNAMICS SIMULATION — VERIFICATION EXPERIMENTS")
    print("  N=500 agents, T=80-100 steps, 3 seeds each")
    print("=" * 60)

    t0 = time.perf_counter()
    all_outputs = {}

    for name, fn in ALL_EXPERIMENTS:
        if experiments is not None and name not in experiments:
            print(f"\n  Skipping: {name}")
            continue
        try:
            result = fn()
            all_outputs[name] = result
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print(f"  Results: {OUTPUT_DIR}")
    print(f"{'='*60}")

    return all_outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run simulation experiments")
    parser.add_argument("--experiments", nargs="*", default=None,
                       help="Experiment names to run (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Override global seed (for debugging)")
    args = parser.parse_args()

    if args.seed is not None:
        SEEDS.clear()
        SEEDS.append(args.seed)

    run_all(args.experiments)
