"""
Example: Task 3 Integration — LLM-Parametric Hybrid Simulation.

This script demonstrates how Task 3 (LLM multi-agent simulation) uses
the dynamics_simulation API to run a hybrid simulation where a subset
of agents are controlled by LLMs and the rest follow parametric rules.

Pattern:
  1. Initialize simulation with llm_agent_ids
  2. Each step:
     a. Get snapshots of LLM agents' states
     b. Send snapshots to LLM (DeepSeek API) for decisions
     c. Inject LLM decisions into the simulation
     d. Step parametric agents
     e. Collect metrics

Usage:
    python dynamics_simulation/examples/task3_integration.py
"""
import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from dynamics_simulation.api import (
    Simulation, AgentSnapshot, LLMDecision, StepMetrics,
    load_params_from_json, create_llm_prompt,
)


def mock_llm_decision(snapshot: AgentSnapshot) -> LLMDecision:
    """Mock LLM decision for demonstration.

    In production, this would call DeepSeek API with the prompt from
    create_llm_prompt(snapshot) and parse the JSON response.

    This mock implements a simple parametric decision rule as a stand-in.
    """
    # Simple threshold rule (stand-in for actual LLM)
    if snapshot.z_name == "A":
        # Already active: decide whether to continue or go silent
        if snapshot.f > 0.7:
            return LLMDecision(agent_id=snapshot.agent_id, action="remain_silent")
        else:
            return LLMDecision(
                agent_id=snapshot.agent_id,
                action="express",
                expressed_opinion=snapshot.o,  # express true opinion
            )
    elif snapshot.z_name == "E":
        # Just exposed: decide whether to engage
        if snapshot.n_active_neighbors > 3 and snapshot.h > 0.3:
            return LLMDecision(
                agent_id=snapshot.agent_id,
                action="express",
                expressed_opinion=snapshot.o,
            )
        else:
            return LLMDecision(agent_id=snapshot.agent_id, action="remain_silent")
    elif snapshot.z_name == "D":
        # Dormant: decide whether to re-engage
        if snapshot.local_climate * snapshot.o > 0:
            # Climate aligns with opinion -> safer to express
            return LLMDecision(
                agent_id=snapshot.agent_id,
                action="express",
                expressed_opinion=snapshot.o,
            )
        else:
            return LLMDecision(agent_id=snapshot.agent_id, action="remain_silent")
    else:
        # U state: no decision needed (parametric handles exposure)
        return LLMDecision(agent_id=snapshot.agent_id, action="no_change")


def run_hybrid_simulation():
    """Run a hybrid LLM-parametric simulation."""
    print("=" * 60)
    print("  TASK 3 INTEGRATION EXAMPLE")
    print("  Hybrid LLM-Parametric Simulation")
    print("=" * 60)

    # ── Configuration ──
    n_agents = 300
    n_llm_agents = 30
    T = 60
    seed = 42

    # Select which agents are LLM-controlled (every 10th agent)
    llm_ids = list(range(0, n_agents, n_agents // n_llm_agents))[:n_llm_agents]

    # ── Initialize ──
    print(f"\nInitializing: N={n_agents}, LLM agents={len(llm_ids)}")
    sim = Simulation.init(
        n_agents=n_agents,
        params="default",
        network="sbm",
        network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
        initial_opinion="polarized",
        initial_active=10,
        llm_agent_ids=llm_ids,
        seed=seed,
    )
    print(f"  Network: {(sim._G_s > 0).sum()} edges, "
          f"k_mean={sim._G_s.sum(axis=0).mean():.1f}")

    # ── Main loop ──
    print(f"\nRunning {T} steps...")
    t0 = time.perf_counter()

    for t in range(T):
        # Step 1: Get LLM agent snapshots
        snapshots = sim.get_agent_snapshots(llm_ids)

        # Step 2: Get LLM decisions (mock in this example)
        decisions = [mock_llm_decision(s) for s in snapshots]
        decisions = [d for d in decisions if d.action != "no_change"]

        # Step 3: Inject decisions
        sim.inject_llm_decisions(decisions)

        # Step 4: Step parametric agents with optional shock
        shock = 0.8 if t == 25 else 0.0  # Inject shock at t=25
        metrics = sim.step(shock=shock, novelty=np.exp(-t / 30.0))

        # Step 5: Report progress
        if t % 10 == 9 or t == 0:
            n_llm_active = sum(1 for aid in llm_ids if sim._state.z[aid] == 2)
            print(f"  t={t+1:3d}: A={metrics.n_A:4d} D={metrics.n_D:4d} "
                  f"LLM_active={n_llm_active:3d} "
                  f"o_mean={metrics.o_mean:+.3f} bias={metrics.public_bias:.3f}")

    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # ── Results ──
    summary = sim.summary()
    print(f"\n── Final State ──")
    print(f"  U={summary['state_counts']['U']} "
          f"A={summary['state_counts']['A']} "
          f"D={summary['state_counts']['D']}")
    print(f"  o_mean={summary['o_mean']:+.3f}  h_mean={summary['h_mean']:.3f}")

    # ── LLM agent activity analysis ──
    llm_snapshots_final = sim.get_agent_snapshots(llm_ids)
    n_llm_A = sum(1 for s in llm_snapshots_final if s.z_name == "A")
    n_llm_D = sum(1 for s in llm_snapshots_final if s.z_name == "D")
    print(f"\n── LLM Agent States (t={T}) ──")
    print(f"  Active: {n_llm_A}/{len(llm_ids)}")
    print(f"  Dormant: {n_llm_D}/{len(llm_ids)}")
    print(f"  Mean opinion: {np.mean([s.o for s in llm_snapshots_final]):+.3f}")

    # ── Export ──
    output_dir = Path(__file__).parent.parent.parent / "data" / "sim_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    sim.export_metrics(str(output_dir / "task3_example_metrics.json"))

    # Compare to pure parametric run
    print(f"\n── Comparison: Pure Parametric (no LLM agents) ──")
    sim2 = Simulation.init(
        n_agents=n_agents, params="default", network="sbm",
        network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
        initial_opinion="polarized", initial_active=10, seed=seed,
    )
    for t in range(T):
        shock = 0.8 if t == 25 else 0.0
        sim2.step(shock=shock, novelty=np.exp(-t / 30.0))
    s2 = sim2.summary()
    print(f"  Pure parametric: A={s2['state_counts']['A']} "
          f"D={s2['state_counts']['D']} o_mean={s2['o_mean']:+.3f}")
    print(f"  Hybrid:          A={summary['state_counts']['A']} "
          f"D={summary['state_counts']['D']} o_mean={summary['o_mean']:+.3f}")

    return sim


if __name__ == "__main__":
    run_hybrid_simulation()
