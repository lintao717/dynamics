"""V1.1 Smoke Tests: package import, init, single-step, mini-simulation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dynamics_simulation.config import default_params
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs


def test_network_generation():
    rng = np.random.default_rng(42)
    G_s, G_o, comms = generate_networks("sbm", n=100, rng=rng, n_blocks=3, p_in=0.15, p_out=0.02)
    assert G_s.shape == (100, 100)
    assert len(comms) == 3
    assert (G_s > 0).sum() > 0

def test_agent_initialization():
    rng = np.random.default_rng(42)
    state = initialize_agents(n=100, initial_active=5, rng=rng)
    assert state.n == 100
    assert state.n_A == 5
    assert state.n_U == 95
    assert state.m.sum() == 5  # active agents have m=1

def test_single_step():
    rng = np.random.default_rng(42)
    G_s, G_o, _ = generate_networks("sbm", n=100, rng=rng, n_blocks=3, p_in=0.15, p_out=0.02)
    state = initialize_agents(n=100, initial_active=5, rng=rng)
    engine = TransitionEngine(default_params(), rng)
    o_init = state.o.copy()
    new_state, V, events = engine.step(state, G_s, G_o, None, ExternalInputs(), o_init, t=0)
    assert new_state.n_E == 0, f"E should be 0 after step, got {new_state.n_E}"
    assert 0.0 <= V <= 1.0

def test_m_flag_propagation():
    """V1.1: m_i flag should be set to 1 for newly activated agents."""
    rng = np.random.default_rng(42)
    G_s, G_o, _ = generate_networks("sbm", n=100, rng=rng, n_blocks=3, p_in=0.15, p_out=0.02)
    state = initialize_agents(n=100, initial_active=3, rng=rng)
    assert state.m.sum() == 3  # initial actives have m=1
    engine = TransitionEngine(default_params(), rng)
    o_init = state.o.copy()
    for t in range(10):
        state, _, events = engine.step(state, G_s, G_o, None, ExternalInputs(), o_init, t)
    # All A agents should have m=1
    assert (state.m[state.z == A] == 1).all()

def test_mini_simulation():
    from dynamics_simulation.simulation import SimulationConfig, SimulationRunner
    cfg = SimulationConfig(n_agents=100, initial_active=5, T=10, network_type="sbm",
                           params=default_params(), seed=42, verbose=False)
    runner = SimulationRunner(cfg)
    metrics = runner.run()
    assert metrics.peak_A > 0
    assert len(metrics.steps) > 0

if __name__ == "__main__":
    test_network_generation(); print("[PASS] Network generation")
    test_agent_initialization(); print("[PASS] Agent initialization")
    test_single_step(); print("[PASS] Single step")
    test_m_flag_propagation(); print("[PASS] m_i flag propagation")
    test_mini_simulation(); print("[PASS] Mini simulation")
    print("\nALL SMOKE TESTS PASSED")
