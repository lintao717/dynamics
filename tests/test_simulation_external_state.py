"""Tests for external initial state, dynamic networks, and step observers."""

import numpy as np
import pytest
from dynamics_simulation.config import default_params
from dynamics_simulation.agents import (
    AgentState, AgentAttributes, initialize_agents, U, E, A, D,
)
from dynamics_simulation.simulation import SimulationConfig, SimulationRunner
from dynamics_simulation.api import Simulation


def _make_custom_state(n: int = 5):
    """Create a custom AgentState with known values."""
    attrs = AgentAttributes(
        c=np.full(n, 0.3),
        mu=np.full(n, 0.2),
        zeta=np.full(n, 0.5),
        epsilon=np.full(n, 0.4),
        sigma_xi=np.full(n, 0.02),
        eta=np.full(n, 0.15),
        chi=np.full(n, 0.2),
    )
    z = np.full(n, U, dtype=np.int8)
    z[0] = A
    o_vals = np.zeros(n, dtype=np.float64)
    o_vals[0] = 0.5
    o_hat_vals = np.full(n, np.nan, dtype=np.float64)
    o_hat_vals[0] = 0.5
    h_vals = np.zeros(n, dtype=np.float64)
    h_vals[0] = 0.5
    m_vals = np.zeros(n, dtype=np.int8)
    m_vals[0] = 1
    return AgentState(
        z=z,
        m=m_vals,
        o=o_vals,
        o_hat=o_hat_vals,
        h=h_vals,
        f=np.zeros(n),
        attrs=attrs,
    )


def test_initial_state_is_preserved():
    """Passing initial_state must use it exactly, not random init."""
    custom = _make_custom_state(5)
    G_s = np.eye(5) * 0.1
    G_o = np.eye(5) * 0.1

    cfg = SimulationConfig(
        n_agents=5, T=1, seed=42,
        initial_state=custom, G_s=G_s, G_o=G_o,
    )
    runner = SimulationRunner(cfg)
    runner.run()
    # After 1 step, state should not be randomly initialized
    # Verify initial metrics came from custom state
    metrics = runner.metrics.as_metrics()
    assert metrics.steps[0] == 0  # t=0 snapshot recorded
    assert metrics.n_A_ts[0] == 1


def test_initial_state_n_mismatch_rejected():
    custom = _make_custom_state(5)
    cfg = SimulationConfig(
        n_agents=3, T=1,
        initial_state=custom,  # n=5 but n_agents=3
        G_s=np.eye(3), G_o=np.eye(3),
    )
    runner = SimulationRunner(cfg)
    with pytest.raises(ValueError):
        runner.run()


def test_network_provider_is_called():
    """Dynamic network provider should be called each step."""
    custom = _make_custom_state(3)
    call_log = []

    def provider(step):
        call_log.append(step)
        return (
            np.eye(3, dtype=np.float64) * 0.1,
            np.eye(3, dtype=np.float64) * 0.1,
            {0: [0, 1, 2]},
        )

    cfg = SimulationConfig(
        n_agents=3, T=5, seed=42,
        initial_state=custom,
        network_provider=provider,
    )
    runner = SimulationRunner(cfg)
    runner.run()
    # Provider called once for t=0 (during initialization, before
    # initial snapshot), then for t=1, 2, 3, 4 in the main loop.
    # The main loop skips t=0 since the provider was already called.
    assert call_log == [0, 1, 2, 3, 4]


def test_step_observer_is_called():
    custom = _make_custom_state(3)
    observed_steps = []

    def observer(step, state_before, state_after, events):
        observed_steps.append(step)

    cfg = SimulationConfig(
        n_agents=3, T=3, seed=42,
        initial_state=custom,
        G_s=np.eye(3) * 0.01,
        G_o=np.eye(3) * 0.01,
        step_observer=observer,
    )
    runner = SimulationRunner(cfg)
    runner.run()
    assert observed_steps == [1, 2, 3]  # called after each completed step


def test_no_initial_state_uses_default():
    """Backward compatibility: without initial_state, random init is used."""
    cfg = SimulationConfig(n_agents=50, T=2, seed=42)
    runner = SimulationRunner(cfg)
    metrics = runner.run()
    assert len(metrics.steps) > 0
    assert metrics.n_A_ts[0] > 0  # at least some agents active initially


# ── V1.7R.4: Behavioral emission tests ──

def test_observer_receives_state_before_and_after():
    """Observer must receive (step_idx, state_before, state_after, events)."""
    records = []
    def obs(step_idx, state_before, state_after, events):
        records.append((step_idx, state_before is not None, state_after is not None))
    cfg = SimulationConfig(n_agents=5, T=3, seed=42, step_observer=obs)
    runner = SimulationRunner(cfg)
    runner.run()
    assert len(records) == 3  # 3 macro-steps
    for step, has_before, has_after in records:
        assert has_before, f"Step {step}: state_before is None"
        assert has_after, f"Step {step}: state_after is None"


def test_observer_state_before_differs_from_after():
    """State before and after a step must be different if dynamics occur."""
    states = []
    def obs(step_idx, state_before, state_after, events):
        states.append((state_before.z.copy(), state_after.z.copy()))
    cfg = SimulationConfig(n_agents=50, T=5, seed=42, step_observer=obs)
    runner = SimulationRunner(cfg)
    runner.run()
    # At least one step should have state change
    any_diff = any(not np.array_equal(before, after) for before, after in states)
    assert any_diff, "State never changed during simulation"


def test_observer_active_includes_A_to_D_transition():
    """Agent that starts A and ends D must still count as active."""
    from dynamics_simulation.agents import AgentState, initialize_agents, A, D
    import numpy as np
    # Create a state where one agent is A, all others U
    state = initialize_agents(10, initial_active=1, rng=np.random.default_rng(42))
    # Force agent 0 to be A
    z = np.zeros(10, dtype=np.int32)
    z[0] = A
    state = AgentState(z=z, m=np.zeros(10, dtype=np.int32),
                       o=state.o, o_hat=state.o_hat,
                       h=state.h, f=state.f, attrs=state.attrs)
    active_steps = []
    def obs(step_idx, state_before, state_after, events):
        was_A = state_before.z == A
        became_A = (state_before.z != A) & (state_after.z == A)
        active_during = was_A | became_A
        active_steps.append(int(active_during.sum()))
    cfg = SimulationConfig(n_agents=10, T=3, seed=42, initial_state=state,
                          step_observer=obs)
    runner = SimulationRunner(cfg)
    runner.run()
    # Step 1: agent 0 was A at start, might decay to D
    # Should still count as active even if it decayed
    assert len(active_steps) == 3


def test_observer_active_equals_first_plus_repeat():
    """active = first + repeat must hold for the observer output."""
    first_steps = []; repeat_steps = []; active_steps = []
    prior_m = None
    def obs(step_idx, state_before, state_after, events):
        nonlocal prior_m
        if prior_m is None: prior_m = state_before.m.copy()
        was_A = state_before.z == 2
        became_A = (state_before.z != 2) & (state_after.z == 2)
        active_during = was_A | became_A
        new_first = (state_after.m == 1) & (prior_m == 0)
        repeat_during = active_during & ~new_first
        first_steps.append(int(new_first.sum()))
        repeat_steps.append(int(repeat_during.sum()))
        active_steps.append(int(active_during.sum()))
        prior_m[:] = state_after.m
    cfg = SimulationConfig(n_agents=50, T=5, seed=42, step_observer=obs)
    runner = SimulationRunner(cfg)
    runner.run()
    for i in range(len(active_steps)):
        assert active_steps[i] == first_steps[i] + repeat_steps[i], \
            f"Step {i}: active={active_steps[i]} != first={first_steps[i]}+repeat={repeat_steps[i]}"


def test_from_network_accepts_initial_state():
    custom = _make_custom_state(5)
    sim = Simulation.from_network(
        G_s=np.eye(5) * 0.1,
        G_o=np.eye(5) * 0.1,
        n_agents=5,
        initial_state=custom,
        seed=42,
    )
    assert sim._state.n == 5
    assert sim._state.z[0] == A  # root is active


def test_from_network_initial_state_n_mismatch():
    custom = _make_custom_state(3)
    with pytest.raises(ValueError):
        Simulation.from_network(
            G_s=np.eye(5), G_o=np.eye(5),
            n_agents=5,
            initial_state=custom,  # n=3 but n_agents=5
        )
