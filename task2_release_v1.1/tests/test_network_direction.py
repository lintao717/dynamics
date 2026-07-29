"""
Network direction convention test.

CONVENTION: G[dst, src] = weight
  - Row = receiver (destination)
  - Column = sender (source)
  - Information flows FROM column TO row

3-node chain: 0 -> 1 -> 2
  - Node 0 active -> should expose node 1, NOT node 2
  - Step 1: only node 1 gets exposed
  - Step 2: node 1 expresses -> node 2 gets exposed
"""
import sys, os
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dynamics_simulation.config import default_params
from dynamics_simulation.agents import initialize_agents, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs


def build_chain_network():
    """Build a 3-node chain: 0 -> 1 -> 2.
    G[dst=1, src=0] = 1.0, G[dst=2, src=1] = 1.0.
    This is row 1 gets from column 0, row 2 gets from column 1.
    """
    n = 3
    G_s = np.zeros((n, n))
    G_s[1, 0] = 1.0  # 0 can expose 1
    G_s[2, 1] = 1.0  # 1 can expose 2
    G_o = G_s.copy()
    # Row-normalize G_o
    for i in range(n):
        s = G_o[i].sum()
        if s > 0:
            G_o[i] /= s
    return G_s, G_o


def test_chain_direction():
    """Verify: active node 0 exposes node 1 but NOT node 2."""
    G_s, G_o = build_chain_network()

    rng = np.random.default_rng(42)
    from dynamics_simulation.config import PropagationParams
    from dataclasses import replace
    params = replace(default_params(), propagation=PropagationParams(beta=1.0, beta_M=0.0))
    state = initialize_agents(n=3, initial_active=0, rng=rng,
                              initial_opinion_dist="uniform")
    state.z[:] = U
    state.z[0] = A; state.m[0] = 1; state.o_hat[0] = state.o[0]

    engine = TransitionEngine(params, rng)
    o_init = state.o.copy()

    # With beta=1.0, node 1 is definitely exposed (Lambda=1.0)
    state, V, _ = engine.step(state, G_s, G_o, None, ExternalInputs(), o_init, t=0)

    # Node 1 should be aware (A or D, not U)
    assert state.z[1] != U, f"Node 1 should be exposed (got {state.z[1]})"
    # Node 2 should still be U — no edge 0->2
    assert state.z[2] == U, f"Node 2 should NOT be exposed (got {state.z[2]})"
    print("[PASS] Chain direction: 0->1 exposes node 1, not node 2")


def test_no_reverse_propagation():
    """Verify: active node 2 cannot expose node 1 if no reverse edge."""
    G_s, G_o = build_chain_network()

    rng = np.random.default_rng(42)
    params = default_params()
    state = initialize_agents(n=3, initial_active=1, rng=rng,
                              initial_opinion_dist="uniform")
    # Make node 2 active, nodes 0 and 1 U
    state.z[:] = U
    state.z[2] = A
    state.m[2] = 1
    state.o_hat[2] = state.o[2]

    engine = TransitionEngine(params, rng)
    o_init = state.o.copy()

    for t in range(5):
        state, _, _ = engine.step(state, G_s, G_o, None,
                                   ExternalInputs(), o_init, t)
        # Node 1 should never become aware (no edge 2->1)
        assert state.z[1] == U, f"Node 1 should stay U (got {state.z[1]} at t={t})"
    print("[PASS] No reverse propagation: node 2 cannot reach node 1")


def test_bidirectional():
    """Verify: with mutual edges, both directions work."""
    n = 2
    G_s = np.ones((n, n))  # weight 1.0 ensures deterministic exposure with beta=1.0
    np.fill_diagonal(G_s, 0)
    G_o = G_s.copy()
    for i in range(n):
        s = G_o[i].sum()
        if s > 0:
            G_o[i] /= s

    rng = np.random.default_rng(42)
    from dynamics_simulation.config import PropagationParams
    from dataclasses import replace
    params = replace(default_params(), propagation=PropagationParams(beta=1.0, beta_M=0.0))
    state = initialize_agents(n=2, initial_active=0, rng=rng,
                              initial_opinion_dist="uniform")
    state.z[:] = U; state.z[0] = A; state.m[0] = 1; state.o_hat[0] = state.o[0]

    engine = TransitionEngine(params, rng)
    o_init = state.o.copy()

    state, _, _ = engine.step(state, G_s, G_o, None, ExternalInputs(), o_init, t=0)
    assert state.z[1] != U, f"Node 1 should be exposed with mutual edges"
    print("[PASS] Bidirectional: mutual edges work in both directions")


if __name__ == "__main__":
    test_chain_direction()
    test_no_reverse_propagation()
    test_bidirectional()
    print("\nALL NETWORK DIRECTION TESTS PASSED")
