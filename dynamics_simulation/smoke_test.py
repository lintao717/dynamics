"""Quick smoke test for the dynamics simulation package."""
import numpy as np

from dynamics_simulation.config import default_params, no_coupling
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents, U, E, A, D

rng = np.random.default_rng(42)

# Test 1: Network generation
print("Test 1: Network generation...")
G_s, G_o, communities = generate_networks(
    "sbm", n=100, rng=rng, n_blocks=3, p_in=0.15, p_out=0.02
)
print(f"  N={G_s.shape[0]}, edges={(G_s>0).sum()}, blocks={len(communities)}")
print(f"  Block sizes: {[len(v) for v in communities.values()]}")
assert G_s.shape == (100, 100)
assert G_o.shape == (100, 100)
assert len(communities) == 3

# Test 2: Agent initialization
print("Test 2: Agent initialization...")
state = initialize_agents(n=100, initial_active=5, rng=rng)
counts = state.state_counts()
print(f"  States: U={counts['U']} E={counts['E']} A={counts['A']} D={counts['D']}")
print(f"  o mean={state.o.mean():.3f} std={state.o.std():.3f}")
assert state.n == 100
assert state.n_A == 5
assert state.n_U == 95

# Test 3: TransitionEngine
print("Test 3: TransitionEngine (single step)...")
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs
from dynamics_simulation.config import ModelParams

params = default_params()
engine = TransitionEngine(params, rng)
o_initial = state.o.copy()
inputs = ExternalInputs()

new_state, _, _ = engine.step(state, G_s, G_o, None, inputs, o_initial, t=0)
new_counts = new_state.state_counts()
print(f"  After 1 step: U={new_counts['U']} E={new_counts['E']} A={new_counts['A']} D={new_counts['D']}")
print(f"  o mean={new_state.o.mean():.3f} h mean={new_state.h.mean():.3f} f mean={new_state.f.mean():.3f}")
# After step, no agents should be in E (transient)
assert new_state.n_E == 0, f"E state should be 0 after step, got {new_state.n_E}"

# Test 4: Full simulation (mini)
print("Test 4: Mini simulation (N=100, T=10)...")
from dynamics_simulation.simulation import SimulationConfig, SimulationRunner

cfg = SimulationConfig(
    n_agents=100,
    initial_active=5,
    T=10,
    network_type="sbm",
    network_kwargs={"n_blocks": 3, "p_in": 0.15, "p_out": 0.02},
    params=default_params(),
    seed=42,
    snapshot_interval=1,
)
runner = SimulationRunner(cfg)
metrics = runner.run()

print(f"  Peak A: {metrics.peak_A} at step {metrics.peak_A_step}")
print(f"  Final o std: {metrics.final_opinion_std:.4f}")
print(f"  Max public bias: {metrics.max_public_bias:.4f}")
print(f"  Reactivations: {metrics.total_reactivations}")
assert metrics.peak_A > 0, "Should have some active agents"
assert len(metrics.steps) > 0, "Should have time series data"

print("\nALL SMOKE TESTS PASSED")
