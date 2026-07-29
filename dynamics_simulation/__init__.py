"""
Integrated Propagation-Opinion Dynamics Simulation Kernel.

Implements the 7-step coupled update cycle defined in:
    docs/task2_model_definition_v1.md

State vector X_i(t) = (z_i, o_i, o_hat_i, h_i, f_i)
- z_i in {U=0, E=1, A=2, D=3}: propagation state
- o_i in [-1, 1]: private opinion
- o_hat_i in [-1, 1] U {NaN}: public expression
- h_i in [0, 1]: emotional arousal
- f_i in [0, 1]: information fatigue

Public API (for Task 3 integration):
    from dynamics_simulation.api import Simulation, TextGenerationRequest, GeneratedText
    sim = Simulation.init(n_agents=500, params="default")
    metrics = sim.step()
    requests = sim.get_text_requests()  # agents in state A -> LLM prompt

Quick start:
    from dynamics_simulation.api import Simulation
    sim = Simulation.init(n_agents=300, params="default")
    for t in range(100):
        metrics = sim.step()
    print(sim.summary())
"""

__version__ = "1.1.0"

# Public API exports (LLM integration is text-generation only)
from dynamics_simulation.api import (
    Simulation,
    AgentSnapshot,
    TextGenerationRequest,
    GeneratedText,
    StepMetrics,
    load_params_from_json,
    create_llm_prompt,
)
