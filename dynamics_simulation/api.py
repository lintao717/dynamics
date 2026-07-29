"""
Public API for Task 3 (LLM Multi-Agent Simulation) integration.

Provides a clean interface for:
  - Initializing agent populations with configurable parameters
  - Running single-step state transitions
  - Reading/writing individual agent states
  - Injecting external shocks and media events
  - Querying network neighborhoods
  - Exporting aggregate metrics

Design principle: Task 3 controls the simulation loop. This API provides
stateless transition functions that Task 3 calls per step.

Typical Task 3 integration pattern:

    sim = Simulation.init(config)
    for t in range(T):
        # 1. Query agent states for LLM prompting
        states = sim.get_agent_states(agent_ids)
        # 2. LLM agents make decisions
        llm_decisions = call_llm(states, context)
        # 3. Inject LLM decisions into the state
        sim.inject_llm_decisions(llm_decisions)
        # 4. Run one step for parametric agents
        sim.step()
        # 5. Collect metrics
        metrics = sim.get_metrics()
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import json

from dynamics_simulation.config import ModelParams, default_params
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import (
    AgentState, AgentAttributes, initialize_agents,
    U, E, A, D, STATE_NAMES,
)
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs


@dataclass
class AgentSnapshot:
    """Lightweight snapshot of one agent's state for LLM prompting."""
    agent_id: int
    z: int           # 0=U, 1=E, 2=A, 3=D
    z_name: str      # "U", "E", "A", "D"
    o: float         # private opinion [-1, 1]
    o_hat: float     # public expression (NaN if not A)
    h: float         # emotional arousal [0, 1]
    f: float         # information fatigue [0, 1]
    c: float         # expression cost (fixed trait)
    n_active_neighbors: int
    local_climate: float       # mean opinion of active neighbors
    climate_visible: bool      # whether climate can be perceived
    climate_visibility: float  # v_i value [0, 1]


@dataclass
class TextGenerationRequest:
    """Request for LLM text generation for an agent in state A.

    The dynamics kernel has ALREADY determined that this agent is active (z=A)
    and has ALREADY computed its expressed opinion o_hat. The LLM's job is
    purely linguistic: generate text consistent with the pre-computed stance.
    """
    agent_id: int
    target_stance: float       # o_hat value to express (-1 to +1)
    target_arousal: float      # emotional intensity [0,1]
    private_opinion: float     # o_i for context
    local_climate: float       # perceived opinion climate
    climate_visible: bool
    event_context: str = ""    # description of current event state
    facts: list = None         # relevant facts/updates for this step


@dataclass
class GeneratedText:
    """LLM-generated text with validation scores."""
    agent_id: int
    text: str                  # the generated post/comment
    stance_score: float        # NLP-computed stance [-1,1] for validation
    arousal_score: float       # NLP-computed arousal [0,1] for validation


@dataclass
class StepMetrics:
    """Aggregate metrics after one simulation step."""
    step: int
    n_U: int
    n_E: int
    n_A: int
    n_D: int
    o_mean: float
    o_std: float
    h_mean: float
    f_mean: float
    public_bias: float
    mean_active_neighbors: float
    V: float  # platform viral intensity


class Simulation:
    """Main simulation interface for Task 3.

    Usage:
        sim = Simulation.init(n_agents=500, params="default", network="sbm")
        for t in range(100):
            # LLM agents decide (Task 3's responsibility)
            snapshots = sim.get_agent_snapshots(llm_agent_ids)
            decisions = your_llm_function(snapshots)
            sim.inject_llm_decisions(decisions)
            # Step parametric agents
            metrics = sim.step()
    """

    def __init__(self):
        self._state: Optional[AgentState] = None
        self._G_s: Optional[np.ndarray] = None
        self._G_o: Optional[np.ndarray] = None
        self._communities: Optional[Dict[int, List[int]]] = None
        self._engine: Optional[TransitionEngine] = None
        self._params: Optional[ModelParams] = None
        self._o_initial: Optional[np.ndarray] = None
        self._t: int = 0
        self._rng: Optional[Generator] = None
        self._llm_texts: Dict[int, str] = {}  # agent_id -> generated text (per step)
        self._metrics_history: List[StepMetrics] = []
        self._V: float = 0.0  # platform viral intensity, persisted across steps

    # ═══════════════════════════════════════════════════════════
    # Initialization
    # ═══════════════════════════════════════════════════════════

    @classmethod
    def init(
        cls,
        n_agents: int = 500,
        params: str | ModelParams = "default",
        network: str = "sbm",
        network_kwargs: Optional[dict] = None,
        initial_opinion: str = "polarized",
        initial_active: int = 10,
        seed: int = 42,
    ) -> "Simulation":
        """Initialize a simulation with the given configuration.

        The dynamics kernel controls ALL state transitions. LLM integration
        is for TEXT GENERATION only — agents in state A can have their
        public expressions rendered as natural language by an LLM.

        Args:
            n_agents: Number of agents in the population.
            params: "default", a preset name, or a ModelParams instance.
            network: "er", "ba", "ws", or "sbm".
            network_kwargs: Passed to the network generator.
            initial_opinion: "polarized", "uniform", "moderate", or "consensus".
            initial_active: Number of initially active agents.
            seed: Random seed for reproducibility.

        Returns:
            Initialized Simulation ready for step() calls.
        """
        sim = cls()
        sim._rng = np.random.default_rng(seed)

        # Resolve params
        if isinstance(params, str):
            from dynamics_simulation.config import PRESETS
            sim._params = PRESETS.get(params, default_params())
        else:
            sim._params = params

        # Generate network
        if network_kwargs is None:
            network_kwargs = {}
        sim._G_s, sim._G_o, sim._communities = generate_networks(
            network_type=network,
            n=n_agents,
            rng=sim._rng,
            **network_kwargs,
        )

        # Initialize agents
        sim._state = initialize_agents(
            n=n_agents,
            initial_active=initial_active,
            initial_opinion_dist=initial_opinion,
            rng=sim._rng,
            opinion_params=sim._params.opinion,
        )
        sim._o_initial = sim._state.o.copy()
        sim._engine = TransitionEngine(sim._params, sim._rng)
        sim._t = 0

        # Record initial metrics
        sim._record_metrics()

        return sim

    # ═══════════════════════════════════════════════════════════
    # State access for Task 3
    # ═══════════════════════════════════════════════════════════

    @property
    def n_agents(self) -> int:
        return self._state.n

    @property
    def t(self) -> int:
        return self._t

    @property
    def all_agent_ids(self) -> List[int]:
        return list(range(self._state.n))

    def get_agent_snapshot(self, agent_id: int) -> AgentSnapshot:
        """Get a single agent's state for LLM prompting."""
        st = self._state
        # Count active neighbors
        active_vec = (st.z == A).astype(np.float64)
        n_active = int(self._G_s[agent_id].dot(active_vec))

        # Local climate with visibility threshold
        expressed_mask = (st.z == A) & np.isfinite(st.o_hat)
        V_MIN = self._params.opinion.climate_visibility_threshold
        if expressed_mask.any():
            expressed = expressed_mask.astype(np.float64)
            neighbor_active = self._G_o[agent_id] * expressed
            v_i = float(neighbor_active.sum())
            climate_visible = v_i >= V_MIN
            if v_i > 1e-8:
                climate = float(
                    (self._G_o[agent_id] * np.where(expressed_mask, st.o_hat, 0.0)).sum() / v_i
                )
            else:
                climate = 0.0
        else:
            v_i = 0.0
            climate = 0.0
            climate_visible = False

        return AgentSnapshot(
            agent_id=agent_id,
            z=int(st.z[agent_id]),
            z_name=STATE_NAMES.get(int(st.z[agent_id]), "?"),
            o=float(st.o[agent_id]),
            o_hat=float(st.o_hat[agent_id]) if not np.isnan(st.o_hat[agent_id]) else float('nan'),
            h=float(st.h[agent_id]),
            f=float(st.f[agent_id]),
            c=float(st.attrs.c[agent_id]),
            n_active_neighbors=n_active,
            local_climate=climate,
            climate_visible=climate_visible,
            climate_visibility=v_i,
        )

    def get_agent_snapshots(self, agent_ids: Optional[List[int]] = None) -> List[AgentSnapshot]:
        """Get snapshots for multiple agents. If agent_ids is None, returns all."""
        if agent_ids is None:
            agent_ids = self.all_agent_ids
        return [self.get_agent_snapshot(aid) for aid in agent_ids]

    def get_active_agents(self) -> List[int]:
        """Return IDs of all agents currently in state A."""
        return [i for i in range(self._state.n) if self._state.z[i] == A]

    def get_aware_agents(self) -> List[int]:
        """Return IDs of all agents not in state U."""
        return [i for i in range(self._state.n) if self._state.z[i] != U]

    def get_neighbors(self, agent_id: int, network: str = "propagation") -> List[int]:
        """Get neighbor IDs for an agent.

        Args:
            agent_id: The agent to query.
            network: "propagation" (G_s) or "opinion" (G_o).
        """
        G = self._G_s if network == "propagation" else self._G_o
        return [j for j in range(self._state.n) if G[agent_id, j] > 0]

    # ═══════════════════════════════════════════════════════════
    # Text generation for Task 3 (LLM renders language for A-state agents)
    # ═══════════════════════════════════════════════════════════

    def get_text_requests(self, agent_ids: Optional[List[int]] = None) -> List[TextGenerationRequest]:
        """Get text generation requests for agents currently in state A.

        The dynamics kernel has ALREADY determined that these agents are active
        and has ALREADY computed their o_hat stance. The LLM's job is to
        generate natural language text consistent with this stance.

        Args:
            agent_ids: Specific agents to generate text for. None = all A-state agents.

        Returns:
            List of TextGenerationRequest objects for LLM processing.
        """
        st = self._state
        if agent_ids is None:
            agent_ids = [i for i in range(st.n) if st.z[i] == A]
        else:
            agent_ids = [i for i in agent_ids if st.z[i] == A]

        requests = []
        for aid in agent_ids:
            snap = self.get_agent_snapshot(aid)
            requests.append(TextGenerationRequest(
                agent_id=aid,
                target_stance=float(st.o_hat[aid]) if not np.isnan(st.o_hat[aid]) else float(st.o[aid]),
                target_arousal=float(st.h[aid]),
                private_opinion=float(st.o[aid]),
                local_climate=snap.local_climate,
                climate_visible=snap.climate_visible,
            ))
        return requests

    def record_generated_texts(self, texts: List[GeneratedText]) -> None:
        """Record LLM-generated texts for this step (no state modification).

        The texts are stored for export/analysis. The dynamics kernel's
        state (z, o, o_hat, h, f) is NOT modified by this call.

        Args:
            texts: List of GeneratedText objects from LLM.
        """
        for t in texts:
            self._llm_texts[t.agent_id] = t.text

    # ═══════════════════════════════════════════════════════════
    # Step execution
    # ═══════════════════════════════════════════════════════════

    def step(
        self,
        shock: float = 0.0,
        novelty: float = 0.0,
        media_exposure: Optional[np.ndarray] = None,
    ) -> StepMetrics:
        """Execute one simulation step for PARAMETRIC agents only.

        LLM-controlled agents (registered via init or marked via
        inject_llm_decisions) are frozen during the parametric transition.
        Their state changes are applied only through inject_llm_decisions().

        Args:
            shock: External shock intensity [0, 1] for this step.
            novelty: Novelty intensity [0, 1] for this step.
            media_exposure: Per-agent media exposure array, or None.

        Returns:
            StepMetrics with aggregate statistics after the step.
        """
        # Dynamics kernel controls ALL state transitions
        inputs = ExternalInputs(
            V=self._V, shock=shock, novelty=novelty,
            staleness=self._t / max(self._t + 1, 1),
            media_exposure=media_exposure,
        )

        new_state, V_next, events = self._engine.step(
            state=self._state, G_s=self._G_s, G_o=self._G_o, G_h=None,
            inputs=inputs, o_initial=self._o_initial, t=self._t,
        )
        self._V = V_next
        self._state = new_state
        self._t += 1
        self._llm_texts = {}

        return self._record_metrics()

    def _record_metrics(self) -> StepMetrics:
        """Compute and store aggregate metrics for the current step."""
        st = self._state
        a_mask = st.z == A

        if a_mask.sum() > 0:
            o_hat_mean = float(np.nanmean(st.o_hat[a_mask]))
            bias = abs(st.o.mean() - o_hat_mean)
        else:
            bias = 0.0

        active_vec = (st.z == A).astype(np.float64)
        mean_active_neighbors = float(self._G_s.dot(active_vec).mean())

        m = StepMetrics(
            step=self._t,
            n_U=st.n_U,
            n_E=st.n_E,
            n_A=st.n_A,
            n_D=st.n_D,
            o_mean=float(st.o.mean()),
            o_std=float(st.o.std()),
            h_mean=float(st.h.mean()),
            f_mean=float(st.f.mean()),
            public_bias=bias,
            mean_active_neighbors=mean_active_neighbors,
            V=self._V,
        )
        self._metrics_history.append(m)
        return m

    # ═══════════════════════════════════════════════════════════
    # Metrics and export
    # ═══════════════════════════════════════════════════════════

    def get_metrics(self) -> List[StepMetrics]:
        """Return all recorded step metrics."""
        return self._metrics_history

    def get_metrics_dataframe(self):
        """Return metrics as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([{
            "step": m.step, "n_U": m.n_U, "n_E": m.n_E,
            "n_A": m.n_A, "n_D": m.n_D,
            "o_mean": m.o_mean, "o_std": m.o_std,
            "h_mean": m.h_mean, "f_mean": m.f_mean,
            "public_bias": m.public_bias,
            "mean_active_neighbors": m.mean_active_neighbors,
        } for m in self._metrics_history])

    def export_metrics(self, path: str) -> None:
        """Export metrics to a JSON file."""
        df = self.get_metrics_dataframe()
        df.to_json(path, orient="records", indent=2)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the current simulation state."""
        st = self._state
        return {
            "t": self._t,
            "n_agents": st.n,
            "state_counts": {"U": st.n_U, "E": st.n_E, "A": st.n_A, "D": st.n_D},
            "o_mean": float(st.o.mean()),
            "o_std": float(st.o.std()),
            "h_mean": float(st.h.mean()),
            "f_mean": float(st.f.mean()),
            "n_agents_A": st.n_A,
            "network": {
                "n_edges": int((self._G_s > 0).sum()),
                "k_mean": float(self._G_s.sum(axis=0).mean()),
            },
        }


# ═══════════════════════════════════════════════════════════════
# Convenience functions for Task 3
# ═══════════════════════════════════════════════════════════════

def load_params_from_json(path: str) -> ModelParams:
    """Load ModelParams from a JSON configuration file.

    JSON format:
    {
        "propagation": {"beta": 0.15, "beta_M": 0.05},
        "activation": {"alpha_0": -2.5, "alpha_1": 1.5},
        ...
    }
    """
    from dataclasses import replace
    from dynamics_simulation.config import (
        PropagationParams, ActivationParams, DecayParams,
        ReactivationParams, OpinionParams, EmotionFatigueParams,
    )

    with open(path, "r") as f:
        data = json.load(f)

    p = default_params()
    for section, values in data.items():
        if section == "propagation":
            p = replace(p, propagation=PropagationParams(**values))
        elif section == "activation":
            p = replace(p, activation=ActivationParams(**values))
        elif section == "decay":
            p = replace(p, decay=DecayParams(**values))
        elif section == "reactivation":
            p = replace(p, reactivation=ReactivationParams(**values))
        elif section == "opinion":
            p = replace(p, opinion=OpinionParams(**values))
        elif section == "emotion_fatigue":
            p = replace(p, emotion_fatigue=EmotionFatigueParams(**values))
    return p


def create_llm_prompt(req: TextGenerationRequest, event_context: str = "") -> str:
    """Create a text-generation prompt for an agent already determined to be active.

    The dynamics kernel has ALREADY decided this agent is in state A and has
    ALREADY computed the target stance (req.target_stance). The LLM's job is
    ONLY to generate natural language text consistent with that stance.

    Args:
        req: TextGenerationRequest with pre-computed stance and arousal.
        event_context: Description of the event being discussed.

    Returns:
        A prompt string for the LLM.
    """
    stance_desc = (
        "strongly supportive" if req.target_stance > 0.5
        else "moderately supportive" if req.target_stance > 0.1
        else "neutral" if req.target_stance > -0.1
        else "moderately opposed" if req.target_stance > -0.5
        else "strongly opposed"
    )
    arousal_desc = (
        "highly emotional" if req.target_arousal > 0.6
        else "somewhat emotional" if req.target_arousal > 0.3
        else "calm and matter-of-fact"
    )
    climate_desc = (
        "supportive of the measures" if req.local_climate > 0.2
        else "opposed to the measures" if req.local_climate < -0.2
        else "mixed or neutral"
    ) if req.climate_visible else "unclear (insufficient visible expression)"

    prompt = f"""You are a social media user in a simulation about a public event.

{event_context}

YOUR TASK: Write a short social media post (1-3 sentences, in Chinese).

CONSTRAINTS:
- Your stance should be: {stance_desc} (numerical score: {req.target_stance:+.2f})
- Your emotional tone should be: {arousal_desc} (numerical level: {req.target_arousal:.2f})
- Your private opinion is {req.private_opinion:+.2f}
- The local opinion climate is {climate_desc}

Return ONLY a JSON object: {{"text": "your post here"}}
"""
    return prompt
