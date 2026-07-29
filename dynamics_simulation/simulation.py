"""
SimulationRunner: orchestrates the multi-step simulation.

Manages:
  - Network generation (fixed for entire run)
  - Agent initialization
  - External input timeline
  - Step-by-step execution via TransitionEngine
  - Metrics collection

Supports reproducible runs via seed control.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, SeedSequence
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

from dynamics_simulation.config import ModelParams, default_params
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import (
    AgentState, initialize_agents, U, E, A, D, STATE_NAMES,
)
from dynamics_simulation.transitions import (
    TransitionEngine, ExternalInputs, default_inputs,
)
from dynamics_simulation.metrics import MetricsCollector, SimulationMetrics


@dataclass
class SimulationConfig:
    """Complete configuration for one simulation run.

    All fields have sensible defaults for a 500-agent × 100-step run.
    """

    # ── Agent population ──
    n_agents: int = 500
    initial_active: int = 10
    initial_opinion_dist: str = "polarized"  # "polarized" | "uniform" | "moderate"

    # ── Network ──
    network_type: str = "sbm"   # "er" | "ba" | "ws" | "sbm"
    network_kwargs: dict = field(default_factory=lambda: {
        "n_blocks": 3,
        "p_in": 0.15,
        "p_out": 0.02,
    })

    # ── Time ──
    T: int = 100               # Number of time steps
    delta_t_hours: float = 24.0  # Hours per step

    # ── Model parameters ──
    params: ModelParams = field(default_factory=default_params)

    # ── External input timeline (override for shock scenarios) ──
    input_fn: Optional[Callable[[int, int, int], ExternalInputs]] = None
    """Function (n, t, T) → ExternalInputs. Default uses default_inputs."""

    # ── Random seed ──
    seed: int = 42

    # ── External network (optional) ──
    G_s: Optional[np.ndarray] = None  # Pre-built propagation network
    G_o: Optional[np.ndarray] = None  # Pre-built opinion network
    communities: Optional[dict] = None

    # ── Output control ──
    verbose: bool = True
    snapshot_interval: int = 1   # Save snapshot every N steps


class SimulationRunner:
    """Runs one simulation and collects metrics.

    Usage:
        config = SimulationConfig(T=100, seed=42)
        runner = SimulationRunner(config)
        metrics = runner.run()
        print(metrics.summary())
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Initialize sub-generators from the same seed
        self._net_rng = np.random.default_rng(
            self.rng.integers(0, 2**31 - 1)
        )
        self._agent_rng = np.random.default_rng(
            self.rng.integers(0, 2**31 - 1)
        )

        self.engine = TransitionEngine(config.params, self.rng)
        self.metrics = MetricsCollector()

        # These are set in run()
        self.G_s: Optional[np.ndarray] = None
        self.G_o: Optional[np.ndarray] = None
        self.G_h: Optional[np.ndarray] = None
        self.communities: Optional[dict] = None

    def run(self) -> SimulationMetrics:
        """Execute the full simulation and return collected metrics."""
        cfg = self.config

        if cfg.verbose:
            print(f"=== Simulation: N={cfg.n_agents}, T={cfg.T}, "
                  f"net={cfg.network_type}, seed={cfg.seed} ===")

        # ── Generate or use external networks ──
        if cfg.G_s is not None and cfg.G_o is not None:
            self.G_s = cfg.G_s
            self.G_o = cfg.G_o
            self.communities = cfg.communities or {0: list(range(cfg.n_agents))}
            if cfg.verbose:
                print(f"Using external network: {(self.G_s > 0).sum()} edges")
        else:
            if cfg.verbose:
                print("Generating networks...")
            self.G_s, self.G_o, self.communities = generate_networks(
                network_type=cfg.network_type,
                n=cfg.n_agents,
                rng=self._net_rng,
                **cfg.network_kwargs,
            )
        # G_h defaults to G_o
        self.G_h = self.G_o.copy()

        # ── Initialize agents ──
        if cfg.verbose:
            print("Initializing agents...")
        state = initialize_agents(
            n=cfg.n_agents,
            initial_active=cfg.initial_active,
            initial_opinion_dist=cfg.initial_opinion_dist,
            rng=self._agent_rng,
            opinion_params=cfg.params.opinion,
        )
        o_initial = state.o.copy()  # Store for anchoring term

        # ── Set up input function ──
        input_fn = cfg.input_fn or default_inputs

        # ── Record initial snapshot ──
        self.metrics.record(state, communities=self.communities,
                           G_s=self.G_s, G_o=self.G_o)

        if cfg.verbose:
            counts = state.state_counts()
            print(f"  t=0: U={counts['U']} E={counts['E']} "
                  f"A={counts['A']} D={counts['D']}  "
                  f"o_mean={state.o.mean():.3f}  h_mean={state.h.mean():.3f}")

        # ── Main loop ──
        V_current = 0.0  # Track viral intensity across steps
        for t in range(cfg.T):
            inputs = input_fn(cfg.n_agents, t, cfg.T)
            inputs.V = V_current  # Feed V into this step

            state, V_next, events = self.engine.step(
                state=state,
                G_s=self.G_s,
                G_o=self.G_o,
                G_h=self.G_h,
                inputs=inputs,
                o_initial=o_initial,
                t=t,
            )
            V_current = V_next  # Carry forward to next step

            # Record snapshot with exact transition events + actual time step
            if t % cfg.snapshot_interval == 0 or t == cfg.T - 1:
                self.metrics.record(
                    state, communities=self.communities,
                    G_s=self.G_s, G_o=self.G_o, events=events, t=t,
                )

            if cfg.verbose and t % 10 == 9:
                counts = state.state_counts()
                print(f"  t={t+1:3d}: U={counts['U']:4d} E={counts['E']} "
                      f"A={counts['A']:4d} D={counts['D']:4d}  "
                      f"o_mean={state.o.mean():+.3f}  h_mean={state.h.mean():.3f}  "
                      f"f_mean={state.f.mean():.3f}  "
                      f"|o_hat|_mean={np.nanmean(np.abs(state.o_hat)):.3f}")

        # ── Finalize metrics ──
        final_counts = state.state_counts()
        if cfg.verbose:
            print(f"  FINAL: U={final_counts['U']} E={final_counts['E']} "
                  f"A={final_counts['A']} D={final_counts['D']}")

        self.metrics.finalize()
        return self.metrics.as_metrics()
