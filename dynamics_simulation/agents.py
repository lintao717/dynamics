"""
Agent state initialization and management.

Stores all agent states as numpy arrays for efficient vectorized operations.
State encoding:
  z:  int8   {0=U, 1=E, 2=A, 3=D}
  o:  float64 ∈ [-1, 1]    private opinion
  o_hat: float64 ∈ [-1, 1] ∪ {NaN}  public expression
  h:  float64 ∈ [0, 1]     emotional arousal
  f:  float64 ∈ [0, 1]     information fatigue

Fixed attributes (per-agent, immutable after init):
  c:      expression cost
  mu:     opinion update speed
  zeta:   initial opinion anchoring
  epsilon: bounded confidence threshold
  sigma_xi: noise sensitivity
  eta:    information evidence sensitivity
  chi:    official information trust
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from typing import Optional, Dict, Any
from dataclasses import dataclass


# State constants
U = 0  # Uncertain
E = 1  # Exposed (transient)
A = 2  # Active
D = 3  # Dormant

STATE_NAMES = {0: "U", 1: "E", 2: "A", 3: "D"}


@dataclass
class AgentState:
    """Complete state of all agents at one time step.

    All arrays are 1-D with length N.
    """

    z: np.ndarray       # int8, propagation state {0=U, 1=E, 2=A, 3=D}
    m: np.ndarray       # int8, history flag: 0=never expressed, 1=has expressed before
    o: np.ndarray       # float64, private opinion [-1, 1]
    o_hat: np.ndarray   # float64, public expression [-1, 1] ∪ {NaN}
    h: np.ndarray       # float64, emotional arousal [0, 1]
    f: np.ndarray       # float64, information fatigue [0, 1]

    # Fixed attributes
    attrs: "AgentAttributes"

    @property
    def n(self) -> int:
        return len(self.z)

    @property
    def n_U(self) -> int:
        return int((self.z == U).sum())

    @property
    def n_E(self) -> int:
        return int((self.z == E).sum())

    @property
    def n_A(self) -> int:
        return int((self.z == A).sum())

    @property
    def n_D(self) -> int:
        return int((self.z == D).sum())

    def state_counts(self) -> dict[str, int]:
        """Return dict of state name → count."""
        return {
            "U": self.n_U,
            "E": self.n_E,
            "A": self.n_A,
            "D": self.n_D,
        }

    def copy(self) -> "AgentState":
        """Deep copy of all state arrays."""
        return AgentState(
            z=self.z.copy(),
            m=self.m.copy(),
            o=self.o.copy(),
            o_hat=self.o_hat.copy(),
            h=self.h.copy(),
            f=self.f.copy(),
            attrs=self.attrs,
        )

    def clone_with_new_z(self, z_new: np.ndarray, m_new: np.ndarray = None) -> "AgentState":
        """Return a new AgentState with updated z and m (other fields shallow-copied)."""
        if m_new is None:
            m_new = self.m.copy()
        return AgentState(
            z=z_new,
            m=m_new,
            o=self.o.copy(),
            o_hat=self.o_hat.copy(),
            h=self.h.copy(),
            f=self.f.copy(),
            attrs=self.attrs,
        )

    def snapshot(self, step: int) -> dict:
        """Return a lightweight snapshot for metrics collection."""
        return {
            "step": step,
            "n_U": self.n_U,
            "n_E": self.n_E,
            "n_A": self.n_A,
            "n_D": self.n_D,
            "o_mean": float(self.o.mean()),
            "o_std": float(self.o.std()),
            "o_hat_mean": float(np.nanmean(self.o_hat)) if self.n_A > 0 else np.nan,
            "o_hat_std": float(np.nanstd(self.o_hat)) if self.n_A > 0 else np.nan,
            "h_mean": float(self.h.mean()),
            "f_mean": float(self.f.mean()),
        }


@dataclass
class AgentAttributes:
    """Fixed per-agent attributes (heterogeneous parameters).

    All arrays are 1-D with length N.
    """

    c: np.ndarray       # expression cost [0, 1]
    mu: np.ndarray      # opinion update speed [0, 1]
    zeta: np.ndarray    # initial opinion anchoring [0, 1]
    epsilon: np.ndarray # bounded confidence threshold [0, 2]
    sigma_xi: np.ndarray # noise sensitivity [0, 0.5]
    eta: np.ndarray     # information evidence sensitivity [0, 1]
    chi: np.ndarray     # official information trust [0, 1]

    @property
    def n(self) -> int:
        return len(self.c)


# ─────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────


def initialize_agents(
    n: int = 500,
    initial_active: int = 10,
    initial_opinion_dist: str = "polarized",
    opinion_seeds: Optional[tuple[float, ...]] = None,
    rng: Optional[Generator] = None,
    # Attribute distribution parameters
    c_beta: tuple[float, float] = (2.0, 5.0),          # expression cost: skewed low
    mu_beta: tuple[float, float] = (2.5, 7.5),         # update speed: skewed low-moderate
    zeta_beta: tuple[float, float] = (5.5, 4.5),       # anchoring: centered ~0.55
    epsilon_beta: tuple[float, float] = (3.0, 4.5),    # bounded confidence: ~0.40
    sigma_xi_beta: tuple[float, float] = (1.5, 8.0),   # noise: skewed low
    eta_beta: tuple[float, float] = (1.5, 8.5),        # info sensitivity: skewed low
    chi_beta: tuple[float, float] = (1.5, 6.0),        # official info trust: varied
) -> AgentState:
    """Initialize N agents with realistic heterogeneous attributes.

    Args:
        n: Number of agents.
        initial_active: How many agents start in A state.
        initial_opinion_dist: "polarized" | "uniform" | "moderate" | "consensus".
            - polarized: bimodal distribution (two camps)
            - uniform: uniform on [-1, 1]
            - moderate: normal centered at 0 with sd=0.3
            - consensus: normal centered at 0.5 with sd=0.15
        opinion_seeds: Optional tuple of opinion values for seed agents.
            If provided, these agents get exact initial opinions.
        rng: NumPy random generator.

    Returns:
        Initialized AgentState.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Fixed attributes from Beta distributions ──

    def _beta(a, b, size, scale=1.0):
        return rng.beta(a, b, size=size) * scale

    attrs = AgentAttributes(
        c=_beta(*c_beta, n),
        mu=_beta(*mu_beta, n),
        zeta=_beta(*zeta_beta, n),
        epsilon=_beta(*epsilon_beta, n, scale=2.0),
        sigma_xi=_beta(*sigma_xi_beta, n, scale=0.5),
        eta=_beta(*eta_beta, n),
        chi=_beta(*chi_beta, n),
    )

    # ── Initial opinions ──

    if initial_opinion_dist == "uniform":
        o = rng.uniform(-1.0, 1.0, size=n)
    elif initial_opinion_dist == "polarized":
        # Two camps: one around +0.6, one around -0.6
        camp = rng.random(n) < 0.5
        o = np.where(
            camp,
            rng.normal(0.65, 0.20, size=n),
            rng.normal(-0.65, 0.20, size=n),
        )
    elif initial_opinion_dist == "moderate":
        o = rng.normal(0.0, 0.30, size=n)
    elif initial_opinion_dist == "consensus":
        o = rng.normal(0.50, 0.15, size=n)
    else:
        raise ValueError(f"Unknown opinion distribution: {initial_opinion_dist}")

    o = np.clip(o, -1.0, 1.0)

    # Apply seed opinions if provided
    if opinion_seeds is not None:
        for idx, val in enumerate(opinion_seeds):
            if idx < n:
                o[idx] = np.clip(val, -1.0, 1.0)

    # ── Initial propagation state ──
    z = np.full(n, U, dtype=np.int8)
    m = np.zeros(n, dtype=np.int8)  # history flag: 0=never expressed
    if initial_active > 0:
        # First `initial_active` agents start as Active
        active_indices = rng.choice(n, size=min(initial_active, n), replace=False)
        z[active_indices] = A
        m[active_indices] = 1  # they have expressed

    # ── Initial emotion and fatigue ──
    h = np.zeros(n, dtype=np.float64)
    # Active agents have moderate initial arousal
    h[z == A] = rng.uniform(0.3, 0.6, size=int((z == A).sum()))

    f = np.zeros(n, dtype=np.float64)

    # ── Initial public expression ──
    o_hat = np.full(n, np.nan, dtype=np.float64)
    o_hat[z == A] = o[z == A].copy()

    return AgentState(z=z, m=m, o=o, o_hat=o_hat, h=h, f=f, attrs=attrs)
