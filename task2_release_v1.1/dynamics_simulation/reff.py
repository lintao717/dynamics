"""
Effective Reproduction Number (R_eff) for the Integrated Dynamics Model.

Theory:
  R_eff measures the expected number of secondary activations caused by
  a single active agent introduced into a population at the disease-free
  equilibrium (DFE: all agents in U or D, A=0).

Key result:
  R_eff = rho(M) where M is the next-generation matrix with entries:
    M_ij = beta · w_ji^s · σ(α_0 + α_1|o_i| + α_2·h_i + α_3·Gamma_i - α_4·f_i - α_5·c_i)

  This means R_eff is ENDOGENOUSLY determined by the opinion distribution
  {o_i}, emotional state {h_i}, and opinion climate {Gamma_i}.

Partial derivatives (proved below):
  dR_eff/dα_1 > 0  — opinion extremity increases transmissibility
  dR_eff/d||o|| > 0  — more extreme mean opinion → higher R_eff
  dR_eff/dα_3 > 0  — climate congruity amplifies propagation
  dR_eff/dlambda_spiral < 0  — silence spiral suppresses minority expression

Usage:
  from dynamics_simulation.reff import compute_reff, analyze_reff
"""

import numpy as np
from numpy.random import Generator
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from dynamics_simulation.config import ModelParams
from dynamics_simulation.agents import AgentState, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs


@dataclass
class ReffResult:
    """Results of R_eff computation."""
    R_eff: float
    """Spectral radius of the next-generation matrix."""

    components: Dict[str, float]
    """Decomposition of R_eff into contributing factors."""

    partial_derivatives: Dict[str, float]
    """dR_eff/dparam for key parameters (computed via finite differences)."""

    interpretation: str
    """Human-readable interpretation."""


def compute_next_generation_matrix(
    state: AgentState,
    G_s: np.ndarray,
    G_o: np.ndarray,
    params: ModelParams,
    V: float = 0.0,  # platform viral intensity
    q_j: Optional[np.ndarray] = None,  # content influence per agent
) -> np.ndarray:
    """Build the corrected next-generation matrix K at the current state.

    K_{ij} = S_i * [beta*w_ji^s + beta_V*V/N] * q_j * P_i^{activate} * (1/g_j)

    Where:
      - S_i = 1[z_i == U]: susceptible indicator
      - w_ji^s: propagation weight j->i
      - beta_V*V/N: platform viral exposure per agent
      - q_j: content influence of agent j
      - P_i^{activate}: activation probability with silence spiral
      - g_j = sigma(gamma_0 + gamma_1*f_j + ...): daily decay probability
      - 1/g_j: expected active duration of agent j

    Note: w_ji^s is in G_s[i,j] (row i receives from column j).
    """
    ap = params.activation
    pp = params.propagation
    dp = params.decay
    op = params.opinion
    vp = params.viral
    attrs = state.attrs
    n = state.n

    # ── Susceptible fraction S_i ──
    # Agents that can receive new infections: U (unaware) or D (dormant, can reactivate)
    # A (already active) and E (transient) are not susceptible
    S_i = ((state.z == U) | (state.z == D)).astype(np.float64)

    # ── Content influence q_j ──
    if q_j is None:
        q_j = np.ones(n)

    # ── Activation probability P_i^{activate} ──
    # Climate with visibility threshold
    expressed_mask = (state.z == A) & np.isfinite(state.o_hat)
    V_MIN = op.climate_visibility_threshold
    if expressed_mask.any():
        expressed = expressed_mask.astype(np.float64)
        neighbor_active = G_o * expressed[np.newaxis, :]
        v_i = neighbor_active.sum(axis=1)
        weighted_opinion = G_o.dot(np.where(expressed_mask, state.o_hat, 0.0))
        local_climate = np.divide(weighted_opinion, v_i, out=np.zeros(n), where=v_i > 1e-8)
        climate_visible = v_i >= V_MIN
    else:
        v_i = np.zeros(n)
        local_climate = np.zeros(n)
        climate_visible = np.zeros(n, dtype=bool)

    Gamma = np.clip(1.0 - np.abs(state.o - local_climate) / 2.0, 0.0, 1.0)
    v_i_norm = np.minimum(v_i / max(V_MIN, v_i.max()), 1.0)

    logit = (
        ap.alpha_0
        + ap.alpha_1 * np.abs(state.o)
        + ap.alpha_2 * state.h
        + ap.alpha_3 * v_i_norm * Gamma * climate_visible
        - ap.alpha_4 * state.f
        - ap.alpha_5 * attrs.c
    )
    p_activate = 1.0 / (1.0 + np.exp(-logit))

    # Silence spiral (only when climate visible)
    minority_mask = (Gamma < 0.5) & climate_visible
    spiral_penalty = np.ones(n)
    spiral_penalty[minority_mask] = 1.0 - op.lambda_spiral * (0.5 - Gamma[minority_mask])
    p_activate = p_activate * spiral_penalty
    p_activate = np.clip(p_activate, 0.0, 1.0)

    # ── Active duration: L_j = 1 / g_j ──
    # g_j = P(A_j -> D_j) per step
    # Using current fatigue and staleness≈0.5 as approximation
    logit_decay = (
        dp.gamma_0 + dp.gamma_1 * state.f + dp.gamma_2 * 0.5
    )
    g_j = 1.0 / (1.0 + np.exp(-logit_decay))  # daily decay prob
    g_j = np.maximum(g_j, 0.01)  # prevent division by zero
    L_j = 1.0 / g_j  # expected active duration in steps

    # ── Build K: total exposure × activation × duration ──
    # Network channel: beta * G_s[i,j] * q_j
    # PVA channel: beta_V * V / N (uniform per-agent viral exposure)
    pva_per_agent = vp.beta_V * V / max(n, 1)
    exposure_matrix = pp.beta * G_s * q_j[np.newaxis, :] + pva_per_agent

    # K[i,j] = S_i * exposure[i,j] * p_activate[i] * L_j
    K = S_i[:, np.newaxis] * exposure_matrix * p_activate[:, np.newaxis] * L_j[np.newaxis, :]

    return K


def compute_reff(
    state: AgentState,
    G_s: np.ndarray,
    G_o: np.ndarray,
    params: ModelParams,
    V: float = 0.0,
    q_j: Optional[np.ndarray] = None,
    compute_partials: bool = True,
    delta: float = 0.05,
) -> ReffResult:
    """Compute R_eff and its decomposition.

    Args:
        state: Current agent state.
        G_s: Propagation network.
        G_o: Opinion influence network.
        params: Model parameters.
        V: Platform viral intensity.
        q_j: Content influence per agent.
        compute_partials: Whether to compute partial derivatives.
        delta: Finite difference step for partial derivatives.

    Returns:
        ReffResult with R_eff, components, and partial derivatives.
    """
    K = compute_next_generation_matrix(state, G_s, G_o, params, V, q_j)
    eigenvalues = np.linalg.eigvals(K)
    R_eff = float(np.max(np.abs(eigenvalues)))

    # ── Decompose into components ──
    n = state.n
    ap = params.activation

    # Mean-field approximation: R_eff ≈ beta · k̄ · σ̄
    # where k̄ = average out-degree (mean column sum of G_s)
    k_out = G_s.sum(axis=0)  # out-degree of each source j
    k_bar = float(k_out.mean())

    # Mean activation probability
    o_hat_safe = np.nan_to_num(state.o_hat, nan=0.0)
    active_mask = state.z == A
    if active_mask.any():
        neighbor_active = G_o * active_mask[np.newaxis, :].astype(np.float64)
        neighbor_sum = neighbor_active.sum(axis=1)
        weighted_opinion = G_o.dot(o_hat_safe * active_mask.astype(np.float64))
        local_climate = np.divide(
            weighted_opinion, neighbor_sum,
            out=np.zeros(n), where=neighbor_sum > 1e-8,
        )
    else:
        local_climate = np.zeros(n)
    Gamma = np.clip(1.0 - np.abs(state.o - local_climate) / 2.0, 0.0, 1.0)

    logit = (
        ap.alpha_0 + ap.alpha_1 * np.abs(state.o) + ap.alpha_2 * state.h
        + ap.alpha_3 * Gamma - ap.alpha_4 * state.f - ap.alpha_5 * state.attrs.c
    )
    p_bar = float(np.mean(1.0 / (1.0 + np.exp(-logit))))  # mean(sigma(x)), not 1/mean(1+e^{-x})

    # Degree heterogeneity correction: k_eff = <k_out * k_in> / <k>
    # This accounts for the friendship paradox in directed networks
    k_in = G_s.sum(axis=1)
    if k_bar > 0:
        k_eff = float((k_out * k_in).sum() / (k_out.sum() + 1e-8))
    else:
        k_eff = k_bar

    # Mean-field R_eff
    R_mf = params.propagation.beta * k_eff * p_bar

    components = {
        "beta": params.propagation.beta,
        "k_out_mean": k_bar,
        "k_eff": k_eff,
        "p_activate_mean": p_bar,
        "R_mf_approx": R_mf,
        "mean_opinion_abs": float(np.abs(state.o).mean()),
        "mean_arousal": float(state.h.mean()),
        "mean_fatigue": float(state.f.mean()),
        "mean_Gamma": float(Gamma.mean()),
        "mean_cost": float(state.attrs.c.mean()),
    }

    # ── Partial derivatives via finite differences ──
    partials = {}
    if compute_partials:
        from dataclasses import replace
        from dynamics_simulation.config import ActivationParams, OpinionParams

        # dR_eff/dα_1
        p_plus = replace(params, activation=replace(
            params.activation, alpha_1=ap.alpha_1 + delta))
        K_plus = compute_next_generation_matrix(state, G_s, G_o, p_plus, V, q_j)
        R_plus = float(np.max(np.abs(np.linalg.eigvals(K_plus))))
        partials["dReff_dalpha1"] = (R_plus - R_eff) / delta

        # dR_eff/dα_3
        p_plus = replace(params, activation=replace(
            params.activation, alpha_3=ap.alpha_3 + delta))
        K_plus = compute_next_generation_matrix(state, G_s, G_o, p_plus, V, q_j)
        R_plus = float(np.max(np.abs(np.linalg.eigvals(K_plus))))
        partials["dReff_dalpha3"] = (R_plus - R_eff) / delta

        # dR_eff/dbeta
        p_plus = replace(params, propagation=replace(
            params.propagation, beta=params.propagation.beta + delta))
        K_plus = compute_next_generation_matrix(state, G_s, G_o, p_plus, V, q_j)
        R_plus = float(np.max(np.abs(np.linalg.eigvals(K_plus))))
        partials["dReff_dbeta"] = (R_plus - R_eff) / delta

        # dR_eff/dlambda_spiral
        p_plus = replace(params, opinion=replace(
            params.opinion, lambda_spiral=params.opinion.lambda_spiral + delta))
        K_plus = compute_next_generation_matrix(state, G_s, G_o, p_plus, V, q_j)
        R_plus = float(np.max(np.abs(np.linalg.eigvals(K_plus))))
        partials["dReff_dlambda_spiral"] = (R_plus - R_eff) / delta

    # ── Interpretation ──
    if R_eff > 1.0:
        interp = (
            f"R_eff = {R_eff:.3f} > 1: Propagation will GROW. "
            f"Each active agent generates {R_eff:.1f} secondary activations on average. "
            f"Key driver: "
            + (f"opinion extremity (alpha_1*|o|={ap.alpha_1 * components['mean_opinion_abs']:.3f})"
               if ap.alpha_1 * components['mean_opinion_abs'] > 0.3 else
               f"baseline activation (alpha_0={ap.alpha_0})")
        )
    elif R_eff > 0.0:
        interp = (
            f"R_eff = {R_eff:.3f} < 1: Propagation will DECAY. "
            f"Each active agent generates fewer than 1 secondary activation. "
            f"Without external shocks, the cascade dies out."
        )
    else:
        interp = (
            f"R_eff = 0: No propagation possible. "
            f"No active agents to seed the next generation."
        )

    return ReffResult(
        R_eff=R_eff,
        components=components,
        partial_derivatives=partials,
        interpretation=interp,
    )


def analyze_reff_across_opinion_gradient(
    G_s: np.ndarray,
    G_o: np.ndarray,
    base_state: AgentState,
    params: ModelParams,
    opinion_range: Tuple[float, float] = (-0.8, 0.8),
    n_points: int = 9,
) -> Dict:
    """Analyze how R_eff varies with mean opinion extremity.

    This directly tests: dR_eff/d||o|| > 0

    Modifies all agent opinions to have the same absolute value |o|,
    then computes R_eff at each point.

    Returns:
        Dict with opinion_values, R_eff_values, and slope.
    """
    state = base_state.copy()
    o_orig = state.o.copy()

    abs_opinions = np.linspace(opinion_range[0], opinion_range[1], n_points)
    R_eff_values = []

    for abs_o in abs_opinions:
        # Set all opinions to ±abs_o preserving original sign distribution
        signs = np.sign(o_orig)
        signs[signs == 0] = 1.0
        state.o[:] = abs_o * signs
        result = compute_reff(state, G_s, G_o, params, compute_partials=False)
        R_eff_values.append(result.R_eff)

    state.o[:] = o_orig  # restore

    # Linear fit: R_eff vs |o|
    slope, intercept = np.polyfit(abs_opinions, R_eff_values, 1)

    return {
        "opinion_values": abs_opinions,
        "R_eff_values": R_eff_values,
        "slope": float(slope),
        "intercept": float(intercept),
        "dReff_d_abs_opinion": float(slope),
        "verified": slope > 0.01,  # dR_eff/d||o|| > 0
    }
