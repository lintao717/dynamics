"""
Default parameters for the Integrated Propagation-Opinion Dynamics Model.

All parameters correspond to §4 of the Model Definition Document.
Parameters are organized into six categories and stored in immutable
frozen dataclasses for reproducibility.

Usage:
    from dynamics_simulation.config import ModelParams, default_params
    p = default_params()
    p = replace(default_params(), propagation=replace(default_params().propagation, beta=0.20))  # override specific params
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class PropagationParams:
    """§4.1: Parameters governing information exposure and spread."""

    beta: float = 0.15
    """Base propagation rate: prob that a single active neighbor exposes U agent.

    Calibrated to sustain propagation on ER/BA/WS networks with moderate activation.
    Previous value 0.12 was below the critical threshold for non-SBM topologies.
    """

    beta_M: float = 0.08
    """Media propagation rate: prob that media exposure converts U → E."""

    theta_Lambda: float = 0.0
    """Optional sigmoid threshold for exposure. 0 = use min(Λ, 1) clipping."""


@dataclass(frozen=True)
class ActivationParams:
    """§4.2: Parameters governing E → A transition (activation decision)."""

    alpha_0: float = -2.5
    """Baseline activation log-odds. Negative = most exposed do not act."""

    alpha_1: float = 1.5
    """Opinion extremity weight: |o_i| drives activation."""

    alpha_2: float = 2.0
    """Emotional arousal weight: higher arousal → more likely to express."""

    alpha_3: float = 1.8
    """Opinion climate congruity weight: feeling "safe" → more likely to express."""

    alpha_4: float = 1.5
    """Fatigue suppression weight: higher fatigue → less likely to express."""

    alpha_5: float = 1.0
    """Expression cost weight: higher inherent cost → less likely to express."""


@dataclass(frozen=True)
class DecayParams:
    """§4.3: Parameters governing A → D transition (active decay)."""

    gamma_0: float = 0.0
    """Baseline decay log-odds. σ(0)=0.5 → ~2 time-step expected active duration."""

    gamma_1: float = 2.0
    """Fatigue-accelerated decay weight."""

    gamma_2: float = 1.5
    """Information staleness weight: older info → faster decay."""

    gamma_3: float = 1.2
    """Novelty offset weight: new content → slower decay."""


@dataclass(frozen=True)
class ReactivationParams:
    """§4.4: Parameters governing D -> A transition.

    Split by history flag m_i:
      - m_i=0: delayed activation (never expressed before)
      - m_i=1: true reactivation (previously active, now dormant)
    """

    # Delayed activation (m=0): first-time expression after initial exposure
    r_0_0: float = -3.0
    """Baseline delayed-activation log-odds. Higher than r_0_1 because
    never-expressed agents are more likely to eventually express."""

    r_1_0: float = 2.5
    """Shock response for delayed activation. Lower than r_1_1 because
    external shocks primarily re-engage those with prior expression history."""

    # True reactivation (m=1): re-engagement after previous active period
    r_0_1: float = -4.0
    """Baseline true-reactivation log-odds. Very negative = rarely spontaneous."""

    r_1_1: float = 3.5
    """Shock response for true reactivation. Higher than r_1_0 because
    shocks resonate more with those who have prior engagement history."""

    # Shared parameters
    r_2: float = 1.0
    """Novelty-driven reactivation weight (shared across m_i)."""

    r_3: float = 1.2
    """Residual emotion-driven reactivation weight (shared across m_i)."""


@dataclass(frozen=True)
class OpinionParams:
    """§4.5: Parameters governing opinion dynamics."""

    # Private opinion update
    mu_mean: float = 0.25
    """Mean opinion update speed (drawn from Beta distribution per agent)."""

    mu_conc: float = 5.0
    """Concentration parameter for mu Beta distribution (higher = tighter)."""

    zeta_mean: float = 0.55
    """Mean initial opinion anchoring strength."""

    zeta_conc: float = 4.0
    """Concentration for zeta Beta distribution."""

    epsilon_mean: float = 0.40
    """Mean bounded confidence threshold (max opinion gap = 2)."""

    epsilon_conc: float = 4.0
    """Concentration for epsilon Beta distribution."""

    sigma_xi: float = 0.02
    """Std dev of Gaussian noise added to opinion each step."""

    # Information evidence
    eta_mean: float = 0.15
    """Mean sensitivity to information evidence."""

    eta_conc: float = 5.0
    """Concentration for eta Beta distribution."""

    # Official information
    chi_mean: float = 0.20
    """Mean trust/responsiveness to official information."""

    chi_conc: float = 3.0
    """Concentration for chi Beta distribution. Lower = more heterogeneous."""

    # Public expression bias
    lambda_c: float = 0.35
    """Conformity intensity: how much expression shifts toward local climate."""

    lambda_h: float = 0.15
    """Emotional amplification: how much arousal polarizes expression."""

    # Silence spiral
    lambda_spiral: float = 0.50
    """Silence spiral strength: penalty on minority-opinion expression."""

    climate_visibility_threshold: float = 0.10
    """Minimum weighted active-neighbor sum v_i to perceive opinion climate.
    With row-normalized G_o, v_i in [0,1]. 0.10 means ~10% of influence
    weight must come from active neighbors. Lower = more sensitive."""


@dataclass(frozen=True)
class EmotionFatigueParams:
    """§4.6: Parameters governing emotion and fatigue dynamics."""

    # Emotion
    delta_h: float = 0.30
    """Emotion natural decay rate per step. Half-life ≈ 2.3 steps."""

    eta_h: float = 0.15
    """Exposure-induced emotion coefficient."""

    omega_h: float = 0.10
    """Emotional contagion rate from neighbors."""

    chi_h: float = 0.35
    """Shock-induced emotion coefficient."""

    nu_h: float = 0.12
    """Fatigue suppression of emotion coefficient."""

    # Fatigue
    delta_f: float = 0.15
    """Fatigue natural recovery rate per step. Half-life ≈ 4.6 steps."""

    eta_f: float = 0.05
    """Exposure-induced fatigue coefficient (per exposed item)."""

    omega_f: float = 0.08
    """Active-expression fatigue accumulation per step in A state."""


@dataclass(frozen=True)
class ViralParams:
    """Parameters governing platform viral amplification (PVA).

    Models how trending/hot-search mechanisms amplify exposure beyond
    direct network connections. Inspired by Bass diffusion (publicity
    channel) + Hawkes self-exciting processes.

    The viral intensity V(t) evolves as:
      V(t+1) = (1 - delta_V) * V(t) + eta_V * n_new_posts(t) * mean_q(t)

    And contributes to exposure via:
      Lambda_i += beta_V * V(t)
    """

    beta_V: float = 0.012
    """Platform viral amplification: prob of exposure per unit of viral intensity.
    Calibrated from Weibo cascade size distribution (mean 147 reposts/post)."""

    delta_V: float = 0.40
    """Viral decay rate per step (24h). Half-life = ln(2)/delta_V ≈ 1.7 days.
    Calibrated from Weibo cascade temporal decay (half-life ~1-2 days)."""

    eta_V: float = 0.08
    """Viral excitation per new post. How much each new post contributes to V(t).
    Calibrated to match observed cascade size amplification (~100x over network-only)."""


@dataclass(frozen=True)
class ModelParams:
    """Complete parameter set for one simulation run.

    All fields are frozen dataclasses — use `replace()` to create variants:

        p2 = replace(p, propagation=replace(p.propagation, beta=0.20))
    """

    propagation: PropagationParams = PropagationParams()
    activation: ActivationParams = ActivationParams()
    decay: DecayParams = DecayParams()
    reactivation: ReactivationParams = ReactivationParams()
    opinion: OpinionParams = OpinionParams()
    emotion_fatigue: EmotionFatigueParams = EmotionFatigueParams()
    viral: ViralParams = ViralParams()


def default_params() -> ModelParams:
    """Return the default (baseline) parameter set.

    These defaults are calibrated to produce plausible baseline behavior
    for N=500 agents on a random-block network over T=100 steps.
    """
    return ModelParams()


# ── Named presets for sensitivity analysis ──

def high_propagation() -> ModelParams:
    """Higher propagation rate → faster spread, higher peak."""
    return replace(default_params(),
        propagation=PropagationParams(beta=0.25, beta_M=0.15)
    )


def low_propagation() -> ModelParams:
    """Lower propagation rate → slower spread, lower peak."""
    return replace(default_params(),
        propagation=PropagationParams(beta=0.05, beta_M=0.03)
    )


def high_opinion_extremity() -> ModelParams:
    """Stronger opinion extremity effect on activation."""
    return replace(default_params(),
        activation=ActivationParams(alpha_1=3.0)
    )


def strong_silence_spiral() -> ModelParams:
    """Stronger silence spiral → faster minority silencing."""
    return replace(default_params(),
        opinion=OpinionParams(lambda_spiral=0.85)
    )


def no_silence_spiral() -> ModelParams:
    """No silence spiral (control condition)."""
    return replace(default_params(),
        opinion=OpinionParams(lambda_spiral=0.0)
    )


def high_emotional_contagion() -> ModelParams:
    """Higher emotional contagion → faster emotional cascades."""
    return replace(default_params(),
        emotion_fatigue=EmotionFatigueParams(omega_h=0.25)
    )


def fast_fatigue() -> ModelParams:
    """Faster fatigue accumulation → shorter active periods."""
    return replace(default_params(),
        emotion_fatigue=EmotionFatigueParams(eta_f=0.12, omega_f=0.15)
    )


def no_coupling() -> ModelParams:
    """No opinion-propagation coupling: α_1=α_3=0, λ_spiral=0.

    Propagation decisions are independent of opinion state.
    """
    return replace(default_params(),
        activation=ActivationParams(alpha_1=0.0, alpha_3=0.0),
        opinion=OpinionParams(lambda_spiral=0.0),
    )


def one_way_coupling() -> ModelParams:
    """One-way coupling: opinion → propagation but not propagation → opinion.

    Opinion affects activation, but propagation doesn't feed back to opinion.
    """
    return replace(default_params(),
        opinion=OpinionParams(lambda_c=0.0, lambda_h=0.0),
    )


# Registry for experiment lookup
PRESETS: dict[str, ModelParams] = {
    "default": default_params(),
    "high_propagation": high_propagation(),
    "low_propagation": low_propagation(),
    "high_opinion_extremity": high_opinion_extremity(),
    "strong_silence_spiral": strong_silence_spiral(),
    "no_silence_spiral": no_silence_spiral(),
    "high_emotional_contagion": high_emotional_contagion(),
    "fast_fatigue": fast_fatigue(),
    "no_coupling": no_coupling(),
    "one_way_coupling": one_way_coupling(),
}
