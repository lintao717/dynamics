"""
TransitionEngine: the 7-step coupled update cycle.

Implements §8 of the Model Definition Document.
Each step is a separate method for clarity and testability.

All methods are static and pure: they take state arrays + parameters,
return new arrays. No internal mutable state.

External inputs (shocks, media, novelty, etc.) are injected via an
ExternalInputs callable/object, allowing the same engine to drive
both synthetic experiments and data-calibrated runs.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from typing import Optional, Protocol, Dict, Any
from dataclasses import dataclass

from dynamics_simulation.config import ModelParams
from dynamics_simulation.agents import (
    AgentState, AgentAttributes, U, E, A, D,
)


@dataclass
class TransitionEvents:
    """Per-step transition counts for identifiability and metrics."""
    U_to_E: int = 0
    E_to_A: int = 0
    E_to_D: int = 0
    A_to_D: int = 0
    D0_to_A: int = 0   # delayed activation (m=0)
    D1_to_A: int = 0   # true reactivation (m=1)
    new_posts: int = 0  # newly activated agents (for V update)

# ─────────────────────────────────────────────────────────────
# External Inputs Protocol
# ─────────────────────────────────────────────────────────────


@dataclass
class ExternalInputs:
    """External signals at a given time step.

    All values are scalars (global) or per-agent arrays of shape (N,).
    """

    # Global scalars
    shock: float = 0.0           # Shock(t) ∈ [0, 1]
    novelty: float = 0.0         # Novelty(t) ∈ [0, 1]
    staleness: float = 0.0       # s(t) ∈ [0, 1]
    V: float = 0.0               # V(t): platform viral intensity ∈ [0, 1]

    # Per-agent arrays (shape N, default = zeros)
    media_exposure: Optional[np.ndarray] = None   # M_i(t) ∈ [0, 1]
    info_evidence: Optional[np.ndarray] = None    # I_i(t) ∈ [-1, 1]
    official_info: Optional[np.ndarray] = None    # u_i(t) ∈ [-1, 1]
    info_emotion: Optional[np.ndarray] = None     # I_m(t) ∈ [0, 1]
    content_influence: Optional[np.ndarray] = None # q_j(t) ∈ [0, 1] (for each agent as source)

    def resolve(self, n: int) -> Dict[str, np.ndarray]:
        """Resolve all fields to numpy arrays of length n with range validation.

        Ranges:
          shock, novelty, staleness, V: [0, 1]
          media_exposure, info_emotion, content_influence: [0, 1]
          info_evidence, official_info: [-1, 1]
          All arrays: exact shape (n,), finite, no NaN.
        """
        # ── Validate scalars ──
        for name, low, high in [
            ("shock", 0.0, 1.0),
            ("novelty", 0.0, 1.0),
            ("staleness", 0.0, 1.0),
            ("V", 0.0, 1.0),
        ]:
            val = getattr(self, name)
            if not (low <= val <= high):
                raise ValueError(
                    f"{name}={val} is outside [{low}, {high}]"
                )

        defaults = {
            "media_exposure": np.zeros(n),
            "info_evidence": np.zeros(n),
            "official_info": np.zeros(n),
            "info_emotion": np.zeros(n),
            "content_influence": np.ones(n),  # default: all agents equal influence
        }
        range_checks = {
            "media_exposure": (0.0, 1.0),
            "info_evidence": (-1.0, 1.0),
            "official_info": (-1.0, 1.0),
            "info_emotion": (0.0, 1.0),
            "content_influence": (0.0, 1.0),
        }

        for name, default in defaults.items():
            val = getattr(self, name)
            if val is None:
                setattr(self, name, default)
                val = default

            arr = np.asarray(val, dtype=np.float64)
            if arr.shape != (n,):
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected ({n},)"
                )
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )
            low, high = range_checks[name]
            if np.any(arr < low) or np.any(arr > high):
                raise ValueError(
                    f"{name} values outside [{low}, {high}]"
                )

        return {
            "media_exposure": self.media_exposure,
            "info_evidence": self.info_evidence,
            "official_info": self.official_info,
            "info_emotion": self.info_emotion,
            "content_influence": self.content_influence,
        }


def default_inputs(n: int, t: int, T: int) -> ExternalInputs:
    """Generate default external inputs for synthetic experiments.

    A shock pulse can be injected at a specific time for reactivation tests.
    Override this for scenario-specific inputs.
    """
    return ExternalInputs(
        shock=0.0,
        novelty=0.0,
        staleness=t / max(T, 1),
    )


# ─────────────────────────────────────────────────────────────
# Transition Engine
# ─────────────────────────────────────────────────────────────


class TransitionEngine:
    """Stateless transition calculator implementing the 7-step cycle.

    Usage:
        engine = TransitionEngine(params, rng)
        new_state = engine.step(state, inputs, t)
    """

    def __init__(
        self,
        params: ModelParams,
        rng: Optional[Generator] = None,
    ):
        self.p = params
        self.rng = rng if rng is not None else np.random.default_rng()

    # ── Step 1: Information Exposure ──────────────────────────

    def compute_exposure(
        self,
        state: AgentState,
        G_s: np.ndarray,
        q_j: np.ndarray,       # content influence of each agent as source
        M_i: np.ndarray,       # media exposure per agent
        V: float = 0.0,        # platform viral intensity V(t)
    ) -> np.ndarray:
        """Step 1: Compute Lambda_i(t) — information exposure intensity.

        Lambda_i = beta * sum_j w_ji^s * 1[z_j=A] * q_j    [network]
                 + beta_M * M_i                              [media]
                 + beta_V * V                                [platform viral]

        Args:
            state: Current agent state.
            G_s: Propagation network adjacency (row=i receives from col=j).
            q_j: Content influence strength for each agent j.
            M_i: Media exposure for each agent i.
            V: Platform viral intensity V(t) in [0, 1].

        Returns:
            Lambda array of shape (N,), exposure intensity in [0, inf).
        """
        pp = self.p.propagation
        vp = self.p.viral

        # Active source vector: which agents are in A state
        active = (state.z == A).astype(np.float64)

        # Weighted active neighbors: G_s[i, j] * active[j] * q_j[j]
        neighbor_exposure = G_s.dot(active * q_j)

        Lambda = pp.beta * neighbor_exposure + pp.beta_M * M_i + vp.beta_V * V
        return Lambda

    # ── Step 2: U → E ─────────────────────────────────────────

    def expose_agents(
        self,
        state: AgentState,
        Lambda: np.ndarray,
    ) -> np.ndarray:
        """Step 2: Determine which U agents become E (exposed).

        P(U → E) = min(Λ_i, 1)  (or sigmoid if θ_Λ > 0)

        Returns:
            Updated z array (same shape, modified in-place conceptually).
            Newly exposed agents marked as E.
        """
        pp = self.p.propagation
        z = state.z.copy()

        if pp.theta_Lambda > 0:
            # Sigmoid version
            prob = 1.0 / (1.0 + np.exp(-(Lambda - pp.theta_Lambda)))
        else:
            prob = np.minimum(Lambda, 1.0)

        # Only U agents can be exposed
        u_mask = z == U
        expose = (self.rng.random(state.n) < prob) & u_mask
        z[expose] = E
        return z

    # ── Step 3: Private Opinion Update ─────────────────────────

    def update_opinions(
        self,
        state: AgentState,
        z_before_expose: np.ndarray,  # state BEFORE Step 2 (who was already not-U)
        G_o: np.ndarray,
        I_i: np.ndarray,      # information evidence per agent
        u_i: np.ndarray,      # official information per agent
    ) -> np.ndarray:
        """Step 3: Update private opinions o_i(t+1).

        Only agents that were already not-U before Step 2 receive opinion updates.
        U agents' opinions don't change (they have no new information).

        o_i(t+1) = Π_{[-1,1]}[ o_i(t) + μ_i · (ζ_i[o_i(0)-o_i(t)]
                    + (1-ζ_i)·Σ_j w_ji^o·Φ_i(ô_j(t)-o_i(t))
                    + η_i·I_i(t) + χ_i·u_i(t) ) + ξ_i(t) ]
        """
        op = self.p.opinion
        attrs = state.attrs
        n = state.n
        o = state.o.copy()
        o_hat = state.o_hat.copy()

        # Replace NaN in o_hat with 0 for computation (NaN agents are not A, contribute 0)
        # ── Social influence term ──
        # CRITICAL: Only agents who are ACTIVELY EXPRESSING (z==A AND o_hat valid)
        # can influence others. Silent agents (D, U) and agents with NaN o_hat
        # contribute NOTHING — their silence is NOT interpreted as "opinion 0."
        #
        # For each agent i: Σ_{j in A(t)} w_ji^o · Φ_i(ô_j - o_i) / Σ_{j in A(t)} w_ji^o
        # where A(t) = {j: z_j=A and o_hat_j is finite}

        expressed_mask = (state.z == A) & np.isfinite(state.o_hat)
        n_expressed = expressed_mask.sum()

        if n_expressed > 0:
            # Only compute differences between i and expressed agents j
            o_hat_valid = np.where(expressed_mask, state.o_hat, 0.0)
            D = o_hat_valid[np.newaxis, :] - o[:, np.newaxis]  # (N, N), but j not in A => D[i,j]=0-o_i

            # Bounded confidence: only consider expressed j within epsilon
            eps = attrs.epsilon[:, np.newaxis]
            valid_influence = expressed_mask[np.newaxis, :] & (np.abs(D) <= eps)

            # Weighted average: normalize by sum of valid influence weights
            weights = G_o * valid_influence.astype(np.float64)
            weight_sum = weights.sum(axis=1)

            social_influence = np.divide(
                (weights * D).sum(axis=1),
                weight_sum,
                out=np.zeros(n),
                where=weight_sum > 1e-8,
            )
        else:
            social_influence = np.zeros(n)

        # ── Anchoring term ──
        # For anchoring we need o_i(0). In v1.0, we don't store o_i(0) separately.
        # Instead, we use a stored initial_opinion array.
        # For now, anchoring pulls toward the current mean of the agent's camp.
        # This is a simplification; full anchoring to o_i(0) requires storing init.
        # We store o_i(0) in the simulation runner.
        anchoring = np.zeros(n)  # Will be set by caller with actual o_0

        # ── Information evidence + official info ──
        info_term = attrs.eta * I_i
        official_term = attrs.chi * u_i

        # ── Noise ──
        xi = self.rng.normal(0, attrs.sigma_xi, size=n)

        # ── Assemble update ──
        # Only apply to agents that were not-U before Step 2
        update_mask = z_before_expose != U

        delta_o = attrs.mu * (
            anchoring
            + (1.0 - attrs.zeta) * social_influence
            + info_term
            + official_term
        ) + xi

        o[update_mask] = o[update_mask] + delta_o[update_mask]
        o = np.clip(o, -1.0, 1.0)

        return o

    # ── Step 4: Emotion & Fatigue ──────────────────────────────

    def update_emotion_fatigue(
        self,
        state: AgentState,
        G_h: np.ndarray,      # emotional contagion network
        I_m: np.ndarray,      # information emotion intensity per agent
        shock: float,         # Shock(t)
        N_exp: np.ndarray,    # number of exposures per agent in this step
    ) -> tuple[np.ndarray, np.ndarray]:
        """Step 4: Update h_i(t+1) and f_i(t+1).

        h_i(t+1) = Π[ (1-δ_h)h_i + η_h·I_m + ω_h·Σ w_ji^h·h_j + χ_h·Shock - ν_h·f_i ]
        f_i(t+1) = Π[ (1-δ_f)f_i + η_f·N_i^exp + ω_f·𝟙[z_i=A] ]
        """
        ef = self.p.emotion_fatigue

        # ── Emotion update ──
        emotional_contagion = G_h.dot(state.h)  # Σ_j w_ji^h · h_j

        h_new = (
            (1.0 - ef.delta_h) * state.h
            + ef.eta_h * I_m
            + ef.omega_h * emotional_contagion
            + ef.chi_h * shock
            - ef.nu_h * state.f
        )
        h_new = np.clip(h_new, 0.0, 1.0)

        # ── Fatigue update ──
        f_new = (
            (1.0 - ef.delta_f) * state.f
            + ef.eta_f * N_exp
            + ef.omega_f * (state.z == A).astype(np.float64)
        )
        f_new = np.clip(f_new, 0.0, 1.0)

        return h_new, f_new

    # ── Step 5: Resolve E → A or D ─────────────────────────────

    def resolve_exposed(
        self,
        state: AgentState,
        z: np.ndarray,            # z after Step 2 (has E agents)
        h_new: np.ndarray,        # h(t+1) from Step 4
        f_new: np.ndarray,        # f(t+1) from Step 4
        G_o: np.ndarray,          # for computing Γ_i
    ) -> np.ndarray:
        """Step 5: Resolve E agents → A or D.

        Uses updated opinion, emotion, fatigue for activation decision.
        Applies silence spiral correction.

        Returns updated z array.
        """
        ap = self.p.activation
        op = self.p.opinion
        attrs = state.attrs
        n = state.n

        z_new = z.copy()
        e_mask = z_new == E
        if not e_mask.any():
            return z_new

        # ── Compute Γ_i (opinion climate congruity) for E agents ──
        # Only agents who are ACTIVELY EXPRESSING contribute to climate.
        expressed_mask = (state.z == A) & np.isfinite(state.o_hat)
        V_MIN = self.p.opinion.climate_visibility_threshold

        if expressed_mask.any():
            expressed = expressed_mask.astype(np.float64)
            neighbor_active = G_o * expressed[np.newaxis, :]
            v_i = neighbor_active.sum(axis=1)  # climate visibility
            # Use valid o_hat values (NaN already filtered by expressed_mask)
            weighted_opinion = G_o.dot(np.where(expressed_mask, state.o_hat, 0.0))
            local_climate = np.divide(
                weighted_opinion,
                v_i,
                out=np.zeros(n),
                where=v_i > 1e-8,
            )
        else:
            v_i = np.zeros(n)
            local_climate = np.zeros(n)

        # Γ_i = 1 - |o_i - climate| / 2, only when climate is visible
        Gamma = 1.0 - np.abs(state.o - local_climate) / 2.0
        Gamma = np.clip(Gamma, 0.0, 1.0)
        climate_visible = v_i >= V_MIN

        # ── Base activation probability ──
        # Climate effect scaled by visibility: no visible climate -> no climate effect
        v_i_normalized = np.minimum(v_i / max(V_MIN, v_i.max()), 1.0)
        logit = (
            ap.alpha_0
            + ap.alpha_1 * np.abs(state.o)
            + ap.alpha_2 * h_new
            + ap.alpha_3 * v_i_normalized * Gamma * climate_visible
            - ap.alpha_4 * f_new
            - ap.alpha_5 * attrs.c
        )
        p_activate = 1.0 / (1.0 + np.exp(-logit))

        # ── Silence spiral correction ──
        # Only apply when climate is visible AND agent is in minority
        minority_mask = (Gamma < 0.5) & climate_visible
        spiral_penalty = np.ones(n)
        spiral_penalty[minority_mask] = (
            1.0 - op.lambda_spiral * (0.5 - Gamma[minority_mask])
        )
        p_final = p_activate * spiral_penalty
        p_final = np.clip(p_final, 0.0, 1.0)

        # ── Decision ──
        activate = self.rng.random(n) < p_final
        activate = activate & e_mask  # Only E agents decide

        z_new[activate] = A
        z_new[e_mask & ~activate] = D

        return z_new

    # ── Step 6: A → D and D → A ────────────────────────────────

    def process_decay_reactivation(
        self,
        state: AgentState,
        z: np.ndarray,           # z after Step 5
        h_new: np.ndarray,
        f_new: np.ndarray,
        shock: float,
        novelty: float,
        staleness: float,
        e_mask_before_step5: np.ndarray,  # which agents were E before Step 5
    ) -> np.ndarray:
        """Step 6: A -> D decay and D -> A reactivation.

        Uses history flag m_i to distinguish:
          - m=0: delayed activation (higher baseline, lower shock response)
          - m=1: true reactivation (lower baseline, higher shock response)

        Excludes agents that just became A or D in Step 5.
        """
        dp = self.p.decay
        rp = self.p.reactivation

        z_new = z.copy()

        # ── A → D decay ──
        # Only agents that were A BEFORE Step 5 (not newly activated)
        a_mask = (state.z == A) & ~e_mask_before_step5
        if a_mask.any():
            logit_decay = (
                dp.gamma_0
                + dp.gamma_1 * f_new
                + dp.gamma_2 * staleness
                - dp.gamma_3 * novelty
            )
            p_decay = 1.0 / (1.0 + np.exp(-logit_decay))
            decay = self.rng.random(z_new.shape[0]) < p_decay
            z_new[a_mask & decay] = D

        # ── D → A reactivation ──
        # Only agents that were D BEFORE Step 5 (not newly dormant)
        d_mask = (state.z == D) & ~e_mask_before_step5
        if d_mask.any():
            m_vals = state.m[d_mask]
            # Select params based on m_i: m=0 -> delayed, m=1 -> true reactivation
            r0_vec = np.where(m_vals == 0, rp.r_0_0, rp.r_0_1)
            r1_vec = np.where(m_vals == 0, rp.r_1_0, rp.r_1_1)

            logit_react = (
                r0_vec
                + r1_vec * shock
                + rp.r_2 * novelty
                + rp.r_3 * h_new[d_mask]
            )
            p_react = 1.0 / (1.0 + np.exp(-logit_react))
            react = self.rng.random(d_mask.sum()) < p_react
            react_indices = np.where(d_mask)[0][react]
            z_new[react_indices] = A

        return z_new

    # ── Step 7: Public Expression ───────────────────────────────

    def generate_expressions_with_o(
        self,
        o_now: np.ndarray,       # current private opinion (from Step 3)
        z_new: np.ndarray,       # new propagation states (after Step 6)
        o_hat_old: np.ndarray,   # old public expressions (from Step 7, t)
        z_old: np.ndarray,       # old propagation states (from Step 6 input)
        h_now: np.ndarray,       # current emotional arousal (from Step 4)
        G_o: np.ndarray,
    ) -> np.ndarray:
        """Step 7: Generate ô_i(t+1) for agents in A state.

        Uses the UPDATED opinion o_now and emotion h_now.
        """
        op = self.p.opinion
        n = len(o_now)

        o_hat_new = np.full(n, np.nan, dtype=np.float64)

        a_mask = z_new == A
        if not a_mask.any():
            return o_hat_new

        # Compute local climate from old expressions, with visibility threshold
        V_MIN = self.p.opinion.climate_visibility_threshold
        expressed_mask_old = (z_old == A) & np.isfinite(o_hat_old)

        if expressed_mask_old.any():
            expressed = expressed_mask_old.astype(np.float64)
            neighbor_active = G_o * expressed[np.newaxis, :]
            v_i = neighbor_active.sum(axis=1)
            weighted_opinion = G_o.dot(np.where(expressed_mask_old, o_hat_old, 0.0))
            local_climate = np.divide(
                weighted_opinion,
                v_i,
                out=np.zeros(n),
                where=v_i > 1e-8,
            )
            climate_visible = v_i >= V_MIN
        else:
            local_climate = np.zeros(n)
            climate_visible = np.zeros(n, dtype=bool)

        # Conformity deviation (only when climate is visible)
        conformity = op.lambda_c * (local_climate - o_now) * climate_visible

        # Emotional amplification
        emotional_amp = op.lambda_h * h_now * o_now

        o_hat_new[a_mask] = (
            o_now[a_mask]
            + conformity[a_mask]
            + emotional_amp[a_mask]
        )
        o_hat_new = np.clip(o_hat_new, -1.0, 1.0)

        return o_hat_new

    def generate_expressions(
        self,
        state: AgentState,
        z_new: np.ndarray,
        G_o: np.ndarray,
    ) -> np.ndarray:
        """Legacy wrapper — uses state attributes. Prefer generate_expressions_with_o."""
        return self.generate_expressions_with_o(
            o_now=state.o, z_new=z_new, o_hat_old=state.o_hat,
            z_old=state.z, h_now=state.h, G_o=G_o,
        )

    # ── Full Step ───────────────────────────────────────────────

    def step(
        self,
        state: AgentState,
        G_s: np.ndarray,
        G_o: np.ndarray,
        G_h: Optional[np.ndarray],
        inputs: ExternalInputs,
        o_initial: np.ndarray,    # o_i(0) for anchoring term
        t: int = 0,
    ) -> tuple[AgentState, float, TransitionEvents]:
        """Execute one complete 7-step update cycle.

        Args:
            state: Current agent state at beginning of time step t.
            G_s: Propagation network adjacency.
            G_o: Opinion influence network (row-normalized).
            G_h: Emotional contagion network (defaults to G_o).
            inputs: External input signals for this time step.
            o_initial: Initial opinions o_i(0) for anchoring.
            t: Current time step index (for logging/debugging).

        Returns:
            (New AgentState for time t+1, V_next, TransitionEvents).
        """
        if G_h is None:
            G_h = G_o

        n = state.n
        inputs_dict = inputs.resolve(n)
        M_i = inputs_dict["media_exposure"]
        q_j = inputs_dict["content_influence"]
        I_i = inputs_dict["info_evidence"]
        u_i = inputs_dict["official_info"]
        I_m = inputs_dict["info_emotion"]

        # ── Step 1: Compute exposure ──
        Lambda = self.compute_exposure(state, G_s, q_j, M_i, inputs.V)

        # ── Step 2: U → E ──
        z_after_expose = self.expose_agents(state, Lambda)

        # V1.1 fix: newly exposed agents (U->E) must absorb information
        # before E->A decision. Use post-exposure state for update mask.
        aware_after_exposure = z_after_expose != U

        # Track which agents are newly E (for Step 5 and Step 6 exclusion)
        new_e_mask = (state.z == U) & (z_after_expose == E)
        was_e_before_step5 = new_e_mask.copy()

        # ── Step 3: Update private opinions ──
        # Anchoring term: ζ_i * [o_i(0) - o_i(t)]
        anchoring = state.attrs.zeta * (o_initial - state.o)

        # For Step 3, we need to compute the full update manually
        # because we need to inject the anchoring term
        op = self.p.opinion
        attrs = state.attrs

        # Only agents who are actively expressing can influence others
        expressed_mask = (state.z == A) & np.isfinite(state.o_hat)
        if expressed_mask.any():
            o_hat_valid = np.where(expressed_mask, state.o_hat, 0.0)
            D = o_hat_valid[np.newaxis, :] - state.o[:, np.newaxis]
            eps = attrs.epsilon[:, np.newaxis]
            valid = expressed_mask[np.newaxis, :] & (np.abs(D) <= eps)
            weights = G_o * valid.astype(np.float64)
            weight_sum = weights.sum(axis=1)
            social_influence = np.divide(
                (weights * D).sum(axis=1), weight_sum,
                out=np.zeros(n), where=weight_sum > 1e-8,
            )
        else:
            social_influence = np.zeros(n)

        info_term = attrs.eta * I_i
        official_term = attrs.chi * u_i
        xi = self.rng.normal(0, attrs.sigma_xi, size=n)

        delta_o = attrs.mu * (
            anchoring
            + (1.0 - attrs.zeta) * social_influence
            + info_term
            + official_term
        ) + xi

        o_new = state.o.copy()
        o_new[aware_after_exposure] = o_new[aware_after_exposure] + delta_o[aware_after_exposure]
        o_new = np.clip(o_new, -1.0, 1.0)

        # ── Step 4: Update emotion and fatigue ──
        # N_i^exp: number of exposures = number of active neighbors + media
        N_exp = (G_s > 0).dot((state.z == A).astype(np.float64)) + M_i
        # Normalize to [0, 1] range
        N_exp_norm = np.clip(N_exp / max(N_exp.max(), 1.0), 0.0, 1.0)

        h_new, f_new = self.update_emotion_fatigue(
            state, G_h, I_m, inputs.shock, N_exp_norm,
        )

        # ── Step 5: Resolve E → A or D ──
        # Build temporary state with updated o, h, f for activation decision
        temp_state = AgentState(
            z=state.z,           # original z (before Step 2)
            m=state.m,           # history flag
            o=o_new,             # updated opinion
            o_hat=state.o_hat,   # old public expression (new not yet generated)
            h=h_new,             # updated emotion
            f=f_new,             # updated fatigue
            attrs=state.attrs,
        )
        z_after_resolve = self.resolve_exposed(
            temp_state, z_after_expose, h_new, f_new, G_o,
        )

        # ── Step 6: A → D and D → A ──
        z_after_decay = self.process_decay_reactivation(
            state, z_after_resolve, h_new, f_new,
            inputs.shock, inputs.novelty, inputs.staleness,
            was_e_before_step5,
        )

        # ── Step 7: Public expression ──
        # Use o_new (from Step 3) and h_new (from Step 4), not stale values
        o_hat_new = self.generate_expressions_with_o(
            o_new, z_after_decay, state.o_hat, state.z, h_new, G_o,
        )

        # ── Compute V_next (platform viral amplification) ──
        vp = self.p.viral
        new_posts_mask = (state.z != A) & (z_after_decay == A)
        n_new_posts = int(new_posts_mask.sum())
        if n_new_posts > 0:
            mean_q_new = float(q_j[new_posts_mask].mean())
        else:
            mean_q_new = 0.0
        V_next = (1.0 - vp.delta_V) * inputs.V + vp.eta_V * n_new_posts * mean_q_new
        V_next = float(np.clip(V_next, 0.0, 1.0))

        # ── Record transition events (use int constants to avoid variable shadowing) ──
        events = TransitionEvents(
            U_to_E=int(new_e_mask.sum()),
            E_to_A=int((was_e_before_step5 & (z_after_decay == 2)).sum()),   # 2 = A
            E_to_D=int((was_e_before_step5 & (z_after_decay == 3)).sum()),   # 3 = D
            A_to_D=int(((state.z == 2) & ~was_e_before_step5 & (z_after_decay == 3)).sum()),
            D0_to_A=int(((state.z == 3) & (state.m == 0) & ~was_e_before_step5 & (z_after_decay == 2)).sum()),
            D1_to_A=int(((state.z == 3) & (state.m == 1) & ~was_e_before_step5 & (z_after_decay == 2)).sum()),
            new_posts=n_new_posts,
        )

        # ── Update m: agents who just activated get m=1 ──
        m_updated = state.m.copy()
        newly_activated = (state.z != A) & (z_after_decay == A)
        m_updated[newly_activated] = 1

        # ── Build final state ──
        new_state = AgentState(
            z=z_after_decay,
            m=m_updated,
            o=o_new,
            o_hat=o_hat_new,
            h=h_new,
            f=f_new,
            attrs=state.attrs,
        )
        return new_state, V_next, events
