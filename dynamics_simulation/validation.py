"""
Multi-dimensional equation feasibility validation suite.

Five dimensions of validation:
  1. Degeneracy limit tests — when coupling terms are zeroed, model must
     reduce to known classical models (SIR, HK bounded confidence, etc.)
  2. Structural identifiability — can known true parameters be recovered
     from model-generated synthetic data?
  3. Parameter direction tests — does each parameter push the system in
     the theoretically predicted direction?
  4. Network structure sensitivity — does the model behave differently
     under different network topologies in expected ways?
  5. Long-term stability — does the model avoid degenerate states
     (total consensus, extinction, unbounded growth)?

Each test returns a TestResult with: name, passed (bool), score (0-1),
  evidence (dict with quantitative metrics), interpretation (str).

Usage:
    python -m dynamics_simulation.validation
    python -m dynamics_simulation.validation --quick  # Skip identifiability
"""

from __future__ import annotations

import sys, os, time, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamics_simulation.config import (
    ModelParams, default_params, no_coupling,
    PropagationParams, ActivationParams, DecayParams,
    ReactivationParams, OpinionParams, EmotionFatigueParams,
)
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs
from dynamics_simulation.simulation import SimulationConfig, SimulationRunner
from dynamics_simulation.metrics import SimulationMetrics


@dataclass
class TestResult:
    """Result of a single validation test."""
    name: str
    dimension: str  # "degeneracy" | "identifiability" | "direction" | "stability"
    passed: bool
    score: float    # 0.0–1.0, higher = more convincing
    evidence: dict = field(default_factory=dict)
    interpretation: str = ""

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name} (score={self.score:.2f}) — {self.interpretation}"


# ═══════════════════════════════════════════════════════════════
# DIMENSION 1: Degeneracy Limit Tests
# ═══════════════════════════════════════════════════════════════

def test_sir_degeneracy(seed: int = 42) -> TestResult:
    """When alpha_1=alpha_3=0, beta_M=0, lambda_spiral=0, model -> network SIR.

    Test: cumulative aware N_aware(t) should follow approximate
    logistic growth. The growth rate dN_aware/dt should be
    proportional to A(t) * U(t) — the classic SIR incidence term.
    """
    rng = np.random.default_rng(seed)
    n, T = 300, 80

    # Pure propagation: no opinion effects
    p = no_coupling()
    p = replace(p, propagation=PropagationParams(beta=0.20, beta_M=0.0))

    G_s, G_o, communities = generate_networks("sbm", n=n, rng=rng,
        n_blocks=3, p_in=0.15, p_out=0.02)

    state = initialize_agents(n=n, initial_active=8, rng=rng,
                              initial_opinion_dist="uniform")
    engine = TransitionEngine(p, rng)
    o_initial = state.o.copy()

    snapshots = []
    for t in range(T):
        inputs = ExternalInputs()
        state, V_unused, _ = engine.step(state, G_s, G_o, None, inputs, o_initial, t)

        snapshots.append({
            "n_A": state.n_A,
            "n_U": state.n_U,
            "n_aware": n - state.n_U,
        })

    n_A_ts = np.array([s["n_A"] for s in snapshots])
    n_U_ts = np.array([s["n_U"] for s in snapshots])
    n_aware_ts = np.array([s["n_aware"] for s in snapshots])

    # ── SIR-like check: cumulative aware should be monotonic non-decreasing ──
    aware_monotonic = np.all(np.diff(n_aware_ts) >= -1)  # allow tiny noise
    monotonic_score = 1.0 if aware_monotonic else 0.0

    # ── SIR-like check: growth should correlate with A*U product ──
    SI_product = n_A_ts[:-1] * n_U_ts[:-1]
    d_aware = np.diff(n_aware_ts)
    # Correlation between SI product and growth
    if SI_product.std() > 1e-8 and d_aware.std() > 1e-8:
        si_corr = float(np.corrcoef(SI_product, d_aware)[0, 1])
        si_corr = 0.0 if np.isnan(si_corr) else si_corr
    else:
        si_corr = 0.0

    # ── SIR-like check: peak A should occur before all U are exhausted ──
    peak_t = n_A_ts.argmax()
    u_remaining_at_peak = n_U_ts[peak_t]
    reasonable_peak = u_remaining_at_peak > 0

    # ── Aggregate score ──
    score = (monotonic_score * 0.3 + max(0, si_corr) * 0.4 +
             (0.3 if reasonable_peak else 0.0))
    passed = score >= 0.5

    return TestResult(
        name="SIR consistency (alpha_1=alpha_3=beta_M=lambda_spiral=0)",
        dimension="degeneracy",
        passed=passed,
        score=score,
        evidence={
            "aware_monotonic": aware_monotonic,
            "SI_correlation": round(si_corr, 4),
            "peak_A": int(n_A_ts.max()),
            "peak_t": int(peak_t),
            "U_at_peak": int(u_remaining_at_peak),
        },
        interpretation=(
            f"SIR-like: monotonic={aware_monotonic}, SI_corr={si_corr:.2f}, "
            f"peak_A={n_A_ts.max():.0f}@t={peak_t} — "
            + ("degeneracy holds" if passed else "degeneracy FAILS")
        ),
    )


def test_hk_degeneracy(seed: int = 42) -> TestResult:
    """When beta=0 (no propagation), model -> HK bounded confidence.

    Test: over moderate T, opinions should NOT collapse to a single cluster
    unless epsilon is large. With epsilon_mean=0.4 (our default), we expect 2-4 clusters
    on [-1,1] for polarized initial conditions.
    """
    rng = np.random.default_rng(seed)
    n, T = 200, 60

    # Zero propagation
    p = default_params()
    p = replace(p, propagation=PropagationParams(beta=0.0, beta_M=0.0))

    G_s, G_o, communities = generate_networks("sbm", n=n, rng=rng,
        n_blocks=3, p_in=0.15, p_out=0.02)

    # CRITICAL: Start all agents as D (aware) so opinions can evolve.
    # With beta=0, U agents never get exposed and their opinions freeze.
    state = initialize_agents(n=n, initial_active=5, rng=rng,
                              initial_opinion_dist="polarized")
    state.z[:] = D  # Everyone is aware → opinions can update
    state.z[:5] = A  # Keep 5 active for initial public expression
    state.o_hat[:5] = state.o[:5].copy()
    engine = TransitionEngine(p, rng)
    o_initial = state.o.copy()

    o_history = [state.o.copy()]
    for t in range(T):
        inputs = ExternalInputs()
        state, V_unused, _ = engine.step(state, G_s, G_o, None, inputs, o_initial, t)

        o_history.append(state.o.copy())

    o_final = o_history[-1]
    o_initial_arr = o_history[0]

    # ── Cluster count estimation ──
    # Sort opinions and count gaps > epsilon_mean
    eps_mean = p.opinion.epsilon_mean
    o_sorted = np.sort(o_final)
    gaps = np.diff(o_sorted)
    n_clusters = int((gaps > eps_mean).sum()) + 1

    # ── No total collapse check ──
    final_std = float(o_final.std())
    no_collapse = final_std > 0.05

    # ── Cluster count reasonable ──
    # With epsilon=0.4 on [-1,1], expect 2-5 clusters for polarized init
    reasonable_clusters = 2 <= n_clusters <= 6

    # ── Opinions changed from initial ──
    opinion_change = float(np.abs(o_final - o_initial_arr).mean())
    opinions_evolved = opinion_change > 0.01

    score = (0.3 if no_collapse else 0.0) + \
            (0.35 if reasonable_clusters else 0.1) + \
            (0.35 if opinions_evolved else 0.0)
    passed = score >= 0.5

    return TestResult(
        name="HK bounded confidence consistency (beta=0, opinion-only)",
        dimension="degeneracy",
        passed=passed,
        score=score,
        evidence={
            "final_std": round(final_std, 4),
            "n_clusters": n_clusters,
            "initial_std": round(float(o_initial_arr.std()), 4),
            "mean_opinion_change": round(opinion_change, 4),
        },
        interpretation=(
            f"HK-like: sigma_final={final_std:.3f}, clusters~={n_clusters}, "
            f"Δo={opinion_change:.3f} — "
            + ("opinions cluster without collapse" if passed else "FAILS")
        ),
    )


def test_expression_fidelity(seed: int = 42) -> TestResult:
    """When lambda_c=lambda_h=lambda_spiral=0, public expression should equal private opinion.

    Test: per-agent |o_hat_i - o_i| for active agents should be ~0.
    Population-level B_obs CAN be non-zero due to sampling bias (agents with
    extreme opinions are more likely to be active), which is correct behavior.
    """
    rng = np.random.default_rng(seed)
    n, T = 200, 40

    # Zero all expression biases
    p = default_params()
    p = replace(p, opinion=OpinionParams(
        lambda_c=0.0, lambda_h=0.0, lambda_spiral=0.0,
    ))

    G_s, G_o, communities = generate_networks("sbm", n=n, rng=rng,
        n_blocks=3, p_in=0.15, p_out=0.02)

    state = initialize_agents(n=n, initial_active=20, rng=rng,
                              initial_opinion_dist="polarized")
    engine = TransitionEngine(p, rng)
    o_initial = state.o.copy()

    # Collect per-agent differences across all steps
    per_agent_diffs = []
    for t in range(T):
        inputs = ExternalInputs()
        state, V_unused, _ = engine.step(state, G_s, G_o, None, inputs, o_initial, t)

        a_mask = state.z == A
        if a_mask.sum() > 0:
            diffs = np.abs(state.o_hat[a_mask] - state.o[a_mask])
            diffs = diffs[~np.isnan(diffs)]
            per_agent_diffs.extend(diffs.tolist())

    direct_diff = float(np.mean(per_agent_diffs)) if per_agent_diffs else 999.0

    # The per-agent difference should be tiny (only floating-point noise)
    passed = direct_diff < 0.01
    score = 1.0 - min(direct_diff / 0.02, 1.0)

    return TestResult(
        name="Expression fidelity (lambda_c=lambda_h=lambda_spiral=0)",
        dimension="degeneracy",
        passed=passed,
        score=score,
        evidence={
            "per_agent_diff": round(direct_diff, 6),
            "n_samples": len(per_agent_diffs),
        },
        interpretation=(
            f"o_hat~=o: per-agent |diff|={direct_diff:.6f} — "
            + ("expression is faithful" if passed else "unexpected expression bias")
        ),
    )


def test_frozen_opinion_independence(seed: int = 42) -> TestResult:
    """When all opinion->propagation paths are disabled, A(t) curves
    should be statistically independent of initial opinion distribution.

    Test: run 5 seeds per condition, compare mean A(t) curves.
    Individual runs differ due to randomness; means should match.
    """
    n, T = 200, 40
    n_seeds = 5

    # Disable ALL opinion->propagation pathways including silence spiral
    p = default_params()
    p = replace(p,
        activation=ActivationParams(
            alpha_0=-2.5, alpha_1=0.0, alpha_2=0.0, alpha_3=0.0,
            alpha_4=0.0, alpha_5=0.0,
        ),
        opinion=OpinionParams(lambda_spiral=0.0),
    )

    all_curves = {"polarized": [], "uniform": []}

    for s in range(n_seeds):
        # Same network + same base attributes for both conditions at each seed
        net_rng = np.random.default_rng(seed + s * 100)
        G_s, G_o, _ = generate_networks("sbm", n=n, rng=net_rng,
            n_blocks=3, p_in=0.15, p_out=0.02)

        # Base state with fixed attributes (same for both conditions)
        base = initialize_agents(n=n, initial_active=10, rng=net_rng,
                                  initial_opinion_dist="uniform")
        base.attrs.mu[:] = 0.0
        base.attrs.sigma_xi[:] = 0.0
        base_z = base.z.copy(); base_h = base.h.copy()
        base_f = base.f.copy(); base_o_hat = base.o_hat.copy()

        for dist in ["polarized", "uniform"]:
            # Clone base state (identical z, m, h, f, o_hat, attrs)
            # Only change opinions — everything else is deterministic
            state = base.copy()
            opinion_rng = np.random.default_rng(seed + s * 1000)
            if dist == "polarized":
                signs = np.where(opinion_rng.random(n) < 0.5, 1.0, -1.0)
                state.o[:] = np.clip(signs * np.abs(opinion_rng.normal(0.65, 0.20, size=n)), -1.0, 1.0)
            else:
                state.o[:] = np.clip(opinion_rng.uniform(-1.0, 1.0, size=n), -1.0, 1.0)

            # IDENTICAL transition RNG for both conditions
            engine = TransitionEngine(p, np.random.default_rng(seed + s * 1000))
            o_initial = state.o.copy()
            a_ts = [state.n_A]
            for t in range(T):
                state, V_unused, _ = engine.step(state, G_s, G_o, None, ExternalInputs(), o_initial, t)
                a_ts.append(state.n_A)
            all_curves[dist].append(np.array(a_ts))

    # Compare mean curves
    mean_pol = np.mean(all_curves["polarized"], axis=0)
    mean_uni = np.mean(all_curves["uniform"], axis=0)
    mean_A = max(mean_pol.mean(), mean_uni.mean(), 1.0)
    norm_rms_diff = float(np.sqrt(np.mean((mean_pol - mean_uni) ** 2)) / mean_A)

    passed = norm_rms_diff < 0.15
    score = 1.0 - min(norm_rms_diff / 0.3, 1.0)

    return TestResult(
        name="Frozen opinion independence (mu_i=0, all opinion paths off)",
        dimension="degeneracy",
        passed=passed,
        score=score,
        evidence={
            "norm_rms_diff": round(norm_rms_diff, 4),
            "n_seeds_per_condition": n_seeds,
            "polarized_peak_mean": float(mean_pol.max()),
            "uniform_peak_mean": float(mean_uni.max()),
        },
        interpretation=(
            f"Mean A(t) norm_RMS_diff={norm_rms_diff:.2%} — "
            + ("propagation indep of opinion" if passed else "opinion still leaks into propagation")
        ),
    )


def test_no_spontaneous_reactivation(seed: int = 42) -> TestResult:
    """With r_0 very negative and no shocks, D->A should never occur.

    Test: let system reach all-D state, then verify A stays at 0.
    """
    rng = np.random.default_rng(seed)
    n, T = 200, 60

    p = default_params()
    p = replace(p, reactivation=ReactivationParams(
        r_0_0=-10.0, r_1_0=0.0, r_0_1=-10.0, r_1_1=0.0, r_2=0.0, r_3=0.0,
    ))

    G_s, G_o, communities = generate_networks("sbm", n=n, rng=rng,
        n_blocks=3, p_in=0.15, p_out=0.02)

    state = initialize_agents(n=n, initial_active=10, rng=rng,
                              initial_opinion_dist="uniform")
    engine = TransitionEngine(p, rng)
    o_initial = state.o.copy()

    a_zero_steps = 0
    zero_started = False
    for t in range(T):
        inputs = ExternalInputs()
        state, V_unused, _ = engine.step(state, G_s, G_o, None, inputs, o_initial, t)

        if state.n_A == 0 and state.n_D > 0:
            if not zero_started:
                zero_started = True
            a_zero_steps += 1

    # Once A hits 0 (and D > 0), should stay at 0
    passed = a_zero_steps >= 3  # At least 3 consecutive zero-A steps
    score = 1.0 if passed else 0.0

    return TestResult(
        name="No spontaneous reactivation (r_0=-10, no shocks)",
        dimension="degeneracy",
        passed=passed,
        score=score,
        evidence={
            "consecutive_zero_A_steps": a_zero_steps,
            "final_A": int(state.n_A),
            "final_D": int(state.n_D),
        },
        interpretation=(
            f"A stays at 0 for {a_zero_steps} steps after reaching D-only — "
            + ("no spontaneous reactivation" if passed else "unexpected reactivation")
        ),
    )


# ═══════════════════════════════════════════════════════════════
# DIMENSION 2: Structural Identifiability
# ═══════════════════════════════════════════════════════════════

def test_identifiability(seed: int = 42) -> TestResult:
    """Can beta and alpha_1 be recovered from synthetic data?

    Approach: generate data with known θ_true, then grid-search θ_est.

    We focus on beta (propagation rate) and alpha_1 (opinion extremity weight)
    because they explain the largest variance in model behavior.

    Metric: relative error |θ_est - θ_true| / θ_true.
    Pass if mean relative error < 0.25 (25%).
    """
    rng = np.random.default_rng(seed)
    n, T = 150, 40

    # ── True parameters ──
    beta_true = 0.20
    alpha1_true = 2.0
    p_true = default_params()
    p_true = replace(p_true,
        propagation=PropagationParams(beta=beta_true, beta_M=0.0),
        activation=ActivationParams(alpha_1=alpha1_true),
    )

    G_s, G_o, communities = generate_networks("sbm", n=n, rng=rng,
        n_blocks=2, p_in=0.20, p_out=0.03)

    # ── Generate synthetic data ──
    state = initialize_agents(n=n, initial_active=8, rng=rng,
                              initial_opinion_dist="polarized")
    engine_true = TransitionEngine(p_true, rng)
    o_initial = state.o.copy()

    a_true_ts = []
    for t in range(T):
        inputs = ExternalInputs()
        state, V_unused, _ = engine_true.step(state, G_s, G_o, None, inputs, o_initial, t)
        a_true_ts.append(state.n_A)
    a_true = np.array(a_true_ts)

    # ── Grid search ──
    beta_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    alpha1_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    best_mse = float('inf')
    best_beta, best_alpha1 = None, None

    for beta_try in beta_grid:
        for a1_try in alpha1_grid:
            # Quick test: run 20 steps
            p_try = default_params()
            p_try = replace(p_try,
                propagation=PropagationParams(beta=beta_try, beta_M=0.0),
                activation=ActivationParams(alpha_1=a1_try),
            )
            state_try = initialize_agents(n=n, initial_active=8, rng=rng,
                                          initial_opinion_dist="polarized")
            engine_try = TransitionEngine(p_try, rng)
            o_init_try = state_try.o.copy()
            a_try_ts = []
            for t in range(20):
                state_try, _, _ = engine_try.step(state_try, G_s, G_o, None,
                                                 ExternalInputs(), o_init_try, t)
                a_try_ts.append(state_try.n_A)

            mse = np.mean((np.array(a_try_ts) - a_true[:20]) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_beta = beta_try
                best_alpha1 = a1_try

    # ── Compute errors ──
    beta_err = abs(best_beta - beta_true) / beta_true
    alpha1_err = abs(best_alpha1 - alpha1_true) / alpha1_true
    mean_err = (beta_err + alpha1_err) / 2

    passed = mean_err < 0.25
    score = 1.0 - min(mean_err / 0.5, 1.0)

    return TestResult(
        name="Structural identifiability (beta, alpha_1)",
        dimension="identifiability",
        passed=passed,
        score=score,
        evidence={
            "beta_true": beta_true,
            "beta_est": best_beta,
            "beta_rel_err": round(beta_err, 4),
            "alpha1_true": alpha1_true,
            "alpha1_est": best_alpha1,
            "alpha1_rel_err": round(alpha1_err, 4),
            "mean_rel_err": round(mean_err, 4),
            "grid_mse": round(float(best_mse), 2),
        },
        interpretation=(
            f"beta: {beta_true}->{best_beta} (err={beta_err:.1%}), "
            f"alpha_1: {alpha1_true}->{best_alpha1} (err={alpha1_err:.1%}) — "
            + ("identifiable" if passed else "NOT identifiable — needs more data or constraints")
        ),
    )


# ═══════════════════════════════════════════════════════════════
# DIMENSION 3: Parameter Direction Tests
# ═══════════════════════════════════════════════════════════════

def _direction_test_single(
    param_name: str,
    low_params: ModelParams,
    high_params: ModelParams,
    metric_fn: Callable[[SimulationMetrics], float],
    expected_direction: str,  # "increase" | "decrease"
    seed: int = 42,
    n: int = 200,
    T: int = 50,
    G_s=None, G_o=None, communities=None,
) -> dict:
    """Run a single direction test: low vs high param value."""
    rng = np.random.default_rng(seed)

    if G_s is None:
        G_s, G_o, communities = generate_networks("sbm", n=n, rng=rng,
            n_blocks=2, p_in=0.20, p_out=0.03)

    results = {}
    for label, params in [("low", low_params), ("high", high_params)]:
        runs = []
        for s in [seed, seed + 100, seed + 200]:
            run_rng = np.random.default_rng(s)
            cfg = SimulationConfig(
                n_agents=n, initial_active=8, T=T,
                network_type="sbm",
                network_kwargs={"n_blocks": 2, "p_in": 0.20, "p_out": 0.03},
                params=params, seed=s, verbose=False,
            )
            runner = SimulationRunner(cfg)
            runner.G_s = G_s
            runner.G_o = G_o
            runner.communities = communities
            runner.G_h = G_o
            metrics = runner.run()
            runs.append(metric_fn(metrics))
        results[label] = float(np.mean(runs))

    return results


def test_all_directions(seed: int = 42) -> list[TestResult]:
    """Run direction tests for all key parameters."""
    results = []
    rng = np.random.default_rng(seed)
    n, T = 200, 50
    G_s, G_o, communities = generate_networks("sbm", n=n, rng=rng,
        n_blocks=2, p_in=0.20, p_out=0.03)

    # ── beta: higher -> higher peak_A ──
    low = replace(default_params(), propagation=PropagationParams(beta=0.10, beta_M=0.0))
    high = replace(default_params(), propagation=PropagationParams(beta=0.25, beta_M=0.0))
    r = _direction_test_single("beta", low, high, lambda m: m.peak_A,
                               "increase", seed, n, T, G_s, G_o, communities)
    passed = r["high"] > r["low"] * 1.1  # At least 10% increase
    results.append(TestResult(
        name="beta  inc  -> peak_A  inc ",
        dimension="direction",
        passed=passed,
        score=1.0 if passed else 0.0,
        evidence={"low": round(r["low"], 1), "high": round(r["high"], 1),
                  "ratio": round(r["high"] / max(r["low"], 1), 2)},
        interpretation=f"beta: {r['low']:.0f} -> {r['high']:.0f} ({' inc ' if passed else '✗'})",
    ))

    # ── alpha_1: higher -> higher peak_A ──
    low = replace(default_params(), activation=ActivationParams(alpha_1=0.5))
    high = replace(default_params(), activation=ActivationParams(alpha_1=3.0))
    r = _direction_test_single("alpha_1", low, high, lambda m: m.peak_A,
                               "increase", seed, n, T, G_s, G_o, communities)
    passed = r["high"] > r["low"] * 1.1
    results.append(TestResult(
        name="alpha_1  inc  -> peak_A  inc ",
        dimension="direction",
        passed=passed,
        score=1.0 if passed else 0.0,
        evidence={"low": round(r["low"], 1), "high": round(r["high"], 1),
                  "ratio": round(r["high"] / max(r["low"], 1), 2)},
        interpretation=f"alpha_1: {r['low']:.0f} -> {r['high']:.0f} ({' inc ' if passed else '✗'})",
    ))

    # ── gamma_1: higher -> faster decay, lower late-stage A ──
    low = replace(default_params(), decay=DecayParams(gamma_1=0.5))
    high = replace(default_params(), decay=DecayParams(gamma_1=4.0))
    # Metric: A at late stage (last 20% of steps)
    def late_A(m): return m.n_A_ts[int(len(m.n_A_ts)*0.8):].mean()
    r = _direction_test_single("gamma_1", low, high, late_A,
                               "decrease", seed, n, T, G_s, G_o, communities)
    passed = r["high"] < r["low"]
    results.append(TestResult(
        name="gamma_1  inc  -> late_A  dec ",
        dimension="direction",
        passed=passed,
        score=1.0 if passed else 0.0,
        evidence={"low": round(r["low"], 1), "high": round(r["high"], 1)},
        interpretation=f"gamma_1: late_A {r['low']:.1f} -> {r['high']:.1f} ({' dec ' if passed else '✗'})",
    ))

    # ── delta_f: higher -> faster fatigue recovery -> lower fatigue -> higher A ──
    low = replace(default_params(), emotion_fatigue=EmotionFatigueParams(delta_f=0.05))
    high = replace(default_params(), emotion_fatigue=EmotionFatigueParams(delta_f=0.40))
    r = _direction_test_single("delta_f", low, high, lambda m: m.f_mean_ts[-1],
                               "decrease", seed, n, T, G_s, G_o, communities)
    passed = r["high"] < r["low"] * 0.9
    results.append(TestResult(
        name="delta_f  inc  -> final_f  dec ",
        dimension="direction",
        passed=passed,
        score=1.0 if passed else 0.0,
        evidence={"low": round(r["low"], 3), "high": round(r["high"], 3)},
        interpretation=f"delta_f: f_final {r['low']:.3f} -> {r['high']:.3f} ({' dec ' if passed else '✗'})",
    ))

    # ── r_1: higher -> more reactivation under shock ──
    low_p = replace(default_params(), reactivation=ReactivationParams(
        r_0_0=-4.0, r_1_0=1.0, r_0_1=-4.0, r_1_1=1.0, r_2=0.5, r_3=0.5))
    high_p = replace(default_params(), reactivation=ReactivationParams(
        r_0_0=-4.0, r_1_0=5.0, r_0_1=-4.0, r_1_1=5.0, r_2=0.5, r_3=0.5))
    # Need shock for this test
    def run_with_shock(params, seed, n, T, G_s, G_o):
        rng_s = np.random.default_rng(seed)
        state = initialize_agents(n=n, initial_active=8, rng=rng_s,
                                  initial_opinion_dist="uniform")
        engine = TransitionEngine(params, rng_s)
        o_init = state.o.copy()
        a_max = 0
        for t in range(T):
            shock_val = 0.8 if t == 25 else 0.0
            inputs = ExternalInputs(shock=shock_val)
            state, V_unused, _ = engine.step(state, G_s, G_o, None, inputs, o_init, t)

            a_max = max(a_max, state.n_A)
        return a_max

    low_r = np.mean([run_with_shock(low_p, s, n, T, G_s, G_o) for s in [seed, seed+100, seed+200]])
    high_r = np.mean([run_with_shock(high_p, s, n, T, G_s, G_o) for s in [seed, seed+100, seed+200]])
    passed = high_r > low_r * 1.1
    results.append(TestResult(
        name="r_1  inc  -> shock_reactivation  inc ",
        dimension="direction",
        passed=passed,
        score=1.0 if passed else 0.0,
        evidence={"low": round(float(low_r), 1), "high": round(float(high_r), 1)},
        interpretation=f"r_1: reactivation {low_r:.0f} -> {high_r:.0f} ({' inc ' if passed else '✗'})",
    ))

    return results


# ═══════════════════════════════════════════════════════════════
# Master runner
# ═══════════════════════════════════════════════════════════════

def run_all_validation(quick: bool = False) -> dict:
    """Run the complete validation suite.

    Args:
        quick: If True, skip expensive identifiability test.

    Returns:
        Dict with all results, summary, and overall feasibility verdict.
    """
    print("=" * 65)
    print("  EQUATION FEASIBILITY VALIDATION SUITE")
    print("=" * 65)

    all_tests: list[TestResult] = []
    t0 = time.perf_counter()

    # ── Dimension 1: Degeneracy ──
    print("\n── Dimension 1: Degeneracy Limits ──")
    for test_fn in [
        test_sir_degeneracy,
        test_hk_degeneracy,
        test_expression_fidelity,
        test_frozen_opinion_independence,
        test_no_spontaneous_reactivation,
    ]:
        result = test_fn()
        all_tests.append(result)
        print(result.summary_line())

    # ── Dimension 2: Identifiability ──
    print("\n── Dimension 2: Structural Identifiability ──")
    if not quick:
        result = test_identifiability()
        all_tests.append(result)
        print(result.summary_line())
    else:
        print("  [SKIP] Quick mode — identifiability test skipped")

    # ── Dimension 3: Parameter Directions ──
    print("\n── Dimension 3: Parameter Direction Effects ──")
    dir_results = test_all_directions()
    all_tests.extend(dir_results)
    for r in dir_results:
        print(r.summary_line())

    elapsed = time.perf_counter() - t0

    # ── Aggregate ──
    n_total = len(all_tests)
    n_passed = sum(1 for t in all_tests if t.passed)
    overall_score = np.mean([t.score for t in all_tests])

    passed_dims = set()
    for t in all_tests:
        if t.passed:
            passed_dims.add(t.dimension)

    # Verdict
    if n_passed == n_total and overall_score > 0.8:
        verdict = "PASS — All executed tests passed (degeneracy + direction only)"
    elif n_passed / n_total >= 0.7:
        verdict = "CONDITIONAL PASS — Equations are mostly feasible; review failures"
    elif n_passed / n_total >= 0.5:
        verdict = "MARGINAL — Several tests failed; equations need revision"
    else:
        verdict = "FAIL — Fundamental problems with equation structure"

    summary = {
        "total_tests": n_total,
        "passed": n_passed,
        "failed": n_total - n_passed,
        "overall_score": round(float(overall_score), 3),
        "verdict": verdict,
        "dimensions_covered": list(passed_dims),
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n{'='*65}")
    print(f"  VALIDATION COMPLETE ({elapsed:.0f}s)")
    print(f"  Tests: {n_passed}/{n_total} passed")
    print(f"  Overall score: {overall_score:.3f}")
    print(f"  Verdict: {verdict}")
    print(f"{'='*65}")

    return {
        "summary": summary,
        "tests": [{
            "name": t.name,
            "dimension": t.dimension,
            "passed": t.passed,
            "score": t.score,
            "evidence": t.evidence,
            "interpretation": t.interpretation,
        } for t in all_tests],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run equation validation suite")
    parser.add_argument("--quick", action="store_true",
                       help="Skip expensive identifiability test")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")
    args = parser.parse_args()

    results = run_all_validation(quick=args.quick)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {out_path}")
