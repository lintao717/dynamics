"""
Calibrated simulation: run the dynamics model with v0.1-calibrated parameters
and compare output to the observed posting pattern.

Focus on the June-July 2026 window where data density is highest.
"""
import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamics_simulation.config import (
    ModelParams, default_params,
    PropagationParams, ActivationParams, DecayParams,
    ReactivationParams, OpinionParams,
)
from dynamics_simulation.networks import generate_networks, stochastic_block
from dynamics_simulation.agents import initialize_agents, U, E, A, D
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs

OUTPUT = Path(__file__).parent.parent / "data" / "sim_results" / "calibrated_simulation.json"
UEAD_PATH = "d:/舆情分析/Tibet_data_collector/data/exported/TIB-2025-001_uead.parquet"


def get_observed_A_t():
    """Extract A(t) for June-July 2026 only."""
    df = pd.read_parquet(UEAD_PATH)
    df['time_window'] = pd.to_datetime(df['time_window'])

    # Focus on June 1 - July 23, 2026
    mask = (df['time_window'] >= '2026-06-01') & (df['time_window'] <= '2026-07-23')
    df_focus = df[mask]

    daily = df_focus.groupby('time_window').size().sort_index()
    full_idx = pd.date_range('2026-06-01', '2026-07-23', freq='D')
    daily = daily.reindex(full_idx, fill_value=0)

    return daily


def build_calibrated_params():
    """Build ModelParams calibrated to v0.1 estimates.

    From calibration:
      - gamma_0 ~ -3.84 (slow decay) BUT median interval is 1 day
        -> Use median: P(A->D per day) ~ 0.5 -> gamma_0 ~ 0.0 (matches default)
      - r_0 ~ -1.82 (higher than default -4.0, meaning easier reactivation)
      - alpha_1: undetectable -> keep default 1.5
      - r_1: positive shock effect -> keep default 3.5
      - beta: unknown -> use 0.15 (calibrated default)
      - Initial active: 732 agents over 53 days ~ 14 new agents/day
    """
    p = default_params()

    # Calibrated parameters
    p = replace(p,
        propagation=PropagationParams(beta=0.15, beta_M=0.05),
        decay=DecayParams(gamma_0=-0.5),  # sigma(-0.5)~0.38, E[duration]~2.6 days
        reactivation=ReactivationParams(r_0_0=-3.0, r_1_0=2.5, r_0_1=-1.82, r_1_1=3.5),  # calibrated: r_0_1 from 18% reactivation
    )

    return p


def build_realistic_network_and_state(n_agents, seed):
    """Build network calibrated to v0.1 scale.

    v0.1 has 758 agents with 2171 reply edges.
    k ~ 2171/758 ~ 2.9 reply edges per agent.
    Use SBM with adjusted density.
    """
    rng = np.random.default_rng(seed)

    # Build SBM with degree ~3
    G_s, G_o, communities = stochastic_block(
        n=n_agents, n_blocks=3,
        p_in=0.04,   # Lower density to match k~3
        p_out=0.005,
        block_sizes=None,
        directed=True, rng=rng,
    )

    # Initialize with opinion distribution matching v0.1 stance distribution
    # v0.1: 69% informational, 19% emotional_outburst, 14% support_gov, etc.
    state = initialize_agents(n=n_agents, initial_active=10, rng=rng,
                              initial_opinion_dist="polarized")

    # Override opinions to match v0.1 stance distribution
    # informational (neutral, low arousal) -> o~0
    # emotional_outburst (extreme) -> o~±0.7
    # support_gov -> o~+0.6
    n_info = int(n_agents * 0.69)
    n_emo = int(n_agents * 0.19)
    n_sup = n_agents - n_info - n_emo

    o = np.zeros(n_agents)
    o[:n_info] = rng.normal(0.0, 0.20, size=n_info)
    o[n_info:n_info+n_emo] = rng.normal(0.0, 0.70, size=n_emo)  # wider spread
    o[n_info+n_emo:] = rng.normal(0.55, 0.15, size=n_sup)
    state.o[:] = np.clip(o, -1.0, 1.0)
    state.o_hat[state.z == A] = state.o[state.z == A].copy()

    return G_s, G_o, communities, state


def run_calibrated_simulation():
    """Run simulation with calibrated params, compare to observed A(t)."""
    print("=" * 60)
    print("  CALIBRATED SIMULATION vs v0.1 DATA")
    print("=" * 60)

    observed = get_observed_A_t()
    T = len(observed)
    n_agents = 758  # match v0.1 agent count
    print(f"  Observed: {T} days (2026-06-01 to 2026-07-23)")
    print(f"  Total posts in window: {observed.sum()}")
    print(f"  Peak: {observed.max()} on {observed.idxmax().date()}")

    params = build_calibrated_params()
    # Lower beta to prevent runaway growth — v0.1 is sparse, not a cascade
    params = replace(params, propagation=PropagationParams(beta=0.03, beta_M=0.01))
    seeds = [42, 123, 789]
    all_sim_A = []

    # Extract shock days + intensities
    df = pd.read_parquet(UEAD_PATH)
    df['time_window'] = pd.to_datetime(df['time_window'])
    shock_df = df[df['observed_action'] == 'external_shock']
    shock_by_day = {}
    for _, row in shock_df.iterrows():
        d = row['time_window'].date()
        shock_by_day[d] = max(shock_by_day.get(d, 0), row.get('shock_intensity', 0.5))
    print(f"  Shock days in window: {len(shock_by_day)}")

    for seed in seeds:
        rng = np.random.default_rng(seed)
        G_s, G_o, communities, state = build_realistic_network_and_state(
            n_agents, seed)

        engine = TransitionEngine(params, rng)
        o_initial = state.o.copy()

        sim_A = []
        for t in range(T):
            day = observed.index[t].date()
            shock_val = shock_by_day.get(day, 0.0)
            inputs = ExternalInputs(
                shock=shock_val,
                staleness=t / T,
                novelty=np.exp(-t / 30.0),
            )
            state, V_unused, _ = engine.step(state, G_s, G_o, None, inputs, o_initial, t)
            sim_A.append(int(state.n_A))

        all_sim_A.append(np.array(sim_A))
        print(f"  seed={seed}: peak_A={max(sim_A):3d} at t={np.argmax(sim_A):2d}, "
              f"mean_A={np.mean(sim_A):.1f}, total={sum(sim_A)}")

    # Aggregate across seeds
    mean_sim_A = np.mean(all_sim_A, axis=0)
    std_sim_A = np.std(all_sim_A, axis=0)

    # ── Comparison metrics ──
    obs_A = observed.values.astype(float)

    # Correlation between simulated and observed A(t)
    corr = np.corrcoef(mean_sim_A, obs_A)[0, 1]
    print(f"\n  Correlation(mean_sim_A, obs_A): {corr:.4f}")

    # Peak comparison
    print(f"  Peak: obs={obs_A.max():.0f} (t={obs_A.argmax()}), "
          f"sim={mean_sim_A.max():.0f} (t={mean_sim_A.argmax()})")

    # Total activity
    print(f"  Total: obs={obs_A.sum():.0f}, sim={mean_sim_A.sum():.0f}")

    # Print A(t) comparison for first 10 and last 10 days
    print(f"\n  A(t) comparison (first 10 days):")
    print(f"    Day:  " + " ".join(f"{i:4d}" for i in range(10)))
    print(f"    Obs:  " + " ".join(f"{obs_A[i]:4.0f}" for i in range(min(10, len(obs_A)))))
    print(f"    Sim:  " + " ".join(f"{mean_sim_A[i]:4.0f}" for i in range(min(10, len(mean_sim_A)))))

    # ── Qualitative assessment ──
    if corr > 0.3:
        assessment = "model captures temporal pattern"
    elif corr > 0:
        assessment = "weak positive correlation"
    else:
        assessment = "model does NOT capture temporal pattern"

    print(f"\n  Assessment: {assessment}")

    # Try shock-only variant: big shock at observed peak day
    print(f"\n── VARIANT: Shock-driven spike at peak day ──")
    peak_day = int(obs_A.argmax())
    for seed in [42]:
        rng = np.random.default_rng(seed)
        G_s, G_o, communities, state = build_realistic_network_and_state(
            n_agents, seed)
        # Very low beta: no endogenous propagation, only shock-driven
        p_shock = replace(params, propagation=PropagationParams(beta=0.01, beta_M=0.005))
        engine = TransitionEngine(p_shock, rng)
        o_initial = state.o.copy()

        shock_A = []
        for t in range(T):
            # Big shock at observed peak, smaller shocks at other shock days
            day = observed.index[t].date()
            if t == peak_day:
                shock_val = 0.95
            elif t == peak_day - 1:
                shock_val = 0.7  # build-up
            elif t == peak_day + 1:
                shock_val = 0.5  # after-shock
            else:
                shock_val = 0.0
            inputs = ExternalInputs(
                shock=shock_val,
                staleness=t / T,
                novelty=1.0 if t == peak_day else 0.0,
            )
            state, V_unused, _ = engine.step(state, G_s, G_o, None, inputs, o_initial, t)
            shock_A.append(int(state.n_A))

        shock_corr = np.corrcoef(shock_A, obs_A)[0, 1]
        print(f"    Shock-driven: peak={max(shock_A)} at t={np.argmax(shock_A)}, "
              f"corr={shock_corr:.4f}")
        shock_peak = max(shock_A)
        shock_peak_t = int(np.argmax(shock_A))

    # Save
    result = {
        "T": T,
        "n_agents": n_agents,
        "params": {
            "beta": params.propagation.beta,
            "gamma_0": params.decay.gamma_0,
            "r_0": params.reactivation.r_0,
        },
        "observed": {
            "total_posts": int(obs_A.sum()),
            "peak_A": float(obs_A.max()),
            "peak_day": int(obs_A.argmax()),
            "A_t": obs_A.tolist(),
        },
        "simulated": {
            "mean_A_t": mean_sim_A.tolist(),
            "std_A_t": std_sim_A.tolist(),
            "peak_A_mean": float(mean_sim_A.max()),
            "correlation": float(corr),
        },
        "assessment": assessment,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT}")

    return result


if __name__ == "__main__":
    run_calibrated_simulation()
