"""
Event-level calibration of the integrated dynamics model to v0.1 data.

Key constraints of v0.1 data:
  - One row per post (observed_active is ALWAYS 1 at row level)
  - 758 agents, 902 posts, 151 daily windows
  - 84 agents have >=2 posts (can estimate inter-post intervals)
  - No repost edges, no follower network, no population denominator

What we CAN estimate:
  1. Aggregate A(t) curve parameters (growth/decay rates)
  2. Inter-post interval distribution -> decay rate gamma_0
  3. Reactivation rate from previously_active -> reactivated
  4. Opinion-activation correlation (do extreme opinions post more?)
  5. Shock impact (does A(t) increase after government documents?)

Output: data/sim_results/calibration_v01.json
"""
import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import expon, gamma as gamma_dist

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT = Path(__file__).parent.parent / "data" / "sim_results" / "calibration_v01.json"
UEAD_PATH = "d:/舆情分析/Tibet_data_collector/data/exported/TIB-2025-001_uead.parquet"


def load_and_preprocess():
    """Load v0.1 data and build aggregate time series."""
    df = pd.read_parquet(UEAD_PATH)
    df['time_window'] = pd.to_datetime(df['time_window'])

    # Daily post counts = A(t) observable
    daily_A = df.groupby('time_window').size().sort_index()
    # Fill missing days with 0
    full_idx = pd.date_range(daily_A.index.min(), daily_A.index.max(), freq='D')
    daily_A = daily_A.reindex(full_idx, fill_value=0)

    # Shock counts per day
    daily_shock = df[df['observed_action'] == 'external_shock'].groupby('time_window').size()
    daily_shock = daily_shock.reindex(full_idx, fill_value=0)

    return df, daily_A, daily_shock


def estimate_1_aggregate_growth_rate(daily_A):
    """Estimate the effective growth rate from the A(t) curve.

    Fits: A(t) = A_0 * exp(r * t) for the early growth phase,
    and A(t) = A_peak * exp(-d * t) for the decay phase.

    The effective growth rate r = beta * k_eff * P_activate - gamma.
    """
    print("\n── Estimate 1: Aggregate Growth/Decay Rates ──")
    n = len(daily_A)
    t = np.arange(n)
    A = daily_A.values.astype(float)

    # Find peak
    peak_idx = int(np.argmax(A))
    peak_A = A[peak_idx]
    print(f"  Time range: {daily_A.index[0].date()} to {daily_A.index[-1].date()} ({n} days)")
    print(f"  Peak: A({peak_idx}) = {peak_A:.0f} posts on {daily_A.index[peak_idx].date()}")

    # ── Growth phase (first 20 days or up to peak) ──
    growth_end = min(peak_idx + 1, 30)
    if growth_end > 5:
        t_growth = t[1:growth_end]
        A_growth = A[1:growth_end]
        A_growth = np.maximum(A_growth, 1)  # avoid log(0)

        # log-linear fit: log(A) = log(A0) + r * t
        coeffs = np.polyfit(t_growth, np.log(A_growth), 1)
        r_growth = coeffs[0]
        A0_est = np.exp(coeffs[1])
        print(f"  Growth phase (t=1..{growth_end-1}): r = {r_growth:.6f} per day")
        print(f"    Estimated A0 = {A0_est:.1f}")
        print(f"    Doubling time (if r>0): {np.log(2)/max(r_growth, 1e-6):.1f} days")
    else:
        r_growth = 0.0
        A0_est = float(A[0])
        print(f"  Growth phase: too few points to fit")

    # ── Decay phase (after peak) ──
    decay_start = peak_idx
    decay_end = min(peak_idx + 60, n)
    if decay_end - decay_start > 5:
        t_decay = t[decay_start:decay_end] - peak_idx
        A_decay = A[decay_start:decay_end]
        A_decay = np.maximum(A_decay, 1)

        coeffs = np.polyfit(t_decay, np.log(A_decay), 1)
        r_decay = coeffs[0]
        print(f"  Decay phase (t={peak_idx}..{decay_end-1}): r = {r_decay:.6f} per day")
        if r_decay < 0:
            print(f"    Half-life: {np.log(2)/abs(r_decay):.1f} days")
    else:
        r_decay = 0.0
        print(f"  Decay phase: too few points to fit")

    # ── Aggregate A(t) statistics ──
    print(f"  Mean daily posts: {A.mean():.1f}")
    print(f"  Total posts: {A.sum():.0f}")
    print(f"  Days with >0 posts: {(A>0).sum()}/{n}")

    return {
        "n_days": n,
        "peak_A": float(peak_A),
        "peak_day": int(peak_idx),
        "r_growth": float(r_growth),
        "r_decay": float(r_decay),
        "mean_daily_A": float(A.mean()),
        "total_posts": int(A.sum()),
    }


def estimate_2_decay_rate(df):
    """Estimate gamma_0 from inter-post intervals of repeat posters.

    For agents with >=2 posts, the time between consecutive posts follows
    approximately an exponential distribution with rate = P(A->D).

    P(stay active one more day) = 1 - sigma(gamma_0)
    Expected active duration = 1 / sigma(gamma_0)

    From inter-post intervals, we can estimate the probability that an
    agent posts again within k days of their previous post.
    """
    print("\n── Estimate 2: Decay Rate from Inter-Post Intervals ──")

    df = df.copy()
    df['time_window'] = pd.to_datetime(df['time_window'])
    df = df.sort_values(['agent_id', 'time_window'])

    # For each agent, compute inter-post intervals (days)
    intervals = []
    for agent_id, group in df.groupby('agent_id'):
        times = group['time_window'].values
        if len(times) >= 2:
            for i in range(len(times) - 1):
                delta = (times[i+1] - times[i]) / np.timedelta64(1, 'D')
                intervals.append(delta)

    intervals = np.array(intervals)
    if len(intervals) == 0:
        print("  No repeat posters found — cannot estimate decay rate")
        return {"n_intervals": 0, "mean_interval": None, "gamma_0_est": None}

    print(f"  Repeat-post intervals: n={len(intervals)}, "
          f"mean={intervals.mean():.1f}d, median={np.median(intervals):.1f}d")
    print(f"  Distribution: min={intervals.min():.0f}d, max={intervals.max():.0f}d")

    # Under exponential inter-event times with rate lambda:
    # P(interval > k days) = exp(-lambda * k)
    # Expected interval = 1/lambda
    # lambda = P(A->D per day) = sigma(gamma_0)
    mean_interval = float(intervals.mean())
    lambda_est = 1.0 / max(mean_interval, 1.0)

    # sigma(gamma_0) = lambda
    # gamma_0 = logit(lambda) = log(lambda / (1 - lambda))
    if lambda_est < 1.0:
        gamma_0_est = np.log(lambda_est / (1.0 - lambda_est))
    else:
        gamma_0_est = 5.0  # cap

    print(f"  Estimated lambda (daily A->D rate) = {lambda_est:.4f}")
    print(f"  Estimated gamma_0 = logit(lambda) = {gamma_0_est:.3f}")
    print(f"    (Model default gamma_0 = 0.0, sigma(0)=0.5, E[duration]=2 days)")
    print(f"    (Data implies E[duration] = {1/lambda_est:.1f} days)")

    return {
        "n_intervals": len(intervals),
        "mean_interval_days": round(mean_interval, 1),
        "median_interval_days": float(np.median(intervals)),
        "lambda_est": round(float(lambda_est), 4),
        "gamma_0_est": round(float(gamma_0_est), 3),
    }


def estimate_3_reactivation(df):
    """Estimate reactivation rate from previously_active -> reactivated.

    P(D->A) = sigma(r_0 + r_3 * h + ...)
    From data: reactivation_rate = reactivated / previously_active
    """
    print("\n── Estimate 3: Reactivation Rate ──")

    prev_active = df[df['previously_active'] == 1]
    n_prev = len(prev_active)
    n_react = int(prev_active['reactivated'].sum())
    react_rate = n_react / n_prev if n_prev > 0 else 0

    print(f"  Previously active instances: {n_prev}")
    print(f"  Reactivated: {n_react}")
    print(f"  Reactivation rate: {react_rate:.4f} ({react_rate*100:.1f}%)")

    # P(D->A) = sigma(r_0 + r_3 * h_bar)
    # With h_bar ~ 0.3 (mean arousal for previously active)
    h_bar = float(prev_active['arousal_score'].mean()) if len(prev_active) > 0 else 0.3

    # If we assume r_3 = 1.2 (default), solve for r_0:
    # r_0 = logit(P) - r_3 * h_bar
    if react_rate > 0 and react_rate < 1:
        logit_p = np.log(react_rate / (1.0 - react_rate))
        r_0_est = logit_p - 1.2 * h_bar
    else:
        r_0_est = -4.0  # default

    print(f"  Mean arousal (prev active): {h_bar:.3f}")
    print(f"  logit(P) = {np.log(react_rate/(1-react_rate)):.3f}" if 0 < react_rate < 1 else "  Cannot compute logit")
    print(f"  Estimated r_0 = {r_0_est:.3f} (default: -4.0)")

    return {
        "n_previously_active": n_prev,
        "n_reactivated": n_react,
        "reactivation_rate": round(float(react_rate), 4),
        "mean_arousal_prev_active": round(float(h_bar), 3),
        "r_0_est": round(float(r_0_est), 3),
    }


def estimate_4_opinion_effect(df):
    """Constrain alpha_1 by comparing activation patterns by stance.

    If alpha_1 > 0 (opinion extremity drives activation), then:
    - emotional_outburst agents (extreme) should post more frequently
    - informational agents (moderate) should post less frequently

    We test: do extreme-stance agents have shorter inter-post intervals?
    """
    print("\n── Estimate 4: Opinion Effect (alpha_1 constraint) ──")

    df = df.copy()
    df['time_window'] = pd.to_datetime(df['time_window'])
    df = df.sort_values(['agent_id', 'time_window'])

    # Classify stance extremity
    def stance_extremity(s):
        if s in ['emotional_outburst', 'support_gov', 'oppose_gov',
                 'support_local', 'oppose_local']:
            return 'extreme'
        elif s in ['questioning']:
            return 'moderate'
        else:
            return 'neutral'

    df['extremity'] = df['public_stance'].apply(stance_extremity)

    # For agents that post multiple times, compute mean interval by extremity
    agent_extremity = df.groupby('agent_id')['extremity'].first()
    n_posts_per_agent = df.groupby('agent_id').size()

    extreme_agents = agent_extremity[agent_extremity == 'extreme'].index
    neutral_agents = agent_extremity[agent_extremity == 'neutral'].index

    extreme_posters = extreme_agents[n_posts_per_agent[extreme_agents] >= 2]
    neutral_posters = neutral_agents[n_posts_per_agent[neutral_agents] >= 2]

    # Compute intervals
    def compute_mean_interval(agent_ids):
        intervals = []
        for aid in agent_ids:
            times = df[df['agent_id'] == aid]['time_window'].values
            for i in range(len(times) - 1):
                delta = (times[i+1] - times[i]) / np.timedelta64(1, 'D')
                intervals.append(delta)
        return np.mean(intervals) if intervals else float('inf')

    extreme_interval = compute_mean_interval(extreme_posters)
    neutral_interval = compute_mean_interval(neutral_posters)

    print(f"  Extreme-stance agents: n={len(extreme_posters)} (>=2 posts)")
    print(f"    Mean inter-post interval: {extreme_interval:.1f} days")
    print(f"  Neutral-stance agents: n={len(neutral_posters)} (>=2 posts)")
    print(f"    Mean inter-post interval: {neutral_interval:.1f} days")

    if extreme_interval < neutral_interval:
        ratio = neutral_interval / max(extreme_interval, 1)
        print(f"  Extreme agents post {ratio:.1f}x more frequently than neutral")
        print(f"  -> alpha_1 > 0 SUPPORTED (opinion extremity drives activation)")
        alpha1_direction = "positive"
    else:
        ratio = extreme_interval / max(neutral_interval, 1)
        print(f"  Neutral agents post more frequently than extreme (ratio={ratio:.1f})")
        print(f"  -> alpha_1 effect not detectable at individual level")
        alpha1_direction = "undetectable"

    # Also: stance distribution of first-time vs repeat posters
    first_posts = df.groupby('agent_id').first()
    extreme_first_rate = (first_posts['extremity'] == 'extreme').mean()
    print(f"  Extreme stance fraction (first posts): {extreme_first_rate:.3f}")
    print(f"  Extreme stance fraction (all posts): {(df['extremity']=='extreme').mean():.3f}")

    return {
        "n_extreme_repeat": len(extreme_posters),
        "n_neutral_repeat": len(neutral_posters),
        "extreme_mean_interval": round(float(extreme_interval), 1),
        "neutral_mean_interval": round(float(neutral_interval), 1),
        "alpha1_direction": alpha1_direction,
    }


def estimate_5_shock_impact(daily_A, daily_shock):
    """Estimate r_1 (shock response) from A(t) before/after shock events.

    For each day with shock_intensity > 0, compare A(t+1) to A(t).
    """
    print("\n── Estimate 5: Shock Impact (r_1 constraint) ──")

    shock_days = np.where(daily_shock.values > 0)[0]
    if len(shock_days) == 0:
        print("  No shock days found")
        return {"n_shock_days": 0}

    A = daily_A.values.astype(float)

    # For each shock day, compute delta_A = A(t+1) - A(t)
    deltas = []
    for sd in shock_days:
        if sd + 1 < len(A):
            deltas.append(A[sd + 1] - A[sd])

    deltas = np.array(deltas)
    mean_delta = float(deltas.mean()) if len(deltas) > 0 else 0.0

    # Baseline delta: mean change on non-shock days
    non_shock = np.ones(len(A), dtype=bool)
    non_shock[shock_days] = False
    non_shock[-1] = False  # can't compute delta for last day
    non_shock_deltas = np.diff(A)[non_shock[:-1]]
    baseline_delta = float(non_shock_deltas.mean()) if len(non_shock_deltas) > 0 else 0.0

    shock_effect = mean_delta - baseline_delta

    print(f"  Shock days: {len(shock_days)}/{len(A)}")
    print(f"  Mean A(t+1)-A(t) after shock: {mean_delta:+.1f}")
    print(f"  Mean A(t+1)-A(t) on non-shock days: {baseline_delta:+.1f}")
    print(f"  Shock effect (excess): {shock_effect:+.1f} posts")

    if shock_effect > 0:
        print(f"  -> r_1 > 0 SUPPORTED (shocks increase posting activity)")
    elif shock_effect < 0:
        print(f"  -> Shock effect is negative (counter-intuitive, check data)")
    else:
        print(f"  -> No detectable shock effect")

    return {
        "n_shock_days": len(shock_days),
        "mean_delta_after_shock": round(mean_delta, 1),
        "baseline_delta": round(baseline_delta, 2),
        "shock_effect": round(float(shock_effect), 1),
        "r1_direction": "positive" if shock_effect > 1 else "weak_or_zero",
    }


def estimate_6_reff_bounds(estimates):
    """Compute R_eff bounds from calibrated parameters.

    R_eff ~ beta * k_eff * P_activate

    From the aggregate growth rate r:
    r = beta * k_eff * P_activate - gamma
    where gamma ~ sigma(gamma_0)

    If we have r from data and gamma from inter-post intervals:
    beta * k_eff * P_activate = r + gamma

    And R_eff = (r + gamma) / gamma = 1 + r/gamma
    """
    print("\n── Estimate 6: R_eff Empirical Bounds ──")

    r_growth = estimates["aggregate"]["r_growth"]
    gamma_est = estimates["decay"]

    if gamma_est.get("lambda_est") and r_growth:
        lambda_decay = gamma_est["lambda_est"]
        r_plus_gamma = max(r_growth + lambda_decay, 1e-6)
        R_eff_est = r_plus_gamma / max(lambda_decay, 1e-6)
        print(f"  r_growth = {r_growth:.6f}")
        print(f"  lambda_decay (est. P(A->D)) = {lambda_decay:.4f}")
        print(f"  R_eff = (r + lambda) / lambda = {R_eff_est:.3f}")

        if R_eff_est > 1:
            print(f"  R_eff > 1: Propagation would GROW (supercritical)")
        else:
            print(f"  R_eff < 1: Propagation would DECAY (subcritical)")
    else:
        R_eff_est = None
        print(f"  Cannot compute R_eff: missing growth rate or decay estimate")

    # Also estimate from peak shape
    peak_A = estimates["aggregate"]["peak_A"]
    mean_A = estimates["aggregate"]["mean_daily_A"]
    print(f"  Alternative: peak/mean ratio = {peak_A/max(mean_A, 1):.1f}")
    print(f"    (high ratio -> bursty dynamics, low ratio -> steady state)")

    return {
        "R_eff_est": round(float(R_eff_est), 3) if R_eff_est else None,
        "interpretation": (
            "supercritical (R_eff > 1)" if (R_eff_est and R_eff_est > 1)
            else "subcritical (R_eff < 1)" if R_eff_est
            else "cannot determine"
        ),
        "peak_to_mean_ratio": round(float(peak_A / max(mean_A, 1)), 1),
    }


def main():
    print("=" * 60)
    print("  EVENT-LEVEL CALIBRATION TO v0.1 DATA")
    print("=" * 60)

    df, daily_A, daily_shock = load_and_preprocess()
    print(f"Loaded: {len(df)} posts, {len(daily_A)} daily windows")

    estimates = {}

    # Estimate 1: Aggregate growth/decay rates
    estimates["aggregate"] = estimate_1_aggregate_growth_rate(daily_A)

    # Estimate 2: Decay rate from inter-post intervals
    estimates["decay"] = estimate_2_decay_rate(df)

    # Estimate 3: Reactivation rate
    estimates["reactivation"] = estimate_3_reactivation(df)

    # Estimate 4: Opinion effect
    estimates["opinion"] = estimate_4_opinion_effect(df)

    # Estimate 5: Shock impact
    estimates["shock"] = estimate_5_shock_impact(daily_A, daily_shock)

    # Estimate 6: R_eff bounds
    estimates["reff"] = estimate_6_reff_bounds(estimates)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  CALIBRATION SUMMARY")
    print(f"{'='*60}")

    summary_lines = []

    if estimates["aggregate"]["r_growth"]:
        summary_lines.append(
            f"  Aggregate growth rate r: {estimates['aggregate']['r_growth']:.6f}/day")

    if estimates["decay"].get("gamma_0_est") is not None:
        summary_lines.append(
            f"  Estimated gamma_0: {estimates['decay']['gamma_0_est']:.3f} "
            f"(default: 0.0, sigma(0)=0.5)")

    if estimates["reactivation"].get("r_0_est") is not None:
        summary_lines.append(
            f"  Estimated r_0: {estimates['reactivation']['r_0_est']:.3f} "
            f"(default: -4.0)")
        summary_lines.append(
            f"  Reactivation rate: {estimates['reactivation']['reactivation_rate']:.4f} "
            f"({estimates['reactivation']['reactivation_rate']*100:.1f}%)")

    summary_lines.append(
        f"  alpha_1 direction: {estimates['opinion']['alpha1_direction']}")

    summary_lines.append(
        f"  r_1 direction: {estimates['shock']['r1_direction']}")

    if estimates["reff"].get("R_eff_est"):
        summary_lines.append(
            f"  R_eff estimate: {estimates['reff']['R_eff_est']:.3f} "
            f"({estimates['reff']['interpretation']})")
    else:
        summary_lines.append(
            f"  R_eff: {estimates['reff']['interpretation']}")

    for line in summary_lines:
        print(line)

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(estimates, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT}")

    return estimates


if __name__ == "__main__":
    main()
