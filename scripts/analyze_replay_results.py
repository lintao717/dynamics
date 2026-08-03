"""
Analyze default-parameter replay results for CHECKED matched pairs.

Computes per-case, per-mode metrics using ONLY data steps (not tail),
compares against simple baselines, and generates CSV + Markdown report.

Metrics:
  - Observed / simulated peak magnitude and timing
  - Train / validation NRMSE (via calibration objective)
  - MAE
  - AUC ratio (simulated AUC / observed AUC)
  - Seed statistics (mean, std, CV, 5th/50th/95th percentile)
  - Runtime

Baselines:
  - Persistence: y_hat[t+1] = y[t]
  - Exponential decay: a * exp(-lambda * t)
  - Pulse-decay: b + a * exp(-lambda * (t - tp)) * I[t >= tp]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics_simulation.calibration.objective import (
    compute_replay_loss, LossWeights, _nrmse,
)
from dynamics_simulation.calibration.split import TemporalSplit

# ── Constants ──
REPLAY_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "replay"
OUT_DIR = REPLAY_DIR
STEP_HOURS = 24.0
SEEDS = (11, 23, 37, 53, 71)

PAIRS = [
    ("pair_01", "26f247cc05fd53e12daa91c98e7feab8", "cf5522f51f8be6859840b0b948bb813c", 642, 704),
    ("pair_02", "f983d8f70e265ec265acd9ea93e9b10d", "d293a73970751898b5d83f5525a15fcc", 124, 223),
    ("pair_03", "68665bf53973dea5036336fcc5dc9eea", "97bdff4a4098a06dc2b2597b514b2df3", 365, 434),
    ("pair_04", "7cfe610e01dafc496143664a1a8cf87d", "9a74296ad4c231241c1ac6057c08bbcb", 126, 252),
    ("pair_05", "afc5ba429bc79086f05d6798e4ed978a", "619f582e27e736ebc4c5def47f954535", 206, 344),
    ("pair_06", "25d9ed3994c2d5f030b864867facab47", "f030f59e7579dcda946ab8a3bd2733c6", 183, 259),
    ("pair_07", "b4a497151507853058c8c50b6c6670f5", "6f2b7e632c64ba5835164abe2f1ab28e", 338, 444),
    ("pair_08", "b9801872032a3bee629f6559bbf503ba", "3b7e48be19979f9df3a146e2d0277c58", 142, 237),
    ("pair_09", "dfb9f2af5bb9b16ba717b91d6be5fa2f", "100536e843cd4e307e1a2865b28b1f05", 136, 226),
    ("pair_10", "3dac58ea8cfef832bde25278a699fd45", "22c6a11c223c838fa933b5b4af777fcc", 151, 261),
]


def _fit_exp_decay(y: np.ndarray, t: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Fit exponential decay: a * exp(-lambda * t).  Returns (a, lambda, y_hat)."""
    y_pos = np.maximum(y, 1e-6)
    log_y = np.log(y_pos)
    # Linear regression on log space
    A = np.column_stack([np.ones_like(t), t])
    coeff, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    a = np.exp(coeff[0])
    lam = max(-coeff[1], 1e-6)  # ensure positive decay
    y_hat = a * np.exp(-lam * t)
    return a, lam, y_hat


def _fit_pulse_decay(y: np.ndarray, t: np.ndarray) -> tuple[float, float, float, int, np.ndarray]:
    """Fit pulse decay: b + a * exp(-lambda * (t-tp)) * I[t >= tp].
    tp = argmax(y). Returns (b, a, lambda, tp, y_hat)."""
    tp = int(np.argmax(y))
    b = np.mean(y[-max(1, len(y) // 4):])  # baseline = late-stage mean
    y_adj = np.maximum(y - b, 1e-6)
    log_y = np.log(y_adj)
    t_adj = np.maximum(t - tp, 0)
    mask = t >= tp
    if mask.sum() >= 2:
        A = np.column_stack([np.ones_like(t_adj[mask]), t_adj[mask]])
        coeff, _, _, _ = np.linalg.lstsq(A, log_y[mask], rcond=None)
        a = np.exp(coeff[0])
        lam = max(-coeff[1], 1e-6)
    else:
        a = y[tp] - b
        lam = 0.5
    y_hat = b + a * np.exp(-lam * np.maximum(t - tp, 0))
    y_hat = np.where(t >= tp, y_hat, b + a * np.exp(-lam * (tp - t)))  # symmetric pre-peak (optional)
    # Actually use step function: before peak, use rising; after peak, use decay
    y_hat = np.where(t < tp, y_hat, b + a * np.exp(-lam * (t - tp)))
    return b, a, lam, tp, y_hat


def compute_metrics(obs: np.ndarray, sim: np.ndarray,
                    last_data_step: int, split: TemporalSplit) -> dict:
    """Compute all evaluation metrics on data steps only."""
    # Trim to data steps only
    n_data = last_data_step + 1
    obs_d = obs[:n_data].astype(np.float64)
    sim_d = sim[:n_data].astype(np.float64)
    t = np.arange(n_data, dtype=np.float64)

    # Peak metrics
    obs_peak = float(np.max(obs_d))
    sim_peak = float(np.max(sim_d))
    obs_peak_step = int(np.argmax(obs_d))
    sim_peak_step = int(np.argmax(sim_d))

    # NRMSE via standard loss (only active_count)
    weights = LossWeights(active_count=1.0)
    masks = {"active_count": np.ones(n_data, dtype=bool)}
    loss = compute_replay_loss(
        {"active_count": obs_d},
        {"active_count": sim_d},
        split, weights, masks,
    )

    # MAE
    mae = float(np.mean(np.abs(obs_d - sim_d)))

    # AUC ratio
    obs_auc = float(np.trapezoid(obs_d, t))
    sim_auc = float(np.trapezoid(sim_d, t))

    # Baselines
    _, _, exp_fit = _fit_exp_decay(obs_d, t)
    _, _, _, _, pulse_fit = _fit_pulse_decay(obs_d, t)

    # Persistence baseline loss
    if n_data >= 2:
        persistence_pred = np.zeros_like(obs_d)
        persistence_pred[0] = obs_d[0]
        persistence_pred[1:] = obs_d[:-1]
    else:
        persistence_pred = obs_d.copy()

    # Baseline losses
    base_persist = _nrmse(obs_d, persistence_pred, np.ones(n_data, dtype=bool))
    base_exp = _nrmse(obs_d, exp_fit, np.ones(n_data, dtype=bool))
    base_pulse = _nrmse(obs_d, pulse_fit, np.ones(n_data, dtype=bool))

    return {
        "n_data_steps": n_data,
        "obs_peak": obs_peak,
        "sim_peak": sim_peak,
        "peak_ratio": sim_peak / max(obs_peak, 1),
        "obs_peak_step": obs_peak_step,
        "sim_peak_step": sim_peak_step,
        "peak_step_error": sim_peak_step - obs_peak_step,
        "train_nrmse": loss.train_total,
        "val_nrmse": loss.val_total,
        "mae": mae,
        "obs_auc": obs_auc,
        "sim_auc": sim_auc,
        "auc_ratio": sim_auc / max(obs_auc, 1e-6),
        "baseline_persistence_nrmse": base_persist,
        "baseline_exp_decay_nrmse": base_exp,
        "baseline_pulse_decay_nrmse": base_pulse,
    }


def main():
    rows = []
    md_lines = [
        "# CHECKED Default-Parameter Replay — Evaluation Report",
        "",
        f"**Step**: {STEP_HOURS}h  |  **Seeds**: {SEEDS}  |  **Baselines**: persistence, exp-decay, pulse-decay",
        "",
        "> ⚠️ Observation model: `observed_actor_count_as_proxy_for_latent_A` (direct_state_observation=false)",
        "",
        "## Per-Case Metrics",
        "",
    ]

    for pid, fake_id, real_id, fake_n, real_n in PAIRS:
        for mode in ["broadcast", "cumulative"]:
            for lk, cid, n_users in [("fake", fake_id, fake_n), ("real", real_id, real_n)]:
                fpath = REPLAY_DIR / f"{pid}_{mode}_{lk}_{cid[:12]}.json"
                if not fpath.exists():
                    print(f"  MISSING: {fpath}")
                    continue

                data = json.loads(fpath.read_text(encoding="utf-8"))
                obs = data["observed"]
                sm = data.get("simulated_mean", {})
                ss = data.get("simulated_std", {})
                last_data = data.get("last_data_step", 0)
                elapsed = data.get("elapsed_seconds", 0)

                obs_active = np.array(obs.get("active_count", []), dtype=np.float64)
                sm_active = np.array(sm.get("active_count", []), dtype=np.float64)
                ss_active = np.array(ss.get("active_count", []), dtype=np.float64)

                # Per-seed peak CV
                per_seed_peaks = []
                for run in data.get("per_seed", []):
                    arr = np.array(run.get("active_count", []), dtype=np.float64)
                    per_seed_peaks.append(float(np.max(arr)))
                seed_peak_cv = float(np.std(per_seed_peaks) / max(np.mean(per_seed_peaks), 1e-6))

                # Split (70/30 on data steps only)
                split = TemporalSplit.by_fraction(
                    total_steps=last_data, train_fraction=0.7,
                )

                m = compute_metrics(obs_active, sm_active, last_data, split)

                row = {
                    "pair_id": pid,
                    "label": lk,
                    "mode": mode,
                    "case_id": cid,
                    "n_users": n_users,
                    "n_interactions": data["interaction_count"],
                    "elapsed_s": round(elapsed, 2),
                    **{k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()},
                    "seed_peak_cv": round(seed_peak_cv, 4),
                    "sim_mean_peak": round(float(np.max(sm_active)), 1),
                    "sim_std_peak": round(float(ss_active[int(np.argmax(sm_active))]) if len(ss_active) > 0 else 0, 1),
                }
                rows.append(row)

    # ── CSV ──
    csv_path = OUT_DIR / "replay_metrics.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"CSV: {csv_path} ({len(rows)} rows)")

    # ── Summary by mode ──
    for mode in ["broadcast", "cumulative"]:
        mode_rows = [r for r in rows if r["mode"] == mode]
        if not mode_rows:
            continue

        peak_r = [r["peak_ratio"] for r in mode_rows]
        auc_r = [r["auc_ratio"] for r in mode_rows]
        train_n = [r["train_nrmse"] for r in mode_rows]
        val_n = [r["val_nrmse"] for r in mode_rows]

        md_lines.append(f"### {mode}")
        md_lines.append("")
        md_lines.append("| Metric | Mean | Median | Min | Max |")
        md_lines.append("|--------|------|--------|-----|-----|")
        md_lines.append(f"| Peak ratio | {np.mean(peak_r):.4f} | {np.median(peak_r):.4f} | {np.min(peak_r):.4f} | {np.max(peak_r):.4f} |")
        md_lines.append(f"| AUC ratio | {np.mean(auc_r):.4f} | {np.median(auc_r):.4f} | {np.min(auc_r):.4f} | {np.max(auc_r):.4f} |")
        md_lines.append(f"| Train NRMSE | {np.mean(train_n):.4f} | {np.median(train_n):.4f} | {np.min(train_n):.4f} | {np.max(train_n):.4f} |")
        md_lines.append(f"| Val NRMSE | {np.mean(val_n):.4f} | {np.median(val_n):.4f} | {np.min(val_n):.4f} | {np.max(val_n):.4f} |")
        md_lines.append("")

        # Best baseline comparison
        for lk in ["fake", "real"]:
            lk_rows = [r for r in mode_rows if r["label"] == lk]
            if not lk_rows:
                continue
            best_base = [min(r["baseline_persistence_nrmse"],
                           r["baseline_exp_decay_nrmse"],
                           r["baseline_pulse_decay_nrmse"]) for r in lk_rows]
            model_train = [r["train_nrmse"] for r in lk_rows]
            wins = sum(1 for m, b in zip(model_train, best_base) if m < b)
            md_lines.append(f"- **{lk}**: default model beats best baseline in {wins}/{len(lk_rows)} cases")
        md_lines.append("")

    # ── Per-pair detail table ──
    md_lines.append("## Per-Case Detail")
    md_lines.append("")
    md_lines.append("| Pair | Mode | Label | N | ObsPeak | SimPeak | PeakRatio | TrainNRMSE | ValNRMSE | BestBase | BaseWin? |")
    md_lines.append("|------|------|-------|---|---------|---------|-----------|------------|----------|----------|----------|")
    for r in rows:
        best_base = min(r["baseline_persistence_nrmse"], r["baseline_exp_decay_nrmse"], r["baseline_pulse_decay_nrmse"])
        win = "yes" if r["train_nrmse"] < best_base else "no"
        md_lines.append(
            f"| {r['pair_id']} | {r['mode']} | {r['label']} | {r['n_users']} | "
            f"{r['obs_peak']:.0f} | {r['sim_mean_peak']:.1f} | {r['peak_ratio']:.4f} | "
            f"{r['train_nrmse']:.4f} | {r['val_nrmse']:.4f} | {best_base:.4f} | {win} |"
        )

    # ── Observation model note ──
    md_lines.append("")
    md_lines.append("## Observation Model")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps({
        "observation_model": "observed_actor_count_as_proxy_for_latent_A",
        "direct_state_observation": False,
        "note": "Real CHECKED active_count = users with >=1 action in window. "
                "Simulated active_count = agents in A-state. These are NOT equivalent.",
    }, indent=2))
    md_lines.append("```")

    # ── Write report ──
    report_path = OUT_DIR / "default_replay_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Report: {report_path}")

    # ── Write summary JSON ──
    summary = {
        "observation_model": "observed_actor_count_as_proxy_for_latent_A",
        "direct_state_observation": False,
        "n_cases": len(rows),
        "modes": ["broadcast", "cumulative"],
        "metrics_computed": list(rows[0].keys()) if rows else [],
        "baselines": ["persistence", "exponential_decay", "pulse_decay"],
    }
    summary_path = OUT_DIR / "replay_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
