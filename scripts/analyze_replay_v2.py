"""
V2 replay analysis — corrected evaluation protocol.

Fixes vs V1:
  - Baselines fitted on TRAINING data only (open-loop comparison)
  - Rolling persistence reported separately (not as a peer of free-running models)
  - Multiple metrics: NRMSE, train-peak-NRMSE, RMSLE, MASE
  - Validation NRMSE normalised by TRAINING peak range (not tail range)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics_simulation.calibration.split import TemporalSplit

REPLAY_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "replay"
OUT_DIR = REPLAY_DIR

PAIRS = [
    ("pair_01", "26f247cc05fd53e12daa91c98e7feab8", "cf5522f51f8be6859840b0b948bb813c"),
    ("pair_02", "f983d8f70e265ec265acd9ea93e9b10d", "d293a73970751898b5d83f5525a15fcc"),
    ("pair_03", "68665bf53973dea5036336fcc5dc9eea", "97bdff4a4098a06dc2b2597b514b2df3"),
    ("pair_04", "7cfe610e01dafc496143664a1a8cf87d", "9a74296ad4c231241c1ac6057c08bbcb"),
    ("pair_05", "afc5ba429bc79086f05d6798e4ed978a", "619f582e27e736ebc4c5def47f954535"),
    ("pair_06", "25d9ed3994c2d5f030b864867facab47", "f030f59e7579dcda946ab8a3bd2733c6"),
    ("pair_07", "b4a497151507853058c8c50b6c6670f5", "6f2b7e632c64ba5835164abe2f1ab28e"),
    ("pair_08", "b9801872032a3bee629f6559bbf503ba", "3b7e48be19979f9df3a146e2d0277c58"),
    ("pair_09", "dfb9f2af5bb9b16ba717b91d6be5fa2f", "100536e843cd4e307e1a2865b28b1f05"),
    ("pair_10", "3dac58ea8cfef832bde25278a699fd45", "22c6a11c223c838fa933b5b4af777fcc"),
]


def _nrmse(y_true, y_pred, denom_scale=1.0):
    """NRMSE with explicit denominator.  Returns (value, denominator_used)."""
    se = (y_true - y_pred) ** 2
    rmse = np.sqrt(np.mean(se))
    denom = max(denom_scale, np.max(y_true) - np.min(y_true), 1.0)
    return rmse / denom, denom


def _rmsle(y_true, y_pred):
    """Root mean squared log error for non-negative counts."""
    yt = np.maximum(y_true, 0)
    yp = np.maximum(y_pred, 0)
    return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))


def _mase(y_train, y_val, y_pred_val):
    """Mean absolute scaled error.  Scale = MAE of naive forecast on training."""
    if len(y_train) < 2:
        return float(np.mean(np.abs(y_val - y_pred_val)) / max(np.mean(np.abs(y_train)), 1e-6))
    naive_err = np.mean(np.abs(np.diff(y_train)))
    scale = max(naive_err, 1e-6)
    return float(np.mean(np.abs(y_val - y_pred_val)) / scale)


# ── Open-loop baselines (fitted on TRAINING only) ──

def _baseline_persistence_open(y_train, n_val):
    """Open-loop persistence: predict last training value for all validation steps."""
    last = y_train[-1] if len(y_train) > 0 else 0.0
    return np.full(n_val, last, dtype=np.float64)


def _baseline_exp_decay_train(y_train, t_train, t_val):
    """Exponential decay fitted on TRAINING only, forecast into validation."""
    y_pos = np.maximum(y_train, 1e-6)
    log_y = np.log(y_pos)
    A = np.column_stack([np.ones_like(t_train), t_train])
    coeff, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    a = np.exp(coeff[0])
    lam = max(-coeff[1], 1e-6)
    return a * np.exp(-lam * t_val)


def _baseline_pulse_decay_train(y_train, t_train, t_val):
    """Pulse-decay fitted on TRAINING only."""
    tp = int(np.argmax(y_train))
    b = np.mean(y_train[-max(1, len(y_train) // 4):])
    y_adj = np.maximum(y_train - b, 1e-6)
    log_y = np.log(y_adj)
    t_adj = np.maximum(t_train - tp, 0)
    mask = t_train >= tp
    if mask.sum() >= 2:
        A = np.column_stack([np.ones_like(t_adj[mask]), t_adj[mask]])
        coeff, _, _, _ = np.linalg.lstsq(A, log_y[mask], rcond=None)
        a = np.exp(coeff[0])
        lam = max(-coeff[1], 1e-6)
    else:
        a = max(y_train) - b
        lam = 0.5
    return b + a * np.exp(-lam * np.maximum(t_val - tp, 0))


# ── Rolling persistence (for reference only, NOT a peer of free-running models) ──

def _baseline_rolling_persistence(y_all):
    """Rolling one-step persistence: y_hat[t] = y[t-1].  Uses all data."""
    pred = np.zeros_like(y_all)
    pred[0] = y_all[0]
    pred[1:] = y_all[:-1]
    return pred


def compute_metrics_v2(obs, sim, last_data_step, split):
    """Compute V2 metrics with corrected baselines and multiple loss types."""
    n_data = last_data_step + 1
    obs_d = obs[:n_data].astype(np.float64)
    sim_d = sim[:n_data].astype(np.float64)
    train_end = split.train_end_step
    T = split.total_steps

    t_full = np.arange(n_data, dtype=np.float64)
    t_train = t_full[:train_end + 1]
    t_val = t_full[train_end + 1:]

    obs_train = obs_d[:train_end + 1]
    obs_val = obs_d[train_end + 1:]
    sim_train = sim_d[:train_end + 1]
    sim_val = sim_d[train_end + 1:]

    train_peak = float(np.max(obs_train))

    # ── Standard NRMSE (per-segment) ──
    train_nrmse, train_denom = _nrmse(obs_train, sim_train)
    val_nrmse, val_denom = _nrmse(obs_val, sim_val)

    # ── Train-peak-normalised NRMSE ──
    val_peak_nrmse, _ = _nrmse(obs_val, sim_val, denom_scale=max(train_peak, 1.0))

    # ── RMSLE ──
    train_rmsle = _rmsle(obs_train, sim_train)
    val_rmsle = _rmsle(obs_val, sim_val)

    # ── MASE ──
    val_mase = _mase(obs_train, obs_val, sim_val)

    # ── Peak metrics ──
    obs_peak = float(np.max(obs_d))
    sim_peak = float(np.max(sim_d))
    obs_peak_step = int(np.argmax(obs_d))
    sim_peak_step = int(np.argmax(sim_d))

    # ── Open-loop baselines (trained on TRAIN only) ──
    bl_persist = _baseline_persistence_open(obs_train, len(obs_val))
    bl_exp = _baseline_exp_decay_train(obs_train, t_train, t_val)
    bl_pulse = _baseline_pulse_decay_train(obs_train, t_train, t_val)

    bl_persist_nrmse, _ = _nrmse(obs_val, bl_persist)
    bl_exp_nrmse, _ = _nrmse(obs_val, bl_exp)
    bl_pulse_nrmse, _ = _nrmse(obs_val, bl_pulse)

    bl_persist_rmsle = _rmsle(obs_val, bl_persist)
    bl_exp_rmsle = _rmsle(obs_val, bl_exp)
    bl_pulse_rmsle = _rmsle(obs_val, bl_pulse)

    # ── Rolling persistence (reference only) ──
    roll_pred = _baseline_rolling_persistence(obs_d)
    roll_train_nrmse, _ = _nrmse(obs_train, roll_pred[:train_end + 1])
    roll_val_nrmse, _ = _nrmse(obs_val, roll_pred[train_end + 1:])

    # ── AUC ──
    obs_auc = float(np.trapezoid(obs_d, t_full))
    sim_auc = float(np.trapezoid(sim_d, t_full))

    return {
        "n_data": n_data,
        "train_end": train_end,
        "obs_peak": obs_peak,
        "sim_peak": sim_peak,
        "peak_ratio": sim_peak / max(obs_peak, 1),
        "peak_step_err": sim_peak_step - obs_peak_step,
        "train_nrmse": round(train_nrmse, 6),
        "val_nrmse": round(val_nrmse, 6),
        "val_peak_nrmse": round(val_peak_nrmse, 6),
        "train_rmsle": round(train_rmsle, 6),
        "val_rmsle": round(val_rmsle, 6),
        "val_mase": round(val_mase, 4),
        "train_peak": round(train_peak, 1),
        "val_denom": round(val_denom, 1),
        "auc_ratio": round(sim_auc / max(obs_auc, 1e-6), 6),
        # Open-loop baselines
        "bl_persist_nrmse": round(bl_persist_nrmse, 6),
        "bl_exp_nrmse": round(bl_exp_nrmse, 6),
        "bl_pulse_nrmse": round(bl_pulse_nrmse, 6),
        "bl_persist_rmsle": round(bl_persist_rmsle, 6),
        "bl_exp_rmsle": round(bl_exp_rmsle, 6),
        "bl_pulse_rmsle": round(bl_pulse_rmsle, 6),
        # Rolling reference (NOT a peer)
        "rolling_val_nrmse": round(roll_val_nrmse, 6),
        "rolling_train_nrmse": round(roll_train_nrmse, 6),
    }


def main():
    rows = []
    for pid, fake_id, real_id in PAIRS:
        for mode in ["broadcast", "cumulative"]:
            for lk, cid in [("fake", fake_id), ("real", real_id)]:
                fpath = REPLAY_DIR / f"{pid}_{mode}_{lk}_{cid[:12]}.json"
                if not fpath.exists():
                    continue
                data = json.loads(fpath.read_text(encoding="utf-8"))
                obs_arr = np.array(data["observed"].get("active_count", []), dtype=np.float64)
                sm = data.get("simulated_mean", {})
                if not sm:
                    continue
                sim_arr = np.array(sm.get("active_count", []), dtype=np.float64)
                last_data = data.get("last_data_step", 0)

                split = TemporalSplit.by_fraction(total_steps=last_data, train_fraction=0.7)
                m = compute_metrics_v2(obs_arr, sim_arr, last_data, split)

                row = {
                    "pair": pid, "mode": mode, "label": lk,
                    "users": data["node_count"],
                    "interactions": data["interaction_count"],
                    "elapsed_s": data.get("elapsed_seconds", 0),
                    **m,
                }
                rows.append(row)

    # CSV
    csv_path = OUT_DIR / "replay_metrics_v2.csv"
    keys = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"CSV: {csv_path} ({len(rows)} rows)")

    # Summary
    for mode in ["broadcast", "cumulative"]:
        mr = [r for r in rows if r["mode"] == mode]
        if not mr:
            continue
        print(f"\n=== {mode} ({len(mr)} cases) ===")
        for metric in ["val_peak_nrmse", "val_rmsle", "val_mase"]:
            vals = [r[metric] for r in mr]
            print(f"  {metric}: mean={np.mean(vals):.4f} median={np.median(vals):.4f}")

        # Model vs best OPEN-LOOP baseline
        for lk in ["fake", "real"]:
            lr = [r for r in mr if r["label"] == lk]
            best_base_rmsle = [min(r["bl_persist_rmsle"], r["bl_exp_rmsle"], r["bl_pulse_rmsle"]) for r in lr]
            model_rmsle = [r["val_rmsle"] for r in lr]
            wins = sum(1 for m, b in zip(model_rmsle, best_base_rmsle) if m < b)
            print(f"  {lk}: model RMSLE < best open-loop baseline: {wins}/{len(lr)}")

        # Rolling reference
        roll_vals = [r["rolling_val_nrmse"] for r in mr]
        model_vals = [r["val_peak_nrmse"] for r in mr]
        print(f"  rolling persistence val NRMSE: mean={np.mean(roll_vals):.4f}")
        print(f"  model val-peak NRMSE:         mean={np.mean(model_vals):.4f}")

    # Markdown report summary
    md = [
        "# V2 Replay Evaluation (Corrected Protocol)",
        "",
        "## Key changes from V1",
        "- Baselines fitted on TRAINING data only (open-loop comparison)",
        "- Rolling persistence reported separately (NOT a peer of free-running models)",
        "- Multiple metrics: train-peak-NRMSE, RMSLE, MASE",
        "- Validation NRMSE normalised by training peak range",
        "",
    ]
    for mode in ["broadcast", "cumulative"]:
        mr = [r for r in rows if r["mode"] == mode]
        md.append(f"## {mode}")
        md.append("")
        md.append("| Metric | Mean | Median | Min | Max |")
        md.append("|--------|------|--------|-----|-----|")
        for metric in ["val_peak_nrmse", "val_rmsle", "val_mase", "bl_exp_rmsle"]:
            vals = [r[metric] for r in mr]
            md.append(f"| {metric} | {np.mean(vals):.4f} | {np.median(vals):.4f} | {np.min(vals):.4f} | {np.max(vals):.4f} |")
        md.append("")

    with open(OUT_DIR / "replay_evaluation_v2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nReport: {OUT_DIR / 'replay_evaluation_v2.md'}")


if __name__ == "__main__":
    main()
