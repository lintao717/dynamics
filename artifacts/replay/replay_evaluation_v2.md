# V2 Replay Evaluation (Corrected Protocol)

## Key changes from V1
- Baselines fitted on TRAINING data only (open-loop comparison)
- Rolling persistence reported separately (NOT a peer of free-running models)
- Multiple metrics: train-peak-NRMSE, RMSLE, MASE
- Validation NRMSE normalised by training peak range

## broadcast

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| val_peak_nrmse | 0.0328 | 0.0300 | 0.0131 | 0.0807 |
| val_rmsle | 1.6969 | 1.7366 | 0.3720 | 2.7817 |
| val_mase | 0.0865 | 0.0716 | 0.0202 | 0.1847 |
| bl_exp_rmsle | 0.6045 | 0.5228 | 0.1529 | 2.0939 |

## cumulative

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| val_peak_nrmse | 0.0579 | 0.0575 | 0.0321 | 0.1060 |
| val_rmsle | 2.1456 | 2.2828 | 0.7725 | 3.4594 |
| val_mase | 0.1432 | 0.1366 | 0.0716 | 0.2752 |
| bl_exp_rmsle | 0.6045 | 0.5228 | 0.1529 | 2.0939 |

