# CHECKED Default-Parameter Replay — Evaluation Report

**Step**: 24.0h  |  **Seeds**: (11, 23, 37, 53, 71)  |  **Baselines**: persistence, exp-decay, pulse-decay

> ⚠️ Observation model: `observed_actor_count_as_proxy_for_latent_A` (direct_state_observation=false)

## Per-Case Metrics

### broadcast

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| Peak ratio | 0.0459 | 0.0403 | 0.0346 | 0.0890 |
| AUC ratio | 0.2025 | 0.1704 | 0.0951 | 0.4885 |
| Train NRMSE | 0.4285 | 0.4408 | 0.2565 | 0.5952 |
| Val NRMSE | 5.3781 | 5.5261 | 0.5444 | 12.3016 |

- **fake**: default model beats best baseline in 0/10 cases
- **real**: default model beats best baseline in 0/10 cases

### cumulative

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| Peak ratio | 0.0899 | 0.0883 | 0.0634 | 0.1466 |
| AUC ratio | 0.3971 | 0.3696 | 0.1846 | 0.8782 |
| Train NRMSE | 0.4281 | 0.4443 | 0.2601 | 0.5741 |
| Val NRMSE | 9.7143 | 8.6501 | 0.7148 | 25.2255 |

- **fake**: default model beats best baseline in 0/10 cases
- **real**: default model beats best baseline in 0/10 cases

## Per-Case Detail

| Pair | Mode | Label | N | ObsPeak | SimPeak | PeakRatio | TrainNRMSE | ValNRMSE | BestBase | BaseWin? |
|------|------|-------|---|---------|---------|-----------|------------|----------|----------|----------|
| pair_01 | broadcast | fake | 642 | 236 | 21.0 | 0.0890 | 0.5952 | 0.5444 | 0.3822 | no |
| pair_01 | broadcast | real | 704 | 681 | 27.0 | 0.0385 | 0.4452 | 12.3016 | 0.3692 | no |
| pair_01 | cumulative | fake | 642 | 236 | 34.6 | 0.1466 | 0.5741 | 0.7148 | 0.3822 | no |
| pair_01 | cumulative | real | 704 | 681 | 56.2 | 0.0825 | 0.4473 | 25.2255 | 0.3692 | no |
| pair_02 | broadcast | fake | 124 | 77 | 4.2 | 0.0545 | 0.5224 | 1.5297 | 0.3809 | no |
| pair_02 | broadcast | real | 223 | 184 | 8.2 | 0.0380 | 0.5133 | 2.4042 | 0.3603 | no |
| pair_02 | cumulative | fake | 124 | 77 | 6.6 | 0.0831 | 0.5157 | 3.9319 | 0.3809 | no |
| pair_02 | cumulative | real | 223 | 184 | 19.0 | 0.1033 | 0.5116 | 12.0033 | 0.3603 | no |
| pair_03 | broadcast | fake | 365 | 188 | 11.6 | 0.0617 | 0.3843 | 4.2308 | 0.2915 | no |
| pair_03 | broadcast | real | 434 | 417 | 15.8 | 0.0379 | 0.2565 | 9.3052 | 0.2182 | no |
| pair_03 | cumulative | fake | 365 | 188 | 21.0 | 0.1117 | 0.3792 | 6.5441 | 0.2915 | no |
| pair_03 | cumulative | real | 434 | 417 | 33.0 | 0.0791 | 0.2601 | 13.3970 | 0.2182 | no |
| pair_04 | broadcast | fake | 126 | 119 | 4.8 | 0.0403 | 0.4408 | 1.0770 | 0.3688 | no |
| pair_04 | broadcast | real | 252 | 229 | 10.0 | 0.0402 | 0.4975 | 7.0214 | 0.3285 | no |
| pair_04 | cumulative | fake | 126 | 119 | 13.8 | 0.1160 | 0.4456 | 1.9258 | 0.3688 | no |
| pair_04 | cumulative | real | 252 | 229 | 21.8 | 0.0952 | 0.4964 | 17.8084 | 0.3285 | no |
| pair_05 | broadcast | fake | 206 | 147 | 8.4 | 0.0571 | 0.3692 | 5.5726 | 0.3221 | no |
| pair_05 | broadcast | real | 344 | 321 | 13.2 | 0.0411 | 0.3151 | 8.7510 | 0.2674 | no |
| pair_05 | cumulative | fake | 206 | 147 | 14.8 | 0.1007 | 0.3680 | 7.3258 | 0.3221 | no |
| pair_05 | cumulative | real | 344 | 321 | 29.8 | 0.0928 | 0.3184 | 12.5515 | 0.2674 | no |
| pair_06 | broadcast | fake | 183 | 173 | 6.2 | 0.0358 | 0.4449 | 5.4797 | 0.2816 | no |
| pair_06 | broadcast | real | 259 | 243 | 9.0 | 0.0370 | 0.4028 | 7.1433 | 0.2910 | no |
| pair_06 | cumulative | fake | 183 | 173 | 15.6 | 0.0902 | 0.4478 | 11.8625 | 0.2816 | no |
| pair_06 | cumulative | real | 259 | 243 | 21.8 | 0.0897 | 0.4057 | 14.1534 | 0.2910 | no |
| pair_07 | broadcast | fake | 338 | 265 | 16.0 | 0.0604 | 0.4165 | 11.3284 | 0.3438 | no |
| pair_07 | broadcast | real | 444 | 381 | 16.0 | 0.0420 | 0.3782 | 6.5406 | 0.1059 | no |
| pair_07 | cumulative | fake | 338 | 265 | 23.0 | 0.0868 | 0.4165 | 15.0395 | 0.3438 | no |
| pair_07 | cumulative | real | 444 | 381 | 27.4 | 0.0719 | 0.3773 | 8.0035 | 0.1059 | no |
| pair_08 | broadcast | fake | 142 | 133 | 4.8 | 0.0346 | 0.4437 | 3.4718 | 0.3500 | no |
| pair_08 | broadcast | real | 237 | 222 | 10.4 | 0.0468 | 0.3515 | 6.7961 | 0.2989 | no |
| pair_08 | cumulative | fake | 142 | 133 | 10.2 | 0.0767 | 0.4455 | 7.8154 | 0.3500 | no |
| pair_08 | cumulative | real | 237 | 222 | 16.2 | 0.0730 | 0.3524 | 9.2966 | 0.2989 | no |
| pair_09 | broadcast | fake | 136 | 131 | 4.6 | 0.0351 | 0.4407 | 1.5460 | 0.3507 | no |
| pair_09 | broadcast | real | 226 | 219 | 8.6 | 0.0393 | 0.4058 | 6.2971 | 0.3304 | no |
| pair_09 | cumulative | fake | 136 | 131 | 9.6 | 0.0733 | 0.4430 | 3.8423 | 0.3507 | no |
| pair_09 | cumulative | real | 226 | 219 | 20.6 | 0.0941 | 0.4102 | 13.7845 | 0.3304 | no |
| pair_10 | broadcast | fake | 151 | 145 | 7.4 | 0.0510 | 0.4993 | 3.0438 | 0.4019 | no |
| pair_10 | broadcast | real | 261 | 236 | 8.8 | 0.0373 | 0.4478 | 3.1765 | 0.3272 | no |
| pair_10 | cumulative | fake | 151 | 145 | 9.2 | 0.0634 | 0.5000 | 3.6366 | 0.4019 | no |
| pair_10 | cumulative | real | 261 | 236 | 15.8 | 0.0669 | 0.4470 | 5.4237 | 0.3272 | no |

## Observation Model

```json
{
  "observation_model": "observed_actor_count_as_proxy_for_latent_A",
  "direct_state_observation": false,
  "note": "Real CHECKED active_count = users with >=1 action in window. Simulated active_count = agents in A-state. These are NOT equivalent."
}
```
