# V1.7 Rolling Forecast — INVALIDATED

**Date**: 2026-08-05
**Superseded by**: V1.7R sentinel experiment (`v17r_sentinel_forecast.json`)

## Reasons for Invalidation

1. `run_replay()` received a history-only case with `tail_steps=0`. The simulator stopped at the historical cutoff.
2. `_forecast()` padded missing future trajectory values with the final historical value (`np.full(..., trajectory[-1])`).
3. Reported 24h, 48h, and 72h forecasts were frequently identical constants, not genuine forward simulations.
4. The artifact was labelled `cohort_conditioned=true`, but the simulation contained only users observed by the cutoff — not the full cohort.
5. Real `active_count[t]` (behavioral flow) was compared with simulated `n_A_ts[t]` (state stock) without an emission layer.
6. Event-level fitting estimated three free parameters from only a few pre-cutoff observations.

## Impact

The conclusion "4/36 wins, model cannot forecast" is NOT supported by this artifact.
The question of whether ONE_SHOT can forecast CHECKED events remains OPEN.
