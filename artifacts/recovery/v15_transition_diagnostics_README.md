# V1.5 Transition Diagnostics — INVALIDATED

**Date**: 2026-08-04
**Reason**: `n_U_ts` field does not exist in `ReplayRun` or `simulated_mean`.
  The code used `result.simulated_mean.get("n_U_ts", [0])` which silently
  returned `[0]`, making U_outflow=0 an artifact, not a real finding.

**Impact**: The conclusion "default beta_M too low, U->E never happens"
  is NOT supported by this experiment.

**Fix**: V1.5.1 adds explicit per-step transition flow time series
  (U_to_E_ts, E_to_A_ts, etc.) to MetricsCollector, SimulationMetrics,
  ReplayRun, and the replay aggregation.
