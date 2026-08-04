# V1.3 Ablation — VALIDITY NOTICE

**Status**: INVALIDATED (config wiring bug)

The V1.3 ablation experiment (`run_v13_ablation.py`) did not correctly
pass the `micro_steps` and `broadcast_exposure_config` parameters to
the replay pipeline. As a result:

- Config A (micro_steps=1) and Config B (micro_steps=4) ran IDENTICAL
  configurations — both used the default `ReplayConfig.micro_steps=4`
  from the replay runner.

- Config C and Config D also ran IDENTICAL configurations — the
  `root_shock` flag was set on a `BroadcastExposureConfig` object
  that was never passed to `run_replay()`.

**Impact**: The conclusions "micro-steps don't help" and "root shock
doesn't help" are NOT supported by this experiment. The actual
behavior of micro-steps=1 vs micro-steps=4 was never compared.

**Superseded by**: `v15_ablation.json` (V1.5 corrected ablation)

**Date marked**: 2026-08-04
