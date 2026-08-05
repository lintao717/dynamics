# V1.7R Forecast Integrity Revalidation — Design Specification

## 1. Purpose

V1.7R repairs the scientific and software validity of the current rolling-forecast experiment before any conclusion is drawn about the `ONE_SHOT` propagation structure.

The existing artifact `artifacts/forecast/v17_rolling_forecast.json` must not be used as evidence that `ONE_SHOT` cannot forecast CHECKED events because the current implementation does not simulate the requested future horizon. It runs only through the historical cutoff and pads missing future values with the last simulated historical value.

V1.7R therefore has one goal:

> Produce a genuine forward simulation from a correctly assimilated cutoff state, score behavioral flow observables against hidden future observations, and distinguish temporal-dynamics failure from future-population-entry failure.

V1.7R is a revalidation stage, not an optimization stage. It must not introduce LLMs, opinion-dynamics calibration, synthetic shocks, or new propagation mechanisms.

## 2. Invalidated V1.7 Result

The current result `4/36 wins` is invalidated for forecasting conclusions because:

1. `run_replay()` receives a history-only case with `tail_steps=0`.
2. The simulator therefore stops at the historical cutoff.
3. `_forecast()` pads the missing future trajectory with the final historical value.
4. The reported 24 h, 48 h, and 72 h forecasts are frequently identical constants.
5. The artifact is labelled `cohort_conditioned=true`, but the simulation actually contains only users observed by the cutoff.
6. Real `active_count[t]` is compared with simulated `n_A_ts[t]`, even though one is a behavioral flow and the other is an end-of-step state stock.
7. Event-level fitting still estimates three free parameters from only a few pre-cutoff observations.

The original JSON artifact remains in the repository for auditability. A companion invalidation document must state that the result is superseded by V1.7R.

## 3. Scope

### 3.1 In scope

- Mark the current V1.7 artifact as invalidated.
- Correct reactivation enum string behavior and mode validation.
- Correct cutoff-time and future-user accounting.
- Add a dedicated forecast runner with explicit future horizon.
- Remove all last-value trajectory padding.
- Build a forecast state directly at the cutoff from observed history.
- Add simulated behavioral observables matching real first/repeat/active counts.
- Compare two diagnostic population modes:
  - `oracle_cohort`
  - `observed_closed`
- Run four sentinel cases before any full experiment.
- Define the decision gate for retaining `ONE_SHOT` or moving to an event-intensity model.

### 3.2 Out of scope

- Anonymous open-population forecasting.
- Final five-fold cross-validation.
- Untouched holdout testing.
- Event-gated reactivation.
- LLM agent policies.
- Opinion-dynamics fitting.
- Tibet-specific data ingestion.
- Large-scale parameter-search expansion.

## 4. Required Corrections

### 4.1 Reactivation mode correctness

`ReactivationMode.__str__()` must return only `self.value`.

Unknown modes must raise `ValueError`; they must not silently enter the `FULL` branch.

Required structural semantics:

- `FULL`: D0->A and D1->A allowed.
- `NO_DELAYED_FIRST`: D0->A disabled; D1->A allowed.
- `NO_TRUE_REACTIVATION`: D0->A allowed; D1->A disabled.
- `ONE_SHOT`: D0->A and D1->A disabled.
- `EVENT_GATED`: D0->A disabled; D1->A allowed only when a positive external shock is present.

### 4.2 Event slicing correctness

`EventHistory.cutoff_time` must match the actual end of the visible cutoff window according to `TimeGrid` semantics.

`ForecastTarget.n_future_users` must exclude the root author and every user observed at or before the cutoff.

All slicing tests must cover exact boundary timestamps.

### 4.3 No forecast padding

The following behavior is forbidden:

```python
np.full(missing_length, trajectory[-1])
```

A forecast is valid only if the simulator produces at least:

```text
cutoff_step + max(horizons) + 1
```

time points.

If the trajectory is shorter, the run must fail loudly with a `RuntimeError`.

## 5. ForecastRunner

Introduce a dedicated `ForecastRunner` rather than extending historical replay through implicit padding.

Minimum configuration:

```text
ForecastConfig
- cutoff_step
- horizon_steps
- population_mode
- reactivation_mode
- forecast_seeds
- step_hours
- micro_steps
```

The forecast simulation horizon is explicit:

```text
T_forecast = cutoff_step + horizon_steps
```

The implementation may reuse `tail_steps=horizon_steps`, but the resulting trajectory length must be asserted.

The runner must return:

- state trajectories;
- behavioral-flow trajectories;
- P5/P50/P95 across forecast seeds;
- effective configuration;
- git SHA;
- population mode;
- leakage flags;
- cutoff and horizon metadata.

## 6. Cutoff-State Assimilation

Forecasting must begin from an observed cutoff state, not from the root-post initial condition.

Add:

```text
build_forecast_state(history, population, params, rng)
```

Initial state at cutoff:

- active in the cutoff window -> `A`, `m=1`;
- active before cutoff but not in the cutoff window -> `D`, `m=1`;
- potential user never active by cutoff -> `U`, `m=0`;
- no user begins in `E` unless a separately observed exposure signal is available;
- fatigue may be initialized from recent activity history using a fixed documented rule;
- stance/arousal remain prior-based until text signals are introduced in a later phase.

The root author follows the same observed-history rule rather than being forced to `A` if inactive at the cutoff.

Required invariant:

> Forecast state construction may inspect only the root post, pre-cutoff interactions, and the selected diagnostic population list.

## 7. Behavioral Observation Layer

The primary forecast target is not `n_A_ts`.

Add per-step simulated behavioral outputs:

- `sim_first_actor_count_ts`;
- `sim_repeat_actor_count_ts`;
- `sim_active_count_ts`.

Definitions:

- first actor: an agent produces its first observable action during the step;
- repeat actor: an agent that acted in an earlier step produces an observable action again;
- active actor: unique agent producing at least one action in the step.

Required identity:

```text
sim_active_count[t]
= sim_first_actor_count[t]
+ sim_repeat_actor_count[t]
```

An agent counts as active during the step even if it transitions to `D` before the end-of-step snapshot.

`n_A_ts` remains a diagnostic stock and must not be used as the primary observed-activity forecast.

## 8. Diagnostic Population Modes

### 8.1 Oracle cohort

Population contains the complete set of eventual participants from the full event, but future interactions, future action times, and future network edges remain hidden.

Metadata:

```text
population_mode = oracle_cohort
cohort_conditioned = true
causal_forecast = false
```

Scientific question:

> Assuming the eventual participant risk set is known, can `ONE_SHOT` predict when those agents become active?

### 8.2 Observed closed cohort

Population contains only users observed by the cutoff. No new users may enter after the cutoff.

Metadata:

```text
population_mode = observed_closed
cohort_conditioned = false
causal_forecast = false
```

Scientific question:

> How much future activity can be explained by continuing and repeat behavior among already observed users?

### 8.3 Interpretation

- oracle fails -> temporal/state-transition mechanism is inadequate;
- oracle passes and observed-closed fails -> future population entry is the dominant missing mechanism;
- both pass -> proceed to anonymous open-population modeling;
- observed-closed unexpectedly beats oracle consistently -> inspect population initialization and exposure scaling for dilution artifacts.

## 9. Parameter Protocol for V1.7R

V1.7R must not use two or three observations to estimate three free parameters independently per event.

For the sentinel stage:

- use shared `alpha_0` and `gamma_0` estimated from development events or frozen from a documented training-only procedure;
- allow only `beta_M` to adapt to the current event history;
- for 24 h cutoffs, use a global `beta_M` because event-level adaptation is underidentified;
- parameter selection uses pre-cutoff observations only;
- fake/real labels must not select different parameter sets;
- fit and forecast seeds remain disjoint.

A later full revalidation may introduce grouped cross-validation only after forecast-integrity tests pass.

## 10. Sentinel Experiment

Do not run all twenty development cases initially.

Select four documented sentinel cases:

1. a fake-news case that performed well in V1.5.2;
2. a fake-news case that failed in V1.5.2;
3. a real-news case with rapid post-peak decay;
4. a real-news case with a visible long tail.

For each case, run:

- 48 h cutoff -> next 72 h forecast;
- 72 h cutoff -> next 48 h forecast;
- both `oracle_cohort` and `observed_closed` modes.

Before scoring, verify manually and through assertions:

- the simulation genuinely extends beyond the cutoff;
- no future value is produced through padding;
- the assimilated cutoff state matches observed history;
- `sim_active = sim_first + sim_repeat` at every step;
- oracle mode can create future first actors;
- observed-closed mode cannot create unseen users;
- all three forecast horizons are independently simulated;
- parameters use no future target values.

## 11. Tests

### 11.1 Unit tests

- `str(ReactivationMode.ONE_SHOT) == "one_shot"`;
- invalid mode raises `ValueError`;
- cutoff-time boundary correctness;
- root author excluded from future-user count;
- no last-value padding helper exists in forecast code;
- short forecast trajectory raises `RuntimeError`;
- cutoff-state A/D/U assignment;
- first/repeat/active decomposition;
- behavioral conservation identity;
- population-mode metadata correctness.

### 11.2 Integration tests

- full case -> history/target -> cutoff state -> forward forecast -> evaluation;
- forecast length equals requested cutoff plus horizon;
- hidden future interactions are never read by state assimilation;
- oracle population uses only future identities, not future timestamps/actions;
- observed-closed population cannot emit first actors from unseen users;
- repeated execution with fixed seeds is deterministic.

### 11.3 Scientific regression tests

- current V1.7 artifact is marked invalidated;
- V1.5.2 artifacts remain unchanged;
- forecast metrics use `sim_active_count_ts`, not `n_A_ts`;
- forecast seed sets do not overlap fit seeds;
- sentinel results are stored in a new artifact, never appended to the invalid V1.7 JSON.

## 12. Artifacts

Add:

```text
artifacts/forecast/v17_rolling_forecast.INVALIDATED.md
artifacts/forecast/v17r_sentinel_forecast.json
artifacts/forecast/v17r_sentinel_summary.md
```

Every sentinel result must include:

- git SHA;
- case and pair IDs;
- label;
- cutoff and horizon;
- population mode;
- observed users at cutoff;
- potential population size;
- effective parameters;
- fit and forecast seeds;
- observed first/repeat/active future trajectories;
- simulated P5/P50/P95 trajectories;
- RMSLE and peak metrics;
- leakage flags;
- pass/fail assertions.

## 13. Decision Gate

### Continue with ONE_SHOT only if

- every integrity assertion passes;
- oracle forecasts are nonconstant unless the simulated dynamics genuinely converge to a constant;
- at least 2/4 sentinel events beat the best train-only baseline in oracle mode;
- at least one real-news sentinel beats baseline;
- severe systematic overprediction is absent;
- parameter estimates are not consistently pinned to bounds.

### Move to an event-intensity plus agent-state model if

- oracle mode still fails on at least three of four sentinel cases;
- real-news sentinels remain uniformly worse than baseline;
- predicted activity remains grossly inflated after correct state assimilation;
- the model cannot independently reproduce first-actor and repeat-actor trajectories;
- parameter estimates remain unstable or boundary-dominated.

The fallback model decomposes observed activity as:

```text
Y_t = Y_t_first + Y_t_repeat
```

with first participation driven by event/exposure intensity and repeat participation driven by self-excitation, previous activity, fatigue, emotion, and later opinion/LLM signals.

## 14. Implementation Order

1. Add invalidation document for current V1.7 artifact.
2. Fix enum, cutoff-time, and future-user defects.
3. Add forecast-length assertions and remove padding.
4. Implement `ForecastRunner`.
5. Implement cutoff-state assimilation.
6. Implement behavioral first/repeat/active outputs.
7. Implement oracle and observed-closed population modes.
8. Add unit and integration tests.
9. Run four sentinel cases only.
10. Review sentinel residuals and apply the decision gate.

No full twenty-case run, open-population implementation, opinion-dynamics calibration, or LLM integration is allowed before Step 10 is reviewed and approved.
