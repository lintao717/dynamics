# V1.7 Forecast-Ready ONE_SHOT Model — Design Specification

## 1. Purpose

V1.7 converts the current CHECKED historical replay workflow into a forecast-ready propagation pipeline.

The approved scientific direction is:

- retain `ONE_SHOT` as the current best propagation-structure candidate;
- stop synthetic-shock tuning;
- align simulated observables with real platform behaviors;
- introduce cutoff-based rolling forecasts;
- remove future-participant leakage;
- evaluate by event-level cross-validation and untouched holdout cases;
- postpone opinion-dynamics and LLM integration until the propagation forecast passes an independent test.

V1.7 is not intended to prove that `ONE_SHOT` is the final universal model. It is intended to determine whether the current state-transition framework can produce valid out-of-sample forecasts under a leakage-controlled protocol.

## 2. Current Evidence and Constraints

V1.5.2 provides the strongest current structural evidence:

- `ONE_SHOT` outperforms `FULL` on paired validation loss;
- the paired improvement is statistically significant under the current bootstrap analysis;
- only 6/20 cases are reachable under the hardened protocol;
- performance is weaker on real-news cases than on fake-news cases.

V1.6 does not validate event-gated reactivation. It only shows that a fixed synthetic shock at day 3 is harmful under the tested optimization protocol. Real event-gated reactivation remains deferred until timestamped external events are available.

The current replay is not a causal forecast because the full event participant cohort is known before simulation. The current observed target `active_count[t]` is a per-window behavioral flow, while simulated `n_A_ts[t]` is a state stock. V1.7 must resolve both issues.

## 3. Scope

### 3.1 In scope

1. Correctness hardening for reactivation-mode handling.
2. Explicit behavioral observation mapping.
3. Cutoff-based event history and future targets.
4. Cohort-conditioned rolling forecast as an intermediate diagnostic.
5. Open-population forecasting with anonymous susceptible agents.
6. Hierarchical/shared parameter estimation.
7. Event-pair cross-validation.
8. Untouched holdout evaluation.
9. Forecast metrics and uncertainty reporting.

### 3.2 Out of scope

1. New LLM agent policies.
2. Opinion-dynamics calibration.
3. Synthetic-shock tuning.
4. Real event-gated reactivation without external-event timestamps.
5. Cross-platform Tibet data ingestion.
6. Major refactoring unrelated to forecasting correctness.

## 4. Architecture

V1.7 introduces five isolated layers.

### 4.1 History slicing layer

New object: `EventHistory`.

Responsibilities:

- expose only data at or before a forecast cutoff;
- contain the root post, observed interactions, observed users and observed network edges;
- prevent access to future interactions and future user identities;
- preserve the source `case_id`, cutoff timestamp and data-quality metadata.

New object: `ForecastTarget`.

Responsibilities:

- contain future observations after the cutoff;
- remain unavailable to calibration and simulation code;
- support future active-user, first-actor, repeat-actor, cumulative-user and peak metrics.

The slicing API must return `(EventHistory, ForecastTarget)` from a full `EventCase`.

### 4.2 Observation-alignment layer

Observed trajectories must add:

- `first_actor_count[t]`: users whose first event interaction occurs in window `t`;
- `repeat_actor_count[t]`: users who interacted previously and interact again in window `t`;
- `active_count[t] = first_actor_count[t] + repeat_actor_count[t]`.

The simulator must expose behavioral flow metrics distinct from state stocks:

- `new_actor_count_ts`;
- `continuing_actor_count_ts`;
- `reactivated_actor_count_ts`;
- `simulated_active_count_ts`.

For `ONE_SHOT`, `reactivated_actor_count_ts` is zero by construction.

The initial observable mapping is:

`simulated_active_count_ts = new_actor_count_ts + continuing_actor_count_ts + reactivated_actor_count_ts`.

The implementation must not fit `active_count` directly to `n_A_ts` unless a test proves that every A-state agent emits at least one observable action in the same window.

### 4.3 Forecast engine layer

New object: `ForecastConfig`.

Minimum fields:

- `cutoff_step`;
- `horizons`;
- `population_mode`;
- `reactivation_mode`;
- `fit_seeds`;
- `forecast_seeds`;
- parameter-estimation strategy;
- uncertainty quantiles.

Two forecast modes are required.

#### Mode A: cohort-conditioned diagnostic

- full potential cohort may be retained;
- only pre-cutoff interactions and edges are visible;
- outputs must be marked `cohort_conditioned=true` and `causal_forecast=false`.

Purpose: isolate temporal-model error from population-entry error.

#### Mode B: open-population forecast

- only observed users are instantiated with real identities;
- future users are represented by an anonymous susceptible pool;
- new actors are sampled from that pool;
- future real user identities are never read during prediction;
- outputs are marked `cohort_conditioned=false` and `causal_forecast=true` once all other leakage checks pass.

### 4.4 Parameter-estimation layer

Per-event free fitting of three parameters from a few time points must stop.

V1.7 uses a hierarchical structure:

- global/shared parameters: `alpha_0`, `gamma_0`;
- event-adaptive parameter: `beta_M`;
- optional anonymous-pool scale parameter, introduced only for open-population mode.

Training procedure:

1. estimate shared parameters across training events only;
2. estimate the event-adaptive parameter from the pre-cutoff history only;
3. select parameters using training loss only;
4. never use validation horizons to choose parameter vectors;
5. use disjoint fit, validation and final-test random seeds.

Fake/real labels must not select different parameters at prediction time unless the label is an observed input available at the cutoff. The default V1.7 design therefore uses shared parameters across both labels.

### 4.5 Evaluation layer

Evaluation is performed only on future windows after the cutoff.

Required metrics:

- future RMSLE;
- future MASE;
- cumulative-user AUC error;
- future peak-size ratio;
- future peak-time error;
- interval coverage for P5/P50/P95;
- calibration sharpness;
- fraction of events beating the best train-only baseline.

Required baselines:

- zero activity;
- persistence;
- exponential decay;
- pulse decay;
- best baseline chosen using training history only.

## 5. Data Flow

1. Load full `EventCase` for offline experiment construction.
2. Slice at cutoff into `EventHistory` and hidden `ForecastTarget`.
3. Build observed history trajectories from `EventHistory` only.
4. Estimate shared parameters using training events only.
5. Estimate event-adaptive parameters from the current event history only.
6. Initialize `ONE_SHOT` simulation.
7. Run multiple forecast seeds.
8. Aggregate P5/P50/P95 forecasts.
9. Evaluate exclusively against `ForecastTarget`.
10. Persist full effective configuration, git SHA, seeds, cutoff and leakage flags.

## 6. Rolling-Forecast Protocol

Required cutoffs and horizons:

- first 24 h -> predict next 24/48/72 h;
- first 48 h -> predict next 24/48/72 h;
- first 72 h -> predict next 24/48 h.

A forecast instance is valid only when the event contains enough future observations for the requested horizon.

The existing ten fake/real pairs form the development set. Cross-validation must be grouped by pair so that the fake and real members of a pair remain in the same fold.

Recommended protocol:

- five folds;
- eight pairs for training;
- two pairs for validation;
- all cutoffs generated within each held-out event;
- aggregate first by event, then across events, to avoid long events dominating the score.

Five untouched pairs are reserved for one-time final testing. They must not be inspected during V1.7 model design.

## 7. Open-Population Design

The anonymous susceptible pool is introduced only after cohort-conditioned forecasting is operational.

Minimum pool model:

- pool size is inferred from training events and observable early-event features;
- anonymous agents use distributions of susceptibility, activation cost and opinion-neutral priors estimated from training data;
- activation creates an anonymous actor identifier, not a future real user ID;
- pool-size uncertainty is propagated through forecast seeds.

Initial pool-size predictors may use:

- observed users by cutoff;
- first-window active users;
- root-post interaction rate;
- current cumulative-user growth;
- event duration so far.

No text, LLM or fake/real label is required in the first implementation.

## 8. Reactivation Semantics

`ONE_SHOT` is the V1.7 default:

- D0->A disabled;
- D1->A disabled;
- U->E->A/D and A->D remain active.

`EVENT_GATED` remains implemented but disabled in V1.7 experiments.

Real event-gated reactivation may be reconsidered only when the data provides `ExternalEventRecord` entries containing:

- timestamp;
- event type;
- source;
- strength;
- credibility;
- affected communities.

A fixed synthetic day-3 pulse is not an accepted substitute.

## 9. Correctness and Error Handling

The implementation must:

- fix `ReactivationMode.__str__()` to return only `self.value`;
- reject unknown reactivation modes with `ValueError`;
- forbid silent fallback for missing required trajectory fields;
- assert that history interactions are not later than the cutoff;
- assert that forecast code cannot access future user IDs;
- record whether a result is cohort-conditioned or causal;
- fail when training or forecast horizons are too short;
- preserve deterministic reproducibility for a fixed configuration and seed set.

## 10. Testing Strategy

### 10.1 Unit tests

- enum string conversion;
- mode validation;
- each reactivation-mode transition behavior;
- `EVENT_GATED` equals `ONE_SHOT` when shock is zero;
- event slicing at exact cutoff boundaries;
- first/repeat actor decomposition;
- active-count conservation;
- future-user exclusion;
- missing-field loud failure;
- deterministic seed behavior.

### 10.2 Integration tests

- full event -> history/target split -> forecast -> evaluation;
- no future interactions enter network construction;
- no future user identity enters open-population initialization;
- all persisted artifacts contain effective configuration and git SHA;
- cohort-conditioned and open-population flags are correct.

### 10.3 Scientific regression tests

- V1.5.2 `ONE_SHOT` result can be reproduced within a documented tolerance;
- rolling forecasts are scored only on future windows;
- validation events do not influence parameter selection;
- final-test seeds and events are never used during development.

## 11. Acceptance Gates

V1.7 can proceed to opinion dynamics only if the untouched holdout test satisfies all of the following:

- at least 6/10 events beat the best train-only baseline;
- fake events: at least 3/5 beat baseline;
- real events: at least 3/5 beat baseline;
- median future-RMSLE improvement exceeds 10%;
- median peak-time error is at most one day;
- bootstrap 95% confidence interval for median improvement has lower bound above zero;
- uncertainty interval coverage is reported and is not degenerate;
- no causal-leakage assertion fails.

If the cohort-conditioned model passes but the open-population model fails, the next work targets population-entry modeling.

If both modes fail, no further tuning of `ONE_SHOT` is allowed without a residual-based structural diagnosis. The fallback candidate is an event-intensity plus agent-state model.

## 12. Versioning and Artifacts

Planned versions:

- `V1.6.1`: correctness and observable-alignment hardening;
- `V1.7`: cohort-conditioned rolling forecast;
- `V1.7.1`: open-population forecast;
- `V1.7-final`: frozen cross-validation and holdout protocol.

Each experiment artifact must include:

- git SHA;
- model version;
- event and pair IDs;
- cutoff and horizon;
- observed-history length;
- effective parameters;
- fit/forecast seed groups;
- population mode;
- leakage flags;
- baseline identity;
- all forecast metrics;
- prediction intervals.

## 13. Implementation Order

1. Correct reactivation-mode defects and tests.
2. Add observed first/repeat actor trajectories.
3. Add simulated behavioral-flow observables.
4. Implement `EventHistory` and `ForecastTarget` slicing.
5. Implement cohort-conditioned rolling forecast.
6. Implement grouped five-fold event-pair cross-validation.
7. Freeze metrics and baselines.
8. Implement anonymous susceptible pool.
9. Run open-population cross-validation.
10. Freeze the protocol and run untouched holdout cases once.

No LLM integration or opinion-dynamics calibration begins before Step 10 is evaluated against the acceptance gates.
