# Real-Data Assumptions for V1.2 Historical Replay

## 1. CHECKED Data Characteristics

- CHECKED comments and reposts are **root-linked interactions**, not a complete follower graph.
- The dataset provides a collection of root-post cascades, not a complete social-network event database.
- Each cascade includes: root microblog metadata, nested comments, and reposts.
- There is no complete follower graph, no parent-child repost tree, and no exposure network.
- Therefore: `beta` cannot be claimed as separately identified from CHECKED alone.

## 2. Latent Variables

The following variables are **unobservable** in CHECKED/CED data and remain latent in V1.2:

- **U** (Uncertain): agents unaware of the event — no data source provides this.
- **E** (Exposed): agents who have seen content but not acted — CHECKED records only actions, not exposures.
- **Private opinion** (`o`): CHECKED provides text, not private beliefs.
- **Emotional arousal** (`h`): not directly measured; precomputed signals may serve as noisy proxies.
- **Information fatigue** (`f`): not observable from interaction logs.

Missing observations must never be silently converted to zero. Use NaN and explicit observation masks.

## 3. Replay Modes and Causal Interpretation

### Broadcast (Primary)
- `G_s` is zero for first-activation dynamics; exposure enters through `media_exposure` (broadcast signal).
- `G_o` accumulates only after observed interactions occur.
- This is the **primary evaluation mode** for V1.2.

### Cumulative Interaction (Secondary)
- Root-to-user edge becomes active only after that user's first observed action.
- It may influence later reactivation/opinion dynamics but cannot explain the first action.

### Oracle Static (Sensitivity Only)
- All observed root-to-user edges exist from step 0.
- This is an **upper-bound sensitivity run** and **cannot be reported as causal validation**.
- Every oracle-static result must be explicitly labeled as a non-causal upper bound.

## 4. Calibration Scope (Stage 1)

- First-release calibration targets: aggregate activity, cumulative participants, peak time, and interaction-type counts.
- Only four parameters are fitted: `beta_M`, `alpha_0`, `gamma_0`, `beta_V`.
- Stance and arousal loss weights remain zero until actual labels are supplied.
- 70/30 chronological train/validation split is mandatory.

## 5. Known Limitations

- CHECKED does not provide a complete follower or exposure network.
- Exact parent-child diffusion paths cannot be inferred.
- Direct observation of E (Exposed) state is not possible.
- Private opinion trajectories are not observed.
- Separate causal identification of `beta` and network degree is not possible from CHECKED alone.
- Tibet-specific behavioral parameters cannot be estimated from CHECKED.
- Official-intervention response requires an independently timestamped timeline, which CHECKED does not provide.

## 6. Parameter Release Stages

| Stage | Data | Parameters |
|-------|------|------------|
| Stage 1 | CHECKED broadcast | `beta_M`, `alpha_0`, `gamma_0`, `beta_V` |
| Stage 2 | CHECKED + precomputed stance/arousal | add `alpha_1`, `alpha_2` (after sensitivity tests) |
| Stage 3 | Self-collected parent-child cascades | consider `beta`, `alpha_3`, reactivation shock parameters |
| Stage 4 | Repeated user panels | consider `mu`, `zeta`, `epsilon`, fatigue, emotion-dynamics parameters |
