# Changelog

## V1.1 (2026-07-25)

### Changed

- **D-state semantics**: Added agent history flag $m_i(t) \in \{0,1\}$ to distinguish delayed first activation ($m=0$) from true reactivation ($m=1$). Split $D \to A$ parameters into $r_0^{(0)}, r_1^{(0)}$ (delayed) and $r_0^{(1)}, r_1^{(1)}$ (reactivation).
- **Climate visibility**: Added visibility threshold $v_i(t)$ and $v_{\min}$. When $v_i < v_{\min}$, climate congruity ($\Gamma_i$) and silence spiral are not applied, preventing the model from misinterpreting "no active neighbors" as "neutral climate."
- **$R_{\text{eff}}$ derivation**: Added expected active duration $1/g_j$, susceptible fraction $S_i$, and content influence $q_j$ to the next-generation matrix. Annotated as preliminary approximation.
- **$B_{\text{obs}}$ decomposition**: Replaced erroneous equality $B_{\text{obs}} = B_{\text{sel}} + B_{\text{expr}}$ with signed decomposition $\Delta_{\text{obs}} = \Delta_{\text{sel}} + \Delta_{\text{expr}}$ and triangle inequality $B_{\text{obs}} \leq |\Delta_{\text{sel}}| + |\Delta_{\text{expr}}|$.
- **Theorem downgrade**: Theorems 5-9 reclassified as propositions or candidate sufficient conditions, with explicit scope limitations and missing conditions documented.
- **Terminology**: U renamed from Uncertain to Unaware.
- **Simulation language**: "Empirical validation" downgraded to "internal consistency verification" / "mechanism reproduction." "Calibrated" downgraded to "prior-constrained."
- **Network weights**: Laplace smoothing restricted to existing neighbor sets only (documentation fix; code was already correct).
- **Platform Viral Amplification (PVA)**: Added `ViralParams` ($\beta_V, \delta_V, \eta_V$) and Hawkes-inspired viral intensity $V(t)$ to the exposure equation, bridging network-only propagation with platform-mediated trending exposure.

### Added

- `ViralParams` dataclass in config.py
- `V` field in ExternalInputs
- $v_i$ climate visibility computation in resolve_exposed and generate_expressions
- $m_i$ history flag in AgentState
- `test_reactivation_split.py` and `test_climate_visibility.py`

### Validation

- Smoke tests: all passed
- Validation suite: 9/10 passed, overall score 0.885 (V1.0: 0.866)
- SIR consistency correlation: 0.77 -> 0.95
- Expression fidelity per-agent error: 0.0
- Five parameter-direction tests passed
- Split reactivation parameters verified

### Known Limitations

- Silence spiral effect remains weak under default synthetic configuration (see RELEASE_MANIFEST.md)
- Structural identifiability not yet fully evaluated
- Real-event individual-level calibration not yet completed
- Frozen opinion independence test shows ~26% norm RMS diff (statistical noise with n_seeds=5)

---

## V1.0 (2026-07-17)

Initial model specification with:
- 5-state vector (U, E, A, D + o, o_hat, h, f)
- 29 parameters across 6 categories
- 7-step coupled update cycle
- Dual static networks (G_s, G_o) with community kappa
- 9 theorems (later downgraded to propositions in V1.1)
- 6 verification experiments
- v0.1 event-level prior constraints
- 6,605 Weibo repost edges from cascade collector
