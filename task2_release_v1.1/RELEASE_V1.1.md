# Task 2 V1.1 Release Candidate

**Date**: 2026-07-29
**Status**: **Release Candidate** — core implementation single-sourced; test/doc sync pending.
**Predecessor**: V1.0 (2026-07-17) → V1.1-alpha → V1.1-beta0 → V1.1-RC

---

## Validation Results

```
10/10 tests passed. Overall score: 0.973. Verdict: PASS.
```

| # | Test | Dimension | Score | Status |
|:--:|------|:--:|:---:|:---:|
| 1 | SIR consistency | degeneracy | 0.98 | PASS |
| 2 | HK bounded confidence consistency | degeneracy | 0.75 | PASS |
| 3 | Expression fidelity (per-agent diff=0) | degeneracy | 1.00 | PASS |
| 4 | Frozen opinion independence (RMS diff=0%) | degeneracy | 1.00 | PASS |
| 5 | No spontaneous reactivation | degeneracy | 1.00 | PASS |
| 6 | beta direction | direction | 1.00 | PASS |
| 7 | alpha_1 direction | direction | 1.00 | PASS |
| 8 | gamma_1 direction | direction | 1.00 | PASS |
| 9 | delta_f direction | direction | 1.00 | PASS |
| 10 | r_1 direction | direction | 1.00 | PASS |

### Also verified (not in automated suite)

| Test | Result |
|------|:---:|
| R_eff: dR_eff/d|o| > 0 | Slope 0.249 |
| R_eff: dR_eff/dalpha_1 > 0 | Derivative 0.154 |
| R_eff: dR_eff/dlambda_spiral <= 0 | Delta 0.003 |
| R_eff: SBM R_eff > ER R_eff | 0.268 > 0.214 |
| T=500 long-term stability | 3/3 stable polarized |
| Phase diagram (6x6, T=200) | 35 polarized, 1 fragmented |

---

## V1.0 → V1.1 Changes

### Structural fixes (7 P0 bugs)

| Bug | Fix |
|-----|-----|
| Silent agents treated as expressing opinion 0 | `expressed_mask` + only A-state agents with valid o_hat participate |
| V_MIN=1.0 made climate permanently invisible | Parameterized to `climate_visibility_threshold=0.10` |
| Task 3 API V(t) not persisted | `self._V` tracked across steps |
| LLM agent state inconsistency (only z, o_hat frozen) | All 6 fields frozen; m=1 on express; NaN on silent |
| Full validation crashed on identifiability | Tuple unpacking fixed |
| R_eff mean(sigma) miscalculated | `mean(sigma(x))` replaces `1/mean(1+exp(-x))` |
| No per-step transition event recording | `TransitionEvents` dataclass with 7 counts |

### Formula updates

| Update | Detail |
|--------|--------|
| R_eff next-generation matrix | Added S_i, L_j, q_j, climate visibility, PVA channel |
| Social influence | Weighted average over expressed agents only |
| Climate congruity | v_i * Gamma_i scaling, visibility threshold |
| Delayed activation vs reactivation | m_i flag with split r_0^(0)/r_0^(1) and r_1^(0)/r_1^(1) |
| U renamed | Uncertain → Unaware |

### Engineering

| Feature | Status |
|---------|:---:|
| TransitionEvents → MetricsCollector | Connected |
| LLM posts → PVA V(t) | Connected |
| AgentSnapshot climate_visible | Added |
| Frozen opinion test | Proper state cloning, 0% diff |
| Phase diagram | Extended range, honest results |

---

## Known Limitations

1. **Consensus is extremely hard to achieve**: With the social influence fix (silent agents no longer pull opinions to 0), the model predicts consensus only at very extreme parameters (zeta < 0.001, epsilon > 0.98, T > 200). This is a model prediction, not a bug.

2. **Silence spiral requires dedicated experimental conditions**: The deterministic check confirms the mechanism works (P(E->A) drops from 44% to 33% with lambda_spiral 0→0.85), but in default configurations, community homogeneity keeps Gamma_i > 0.5 for most agents.

3. **alpha_1 identifiability requires shock or extreme opinions**: Grid-search recovery shows 50% error without shocks. This is expected behavior for a parameter that matters only when |o| varies significantly.

4. **N > 10,000 requires sparse matrix implementation**: Current dense O(N^2) matrices limit practical agent count.

5. **Real-event individual-level calibration not completed**: v0.1 data provides prior constraints only.

---

## File Inventory

```
task2_release_v1.1/
├── RELEASE_V1.1.md          ← This file
├── README.md
├── CHANGELOG.md
├── RELEASE_MANIFEST.md
├── docs/
│   ├── model_definition_v1.1.md
│   ├── theoretical_analysis_v1.1.md
│   ├── equation_feasibility_v1.1.md
│   ├── simulation_results_v1.1.md
│   ├── data_requirements_v1.1.md
│   └── task3_interface_v1.1.md
├── dynamics_simulation/
│   ├── config.py             ← 7 parameter groups incl. ViralParams + climate_visibility_threshold
│   ├── agents.py             ← AgentState with m_i flag
│   ├── networks.py           ← ER/BA/WS/SBM generators
│   ├── transitions.py        ← 7-step engine + TransitionEvents + expressed_mask + v_i threshold
│   ├── simulation.py         ← SimulationRunner with V(t) tracking
│   ├── metrics.py            ← MetricsCollector with TransitionEvents accumulation
│   ├── reff.py               ← R_eff with S_i, L_j, q_j, PVA
│   ├── api.py                ← Task 3 Simulation class + LLM freeze + PVA integration
│   ├── validation.py         ← 10-test automated suite (10/10, 0.973)
│   └── odd.md
├── tests/
│   ├── test_smoke.py
│   └── test_identifiability.py
├── configs/
│   └── default_v1.1.yaml
└── data/
    └── validation_v11_final.json
```

---

## Quick Start

```python
from dynamics_simulation.api import Simulation

sim = Simulation.init(n_agents=500, params="default", network="sbm", seed=42)
for t in range(100):
    metrics = sim.step()
    if t % 10 == 0:
        print(f"t={t}: A={metrics.n_A} D={metrics.n_D} V={metrics.V:.3f}")
```

---

## Next Steps (V1.2+)

- Global sensitivity analysis (Sobol/Morris)
- Sparse network implementation for N > 10,000
- Real-event parameter calibration with complete cascade data
- LLM agent integration with DeepSeek API
- Multi-event parallel simulation
