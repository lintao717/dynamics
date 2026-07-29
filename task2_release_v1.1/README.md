# Task 2: Integrated Propagation-Opinion-Emotion Dynamics Model V1.1

**Status**: Stage baseline — model structure frozen, internal consistency verified.
**Date**: 2026-07-25

---

## Quick Start

```python
from dynamics_simulation.api import Simulation

# Initialize with default parameters
sim = Simulation.init(n_agents=500, params="default", network="sbm", seed=42)

# Run 100 steps
for t in range(100):
    metrics = sim.step()
    if t % 10 == 0:
        print(f"t={t}: A={metrics.n_A}, D={metrics.n_D}, bias={metrics.public_bias:.3f}")

# Export results
print(sim.summary())
```

## State Vector

Each agent $i$ at time $t$:

$$X_i(t) = (z_i(t), m_i(t), o_i(t), \hat{o}_i(t), h_i(t), f_i(t))$$

- $z_i \in \{U, E, A, D\}$: propagation state (Unaware, Exposed, Active, Dormant)
- $m_i \in \{0,1\}$: history flag (0=never expressed, 1=has expressed)
- $o_i \in [-1,1]$: private opinion
- $\hat{o}_i \in [-1,1] \cup \{\varnothing\}$: public expression
- $h_i \in [0,1]$: emotional arousal
- $f_i \in [0,1]$: information fatigue

## Key Mechanisms

1. **7-step coupled update cycle**: exposure -> opinion -> emotion -> activation -> decay -> reactivation -> expression
2. **Platform Viral Amplification (PVA)**: Hawkes-inspired global viral intensity $V(t)$
3. **Dual network structure**: propagation $G^s$ + opinion influence $G^o$
4. **Community cross-flow $\kappa_{ab}$**: quantifies inter-community information leakage
5. **Silence spiral**: minority-opinion agents face expression penalty when climate is visible
6. **Delayed activation vs true reactivation**: separated by $m_i$ flag

## Validation

- **9/10 tests passed** (overall score 0.885)
- SIR consistency: SI correlation 0.95
- Expression fidelity: per-agent error 0.0
- All 5 parameter directions: correct
- Known limitation: silence spiral requires dedicated experimental conditions

## Documentation

| Document | Description |
|----------|-------------|
| `RELEASE_MANIFEST.md` | Release metadata, file inventory, validation results |
| `CHANGELOG.md` | V1.0 -> V1.1 changes |
| `docs/model_definition_v1.1.md` | Complete model specification |
| `docs/theoretical_analysis_v1.1.md` | Propositions on R_eff, opinion stability |
| `docs/equation_feasibility_v1.1.md` | 5-dimension validation framework |
| `docs/simulation_results_v1.1.md` | 7 mechanism verification experiments |
| `docs/data_requirements_v1.1.md` | Parameter-to-data mapping |
| `docs/task3_interface_v1.1.md` | ODD protocol + Task 3 API spec |

## Requirements

- Python 3.11+
- numpy, scipy

## Next Steps

1. Structural identifiability test
2. Task 3 interface validation
3. Dedicated silence spiral experiment
4. Global sensitivity analysis
5. Real-event parameter calibration
