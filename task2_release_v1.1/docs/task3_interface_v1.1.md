# ODD Protocol: Integrated Propagation-Opinion Dynamics Model

**Version**: 1.0
**Date**: 2026-07-25
**Format**: ODD (Overview, Design concepts, Details) — Grimm et al. (2006, 2010, 2020)

---

## 1. Purpose

The model simulates the co-evolution of information propagation and public opinion during social risk events in the Tibet region. It integrates three mechanisms:

1. **Propagation dynamics** (U-E-A-D state transitions) — how information spreads through a social network
2. **Opinion dynamics** (private opinion update + public expression bias) — how attitudes form and shift
3. **Emotion-fatigue dynamics** — how arousal and exhaustion modulate behavior

The model is designed to be used in two modes:
- **Standalone**: Parametric simulation with homogeneous decision rules
- **LLM text generation (Task 3)**: Dynamics kernel determines all state transitions (z, o, h, f) and expression decisions. LLM is called only for agents already placed in state A by the kernel, to render their pre-computed stance as natural language text. LLM does NOT control state.

## 2. Entities, State Variables, and Scales

### 2.1 Entities

A single entity type: **Agent** (social media user).

Population size N = 10^2 to 10^5 (default: 500 for synthetic, ~750 for empirical calibration).

### 2.2 State Variables

Each agent i at time t is described by:

| Variable | Symbol | Range | Description |
|----------|--------|-------|-------------|
| Propagation state | z_i(t) | {U, E, A, D} | U=Uncertain, E=Exposed(transient), A=Active, D=Dormant |
| Private opinion | o_i(t) | [-1, 1] | Internal stance on the proposition |
| Public expression | ô_i(t) | [-1, 1] ∪ {∅} | Expressed stance (∅ when z_i ≠ A) |
| Emotional arousal | h_i(t) | [0, 1] | Intensity of emotional activation |
| Information fatigue | f_i(t) | [0, 1] | Cumulative information overload |

**Fixed attributes** (agent heterogeneity):
- c_i: Expression cost [0, 1]
- μ_i: Opinion update speed [0, 1]
- ζ_i: Initial opinion anchoring [0, 1]
- ε_i: Bounded confidence threshold [0, 2]
- σ_ξ,i: Noise sensitivity [0, 0.5]

### 2.3 Scales

- **Time step**: 24 hours (one observation window)
- **Time horizon**: 7-100 steps for synthetic, 53 steps for empirical calibration
- **Spatial scale**: Single social media platform (Weibo)
- **Network**: Static directed weighted graph with 2-3 communities

## 3. Process Overview and Scheduling

Each time step executes 7 sub-steps in fixed order:

```
Step 1: Compute information exposure Lambda_i(t) for all agents
Step 2: Determine new exposures: U -> E (transient)
Step 3: Update private opinions o_i(t+1)
Step 4: Update emotion h_i(t+1) and fatigue f_i(t+1)
Step 5: Resolve E state -> A (activate) or D (dormant)
Step 6: Process A -> D (decay) and D -> A (reactivation)
Step 7: Generate public expressions ô_i(t+1) for active agents
```

In **text-generation mode** (Task 3), ALL agents complete Steps 1-7 through the dynamics kernel. After the kernel determines state, stance, and arousal, agents in state A may be routed to an LLM for natural language rendering via `TextGenerationRequest`. The LLM returns `GeneratedText` without modifying any numerical state.

## 4. Design Concepts

### 4.1 Basic Principles

- **Bounded rationality**: Agents use sigmoid-based probabilistic decision rules
- **Social influence**: Opinions update via bounded confidence (HK model) with anchoring (FJ model)
- **Emotional contagion**: Arousal spreads through the opinion network
- **Silence spiral**: Minority-opinion agents have reduced expression probability
- **Fatigue accumulation**: Continuous exposure and active participation increase fatigue

### 4.2 Emergence

- **Propagation cascades** emerge from micro-level exposure + activation decisions
- **Opinion polarization** emerges from bounded confidence + anchoring
- **Secondary bursts** emerge from shock reactivation of dormant agents
- **Public opinion bias** emerges from selection effects (who speaks) + expression bias (what they say)

### 4.3 Adaptation

Agents do NOT adapt strategically. They follow reactive decision rules determined by the dynamics kernel. LLM text generation does not confer adaptive decision-making capability — it only renders pre-computed states as language.

### 4.4 Interaction

- **Propagation**: Agent j -> Agent i via network edge w_ji^s
- **Opinion influence**: Agent j -> Agent i via network edge w_ji^o
- **Emotional contagion**: Same network as opinion influence (G_h = G_o in v1.0)

### 4.5 Stochasticity

- Random draws for state transitions (U->E, E->A, A->D, D->A)
- Gaussian noise in opinion updates
- Random network generation and agent attribute initialization

### 4.6 Observation

Metrics collected per step:
- State counts: n_U, n_E, n_A, n_D
- Opinion statistics: mean, std, polarization index
- Public opinion bias: |o_bar_private - o_bar_public|
- Cross-community flow: kappa_ab
- Emotion and fatigue means

## 5. Initialization

- **Network**: Static SBM with 3 blocks (p_in=0.15, p_out=0.02) or other topologies
- **Opinions**: Bimodal (polarized), uniform, or moderate normal
- **Propagation state**: 5-10 initially active agents, rest in U
- **Emotion**: Active agents have h_i ~ U(0.3, 0.6), rest h_i=0
- **Fatigue**: All agents start at f_i=0

## 6. Input Data

External inputs per step:
- Shock(t): Exogenous shock intensity [0, 1]
- Novelty(t): Information novelty [0, 1]
- M_i(t): Per-agent media exposure [0, 1]
- I_i(t): Per-agent information evidence direction [-1, 1]
- u_i(t): Per-agent official information stance [-1, 1]

## 7. Submodels

### 7.1 Information Exposure

Lambda_i(t) = beta * sum_j w_ji^s * 1[z_j=A] * q_j(t) + beta_M * M_i(t)

### 7.2 Activation (E -> A)

P(E_i -> A_i)(t) = sigma(alpha_0 + alpha_1*|o_i| + alpha_2*h_i + alpha_3*Gamma_i - alpha_4*f_i - alpha_5*c_i)

With silence spiral correction when Gamma_i < 0.5.

### 7.3 Decay (A -> D)

P(A_i -> D_i)(t) = sigma(gamma_0 + gamma_1*f_i + gamma_2*s(t) - gamma_3*n(t))

### 7.4 Reactivation (D -> A)

P(D_i -> A_i)(t) = sigma(r_0 + r_1*Shock(t) + r_2*Novelty(t) + r_3*h_i)

### 7.5 Opinion Update

o_i(t+1) = Pi_{[-1,1]}[ o_i(t) + mu_i * (zeta_i[o_i(0)-o_i(t)] + (1-zeta_i)*sum_j w_ji^o * Phi_i(o_hat_j-o_i) + eta_i*I_i + chi_i*u_i ) + xi_i(t) ]

Phi_i(d) = d if |d| <= epsilon_i, else 0 (bounded confidence)

### 7.6 Public Expression

o_hat_i(t) = o_i(t) + lambda_c*(climate - o_i) + lambda_h*h_i*o_i  (when z_i=A)

### 7.7 Emotion and Fatigue

h_i(t+1) = Pi_{[0,1]}[ (1-delta_h)*h_i + eta_h*I_m + omega_h*sum w_ji^h*h_j + chi_h*Shock - nu_h*f_i ]

f_i(t+1) = Pi_{[0,1]}[ (1-delta_f)*f_i + eta_f*N_i^exp + omega_f*1[z_i=A] ]

## 8. Parameter Table

See `docs/task2_model_definition_v1.md` Section 4 for the complete 29-parameter table.

## 9. Implementation

- **Language**: Python 3.11+
- **Core dependencies**: NumPy, SciPy
- **Package**: `dynamics_simulation/`
- **API for Task 3**: `dynamics_simulation/api.py` (Simulation class)

## References

- Grimm, V., et al. (2006). A standard protocol for describing individual-based and agent-based models. *Ecological Modelling*, 198(1-2), 115-126.
- Grimm, V., et al. (2010). The ODD protocol: A review and first update. *Ecological Modelling*, 221(23), 2760-2768.
- Grimm, V., et al. (2020). The ODD protocol for describing agent-based and other simulation models: A second update. *JASSS*, 23(2).
