# V1.1 Release Manifest

```
release_name:    Task2-V1.1-RC
release_date:    2026-07-29
status:          Release Candidate (test + doc sync pending)
python_version:  3.11
validation:      10/10 passed (degeneracy + direction), score 0.973
random_seeds:    42, 123, 789
dimensions_covered: degeneracy, direction
```

---

## Test Log (immutable)

| # | Test | Score | Result |
|:--:|------|:---:|:---:|
| 1 | SIR consistency | 0.98 | PASS |
| 2 | HK bounded confidence | 0.75 | PASS |
| 3 | Expression fidelity | 1.00 | PASS |
| 4 | Frozen opinion independence | 1.00 | PASS |
| 5 | No spontaneous reactivation | 1.00 | PASS |
| 6 | beta direction | 1.00 | PASS |
| 7 | alpha_1 direction | 1.00 | PASS |
| 8 | gamma_1 direction | 1.00 | PASS |
| 9 | delta_f direction | 1.00 | PASS |
| 10 | r_1 direction | 1.00 | PASS |

Output: `data/validation_v11_final.json`

## Known Limitations

1. Consensus extremely hard (phase diagram: 35/36 polarized, requires extreme params)
2. Silence spiral requires dedicated experimental conditions (deterministic check confirmed)
3. alpha_1 identifiability needs shock conditions (50% error without)
4. N > 10,000 needs sparse matrix implementation
5. Real-event individual-level calibration not completed

## File Inventory

```
task2_release_v1.1/
├── RELEASE_V1.1.md
├── RELEASE_MANIFEST.md (this file)
├── README.md
├── CHANGELOG.md
├── docs/ (8 docs)
├── tests/ (test_smoke.py, test_identifiability.py, test_network_direction.py)
├── configs/ (default_v1.1.yaml)
└── data/ (validation_v1.1.json)

dynamics_simulation/          ← single source of truth (root of repo)
tests/                        ← test suite (root of repo)
```

## Changelog (V1.0 → V1.1)

- Added m_i history flag for delayed activation vs reactivation
- Split D→A params: r_0^(0)/r_1^(0) vs r_0^(1)/r_1^(1)
- Added climate visibility threshold v_i with V_MIN parameter
- Fixed silent-neighbor-treated-as-zero-opinion bug
- Fixed V_MIN=1.0 making climate permanently invisible
- Fixed Task 3 API V(t) persistence
- Fixed LLM agent state consistency (all 6 fields)
- Fixed R_eff mean(sigma) calculation
- Fixed full validation identifiability crash
- Added TransitionEvents to MetricsCollector
- Added LLM posts to PVA V(t)
- Added climate_visible to AgentSnapshot
- R_eff formula: added S_i, L_j, q_j, PVA channel
- U renamed: Uncertain → Unaware
- Downgraded theorems to propositions
