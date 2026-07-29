"""Extended phase diagram: find consensus, polarized, and fragmented regions."""
import sys, os, json
from pathlib import Path
import numpy as np
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamics_simulation.config import default_params, PropagationParams
from dynamics_simulation.networks import generate_networks
from dynamics_simulation.agents import initialize_agents, D, A
from dynamics_simulation.transitions import TransitionEngine, ExternalInputs

OUTPUT = Path(__file__).parent.parent / "data" / "sim_results" / "phase_diagram_extended.json"

def main():
    print("=" * 60)
    print("  EXTENDED ZETA-EPSILON PHASE DIAGRAM")
    print("=" * 60)

    n, T, seed = 150, 200, 42
    # Wide range to capture consensus region
    zeta_vals = np.array([0.002, 0.03, 0.10, 0.30, 0.60, 0.90])
    eps_vals  = np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95])
    sigma = np.zeros((len(zeta_vals), len(eps_vals)))
    n_clusters = np.zeros((len(zeta_vals), len(eps_vals)), dtype=int)

    p = replace(default_params(), propagation=PropagationParams(beta=0.0, beta_M=0.0))
    rng = np.random.default_rng(seed)
    G_s, G_o, _ = generate_networks("sbm", n=n, rng=rng, n_blocks=3, p_in=0.15, p_out=0.02)

    for zi, zeta in enumerate(zeta_vals):
        for ei, eps in enumerate(eps_vals):
            run_rng = np.random.default_rng(seed)
            state = initialize_agents(n=n, initial_active=5, rng=run_rng,
                                      initial_opinion_dist="polarized")
            state.z[:] = D; state.z[:5] = A
            state.o_hat[:5] = state.o[:5].copy()
            state.attrs.zeta[:] = zeta
            state.attrs.epsilon[:] = eps * 2.0
            state.attrs.mu[:] = 0.3

            engine = TransitionEngine(p, run_rng)
            o_init = state.o.copy()
            for t in range(T):
                state, _, _ = engine.step(state, G_s, G_o, None, ExternalInputs(), o_init, t)

            sigma[zi, ei] = float(state.o.std())
            o_sorted = np.sort(state.o)
            gaps = np.diff(o_sorted)
            n_clusters[zi, ei] = int((gaps > eps * 2.0).sum()) + 1

    # Classify
    classification = np.empty_like(sigma, dtype=object)
    for zi in range(len(zeta_vals)):
        for ei in range(len(eps_vals)):
            s = sigma[zi, ei]; nc = n_clusters[zi, ei]
            if s < 0.12:         classification[zi, ei] = "CONSENSUS"
            elif nc >= 4:        classification[zi, ei] = "FRAGMENTED"
            else:                classification[zi, ei] = "POLARIZED"

    n_con = int((classification == "CONSENSUS").sum())
    n_pol = int((classification == "POLARIZED").sum())
    n_frag = int((classification == "FRAGMENTED").sum())

    print(f"\n  Grid: {len(zeta_vals)}x{len(eps_vals)}, T={T}")
    print(f"  CONSENSUS:  {n_con} cells")
    print(f"  POLARIZED:  {n_pol} cells")
    print(f"  FRAGMENTED: {n_frag} cells")
    print(f"  Three regions: {'YES' if (n_con>0 and n_pol>0 and n_frag>0) else 'PARTIAL' if (n_con>0 or n_frag>0) else 'NO'}")

    print(f"\n  sigma_o grid:")
    header = "zeta\\eps " + " ".join(f"{e:6.2f}" for e in eps_vals)
    print(f"  {header}")
    for zi, zeta in enumerate(zeta_vals):
        row = " ".join(f"{sigma[zi,ei]:6.3f}" for ei in range(len(eps_vals)))
        cls = " ".join(f"{classification[zi,ei]:12s}" for ei in range(len(eps_vals)))
        print(f"  {zeta:8.3f} [{row}]  [{cls}]")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({
            "zeta_vals": zeta_vals.tolist(), "eps_vals": eps_vals.tolist(),
            "sigma": sigma.tolist(), "n_clusters": n_clusters.tolist(),
            "classification": classification.tolist(),
            "n_consensus": n_con, "n_polarized": n_pol, "n_fragmented": n_frag,
        }, f, indent=2)
    print(f"\n  Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
