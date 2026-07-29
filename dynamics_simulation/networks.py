"""
Synthetic network generators for the dynamics simulation.

Provides four canonical network models:
  - erdos_renyi:       Erdős–Rényi random graph
  - barabasi_albert:   Barabási–Albert scale-free graph
  - watts_strogatz:    Watts–Strogatz small-world graph
  - stochastic_block:  Stochastic Block Model (community structure)

Each returns a tuple (G_s, G_o, communities) where:
  - G_s:    propagation network adjacency (N×N, row=j→col=i convention)
  - G_o:    opinion influence network adjacency (row-normalized)
  - communities: dict mapping community_id → list of node indices
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from typing import Optional


def _row_normalize(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-normalize adjacency so each target node's in-edges sum to 1.

    adj[i, j] = weight of edge j → i (row i receives from column j).
    Normalization: for each target row i, Σ_j adj[i,j] = 1.
    """
    row_sums = adj.sum(axis=1, keepdims=True)
    return adj / np.maximum(row_sums, eps)


def _ensure_connected(adj: np.ndarray, rng: Generator) -> np.ndarray:
    """Ensure the graph is weakly connected by adding minimal edges."""
    n = adj.shape[0]
    # Use undirected version for connectivity check
    undirected = (adj + adj.T) > 0
    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        v = stack.pop()
        neighbors = np.where(undirected[v])[0]
        for u in neighbors:
            if not visited[u]:
                visited[u] = True
                stack.append(u)

    unvisited = np.where(~visited)[0]
    for v in unvisited:
        u = rng.integers(0, n)
        adj[v, u] = 1.0
        adj[u, v] = 1.0

    return adj


# ─────────────────────────────────────────────────────────────
# Generator functions
# ─────────────────────────────────────────────────────────────


def erdos_renyi(
    n: int = 500,
    p: float = 0.05,
    directed: bool = True,
    rng: Optional[Generator] = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Erdős–Rényi random graph: each edge exists independently with prob p.

    Args:
        n: Number of nodes.
        p: Edge probability.
        directed: If True, edges are directed (j→i and i→j independent).
        rng: NumPy random generator.

    Returns:
        (G_s, G_o, communities) — no community structure (single community).
    """
    if rng is None:
        rng = np.random.default_rng()

    adj = (rng.random((n, n)) < p).astype(np.float64)
    np.fill_diagonal(adj, 0.0)

    if not directed:
        adj = np.maximum(adj, adj.T)

    adj = _ensure_connected(adj, rng)

    # Add random weights in [0.3, 1.0] to existing edges
    mask = adj > 0
    adj[mask] = rng.uniform(0.3, 1.0, size=mask.sum())

    G_s = adj.copy()
    G_o = _row_normalize(adj.copy())

    communities = {0: list(range(n))}
    return G_s, G_o, communities


def barabasi_albert(
    n: int = 500,
    m: int = 5,
    directed: bool = True,
    rng: Optional[Generator] = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Barabási–Albert scale-free graph via preferential attachment.

    Args:
        n: Number of nodes.
        m: Edges per new node (initial clique size = m).
        directed: If True, new edges are directed from new→existing.
        rng: NumPy random generator.

    Returns:
        (G_s, G_o, communities) — single community.
    """
    if rng is None:
        rng = np.random.default_rng()

    adj = np.zeros((n, n), dtype=np.float64)

    # Start with m-node clique
    for i in range(m):
        for j in range(i + 1, m):
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    degree_sum = 2 * m * (m - 1) / 2  # total degree in initial clique

    # Preferential attachment
    degrees = np.zeros(n, dtype=np.float64)
    for i in range(m):
        degrees[i] = m - 1

    for new_node in range(m, n):
        # Choose m targets proportional to degree
        probs = degrees / max(degrees.sum(), 1e-8)
        targets = rng.choice(n, size=m, replace=False, p=probs)

        for t in targets:
            adj[new_node, t] = 1.0
            if not directed:
                adj[t, new_node] = 1.0

            degrees[t] += 1
            degrees[new_node] += 1

    adj = _ensure_connected(adj, rng)

    mask = adj > 0
    adj[mask] = rng.uniform(0.3, 1.0, size=mask.sum())

    G_s = adj.copy()
    G_o = _row_normalize(adj.copy())

    communities = {0: list(range(n))}
    return G_s, G_o, communities


def watts_strogatz(
    n: int = 500,
    k: int = 10,
    p_rewire: float = 0.10,
    rng: Optional[Generator] = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Watts–Strogatz small-world graph: ring lattice + random rewiring.

    Args:
        n: Number of nodes.
        k: Each node connects to k nearest neighbors (must be even).
        p_rewire: Probability of rewiring each edge.
        rng: NumPy random generator.

    Returns:
        (G_s, G_o, communities) — single community.
    """
    if rng is None:
        rng = np.random.default_rng()

    if k % 2 != 0:
        k += 1

    adj = np.zeros((n, n), dtype=np.float64)
    half_k = k // 2

    # Ring lattice: each node connects to k nearest neighbors (undirected)
    for i in range(n):
        for d in range(1, half_k + 1):
            j = (i + d) % n
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # Rewire
    for i in range(n):
        for d in range(1, half_k + 1):
            j = (i + d) % n
            if rng.random() < p_rewire:
                # Remove edge i→j
                adj[i, j] = 0.0
                adj[j, i] = 0.0
                # Pick new target (not self, not already connected)
                candidates = np.where((adj[i] == 0) & (np.arange(n) != i))[0]
                if len(candidates) > 0:
                    new_target = rng.choice(candidates)
                    adj[i, new_target] = 1.0
                    adj[new_target, i] = 1.0

    adj = _ensure_connected(adj, rng)

    mask = adj > 0
    adj[mask] = rng.uniform(0.3, 1.0, size=mask.sum())

    G_s = adj.copy()
    G_o = _row_normalize(adj.copy())

    communities = {0: list(range(n))}
    return G_s, G_o, communities


def stochastic_block(
    n: int = 500,
    n_blocks: int = 3,
    p_in: float = 0.15,
    p_out: float = 0.02,
    block_sizes: Optional[list[int]] = None,
    directed: bool = True,
    rng: Optional[Generator] = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Stochastic Block Model: dense within-block, sparse between-block.

    This is the most important network generator because community structure
    is central to cross-community flow (κ) and the silence spiral mechanism.

    Args:
        n: Total number of nodes.
        n_blocks: Number of communities.
        p_in: Edge probability within blocks.
        p_out: Edge probability between blocks.
        block_sizes: Optional explicit block sizes. If None, roughly equal.
        directed: If True, edges are directed.
        rng: NumPy random generator.

    Returns:
        (G_s, G_o, communities) — community structure is returned.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine block sizes
    if block_sizes is None:
        base = n // n_blocks
        remainder = n % n_blocks
        block_sizes = [base + (1 if i < remainder else 0) for i in range(n_blocks)]

    # Build block membership list
    membership = np.zeros(n, dtype=np.int32)
    communities: dict[int, list[int]] = {}
    idx = 0
    for b, size in enumerate(block_sizes):
        membership[idx:idx + size] = b
        communities[b] = list(range(idx, idx + size))
        idx += size

    # Generate edges with block-dependent probabilities
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            b_i = membership[i]
            b_j = membership[j]
            prob = p_in if b_i == b_j else p_out
            if rng.random() < prob:
                adj[i, j] = 1.0

    adj = _ensure_connected(adj, rng)

    mask = adj > 0
    adj[mask] = rng.uniform(0.3, 1.0, size=mask.sum())

    G_s = adj.copy()
    G_o = _row_normalize(adj.copy())

    return G_s, G_o, communities


# ─────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────

def generate_networks(
    network_type: str = "sbm",
    n: int = 500,
    rng: Optional[Generator] = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Generate networks by name.

    Args:
        network_type: "er", "ba", "ws", or "sbm".
        n: Number of nodes.
        rng: Random generator.
        **kwargs: Passed to the specific generator.

    Returns:
        (G_s, G_o, communities)
    """
    generators = {
        "er": erdos_renyi,
        "ba": barabasi_albert,
        "ws": watts_strogatz,
        "sbm": stochastic_block,
    }
    if network_type not in generators:
        raise ValueError(
            f"Unknown network type '{network_type}'. "
            f"Choose from: {list(generators.keys())}"
        )
    return generators[network_type](n=n, rng=rng, **kwargs)
