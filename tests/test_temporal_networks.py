"""Tests for temporal network providers — no future leakage."""

from datetime import datetime, timezone
import numpy as np
import pytest
from dynamics_simulation.data.schema import (
    EventCase, RootPost, InteractionRecord,
)
from dynamics_simulation.data.indexing import NodeIndex
from dynamics_simulation.data.timegrid import TimeGrid
from dynamics_simulation.data.networks import (
    ReplayNetworkMode,
    NetworkSnapshot,
    build_network_provider,
)


def _make_case():
    """Two interactions from two users at different times."""
    root = RootPost(
        post_id="ev1", user_id="root",
        timestamp=datetime(2020, 1, 1, 8, 0, tzinfo=timezone.utc),
        text="root", label="fake", expert_analysis=None,
    )
    # u1 interacts at 8:30 → step 1
    c1 = InteractionRecord(
        interaction_id="c1", root_post_id="ev1", user_id="u1",
        timestamp=datetime(2020, 1, 1, 8, 30, tzinfo=timezone.utc),
        kind="comment", text="c1",
    )
    # u2 interacts at 9:30 → step 2
    c2 = InteractionRecord(
        interaction_id="c2", root_post_id="ev1", user_id="u2",
        timestamp=datetime(2020, 1, 1, 9, 30, tzinfo=timezone.utc),
        kind="repost", text="c2",
    )
    return EventCase(
        case_id="ev1", source_dataset="CHECKED",
        root=root, interactions=(c1, c2),
    )


# ── Mode enum ──

def test_replay_network_mode_values():
    assert ReplayNetworkMode.BROADCAST.value == "broadcast"
    assert ReplayNetworkMode.CUMULATIVE_INTERACTION.value == "cumulative_interaction"
    assert ReplayNetworkMode.ORACLE_STATIC.value == "oracle_static"


# ── Broadcast mode: all G_s zero ──

def test_broadcast_g_s_is_all_zero():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    provider = build_network_provider(case, index, grid,
                                      mode=ReplayNetworkMode.BROADCAST)

    for step in range(grid.total_steps + 1):
        snap = provider.snapshot_at(step)
        assert np.all(snap.G_s == 0), f"G_s not zero at step {step}"


# ── Cumulative interaction: no future leakage ──

def test_cumulative_u1_edge_absent_before_interaction():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    provider = build_network_provider(
        case, index, grid,
        mode=ReplayNetworkMode.CUMULATIVE_INTERACTION,
    )

    u1_idx = index.user_to_idx["u1"]
    root_idx = index.user_to_idx["root"]

    # step 0: u1 has not interacted yet → no edge
    snap0 = provider.snapshot_at(0)
    assert snap0.G_s[u1_idx, root_idx] == 0

    # step 1: u1 interacted at 8:30 → edge should now exist
    snap1 = provider.snapshot_at(1)
    assert snap1.G_s[u1_idx, root_idx] > 0


def test_cumulative_u2_edge_absent_at_step_1():
    """u2 interacts at step 2; edge must be absent at step 1."""
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    provider = build_network_provider(
        case, index, grid,
        mode=ReplayNetworkMode.CUMULATIVE_INTERACTION,
    )

    u2_idx = index.user_to_idx["u2"]
    root_idx = index.user_to_idx["root"]

    snap1 = provider.snapshot_at(1)
    assert snap1.G_s[u2_idx, root_idx] == 0

    snap2 = provider.snapshot_at(2)
    assert snap2.G_s[u2_idx, root_idx] > 0


def test_cumulative_edge_direction():
    """G[dst, src] convention: receiver row, source column.
    Root→u1 means G_s[u1, root] > 0, NOT G_s[root, u1]."""
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    provider = build_network_provider(
        case, index, grid,
        mode=ReplayNetworkMode.CUMULATIVE_INTERACTION,
    )

    u1_idx = index.user_to_idx["u1"]
    root_idx = index.user_to_idx["root"]

    snap = provider.snapshot_at(1)
    # u1 (receiver) gets edge from root (source)
    assert snap.G_s[u1_idx, root_idx] > 0
    # root should NOT get edge from u1
    assert snap.G_s[root_idx, u1_idx] == 0


# ── G_o row normalization ──

def test_go_row_sums_are_zero_or_one():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    for mode in [ReplayNetworkMode.CUMULATIVE_INTERACTION,
                 ReplayNetworkMode.ORACLE_STATIC]:
        provider = build_network_provider(case, index, grid, mode=mode)
        snap = provider.snapshot_at(grid.total_steps)
        row_sums = snap.G_o.sum(axis=1)
        for s in row_sums:
            assert s == pytest.approx(0.0, abs=1e-6) or s == pytest.approx(1.0, abs=1e-6), \
                f"G_o row sum {s} not in {{0, 1}}"


# ── oracle_static: all edges from step 0 ──

def test_oracle_static_edges_from_step_zero():
    case = _make_case()
    index = NodeIndex.from_case(case)
    grid = TimeGrid.from_case(case, step_hours=1.0, tail_steps=1)
    provider = build_network_provider(
        case, index, grid,
        mode=ReplayNetworkMode.ORACLE_STATIC,
    )

    u1_idx = index.user_to_idx["u1"]
    u2_idx = index.user_to_idx["u2"]
    root_idx = index.user_to_idx["root"]

    snap0 = provider.snapshot_at(0)
    assert snap0.G_s[u1_idx, root_idx] > 0  # already present!
    assert snap0.G_s[u2_idx, root_idx] > 0  # already present!


# ── NetworkSnapshot validation ──

def test_network_snapshot_rejects_nan():
    with pytest.raises(ValueError):
        NetworkSnapshot(
            G_s=np.array([[np.nan, 0.], [0., 0.]]),
            G_o=np.eye(2),
            communities={0: [0, 1]},
        )


def test_network_snapshot_rejects_negative():
    with pytest.raises(ValueError):
        NetworkSnapshot(
            G_s=np.array([[1., -1.], [0., 0.]]),
            G_o=np.eye(2),
            communities={0: [0, 1]},
        )


def test_network_snapshot_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        NetworkSnapshot(
            G_s=np.eye(3),
            G_o=np.eye(2),
            communities={0: [0, 1]},
        )
