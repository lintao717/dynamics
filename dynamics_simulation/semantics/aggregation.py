"""[V2.0-B] Per-window semantic aggregation.

Converts per-interaction semantic signals into step-level
opinion observation trajectories for dynamics calibration.
"""

from __future__ import annotations

import numpy as np
from collections import Counter

from dynamics_simulation.semantics.schema import (
    InteractionSemanticSignal, WindowSemanticAggregate,
)


def _polarization(signals: list[InteractionSemanticSignal]) -> float:
    """Bimodality index: 1 = strongly polarized, 0 = unimodal.

    Uses variance ratio: low variance of absolute values relative
    to overall variance indicates two opposing clusters.
    """
    if len(signals) < 2:
        return 0.0
    stances = np.array([s.stance for s in signals])
    if stances.std() < 1e-8:
        return 0.0
    return float(1.0 - np.abs(stances).std() / stances.std())


def _emotion_entropy(signals: list[InteractionSemanticSignal]) -> float:
    """Entropy of emotion distribution (normalized to [0, 1])."""
    if not signals:
        return 0.0
    counts = Counter(s.emotion for s in signals)
    total = len(signals)
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    max_entropy = np.log(max(len(counts), 2))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _topic_entropy(signals: list[InteractionSemanticSignal]) -> float:
    """Entropy of topic distribution."""
    if not signals:
        return 0.0
    counts = Counter(s.topic for s in signals)
    total = len(signals)
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    max_entropy = np.log(max(len(counts), 2))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def aggregate_window(signals: list[InteractionSemanticSignal],
                     step: int) -> WindowSemanticAggregate:
    """Aggregate per-interaction signals into window-level metrics.

    Args:
        signals: All annotated interactions in this time window.
        step: Simulation step index.

    Returns:
        WindowSemanticAggregate with distributional metrics.
    """
    n = len(signals)
    if n == 0:
        return WindowSemanticAggregate(
            step=step, n_interactions=0,
            stance_mean=0.0, stance_std=0.0,
            support_ratio=0.0, neutral_ratio=1.0, oppose_ratio=0.0,
            polarization=0.0,
            arousal_mean=0.0, dominant_emotion="neutral",
            emotion_entropy=0.0,
            evidence_net=0.0, official_alignment=0.0,
            credibility_mean=0.5,
        )

    stances = np.array([s.stance for s in signals])
    supports = np.sum(stances > 0.1)
    neutrals = np.sum(np.abs(stances) <= 0.1)
    opposes = np.sum(stances < -0.1)

    # Dominant emotion
    emotion_counts = Counter(s.emotion for s in signals)
    dominant = emotion_counts.most_common(1)[0][0]

    # Top topics
    topic_counts = Counter(s.topic for s in signals)
    top_topics = [t for t, _ in topic_counts.most_common(3)]

    return WindowSemanticAggregate(
        step=step,
        n_interactions=n,
        stance_mean=float(np.mean(stances)),
        stance_std=float(np.std(stances)),
        support_ratio=float(supports / n),
        neutral_ratio=float(neutrals / n),
        oppose_ratio=float(opposes / n),
        polarization=_polarization(signals),
        arousal_mean=float(np.mean([s.arousal for s in signals])),
        dominant_emotion=dominant,
        emotion_entropy=_emotion_entropy(signals),
        evidence_net=float(np.mean([s.info_evidence for s in signals])),
        official_alignment=float(np.mean([s.official_info for s in signals])),
        credibility_mean=float(np.mean([s.credibility for s in signals])),
        top_topics=top_topics,
        topic_entropy=_topic_entropy(signals),
    )
