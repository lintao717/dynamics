"""[V2.0] Interaction semantic signal schema.

Per-interaction annotated semantics derived from comment/repost text.
All values are bounded for safe dynamics input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class InteractionSemanticSignal:
    """Semantic annotation for one interaction record.

    All stance/emotion values validated to be in-bounds.
    """

    interaction_id: str
    user_id: str
    step: int  # simulation step index

    # ── Stance ──
    stance: float  # [-1, 1]: -1=oppose, 0=neutral, 1=support
    stance_label: str  # "support" | "neutral" | "oppose"

    # ── Emotion ──
    emotion: str  # "anger" | "fear" | "joy" | "sadness" | "surprise" | "neutral"
    arousal: float  # [0, 1]: emotional intensity

    # ── Information signals (for dynamics input) ──
    info_evidence: float  # [-1, 1]: direction of evidence presented
    info_emotion: float   # [0, 1]: emotional intensity carried in content
    official_info: float  # [-1, 1]: alignment with official narrative
    credibility: float    # [0, 1]: perceived credibility

    # ── Topic ──
    topic: str  # e.g. "information_transparency", "government_response"
    topic_salience: float  # [0, 1]: how prominently this topic is discussed

    # ── Provenance ──
    annotator: str = "unknown"  # "llm:gpt-4", "llm:claude", "human", etc.
    annotator_version: str = ""  # model version or prompt hash
    confidence: float = 1.0  # [0, 1]: annotation confidence

    def __post_init__(self):
        """Validate all bounded fields."""
        for name, low, high in [
            ("stance", -1.0, 1.0),
            ("arousal", 0.0, 1.0),
            ("info_evidence", -1.0, 1.0),
            ("info_emotion", 0.0, 1.0),
            ("official_info", -1.0, 1.0),
            ("credibility", 0.0, 1.0),
            ("topic_salience", 0.0, 1.0),
            ("confidence", 0.0, 1.0),
        ]:
            val = getattr(self, name)
            if not (low <= val <= high):
                raise ValueError(
                    f"{name}={val} outside [{low}, {high}]"
                )


@dataclass
class WindowSemanticAggregate:
    """Per-window aggregated semantics for dynamics input."""

    step: int
    n_interactions: int

    # ── Stance distribution ──
    stance_mean: float  # mean stance in window
    stance_std: float
    support_ratio: float  # fraction with stance > 0
    neutral_ratio: float
    oppose_ratio: float   # fraction with stance < 0
    polarization: float   # bimodality index

    # ── Emotion ──
    arousal_mean: float
    dominant_emotion: str
    emotion_entropy: float  # diversity of emotions

    # ── Information signals ──
    evidence_net: float   # net evidence direction
    official_alignment: float  # mean alignment with official narrative
    credibility_mean: float

    # ── Topics ──
    top_topics: list[str] = field(default_factory=list)
    topic_entropy: float = 0.0
