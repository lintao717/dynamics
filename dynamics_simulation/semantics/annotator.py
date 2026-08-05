"""[V2.0] LLM-based semantic annotator with caching.

Annotates CHECKED interaction texts with stance, emotion, topic,
and information signal values. Results are cached to disk to avoid
repeated LLM calls.

First version uses rule-based defaults (no live LLM). LLM integration
can be added by implementing _call_llm().
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from dynamics_simulation.semantics.schema import InteractionSemanticSignal


class SemanticAnnotator:
    """Annotates interaction texts with semantic signals.

    Cache: JSON file keyed by SHA-256 of (text, annotator_version).
    """

    def __init__(self, cache_path: Optional[Path] = None,
                 annotator_version: str = "v2.0-rule-based"):
        self._annotator_version = annotator_version
        self._cache: dict[str, InteractionSemanticSignal] = {}
        self._cache_path = cache_path
        if cache_path and cache_path.exists():
            self._load_cache()

    def _cache_key(self, text: str) -> str:
        payload = f"{text}|{self._annotator_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cache(self):
        if not self._cache_path:
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for key, val in raw.items():
                self._cache[key] = InteractionSemanticSignal(**val)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save_cache(self):
        if not self._cache_path:
            return
        raw = {}
        for key, sig in self._cache.items():
            raw[key] = {
                "interaction_id": sig.interaction_id,
                "user_id": sig.user_id,
                "step": sig.step,
                "stance": sig.stance,
                "stance_label": sig.stance_label,
                "emotion": sig.emotion,
                "arousal": sig.arousal,
                "info_evidence": sig.info_evidence,
                "info_emotion": sig.info_emotion,
                "official_info": sig.official_info,
                "credibility": sig.credibility,
                "topic": sig.topic,
                "topic_salience": sig.topic_salience,
                "annotator": sig.annotator,
                "annotator_version": sig.annotator_version,
                "confidence": sig.confidence,
            }
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    def annotate(self, interaction_id: str, user_id: str, step: int,
                 text: str) -> InteractionSemanticSignal:
        """Annotate one interaction. Returns cached result if available."""
        key = self._cache_key(text)
        if key in self._cache:
            sig = self._cache[key]
            # Return a copy with correct IDs
            return InteractionSemanticSignal(
                interaction_id=interaction_id,
                user_id=user_id,
                step=step,
                stance=sig.stance,
                stance_label=sig.stance_label,
                emotion=sig.emotion,
                arousal=sig.arousal,
                info_evidence=sig.info_evidence,
                info_emotion=sig.info_emotion,
                official_info=sig.official_info,
                credibility=sig.credibility,
                topic=sig.topic,
                topic_salience=sig.topic_salience,
                annotator=sig.annotator,
                annotator_version=sig.annotator_version,
                confidence=sig.confidence,
            )

        # Rule-based default (placeholder for LLM)
        sig = self._rule_based(text)
        result = InteractionSemanticSignal(
            interaction_id=interaction_id,
            user_id=user_id,
            step=step,
            annotator="rule-based",
            annotator_version=self._annotator_version,
            confidence=0.5,  # low confidence for rule-based
            **sig,
        )
        self._cache[key] = result
        return result

    def _rule_based(self, text: str) -> dict:
        """Simple keyword-based annotation (placeholder)."""
        text_lower = text.lower()

        # Stance: check for negation/agreement keywords
        support_words = ["支持", "同意", "赞成", "好", "赞", "确实", "对"]
        oppose_words = ["反对", "不同意", "假", "谣言", "骗", "不对", "错"]
        n_support = sum(1 for w in support_words if w in text_lower)
        n_oppose = sum(1 for w in oppose_words if w in text_lower)

        if n_support > n_oppose:
            stance = 0.5
            stance_label = "support"
        elif n_oppose > n_support:
            stance = -0.5
            stance_label = "oppose"
        else:
            stance = 0.0
            stance_label = "neutral"

        # Emotion: basic keyword detection
        anger_words = ["愤怒", "气", "怒", "恶心"]
        fear_words = ["怕", "担心", "恐惧", "慌"]
        joy_words = ["开心", "高兴", "哈哈", "笑"]
        sadness_words = ["难过", "伤心", "悲哀", "哭"]
        n_anger = sum(1 for w in anger_words if w in text_lower)
        n_fear = sum(1 for w in fear_words if w in text_lower)
        n_joy = sum(1 for w in joy_words if w in text_lower)
        n_sad = sum(1 for w in sadness_words if w in text_lower)
        max_emotion = max(n_anger, n_fear, n_joy, n_sad, 0)
        if max_emotion == 0:
            emotion = "neutral"
            arousal = 0.1
        elif n_anger == max_emotion:
            emotion = "anger"
            arousal = 0.6
        elif n_fear == max_emotion:
            emotion = "fear"
            arousal = 0.5
        elif n_joy == max_emotion:
            emotion = "joy"
            arousal = 0.5
        else:
            emotion = "sadness"
            arousal = 0.4

        return {
            "stance": stance,
            "stance_label": stance_label,
            "emotion": emotion,
            "arousal": arousal,
            "info_evidence": stance * 0.5,  # evidence aligned with stance
            "info_emotion": arousal,
            "official_info": 0.0,  # unknown
            "credibility": 0.5,    # unknown
            "topic": "general",
            "topic_salience": 0.5,
        }

    def _call_llm(self, text: str, prompt: str) -> dict:
        """Placeholder for LLM API call.
        Override in subclass for real LLM integration.
        """
        raise NotImplementedError("LLM integration not yet implemented")
