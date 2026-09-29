"""Attention subsystem: salience scoring, focus selection, arousal."""

from cognix.attention.salience import (
    WEIGHTS,
    SalienceSignal,
    batch_score,
    novelty_score,
    relevance_score,
    score_observation,
    surprise_score,
    urgency_score,
)
from cognix.attention.focus import Focus
from cognix.attention.arousal import Arousal
from cognix.attention.curiosity import CuriosityTracker

__all__ = [
    "WEIGHTS",
    "SalienceSignal",
    "batch_score",
    "novelty_score",
    "relevance_score",
    "surprise_score",
    "urgency_score",
    "score_observation",
    "Focus",
    "Arousal",
    "CuriosityTracker",
]
