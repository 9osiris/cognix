"""Intrinsic motivation: familiarity tracking and novelty bonuses.

The tracker counts exposures per keyword with exponential decay, so
topics seen often and recently feel familiar. Unfamiliar topics earn a
novelty bonus that nudges attention, and a streak of overly familiar
observations reports boredom, a signal the runtime can use to widen
exploration or take a break.
"""

import time
from typing import Any, Callable, Dict, List, Optional

from ..perception.pipeline import keywords


class CuriosityTracker:
    """Tracks topic familiarity; rewards novelty, flags boredom."""

    def __init__(
        self,
        decay_rate: float = 0.05,
        bonus_scale: float = 0.15,
        boredom_threshold: float = 0.8,
        boredom_window: int = 10,
        now: Optional[Callable[[], float]] = None,
    ):
        self.decay_rate = decay_rate
        self.bonus_scale = bonus_scale
        self.boredom_threshold = boredom_threshold
        self.boredom_window = max(1, boredom_window)
        self._now = now or time.time
        # keyword -> (exposure count, last seen timestamp)
        self._familiarity: Dict[str, List[float]] = {}
        # recent novelty scores, newest last
        self._history: List[float] = []

    def _decay_counts(self, moment: float) -> None:
        # fade old exposures so familiarity reflects recent experience
        for word, (count, seen) in list(self._familiarity.items()):
            age = max(0.0, moment - seen)
            faded = count * (1.0 - self.decay_rate) ** age
            if faded < 0.05:
                del self._familiarity[word]
            else:
                self._familiarity[word] = [faded, seen]

    def familiarity(self, text: str) -> float:
        """Mean exposure of the text's keywords, 0 when all new."""
        words = keywords(text)
        if not words:
            return 0.0
        self._decay_counts(self._now())
        total = sum(self._familiarity.get(w, [0.0, 0.0])[0] for w in words)
        # normalize: 3+ exposures of every keyword counts as fully familiar
        return min(1.0, total / (3.0 * len(words)))

    def novelty(self, text: str) -> float:
        """How new the text feels: 1 minus familiarity."""
        return 1.0 - self.familiarity(text)

    def bonus(self, text: str) -> float:
        """Attention bonus for novel text, scaled to stay small."""
        return self.bonus_scale * self.novelty(text)

    def expose(self, text: str) -> float:
        """Record seeing the text; returns its novelty before exposure."""
        novelty = self.novelty(text)
        moment = self._now()
        self._decay_counts(moment)
        for word in keywords(text):
            count, _ = self._familiarity.get(word, [0.0, moment])
            self._familiarity[word] = [count + 1.0, moment]
        self._history.append(novelty)
        self._history = self._history[-self.boredom_window:]
        return novelty

    def boredom(self) -> float:
        """Fraction of the recent window that felt familiar (0..1)."""
        if not self._history:
            return 0.0
        familiar = sum(1 for n in self._history if n < 0.3)
        return familiar / len(self._history)

    def is_bored(self) -> bool:
        """True when recent observations were mostly familiar."""
        return (
            len(self._history) >= self.boredom_window
            and self.boredom() >= self.boredom_threshold
        )

    def top_familiar(self, n: int = 5) -> List[str]:
        """Most exposed keywords, most familiar first."""
        ranked = sorted(
            self._familiarity.items(),
            key=lambda kv: kv[1][0],
            reverse=True,
        )
        return [word for word, _ in ranked[: max(0, n)]]

    def reset(self) -> None:
        """Forget all familiarity; everything feels new again."""
        self._familiarity.clear()
        self._history.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decay_rate": self.decay_rate,
            "bonus_scale": self.bonus_scale,
            "boredom_threshold": self.boredom_threshold,
            "boredom_window": self.boredom_window,
            "familiarity": {
                word: {"count": count, "seen": seen}
                for word, (count, seen) in self._familiarity.items()
            },
            "history": list(self._history),
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], now: Optional[Callable[[], float]] = None
    ) -> "CuriosityTracker":
        tracker = cls(
            decay_rate=data.get("decay_rate", 0.05),
            bonus_scale=data.get("bonus_scale", 0.15),
            boredom_threshold=data.get("boredom_threshold", 0.8),
            boredom_window=data.get("boredom_window", 10),
            now=now,
        )
        for word, entry in data.get("familiarity", {}).items():
            tracker._familiarity[word] = [
                float(entry.get("count", 0.0)),
                float(entry.get("seen", 0.0)),
            ]
        tracker._history = [float(n) for n in data.get("history", [])]
        return tracker
