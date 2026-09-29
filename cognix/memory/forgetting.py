"""Forgetting curves: Ebbinghaus retention plus per-item rehearsal tracking."""

from __future__ import annotations

import math
from typing import Dict


# seconds in one day; the base time constant of the retention curve
_TAU = 86400.0


def ebbinghaus_retention(
    age_seconds: float, strength: float = 1.0, decay: float = 1.0
) -> float:
    """Probability of recall after age_seconds, in 0..1.

    R = exp(-decay * age / (strength * TAU)). Rehearsal raises strength,
    which slows the fall. A fresh item always returns 1.0.
    """
    if age_seconds <= 0:
        return 1.0
    strength = max(strength, 0.01)
    decay = max(decay, 0.0)
    value = math.exp(-decay * age_seconds / (strength * _TAU))
    return max(0.0, min(1.0, value))


def half_life(strength: float = 1.0, decay: float = 1.0) -> float:
    """Seconds until retention drops to 0.5 for the given strength."""
    strength = max(strength, 0.01)
    decay = max(decay, 1e-9)
    return math.log(2.0) * strength * _TAU / decay


class ForgettingCurve:
    """Tracks rehearsals per item id and reports retention over time."""

    # each rehearsal adds this much strength on top of the base 1.0
    REHEARSAL_GAIN = 0.5

    def __init__(self) -> None:
        self._rehearsals: Dict[str, int] = {}

    def record_rehearsal(self, item_id: str, count: int = 1) -> int:
        """Log rehearsals for an item; returns the new total."""
        if count < 1:
            raise ValueError("count must be >= 1")
        total = self._rehearsals.get(item_id, 0) + count
        self._rehearsals[item_id] = total
        return total

    def rehearsals(self, item_id: str) -> int:
        return self._rehearsals.get(item_id, 0)

    def strength(self, item_id: str) -> float:
        return 1.0 + self.REHEARSAL_GAIN * self._rehearsals.get(item_id, 0)

    def retention(
        self, item_id: str, age_seconds: float, decay: float = 1.0
    ) -> float:
        return ebbinghaus_retention(
            age_seconds, strength=self.strength(item_id), decay=decay
        )

    def half_life(self, item_id: str, decay: float = 1.0) -> float:
        return half_life(strength=self.strength(item_id), decay=decay)

    def reset(self, item_id: str) -> bool:
        if item_id in self._rehearsals:
            del self._rehearsals[item_id]
            return True
        return False

    def clear(self) -> None:
        self._rehearsals.clear()

    def weakest(self, n: int = 5) -> Dict[str, int]:
        """The n least-rehearsed tracked ids (candidates for forgetting)."""
        ordered = sorted(self._rehearsals.items(), key=lambda kv: kv[1])
        return dict(ordered[: max(0, n)])

    def to_dict(self) -> Dict[str, int]:
        return dict(self._rehearsals)

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "ForgettingCurve":
        curve = cls()
        curve._rehearsals = {str(k): int(v) for k, v in d.items()}
        return curve
