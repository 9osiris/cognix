"""Arousal: a 0..1 activation level driving processing depth."""

import math
import time


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


class Arousal:
    """Activation level that rises with salience and decays to baseline.

    processing_depth maps the level to 1, 2 or 3, telling perception how
    hard to work on the current observation.
    """

    def __init__(self, baseline=0.5, rise=0.4, decay_rate=0.1, now=None):
        self.baseline = _clamp01(baseline)
        self.rise = max(0.0, float(rise))
        self.decay_rate = max(0.0, float(decay_rate))
        self._now = now or time.monotonic
        self._level = self.baseline
        self.last_active = self._now()

    def update(self, salience):
        """Raise the level proportionally to salience, clamped to 0..1."""
        self._level = _clamp01(self._level + self.rise * max(0.0, salience))
        self.last_active = self._now()

    def tick(self, dt=1.0):
        """Decay the level toward baseline over dt seconds."""
        dt = max(0.0, float(dt))
        gap = self._level - self.baseline
        self._level = self.baseline + gap * math.exp(-self.decay_rate * dt)
        self.last_active = self._now()

    def level(self):
        return self._level

    def reset(self):
        """Drop the level back to baseline."""
        self._level = self.baseline
        self.last_active = self._now()

    def processing_depth(self):
        """1 if level < .35, 2 if < .7, else 3."""
        if self._level < 0.35:
            return 1
        if self._level < 0.7:
            return 2
        return 3

    def to_dict(self):
        return {
            "baseline": self.baseline,
            "rise": self.rise,
            "decay_rate": self.decay_rate,
            "level": self._level,
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            baseline=data.get("baseline", 0.5),
            rise=data.get("rise", 0.4),
            decay_rate=data.get("decay_rate", 0.1),
        )
        obj._level = _clamp01(data.get("level", obj.baseline))
        return obj
