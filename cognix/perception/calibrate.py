"""Belief calibration: keep asserted confidence honest against outcomes.

The ledger records checkable predictions (a proposition, the confidence it
was asserted with, whether it turned out right) and uses them to dampen
overconfident assertions at assert time. Underconfident or untested claims
are left alone.
"""

import re

BIN_WIDTH = 0.2


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _normalize(proposition):
    return re.sub(r"\s+", " ", str(proposition).strip().lower())


class CalibrationLedger:
    """Predicted confidence vs observed outcomes, keyed by proposition."""

    def __init__(self, min_samples=5, strength=0.5):
        self.min_samples = max(1, int(min_samples))
        self.strength = _clamp01(strength)
        # normalized proposition -> list of (predicted, correct)
        self._outcomes = {}
        # how many assertions have been dampened so far
        self._dampened = 0

    def record_outcome(self, proposition, predicted, correct):
        """Log that a proposition asserted at `predicted` turned out right or not."""
        key = _normalize(proposition)
        self._outcomes.setdefault(key, []).append(
            (_clamp01(predicted), bool(correct))
        )
        return len(self._outcomes[key])

    def samples(self, proposition):
        return len(self._outcomes.get(_normalize(proposition), []))

    def hit_rate(self, proposition):
        """Observed fraction of correct outcomes, or None if never checked."""
        recs = self._outcomes.get(_normalize(proposition))
        if not recs:
            return None
        return sum(1 for _, ok in recs if ok) / len(recs)

    def adjusted(self, proposition, predicted):
        """Dampen an overconfident prediction toward the observed hit rate.

        No outcomes, too few outcomes, or a prediction at/below the observed
        hit rate all come back unchanged.
        """
        predicted = _clamp01(predicted)
        recs = self._outcomes.get(_normalize(proposition), [])
        if len(recs) < self.min_samples:
            return predicted
        rate = sum(1 for _, ok in recs if ok) / len(recs)
        if predicted <= rate:
            return predicted
        # partial pull toward the evidence; more samples, stronger pull
        weight = min(1.0, len(recs) / (2.0 * self.min_samples))
        out = predicted - self.strength * weight * (predicted - rate)
        self._dampened += 1
        return _clamp01(out)

    def stats(self):
        """Aggregate calibration numbers: brier score and per-bin hit rates."""
        allrecs = [r for recs in self._outcomes.values() for r in recs]
        n = len(allrecs)
        brier = 0.0
        if n:
            brier = sum(
                (p - (1.0 if ok else 0.0)) ** 2 for p, ok in allrecs
            ) / n
        bins = []
        count = int(1.0 / BIN_WIDTH)
        for i in range(count):
            lo, hi = i * BIN_WIDTH, (i + 1) * BIN_WIDTH
            bucket = [
                (p, ok) for p, ok in allrecs
                if lo <= p < hi or (i == count - 1 and p == 1.0)
            ]
            if bucket:
                hit = sum(1 for _, ok in bucket if ok) / len(bucket)
                mean = sum(p for p, _ in bucket) / len(bucket)
                bins.append({
                    "range": "%.1f-%.1f" % (lo, hi),
                    "n": len(bucket),
                    "mean_predicted": round(mean, 3),
                    "hit_rate": round(hit, 3),
                })
        return {
            "outcomes": n,
            "propositions": len(self._outcomes),
            "brier": round(brier, 4),
            "bins": bins,
            "dampened": self._dampened,
        }

    def to_dict(self):
        return {
            "min_samples": self.min_samples,
            "strength": self.strength,
            "dampened": self._dampened,
            "outcomes": [
                {
                    "proposition": key,
                    "records": [
                        {"predicted": p, "correct": ok} for p, ok in recs
                    ],
                }
                for key, recs in self._outcomes.items()
            ],
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        ledger = cls(
            min_samples=data.get("min_samples", 5),
            strength=data.get("strength", 0.5),
        )
        ledger._dampened = data.get("dampened", 0)
        for entry in data.get("outcomes", []):
            key = entry.get("proposition", "")
            ledger._outcomes[key] = [
                (_clamp01(r.get("predicted", 0.5)), bool(r.get("correct")))
                for r in entry.get("records", [])
            ]
        return ledger
