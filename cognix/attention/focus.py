"""Focus: bandwidth-limited selection of what to attend to."""


class Focus:
    """Picks the top-k candidate observations above a salience threshold."""

    def __init__(self, bandwidth=3, threshold=0.3):
        self.bandwidth = max(1, int(bandwidth))
        self.threshold = float(threshold)

    def ranked(self, candidates):
        """(name, score) pairs above threshold, best first."""
        ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
        return [
            (name, score) for name, score in ordered if score >= self.threshold
        ]

    def select(self, candidates):
        """Return up to bandwidth names with score >= threshold, best first.

        Sorting is stable, so ties keep their input order.
        """
        return [name for name, _ in self.ranked(candidates)[: self.bandwidth]]

    def interrupt(self, current_best, candidate_score, margin=0.25):
        """True when a candidate captures attention from the current focus."""
        return candidate_score > current_best + margin

    def to_dict(self):
        return {"bandwidth": self.bandwidth, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, data):
        return cls(
            bandwidth=data.get("bandwidth", 3),
            threshold=data.get("threshold", 0.3),
        )
