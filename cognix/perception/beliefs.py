"""Beliefs: asserted propositions with confidence and provenance."""

import re
import time


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _normalize(proposition):
    return re.sub(r"\s+", " ", str(proposition).strip().lower())


def _split_proposition(proposition):
    """Split into (subject, predicate, object) on whitespace."""
    tokens = _normalize(proposition).split()
    if len(tokens) >= 3:
        return tokens[0], tokens[1], " ".join(tokens[2:])
    if len(tokens) == 2:
        return tokens[0], tokens[1], ""
    if tokens:
        return tokens[0], "", ""
    return "", "", ""


def _contradicts(first, second):
    """Same subject and predicate but a different object."""
    subj_a, pred_a, obj_a = _split_proposition(first)
    subj_b, pred_b, obj_b = _split_proposition(second)
    return (
        bool(subj_a)
        and subj_a == subj_b
        and pred_a == pred_b
        and obj_a != obj_b
    )


class Belief:
    """A single proposition with confidence, sources and timestamps."""

    def __init__(
        self,
        proposition,
        confidence=0.5,
        source="unknown",
        created=None,
        updated=None,
        sources=(),
        conflict=False,
    ):
        self.proposition = str(proposition).strip()
        self.confidence = _clamp01(confidence)
        self.source = source
        self.sources = list(dict.fromkeys([source] + list(sources or ())))
        self.created = created if created is not None else time.time()
        self.updated = updated if updated is not None else self.created
        self.conflict = bool(conflict)

    def __repr__(self):
        return "Belief(%r, %.3f)" % (self.proposition, self.confidence)

    def __eq__(self, other):
        return isinstance(other, Belief) and self.to_dict() == other.to_dict()

    def to_dict(self):
        return {
            "proposition": self.proposition,
            "confidence": self.confidence,
            "source": self.source,
            "sources": list(self.sources),
            "created": self.created,
            "updated": self.updated,
            "conflict": self.conflict,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["proposition"],
            confidence=data.get("confidence", 0.5),
            source=data.get("source", "unknown"),
            created=data.get("created"),
            updated=data.get("updated"),
            sources=data.get("sources", ()),
            conflict=data.get("conflict", False),
        )


class BeliefStore:
    """Holds beliefs keyed by normalized proposition text."""

    # how much a contradiction lowers each involved confidence
    CONTRADICTION_PENALTY = 0.25
    # blend factor when re-asserting an existing proposition
    REVISION_BLEND = 0.5

    def __init__(self, now=None):
        self._now = now or time.time
        self._beliefs = {}

    def __len__(self):
        return len(self._beliefs)

    def __contains__(self, proposition):
        return _normalize(proposition) in self._beliefs

    def assert_belief(self, proposition, confidence, source="unknown"):
        """Assert a proposition, revising or flagging conflicts as needed.

        Re-asserting an existing proposition blends the new confidence in.
        A new proposition contradicting an existing one lowers both and
        marks them as in conflict.
        """
        key = _normalize(proposition)
        moment = self._now()
        existing = self._beliefs.get(key)
        if existing is not None:
            existing.confidence = _clamp01(
                existing.confidence
                + (confidence - existing.confidence) * self.REVISION_BLEND
            )
            if source not in existing.sources:
                existing.sources.append(source)
            existing.updated = moment
            return existing
        belief = Belief(
            proposition.strip(),
            confidence=confidence,
            source=source,
            created=moment,
            updated=moment,
        )
        for other in self._beliefs.values():
            if _contradicts(belief.proposition, other.proposition):
                belief.confidence = max(
                    0.0, belief.confidence - self.CONTRADICTION_PENALTY
                )
                other.confidence = max(
                    0.0, other.confidence - self.CONTRADICTION_PENALTY
                )
                belief.conflict = True
                other.conflict = True
                other.updated = moment
        self._beliefs[key] = belief
        return belief

    def retract(self, proposition):
        """Remove a belief, returning True if it existed."""
        return self._beliefs.pop(_normalize(proposition), None) is not None

    def get(self, proposition):
        """Return the Belief for a proposition, or None."""
        return self._beliefs.get(_normalize(proposition))

    def query(self, keyword, min_confidence=0.0):
        """Beliefs whose proposition contains keyword, strongest first."""
        word = keyword.lower()
        matches = [
            belief
            for belief in self._beliefs.values()
            if word in belief.proposition.lower()
            and belief.confidence >= min_confidence
        ]
        matches.sort(key=lambda belief: (-belief.confidence, belief.proposition))
        return matches

    def contradictions(self):
        """All pairs of beliefs that contradict each other."""
        items = list(self._beliefs.values())
        pairs = []
        for index in range(len(items)):
            for other in items[index + 1:]:
                if _contradicts(items[index].proposition, other.proposition):
                    pairs.append((items[index], other))
        return pairs

    def strongest(self, n=10):
        """The n highest-confidence beliefs."""
        ranked = sorted(
            self._beliefs.values(),
            key=lambda belief: (-belief.confidence, belief.proposition),
        )
        return ranked[: max(0, n)]

    def clear(self):
        """Remove all beliefs."""
        self._beliefs.clear()

    def merge(self, other):
        """Fold another store's beliefs in, keeping sources and conflicts."""
        for belief in other._beliefs.values():
            merged = self.assert_belief(
                belief.proposition, belief.confidence, belief.source
            )
            for source in belief.sources:
                if source not in merged.sources:
                    merged.sources.append(source)
            merged.conflict = merged.conflict or belief.conflict
            merged.updated = max(merged.updated, belief.updated)
        return self

    def to_dict(self):
        return {"beliefs": [belief.to_dict() for belief in self._beliefs.values()]}

    @classmethod
    def from_dict(cls, data, now=None):
        store = cls(now=now)
        for item in data.get("beliefs", []):
            belief = Belief.from_dict(item)
            store._beliefs[_normalize(belief.proposition)] = belief
        return store
