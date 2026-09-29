"""Salience scoring: how much an observation deserves attention.

Four independent components (novelty, relevance, surprise, urgency) are
each computed by their own function, then combined with fixed weights
into a single 0..1 score.
"""

from cognix.perception.pipeline import (
    URGENCY_MARKERS,
    find_urgency_markers,
    keywords,
    mentions_date,
)

# component weights used by score_observation
WEIGHTS = {
    "novelty": 0.3,
    "relevance": 0.3,
    "surprise": 0.2,
    "urgency": 0.2,
}

_RECENT_LIMIT = 50


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _jaccard(first, second):
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


def _episode_text(episode):
    for attr in ("text", "content", "summary", "observation"):
        value = getattr(episode, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(episode, str):
        return episode
    if isinstance(episode, dict):
        for key in ("text", "content", "summary"):
            value = episode.get(key)
            if isinstance(value, str) and value:
                return value
    return str(episode)


def _recent_episodes(episodic, limit=_RECENT_LIMIT):
    if episodic is None:
        return []
    if hasattr(episodic, "recent"):
        return list(episodic.recent(limit))
    return list(episodic)[-limit:]


def _novelty_detail(text, episodic):
    words = set(keywords(text))
    if not words:
        return 0.0, "no keywords in text"
    episodes = _recent_episodes(episodic)
    if not episodes:
        return 1.0, "no prior episodes to compare against"
    best = 0.0
    for episode in episodes:
        similarity = _jaccard(words, set(keywords(_episode_text(episode))))
        if similarity > best:
            best = similarity
    return 1.0 - best, "max keyword overlap %.2f over %d episodes" % (
        best,
        len(episodes),
    )


def novelty_score(text, episodic):
    """1 minus the max keyword similarity to recent episodes (last 50)."""
    return _novelty_detail(text, episodic)[0]


def _relevance_detail(text, goals):
    words = set(keywords(text))
    if not words:
        return 0.0, "no keywords in text"
    goals = list(goals or ())
    if not goals:
        return 0.0, "no goals to compare against"
    best = 0.0
    best_goal = None
    for goal in goals:
        goal_words = set(keywords(getattr(goal, "description", "") or ""))
        if not goal_words:
            continue
        overlap = len(words & goal_words) / len(goal_words)
        if overlap > best:
            best = overlap
            best_goal = getattr(goal, "description", "")
    reason = "best goal overlap %.2f" % best
    if best_goal:
        reason += " (%s)" % best_goal[:60]
    return best, reason


def relevance_score(text, goals):
    """Keyword overlap between text and goal descriptions, best goal wins."""
    return _relevance_detail(text, goals)[0]


def _surprise_detail(text, beliefs):
    if beliefs is None:
        return 1.0, "no beliefs to compare against"
    best = 0.0
    matched = None
    seen = set()
    for word in keywords(text):
        for belief in beliefs.query(word):
            proposition = getattr(belief, "proposition", "")
            if proposition in seen:
                continue
            seen.add(proposition)
            confidence = float(getattr(belief, "confidence", 0.0))
            if confidence > best:
                best = confidence
                matched = proposition
    reason = "no matching beliefs" if matched is None else "matches %r at %.2f" % (
        matched[:60],
        best,
    )
    return 1.0 - best, reason


def surprise_score(text, beliefs):
    """1 minus the max confidence among beliefs sharing keywords with text."""
    return _surprise_detail(text, beliefs)[0]


def _urgency_detail(text):
    markers = find_urgency_markers(text)
    score = sum(URGENCY_MARKERS[marker] for marker in markers)
    dated = mentions_date(text)
    if dated:
        score += 0.25
    parts = []
    if markers:
        parts.append("markers: %s" % ", ".join(markers))
    if dated:
        parts.append("date mention")
    reason = "; ".join(parts) if parts else "no urgency markers"
    return _clamp01(score), reason


def urgency_score(text):
    """Weighted urgency markers plus date/deadline mentions, clamped 0..1."""
    return _urgency_detail(text)[0]


class SalienceSignal:
    """One named salience component with its score and a human reason."""

    def __init__(self, name, score, reason=""):
        self.name = name
        self.score = _clamp01(score)
        self.reason = reason

    def __repr__(self):
        return "SalienceSignal(%r, %.3f, %r)" % (self.name, self.score, self.reason)

    def __eq__(self, other):
        return (
            isinstance(other, SalienceSignal)
            and self.name == other.name
            and self.score == other.score
            and self.reason == other.reason
        )

    def to_dict(self):
        return {"name": self.name, "score": self.score, "reason": self.reason}

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["score"], data.get("reason", ""))


def score_observation(text, *, goals=(), beliefs=None, episodic=None, now=None):
    """Score an observation, returning (total, [SalienceSignal, ...]).

    Total is the weighted sum of the four components, clamped to 0..1.
    The now callable is accepted for interface consistency; scoring is
    stateless and does not use wall-clock time.
    """
    _ = now
    details = {
        "novelty": _novelty_detail(text, episodic),
        "relevance": _relevance_detail(text, goals),
        "surprise": _surprise_detail(text, beliefs),
        "urgency": _urgency_detail(text),
    }
    signals = [
        SalienceSignal(name, score, reason)
        for name, (score, reason) in details.items()
    ]
    total = sum(WEIGHTS[name] * score for name, (score, _) in details.items())
    return _clamp01(total), signals


def batch_score(texts, *, goals=(), beliefs=None, episodic=None, now=None):
    """Score many observations, returning (text, total, signals) best first."""
    scored = [
        (text, *score_observation(text, goals=goals, beliefs=beliefs,
                                 episodic=episodic, now=now))
        for text in texts
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
