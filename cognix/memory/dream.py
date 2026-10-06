"""Dream cycle: seeded recombination of episodes into dream traces.

Replay strengthens what already happened. Dreaming invents: it takes
episodes that are individually salient but unlinked, recombines them
into dream traces, and proposes low-confidence insights that relate
the two halves. Dreams are recorded as episodes tagged "dream", so
the dream journal is just the episodic store. Dream beliefs always
carry source "dream" and stay below normal belief confidence, so the
rest of the runtime treats them as hypotheses, not facts.
"""

import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..perception.pipeline import keywords
from .episodic import EpisodicMemory, Episode
from .replay import ReplayPolicy, select_episodes
from .semantic import SemanticMemory


class DreamPolicy:
    """Knobs for one dream cycle."""

    def __init__(
        self,
        budget: int = 8,
        dreams: int = 3,
        min_salience: float = 0.4,
        seed: Optional[int] = None,
        insight_confidence: float = 0.3,
    ):
        self.budget = max(2, budget)
        self.dreams = max(1, dreams)
        self.min_salience = min_salience
        self.seed = seed
        self.insight_confidence = max(0.0, min(0.5, insight_confidence))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget": self.budget,
            "dreams": self.dreams,
            "min_salience": self.min_salience,
            "seed": self.seed,
            "insight_confidence": self.insight_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DreamPolicy":
        return cls(
            budget=data.get("budget", 8),
            dreams=data.get("dreams", 3),
            min_salience=data.get("min_salience", 0.4),
            seed=data.get("seed"),
            insight_confidence=data.get("insight_confidence", 0.3),
        )


def _already_linked(episodic: EpisodicMemory, a: Episode, b: Episode) -> bool:
    return any(other.id == b.id for other, _ in episodic.neighbors(a.id))


def _pick_pair_words(a: Episode, b: Episode) -> Tuple[Optional[str], Optional[str]]:
    # first keyword of each episode not present in the other, for the insight
    wa = [w for w in keywords(a.summary) if w not in keywords(b.summary)]
    wb = [w for w in keywords(b.summary) if w not in keywords(a.summary)]
    return (wa[0] if wa else None, wb[0] if wb else None)


def _dream_summary(a: Episode, b: Episode) -> str:
    # a dream trace reads like the two episodes bled into each other
    return "dream: %s ... then %s" % (
        a.summary.rstrip("."),
        b.summary.rstrip("."),
    )


def dream(
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    beliefs=None,
    policy: Optional[DreamPolicy] = None,
    now: Optional[Callable[[], float]] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Run one dream cycle; returns a stats dict.

    Salient unlinked episodes are paired at random. Each pair becomes a
    dream trace episode, a low-confidence insight belief, and a proposed
    semantic relation between one keyword from each half. With a seed the
    whole cycle is deterministic.
    """
    policy = policy or DreamPolicy()
    moment = now() if now else time.time()
    rng = rng if rng is not None else random.Random(policy.seed)

    candidates = select_episodes(
        episodic,
        ReplayPolicy(budget=policy.budget, min_salience=policy.min_salience),
        now=lambda: moment,
    )
    stats: Dict[str, Any] = {
        "dreams": 0,
        "insights": 0,
        "links_added": 0,
        "relations_proposed": 0,
        "pairs_examined": 0,
        "dream_ids": [],
    }
    if len(candidates) < 2:
        return stats

    order = list(candidates)
    rng.shuffle(order)
    # every unique pairing, shuffled: dreaming takes the first `dreams`
    # pairs that are not already linked, so one replay link cannot
    # starve the whole cycle
    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    rng.shuffle(pairs)

    for first, second in pairs:
        if stats["dreams"] >= policy.dreams:
            break
        stats["pairs_examined"] += 1
        if _already_linked(episodic, first, second):
            continue
        word_a, word_b = _pick_pair_words(first, second)
        shared = sorted(set(first.tags) & set(second.tags))
        dream_ep = episodic.record(
            summary=_dream_summary(first, second),
            detail={
                "pair": [first.id, second.id],
                "source_episodes": [first.summary[:80], second.summary[:80]],
                "insight_words": [word_a, word_b],
            },
            tags=tuple(["dream"] + shared),
            salience=round(0.8 * (first.salience + second.salience) / 2.0, 3),
            source="dream",
        )
        stats["dream_ids"].append(dream_ep.id)
        stats["dreams"] += 1
        if episodic.link(first.id, second.id, "dreamed-with"):
            stats["links_added"] += 1
        if beliefs is not None and word_a and word_b:
            proposition = "dream insight: %s may relate to %s" % (word_a, word_b)
            belief = beliefs.assert_belief(
                proposition,
                policy.insight_confidence,
                source="dream",
            )
            if belief.confidence <= 0.5 + 1e-9:
                stats["insights"] += 1
            semantic.add_relation(
                word_a,
                "dream-related-to",
                word_b,
                confidence=policy.insight_confidence,
                source="dream",
            )
            stats["relations_proposed"] += 1
    return stats
