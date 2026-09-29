"""Offline replay: re-activate salient episodes to strengthen memory.

During idle cycles the runtime replays a small set of salient episodes,
hippocampal style. Episodes that share tags get linked, their shared
tags become stronger semantic relations, and matching working memory
items get rehearsed. Replay never creates new beliefs or goals.
"""

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..perception.pipeline import keywords
from .episodic import EpisodicMemory, Episode
from .semantic import SemanticMemory
from .working import WorkingMemory


class ReplayPolicy:
    """Knobs for one replay pass."""

    def __init__(
        self,
        budget: int = 5,
        min_salience: float = 0.4,
        recency_tau: float = 86400.0,
        link_threshold: float = 0.3,
        strengthen_amount: float = 0.05,
        rehearse_boost: float = 0.2,
    ):
        self.budget = max(1, budget)
        self.min_salience = min_salience
        self.recency_tau = max(1.0, recency_tau)
        self.link_threshold = link_threshold
        self.strengthen_amount = strengthen_amount
        self.rehearse_boost = rehearse_boost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget": self.budget,
            "min_salience": self.min_salience,
            "recency_tau": self.recency_tau,
            "link_threshold": self.link_threshold,
            "strengthen_amount": self.strengthen_amount,
            "rehearse_boost": self.rehearse_boost,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayPolicy":
        return cls(
            budget=data.get("budget", 5),
            min_salience=data.get("min_salience", 0.4),
            recency_tau=data.get("recency_tau", 86400.0),
            link_threshold=data.get("link_threshold", 0.3),
            strengthen_amount=data.get("strengthen_amount", 0.05),
            rehearse_boost=data.get("rehearse_boost", 0.2),
        )


def _recency_weight(age: float, tau: float) -> float:
    # exponential falloff so fresh episodes replay first
    return math.exp(-max(0.0, age) / tau)


def _tag_overlap(a: Episode, b: Episode) -> float:
    # jaccard similarity over tag sets
    sa, sb = set(a.tags), set(b.tags)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _keyword_overlap(a: Episode, b: Episode) -> float:
    # jaccard similarity over summary keywords
    wa, wb = set(keywords(a.summary)), set(keywords(b.summary))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def select_episodes(
    episodic: EpisodicMemory,
    policy: Optional[ReplayPolicy] = None,
    now: Optional[Callable[[], float]] = None,
) -> List[Episode]:
    """Pick up to budget episodes worth replaying.

    Score is salience times recency weight; episodes below min_salience
    never qualify. Returns best first.
    """
    policy = policy or ReplayPolicy()
    moment = now() if now else time.time()
    scored: List[Tuple[float, Episode]] = []
    for episode in episodic.recent(200):
        if episode.salience < policy.min_salience:
            continue
        score = episode.salience * _recency_weight(
            episode.age(moment), policy.recency_tau
        )
        scored.append((score, episode))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [episode for _, episode in scored[: policy.budget]]


def _rehearse_matches(
    working: WorkingMemory, episode: Episode, boost: float
) -> int:
    # rehearse working items that share keywords with the episode
    words = set(keywords(episode.summary))
    rehearsed = 0
    for item in working.items():
        item_words = set(keywords(item.content))
        if words & item_words and working.rehearse(item.id, boost=boost):
            rehearsed += 1
    return rehearsed


def _strengthen_tags(
    semantic: SemanticMemory,
    a: Episode,
    b: Episode,
    amount: float,
) -> int:
    # shared tags become stronger co-occurrence relations
    shared = set(a.tags) & set(b.tags)
    strengthened = 0
    for tag in shared:
        relation = semantic.add_relation(
            tag, "co-occurs-with", tag, confidence=0.4, source="replay"
        )
        if semantic.strengthen(tag, "co-occurs-with", tag, amount):
            strengthened += 1
        elif relation is not None:
            strengthened += 1
    return strengthened


def replay(
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    working: Optional[WorkingMemory] = None,
    policy: Optional[ReplayPolicy] = None,
    now: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
    """Run one replay pass; returns a stats dict.

    Selected episodes are paired up: pairs with enough tag or keyword
    overlap get linked, their shared tags are strengthened in semantic
    memory, and matching working items are rehearsed.
    """
    policy = policy or ReplayPolicy()
    chosen = select_episodes(episodic, policy, now)
    stats = {
        "replayed": len(chosen),
        "links_added": 0,
        "relations_strengthened": 0,
        "items_rehearsed": 0,
        "pairs_examined": 0,
    }
    for i, first in enumerate(chosen):
        if working is not None:
            stats["items_rehearsed"] += _rehearse_matches(
                working, first, policy.rehearse_boost
            )
        for second in chosen[i + 1:]:
            stats["pairs_examined"] += 1
            overlap = max(_tag_overlap(first, second),
                          _keyword_overlap(first, second))
            if overlap < policy.link_threshold:
                continue
            if episodic.link(first.id, second.id, "replayed-with"):
                stats["links_added"] += 1
            stats["relations_strengthened"] += _strengthen_tags(
                semantic, first, second, policy.strengthen_amount
            )
    return stats
