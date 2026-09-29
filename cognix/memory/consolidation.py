"""Consolidation: move working memory into episodes, mine episodes into concepts."""

from __future__ import annotations

import itertools
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .episodic import STOPWORDS, tokenize
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .working import WorkingMemory, WorkingItem


@dataclass
class ConsolidationPolicy:
    """Tuning knobs for one consolidation pass."""

    activation_threshold: float = 0.6
    min_rehearsals: int = 2
    semantic_support: int = 3
    max_age_days: float = 30.0
    prune_confidence: float = 0.1


def _eligible(item: WorkingItem, policy: ConsolidationPolicy) -> bool:
    return (
        item.activation >= policy.activation_threshold
        or item.rehearsals >= policy.min_rehearsals
    )


def _promote(
    item: WorkingItem, episodic: EpisodicMemory
) -> None:
    tags = [item.kind]
    for t in item.meta.get("tags", []):
        if t not in tags:
            tags.append(t)
    detail = {
        "kind": item.kind,
        "source": item.source,
        "activation": round(item.activation, 4),
        "rehearsals": item.rehearsals,
        "working_id": item.id,
    }
    if item.kind == "chunk":
        detail["chunk_size"] = item.meta.get("chunk_size", 0)
    episodic.record(
        summary=str(item.content)[:500],
        detail=detail,
        tags=tuple(tags),
        salience=max(0.0, min(1.0, item.activation)),
        source="consolidation",
    )


def _mine_tags(
    episodes, semantic: SemanticMemory, policy: ConsolidationPolicy
) -> int:
    counts: Counter = Counter()
    for ep in episodes:
        counts.update(ep.tags)
    added = 0
    for tag, n in counts.items():
        if n >= policy.semantic_support and not semantic.has_concept(tag):
            semantic.add_concept(
                tag,
                attributes={
                    "episode_count": n,
                    "derived_from": "episodic_tags",
                },
                confidence=min(0.9, 0.25 + 0.15 * n),
                source="consolidation",
            )
            added += 1
    return added


def _mine_keywords(
    episodes, semantic: SemanticMemory, policy: ConsolidationPolicy
) -> int:
    counts: Counter = Counter()
    for ep in episodes:
        counts.update(set(tokenize(ep.summary)))
    added = 0
    for word, n in counts.items():
        if (
            n >= policy.semantic_support
            and word not in STOPWORDS
            and not semantic.has_concept(word)
        ):
            semantic.add_concept(
                word,
                attributes={
                    "occurrences": n,
                    "derived_from": "summary_keywords",
                },
                confidence=min(0.9, 0.25 + 0.15 * n),
                source="consolidation",
            )
            added += 1
    return added


def _mine_cooccurrence(
    episodes, semantic: SemanticMemory, policy: ConsolidationPolicy
) -> int:
    pair_counts: Counter = Counter()
    for ep in episodes:
        tags = sorted(set(ep.tags))
        for a, b in itertools.combinations(tags, 2):
            pair_counts[(a, b)] += 1
    added = 0
    for (a, b), n in pair_counts.items():
        if n >= policy.semantic_support:
            before = len(semantic.query(subject=a, predicate="co_occurs_with", obj=b))
            semantic.add_relation(
                a,
                "co_occurs_with",
                b,
                confidence=min(0.9, 0.3 + 0.1 * n),
                source="consolidation",
            )
            if before == 0:
                added += 1
    return added


def _mark_mined(episodes: List) -> None:
    for ep in episodes:
        ep.detail["_mined"] = True


def consolidate(
    working: WorkingMemory,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    now: Optional[Callable[[], float]] = None,
    policy: Optional[ConsolidationPolicy] = None,
) -> Dict[str, Any]:
    """Run one consolidation pass.

    Working items above the activation threshold (or rehearsed enough)
    become episodes. Repeated tags, summary keywords, and tag pairs across
    unmined episodes become concepts and relations. Low-confidence semantic
    entries are pruned. Returns a stats dict.
    """
    _ = now or time.time
    policy = policy or ConsolidationPolicy()

    episodes_added = 0
    for item in list(working.items()):
        if _eligible(item, policy):
            _promote(item, episodic)
            working.remove(item.id)
            episodes_added += 1

    fresh = [e for e in episodic.episodes() if not e.detail.get("_mined")]
    concepts_added = _mine_tags(fresh, semantic, policy)
    concepts_added += _mine_keywords(fresh, semantic, policy)
    relations_added = _mine_cooccurrence(fresh, semantic, policy)
    _mark_mined(fresh)

    pruned = semantic.prune(min_confidence=policy.prune_confidence)

    return {
        "episodes_added": episodes_added,
        "concepts_added": concepts_added,
        "relations_added": relations_added,
        "pruned": pruned,
    }
