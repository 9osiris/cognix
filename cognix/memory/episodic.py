"""Episodic memory: timestamped records with multi-strategy recall."""

from __future__ import annotations

import itertools
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the and or but if then else for with from to of in on at by as is are
    was were be been being it its this that these those i you he she we they
    them his her our their my your his her its not no yes do does did done
    have has had having will would can could should shall may might must
    what when where which who whom how why all any some such than too very
    just about into over after before between during under again once here there
    """.split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _doc_tokens(summary: str, detail: Dict[str, Any], tags: List[str]) -> List[str]:
    parts = [summary]
    parts.extend(str(v) for v in detail.values() if not str(v).startswith("_"))
    parts.extend(tags)
    return tokenize(" ".join(parts))


@dataclass
class Episode:
    """One remembered event."""

    id: str
    timestamp: float
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    salience: float = 0.5
    source: str = "runtime"
    links: Dict[str, str] = field(default_factory=dict)

    def age(self, now: float) -> float:
        return max(0.0, now - self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "detail": dict(self.detail),
            "tags": list(self.tags),
            "salience": self.salience,
            "source": self.source,
            "links": dict(self.links),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Episode":
        return cls(
            id=d["id"],
            timestamp=float(d.get("timestamp", 0.0)),
            summary=d.get("summary", ""),
            detail=dict(d.get("detail", {})),
            tags=list(d.get("tags", [])),
            salience=float(d.get("salience", 0.5)),
            source=d.get("source", "runtime"),
            links=dict(d.get("links", {})),
        )


class EpisodicMemory:
    """Append-mostly store of episodes with recency/keyword/temporal recall."""

    def __init__(
        self,
        now: Optional[Callable[[], float]] = None,
        recency_tau: float = 86400.0,
        temporal_tau: float = 86400.0,
    ) -> None:
        self._now = now or time.time
        self.recency_tau = recency_tau
        self.temporal_tau = temporal_tau
        self._episodes: Dict[str, Episode] = {}
        self._order: List[str] = []
        self._ids = itertools.count(1)

    def _new_id(self) -> str:
        return "ep-%d" % next(self._ids)

    def record(
        self,
        summary: str,
        detail: Optional[Dict[str, Any]] = None,
        tags: Tuple[str, ...] = (),
        salience: float = 0.5,
        source: str = "runtime",
        timestamp: Optional[float] = None,
    ) -> Episode:
        ep = Episode(
            id=self._new_id(),
            timestamp=self._now() if timestamp is None else float(timestamp),
            summary=summary,
            detail=dict(detail or {}),
            tags=list(tags),
            salience=max(0.0, min(1.0, salience)),
            source=source,
        )
        self._episodes[ep.id] = ep
        self._order.append(ep.id)
        return ep

    def get(self, ep_id: str) -> Optional[Episode]:
        return self._episodes.get(ep_id)

    def episodes(self) -> List[Episode]:
        return [self._episodes[i] for i in self._order]

    def count(self) -> int:
        return len(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __contains__(self, ep_id: str) -> bool:
        return ep_id in self._episodes

    def _candidates(
        self, since: Optional[float], tags: Optional[List[str]]
    ) -> List[Episode]:
        eps = [self._episodes[i] for i in self._order]
        if since is not None:
            eps = [e for e in eps if e.timestamp >= since]
        if tags:
            wanted = set(tags)
            eps = [e for e in eps if wanted & set(e.tags)]
        return eps

    def _recency_scores(
        self, eps: List[Episode], now: float
    ) -> Dict[str, float]:
        return {
            e.id: math.exp(-e.age(now) / self.recency_tau) for e in eps
        }

    def _keyword_scores(
        self, eps: List[Episode], query: str
    ) -> Dict[str, float]:
        qtokens = tokenize(query)
        scores = {e.id: 0.0 for e in eps}
        if not qtokens or not eps:
            return scores
        docs = {e.id: _doc_tokens(e.summary, e.detail, e.tags) for e in eps}
        df = Counter()
        for toks in docs.values():
            for t in set(toks):
                df[t] += 1
        n = len(eps)
        idf = {t: math.log(1.0 + n / (1.0 + c)) for t, c in df.items()}
        for e in eps:
            tf = Counter(docs[e.id])
            scores[e.id] = sum(
                (1.0 + math.log(1.0 + tf[t])) * idf[t]
                for t in qtokens
                if t in tf
            )
        top = max(scores.values()) or 1.0
        return {k: v / top for k, v in scores.items()}

    def _temporal_scores(
        self, eps: List[Episode], target: float
    ) -> Dict[str, float]:
        return {
            e.id: math.exp(-abs(e.timestamp - target) / self.temporal_tau)
            for e in eps
        }

    @staticmethod
    def _norm(scores: Dict[str, float]) -> Dict[str, float]:
        top = max(scores.values()) if scores else 0.0
        if top <= 0.0:
            return {k: 0.0 for k in scores}
        return {k: v / top for k, v in scores.items()}

    def recall(
        self,
        query: str = "",
        k: int = 5,
        strategy: str = "mixed",
        since: Optional[float] = None,
        tags: Optional[List[str]] = None,
        around: Optional[float] = None,
        now: Optional[float] = None,
    ) -> List[Tuple[Episode, float]]:
        """Recall episodes ranked by the chosen strategy.

        strategies: recency, keyword, temporal, mixed.
        temporal ranks nearness to `around` (falls back to `since`, then now).
        """
        if k <= 0:
            return []
        t = self._now() if now is None else now
        eps = self._candidates(since, tags)
        if not eps:
            return []
        strategy = strategy.lower()
        if strategy == "recency":
            scores = self._recency_scores(eps, t)
        elif strategy == "keyword":
            scores = self._keyword_scores(eps, query)
            if max(scores.values()) <= 0.0:
                scores = self._recency_scores(eps, t)
        elif strategy == "temporal":
            target = around if around is not None else (
                since if since is not None else t
            )
            scores = self._temporal_scores(eps, target)
        elif strategy == "mixed":
            kw = self._norm(self._keyword_scores(eps, query))
            rc = self._norm(self._recency_scores(eps, t))
            scores = {
                e.id: 0.5 * kw[e.id] + 0.3 * rc[e.id] + 0.2 * e.salience
                for e in eps
            }
        else:
            raise ValueError("unknown recall strategy: %r" % strategy)
        ranked = sorted(eps, key=lambda e: (-scores[e.id], -e.timestamp))
        return [(e, scores[e.id]) for e in ranked[:k]]

    def link(self, a_id: str, b_id: str, relation: str = "follows") -> bool:
        a = self._episodes.get(a_id)
        b = self._episodes.get(b_id)
        if a is None or b is None or a_id == b_id:
            return False
        a.links[b_id] = relation
        return True

    def unlink(self, a_id: str, b_id: str) -> bool:
        a = self._episodes.get(a_id)
        if a is None or b_id not in a.links:
            return False
        del a.links[b_id]
        return True

    def neighbors(self, ep_id: str) -> List[Tuple[Episode, str]]:
        ep = self._episodes.get(ep_id)
        if ep is None:
            return []
        out = []
        for other_id, relation in ep.links.items():
            other = self._episodes.get(other_id)
            if other is not None:
                out.append((other, relation))
        return out

    def backlinks(self, ep_id: str) -> List[Tuple[Episode, str]]:
        # episodes that link TO this one
        out = []
        for e in self._episodes.values():
            if ep_id in e.links:
                out.append((e, e.links[ep_id]))
        return out

    def forget(self, max_age_days: float = 30.0, min_salience: float = 0.2) -> int:
        """Drop episodes older than max_age_days with salience below the floor."""
        cutoff = self._now() - max_age_days * 86400.0
        doomed = [
            e.id
            for e in self._episodes.values()
            if e.timestamp < cutoff and e.salience < min_salience
        ]
        for ep_id in doomed:
            self._drop(ep_id)
        return len(doomed)

    def _drop(self, ep_id: str) -> None:
        self._episodes.pop(ep_id, None)
        if ep_id in self._order:
            self._order.remove(ep_id)
        for e in self._episodes.values():
            e.links.pop(ep_id, None)

    def retag(self, ep_id: str, tags: List[str]) -> bool:
        ep = self._episodes.get(ep_id)
        if ep is None:
            return False
        for t in tags:
            if t not in ep.tags:
                ep.tags.append(t)
        return True

    def boost_salience(self, ep_id: str, amount: float = 0.1) -> bool:
        ep = self._episodes.get(ep_id)
        if ep is None:
            return False
        ep.salience = max(0.0, min(1.0, ep.salience + amount))
        return True

    def sequence(
        self, tags: Optional[List[str]] = None, limit: int = 50
    ) -> List[Episode]:
        eps = self._candidates(None, tags)
        eps.sort(key=lambda e: e.timestamp)
        return eps[: max(0, limit)]

    def recent(self, limit: int = 50) -> List[Episode]:
        # newest first, for novelty scoring and recaps
        eps = sorted(self._episodes.values(), key=lambda e: e.timestamp, reverse=True)
        return eps[: max(0, limit)]

    def tag_counts(self) -> Dict[str, int]:
        counts: Counter = Counter()
        for e in self._episodes.values():
            counts.update(e.tags)
        return dict(counts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recency_tau": self.recency_tau,
            "temporal_tau": self.temporal_tau,
            "episodes": [self._episodes[i].to_dict() for i in self._order],
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any], now: Optional[Callable[[], float]] = None
    ) -> "EpisodicMemory":
        mem = cls(
            now=now,
            recency_tau=float(d.get("recency_tau", 86400.0)),
            temporal_tau=float(d.get("temporal_tau", 86400.0)),
        )
        max_n = 0
        for raw in d.get("episodes", []):
            ep = Episode.from_dict(raw)
            mem._episodes[ep.id] = ep
            mem._order.append(ep.id)
            try:
                n = int(ep.id.split("-")[-1])
                max_n = max(max_n, n)
            except (ValueError, IndexError):
                pass
        mem._ids = itertools.count(max_n + 1)
        return mem
