"""Associative recall: cue-driven retrieval across all three memory stores."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .episodic import tokenize
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .working import WorkingMemory


def _overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    doc = set(doc_tokens)
    hits = sum(1 for t in set(query_tokens) if t in doc)
    return hits / len(set(query_tokens))


class AssociativeRecall:
    """One cue fans out to working, episodic, and semantic memory at once."""

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
    ) -> None:
        self.working = working
        self.episodic = episodic
        self.semantic = semantic

    def _working_hits(
        self, qtokens: List[str], per_store: int
    ) -> List[Dict[str, Any]]:
        scored = []
        for item in self.working.items():
            doc = tokenize(
                "%s %s %s %s"
                % (
                    item.content,
                    item.kind,
                    item.source,
                    " ".join(item.meta.get("tags", [])),
                )
            )
            base = _overlap(qtokens, doc)
            if base > 0.0:
                score = base * (0.5 + 0.5 * item.activation)
                scored.append({"store": "working", "item": item, "score": score})
        scored.sort(key=lambda h: -h["score"])
        return scored[: max(0, per_store)]

    def _episodic_hits(
        self, query: str, qtokens: List[str], per_store: int
    ) -> List[Dict[str, Any]]:
        hits = []
        for ep, score in self.episodic.recall(
            query, k=per_store, strategy="mixed"
        ):
            # only episodes that actually mention the query count as hits
            doc = tokenize(ep.summary + " " + " ".join(ep.tags))
            if score > 0.0 and _overlap(qtokens, doc) > 0.0:
                hits.append({"store": "episodic", "item": ep, "score": score})
        return hits

    def _semantic_hits(
        self, qtokens: List[str], per_store: int
    ) -> List[Dict[str, Any]]:
        scored = []
        for concept in self.semantic.concepts():
            base = _overlap(qtokens, tokenize(concept.name))
            if base > 0.0:
                scored.append(
                    {
                        "store": "semantic",
                        "item": concept,
                        "score": base * (0.5 + 0.5 * concept.confidence),
                    }
                )
        for rel in self.semantic.relations():
            base = _overlap(
                qtokens, tokenize("%s %s %s" % (rel.subject, rel.predicate, rel.obj))
            )
            if base > 0.0:
                scored.append(
                    {
                        "store": "semantic",
                        "item": rel,
                        "score": base * (0.5 + 0.5 * rel.confidence),
                    }
                )
        scored.sort(key=lambda h: -h["score"])
        return scored[: max(0, per_store)]

    @staticmethod
    def _normalize(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hits:
            return hits
        top = max(h["score"] for h in hits)
        if top <= 0.0:
            return hits
        for h in hits:
            h["score"] = h["score"] / top
        return hits

    def cue(
        self, query: str, k: int = 5, per_store: int = 5
    ) -> List[Dict[str, Any]]:
        """Rank hits from every store against one query, merged by score."""
        qtokens = tokenize(query)
        if not qtokens or k <= 0:
            return []
        merged: List[Dict[str, Any]] = []
        merged.extend(self._normalize(self._working_hits(qtokens, per_store)))
        merged.extend(
            self._normalize(self._episodic_hits(query, qtokens, per_store))
        )
        merged.extend(self._normalize(self._semantic_hits(qtokens, per_store)))
        merged.sort(key=lambda h: -h["score"])
        return merged[:k]

    def spread(
        self, seed_text: str, depth: int = 2, width: int = 5
    ) -> Dict[str, float]:
        """Cross-store spreading activation from a seed phrase.

        seed -> semantic.activate -> linked episodes -> working items that
        share tags. Returns id -> activation for concepts, episodes, items.
        """
        seed_tokens = set(tokenize(seed_text))
        if not seed_tokens:
            return {}
        sem_acts: Dict[str, float] = {}
        for tok in seed_tokens:
            for name, act in self.semantic.activate(
                tok, steps=depth, decay=0.5
            ).items():
                if act > sem_acts.get(name, 0.0):
                    sem_acts[name] = act
        ranked = sorted(sem_acts.items(), key=lambda kv: -kv[1])[: max(0, width)]
        top_acts = dict(ranked)

        result: Dict[str, float] = {}
        for name, act in top_acts.items():
            if self.semantic.has_concept(name):
                result[name] = act
        for ep in self.episodic.episodes():
            ep_tokens = set(tokenize(ep.summary)) | set(ep.tags)
            score = sum(a for n, a in top_acts.items() if n in ep_tokens)
            if score > 0.0:
                result[ep.id] = score * (0.5 + 0.5 * ep.salience)
        for item in self.working.items():
            item_tokens = set(tokenize(str(item.content))) | set(
                item.meta.get("tags", [])
            )
            score = sum(a for n, a in top_acts.items() if n in item_tokens)
            if score > 0.0:
                result[item.id] = score * (0.5 + 0.5 * item.activation)
        return result

    def best(self, query: str) -> Optional[Dict[str, Any]]:
        """Single best hit across stores, or None."""
        hits = self.cue(query, k=1)
        return hits[0] if hits else None
