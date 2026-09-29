"""Semantic memory: concepts, relations, and spreading activation."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class Concept:
    """A named node of general knowledge."""

    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)
    created: float = 0.0
    updated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "attributes": dict(self.attributes),
            "confidence": self.confidence,
            "sources": list(self.sources),
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Concept":
        return cls(
            name=d["name"],
            attributes=dict(d.get("attributes", {})),
            confidence=float(d.get("confidence", 0.5)),
            sources=list(d.get("sources", [])),
            created=float(d.get("created", 0.0)),
            updated=float(d.get("updated", 0.0)),
        )


@dataclass
class Relation:
    """A typed edge between two concepts."""

    subject: str
    predicate: str
    obj: str
    confidence: float = 0.5
    source: str = "consolidation"
    created: float = 0.0

    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "obj": self.obj,
            "confidence": self.confidence,
            "source": self.source,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Relation":
        return cls(
            subject=d["subject"],
            predicate=d["predicate"],
            obj=d["obj"],
            confidence=float(d.get("confidence", 0.5)),
            source=d.get("source", "consolidation"),
            created=float(d.get("created", 0.0)),
        )


class SemanticMemory:
    """A small knowledge graph of concepts and relations."""

    def __init__(self, now: Optional[Callable[[], float]] = None) -> None:
        self._now = now or time.time
        self._concepts: Dict[str, Concept] = {}
        self._relations: Dict[Tuple[str, str, str], Relation] = {}

    def add_concept(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 0.5,
        source: str = "consolidation",
    ) -> Concept:
        """Add a concept, merging attributes and bumping confidence on repeat."""
        t = self._now()
        existing = self._concepts.get(name)
        if existing is not None:
            if attributes:
                existing.attributes.update(attributes)
            existing.confidence = _clamp01(existing.confidence + 0.1)
            if source not in existing.sources:
                existing.sources.append(source)
            existing.updated = t
            return existing
        concept = Concept(
            name=name,
            attributes=dict(attributes or {}),
            confidence=_clamp01(confidence),
            sources=[source],
            created=t,
            updated=t,
        )
        self._concepts[name] = concept
        return concept

    def get_concept(self, name: str) -> Optional[Concept]:
        return self._concepts.get(name)

    def concepts(self) -> List[Concept]:
        return [self._concepts[k] for k in sorted(self._concepts)]

    def has_concept(self, name: str) -> bool:
        return name in self._concepts

    def remove_concept(self, name: str) -> bool:
        if name not in self._concepts:
            return False
        del self._concepts[name]
        doomed = [
            k for k in self._relations
            if k[0] == name or k[2] == name
        ]
        for k in doomed:
            del self._relations[k]
        return True

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.5,
        source: str = "consolidation",
    ) -> Relation:
        """Add a relation, strengthening it when the same triple repeats."""
        key = (subject, predicate, obj)
        existing = self._relations.get(key)
        if existing is not None:
            existing.confidence = _clamp01(existing.confidence + 0.1)
            return existing
        rel = Relation(
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=_clamp01(confidence),
            source=source,
            created=self._now(),
        )
        self._relations[key] = rel
        return rel

    def strengthen(
        self, subject: str, predicate: str, obj: str, amount: float = 0.1
    ) -> bool:
        rel = self._relations.get((subject, predicate, obj))
        if rel is None:
            return False
        rel.confidence = _clamp01(rel.confidence + amount)
        return True

    def weaken(
        self, subject: str, predicate: str, obj: str, amount: float = 0.1
    ) -> bool:
        rel = self._relations.get((subject, predicate, obj))
        if rel is None:
            return False
        rel.confidence = _clamp01(rel.confidence - amount)
        return True

    def remove_relation(self, subject: str, predicate: str, obj: str) -> bool:
        key = (subject, predicate, obj)
        if key in self._relations:
            del self._relations[key]
            return True
        return False

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Relation]:
        out = []
        for rel in self._relations.values():
            if subject is not None and rel.subject != subject:
                continue
            if predicate is not None and rel.predicate != predicate:
                continue
            if obj is not None and rel.obj != obj:
                continue
            if rel.confidence < min_confidence:
                continue
            out.append(rel)
        out.sort(key=lambda r: -r.confidence)
        return out

    def relations(self) -> List[Relation]:
        return sorted(self._relations.values(), key=lambda r: -r.confidence)

    def contradict(self, subject: str, predicate: str) -> List[Relation]:
        """Relations with the same subject and predicate but a different object."""
        out = [
            r for r in self._relations.values()
            if r.subject == subject and r.predicate == predicate
        ]
        by_obj: Dict[str, Relation] = {}
        for r in out:
            if r.obj not in by_obj or r.confidence > by_obj[r.obj].confidence:
                by_obj[r.obj] = r
        if len(by_obj) < 2:
            return []
        return sorted(by_obj.values(), key=lambda r: -r.confidence)

    def neighbors_of(self, name: str) -> List[Tuple[str, str, float]]:
        """(neighbor, predicate, confidence) for every edge touching name."""
        out = []
        for rel in self._relations.values():
            if rel.subject == name:
                out.append((rel.obj, rel.predicate, rel.confidence))
            elif rel.obj == name:
                out.append((rel.subject, rel.predicate, rel.confidence))
        return out

    def activate(
        self, seed: str, steps: int = 2, decay: float = 0.5
    ) -> Dict[str, float]:
        """Spread activation from seed over the concept/relation graph."""
        if steps < 0:
            raise ValueError("steps must be >= 0")
        decay = _clamp01(decay)
        adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for rel in self._relations.values():
            adjacency[rel.subject].append((rel.obj, rel.confidence))
            adjacency[rel.obj].append((rel.subject, rel.confidence))
        activations: Dict[str, float] = {seed: 1.0}
        frontier = {seed: 1.0}
        for _ in range(steps):
            nxt: Dict[str, float] = {}
            for node, act in frontier.items():
                for neighbor, weight in adjacency.get(node, []):
                    val = act * decay * weight
                    if val > nxt.get(neighbor, 0.0):
                        nxt[neighbor] = val
            if not nxt:
                break
            for node, val in nxt.items():
                if val > activations.get(node, 0.0):
                    activations[node] = val
            frontier = nxt
        return activations

    def path(
        self, start: str, goal: str, max_depth: int = 4
    ) -> Optional[List[Relation]]:
        """Breadth-first path of relations from start concept to goal concept."""
        if start == goal:
            return []
        adjacency: Dict[str, List[Relation]] = defaultdict(list)
        for rel in self._relations.values():
            adjacency[rel.subject].append(rel)
            adjacency[rel.obj].append(rel)
        seen = {start}
        queue: List[Tuple[str, List[Relation]]] = [(start, [])]
        while queue:
            node, trail = queue.pop(0)
            if len(trail) >= max_depth:
                continue
            for rel in adjacency.get(node, []):
                other = rel.obj if rel.subject == node else rel.subject
                if other in seen:
                    continue
                seen.add(other)
                new_trail = trail + [rel]
                if other == goal:
                    return new_trail
                queue.append((other, new_trail))
        return None

    def prune(self, min_confidence: float = 0.1) -> int:
        """Drop relations and concepts below the confidence floor."""
        removed = 0
        for key in [k for k, r in self._relations.items()
                    if r.confidence < min_confidence]:
            del self._relations[key]
            removed += 1
        for name in [n for n, c in self._concepts.items()
                     if c.confidence < min_confidence]:
            del self._concepts[name]
            removed += 1
        return removed

    def stats(self) -> Dict[str, Any]:
        conns = len(self._concepts)
        rels = len(self._relations)
        mean_conf = (
            sum(c.confidence for c in self._concepts.values()) / conns
            if conns else 0.0
        )
        return {
            "concepts": conns,
            "relations": rels,
            "mean_concept_confidence": mean_conf,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concepts": [c.to_dict() for c in self._concepts.values()],
            "relations": [r.to_dict() for r in self._relations.values()],
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any], now: Optional[Callable[[], float]] = None
    ) -> "SemanticMemory":
        mem = cls(now=now)
        for raw in d.get("concepts", []):
            c = Concept.from_dict(raw)
            mem._concepts[c.name] = c
        for raw in d.get("relations", []):
            r = Relation.from_dict(raw)
            mem._relations[r.key()] = r
        return mem
