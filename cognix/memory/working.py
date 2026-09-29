"""Short-term working memory: bounded, activation-based, with decay and chunking."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .forgetting import ForgettingCurve


@dataclass
class WorkingItem:
    """One item held in working memory."""

    id: str
    content: Any
    kind: str = "observation"
    activation: float = 1.0
    created: float = 0.0
    source: str = "perception"
    meta: Dict[str, Any] = field(default_factory=dict)
    pinned: bool = False
    rehearsals: int = 0

    def touch(self, boost: float = 0.35) -> None:
        # rehearse: raise activation and count it
        self.activation = min(1.0, self.activation + boost)
        self.rehearsals += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "activation": self.activation,
            "created": self.created,
            "source": self.source,
            "meta": dict(self.meta),
            "pinned": self.pinned,
            "rehearsals": self.rehearsals,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkingItem":
        return cls(
            id=d["id"],
            content=d.get("content"),
            kind=d.get("kind", "observation"),
            activation=float(d.get("activation", 1.0)),
            created=float(d.get("created", 0.0)),
            source=d.get("source", "perception"),
            meta=dict(d.get("meta", {})),
            pinned=bool(d.get("pinned", False)),
            rehearsals=int(d.get("rehearsals", 0)),
        )


class WorkingMemory:
    """Fixed-capacity store with exponential activation decay.

    Overflow evicts the lowest-activation unpinned item. Pinned items are
    never evicted by capacity pressure.
    """

    def __init__(
        self,
        capacity: int = 7,
        decay_rate: float = 0.08,
        threshold: float = 0.05,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.threshold = threshold
        self._now = now or time.time
        self._items: Dict[str, WorkingItem] = {}
        self._ids = itertools.count(1)
        self._evictions = 0
        self.curve = ForgettingCurve()

    def _new_id(self) -> str:
        return "wm-%d" % next(self._ids)

    def add(
        self,
        content: Any,
        kind: str = "observation",
        source: str = "perception",
        activation: float = 1.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> WorkingItem:
        # make room if full, evicting the weakest unpinned item
        if len(self._items) >= self.capacity:
            self._evict_one()
        item = WorkingItem(
            id=self._new_id(),
            content=content,
            kind=kind,
            activation=max(0.0, min(1.0, activation)),
            created=self._now(),
            source=source,
            meta=dict(meta or {}),
        )
        self._items[item.id] = item
        return item

    def _evict_one(self) -> Optional[WorkingItem]:
        candidates = [i for i in self._items.values() if not i.pinned]
        pool = candidates or list(self._items.values())
        if not pool:
            return None
        # weakest activation first, oldest breaks ties
        victim = min(pool, key=lambda i: (i.activation, i.created))
        del self._items[victim.id]
        self._evictions += 1
        return victim

    def pin(self, item_id: str) -> bool:
        item = self._items.get(item_id)
        if item is None:
            return False
        item.pinned = True
        return True

    def unpin(self, item_id: str) -> bool:
        item = self._items.get(item_id)
        if item is None:
            return False
        item.pinned = False
        return True

    def rehearse(self, item_id: str, boost: float = 0.35) -> bool:
        # boost activation and log it on the forgetting curve
        item = self._items.get(item_id)
        if item is None:
            return False
        item.touch(boost)
        self.curve.record_rehearsal(item_id)
        return True

    def tick(self, dt: float = 1.0) -> int:
        # exponential decay, drop whatever falls below threshold
        dropped = 0
        factor = math.exp(-self.decay_rate * max(0.0, dt))
        for item_id in list(self._items.keys()):
            item = self._items[item_id]
            item.activation *= factor
            if item.activation < self.threshold:
                del self._items[item_id]
                dropped += 1
        return dropped

    def get(self, item_id: str) -> Optional[WorkingItem]:
        return self._items.get(item_id)

    def items(self) -> List[WorkingItem]:
        return list(self._items.values())

    def top(self, n: int = 5) -> List[WorkingItem]:
        return sorted(
            self._items.values(), key=lambda i: (-i.activation, i.created)
        )[: max(0, n)]

    def remove(self, item_id: str) -> bool:
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def clear(self) -> None:
        self._items.clear()

    def chunk(self, item_ids: List[str], label: str) -> Optional[WorkingItem]:
        # group members into one chunk item, freeing capacity
        members = []
        for item_id in item_ids:
            item = self._items.get(item_id)
            if item is None:
                return None
            members.append(item)
        if not members:
            return None
        mean_activation = sum(m.activation for m in members) / len(members)
        chunk_meta = {
            "members": [
                {
                    "id": m.id,
                    "content": m.content,
                    "kind": m.kind,
                    "activation": m.activation,
                    "source": m.source,
                }
                for m in members
            ],
            "chunk_size": len(members),
            "chunked": True,
        }
        tags = []
        for m in members:
            for t in m.meta.get("tags", []):
                if t not in tags:
                    tags.append(t)
        if tags:
            chunk_meta["tags"] = tags
        for m in members:
            del self._items[m.id]
        chunk_item = WorkingItem(
            id=self._new_id(),
            content=label,
            kind="chunk",
            activation=min(1.0, mean_activation),
            created=self._now(),
            source="chunking",
            meta=chunk_meta,
        )
        self._items[chunk_item.id] = chunk_item
        return chunk_item

    def expand_chunk(self, chunk_id: str) -> Optional[List[WorkingItem]]:
        # split a chunk back into its member items
        chunk = self._items.get(chunk_id)
        if chunk is None or chunk.kind != "chunk":
            return None
        restored = []
        for m in chunk.meta.get("members", []):
            item = WorkingItem(
                id=self._new_id(),
                content=m.get("content"),
                kind=m.get("kind", "observation"),
                activation=float(m.get("activation", 0.5)),
                created=self._now(),
                source=m.get("source", "perception"),
                meta={"restored_from": chunk_id},
            )
            if len(self._items) >= self.capacity:
                self._evict_one()
            self._items[item.id] = item
            restored.append(item)
        del self._items[chunk_id]
        return restored

    def refresh(self, item_id: str, activation: float = 1.0) -> bool:
        # snap an item back to full activation without a rehearsal count
        item = self._items.get(item_id)
        if item is None:
            return False
        item.activation = max(0.0, min(1.0, activation))
        return True

    def by_kind(self, kind: str) -> List[WorkingItem]:
        return [i for i in self._items.values() if i.kind == kind]

    def stats(self) -> Dict[str, Any]:
        items = list(self._items.values())
        mean_act = sum(i.activation for i in items) / len(items) if items else 0.0
        return {
            "count": len(items),
            "capacity": self.capacity,
            "capacity_pressure": len(items) / self.capacity,
            "mean_activation": mean_act,
            "pinned": sum(1 for i in items if i.pinned),
            "chunks": sum(1 for i in items if i.kind == "chunk"),
            "evictions": self._evictions,
        }

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "decay_rate": self.decay_rate,
            "threshold": self.threshold,
            "evictions": self._evictions,
            "items": [i.to_dict() for i in self._items.values()],
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any], now: Optional[Callable[[], float]] = None
    ) -> "WorkingMemory":
        wm = cls(
            capacity=int(d.get("capacity", 7)),
            decay_rate=float(d.get("decay_rate", 0.08)),
            threshold=float(d.get("threshold", 0.05)),
            now=now,
        )
        wm._evictions = int(d.get("evictions", 0))
        max_n = 0
        for raw in d.get("items", []):
            item = WorkingItem.from_dict(raw)
            wm._items[item.id] = item
            try:
                n = int(item.id.split("-")[-1])
                max_n = max(max_n, n)
            except (ValueError, IndexError):
                pass
        wm._ids = itertools.count(max_n + 1)
        return wm
