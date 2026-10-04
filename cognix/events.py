"""Event bus: decoupled pub/sub so subsystems announce what happens.

The tracer records how long things took; the bus records what happened.
Handlers subscribe to event types like "goal.completed" or to everything
("*"), and the bus keeps a bounded log of recent events for inspection.

Payloads must be json-safe (dicts, lists, strings, numbers, bools, None)
because the log is persisted with the rest of the saved state.
A handler that raises does not kill the emitter: the failure is recorded
on bus.errors and the remaining handlers still run.
"""

import time
from collections import deque


class EventBus:
    def __init__(self, max_log=200, enabled=True, now=None):
        self._now = now or time.time
        self.max_log = max_log
        self.enabled = enabled
        self._seq = 0
        self._subs = {}
        self._wildcards = []
        self.log = deque(maxlen=max_log)
        # (event_type, handler repr, error string) for handlers that blew up
        self.errors = []

    def subscribe(self, event_type, handler):
        """Subscribe to one event type, or "*" for every event."""
        if event_type == "*":
            if handler not in self._wildcards:
                self._wildcards.append(handler)
            return
        bucket = self._subs.setdefault(event_type, [])
        if handler not in bucket:
            bucket.append(handler)

    def unsubscribe(self, event_type, handler):
        if event_type == "*":
            if handler in self._wildcards:
                self._wildcards.remove(handler)
            return
        bucket = self._subs.get(event_type, [])
        if handler in bucket:
            bucket.remove(handler)

    def emit(self, event_type, **payload):
        """Announce an event. Returns the event dict, or None when disabled."""
        if not self.enabled:
            return None
        self._seq += 1
        event = {
            "seq": self._seq,
            "type": event_type,
            "at": self._now(),
            "payload": dict(payload),
        }
        self.log.append(event)
        handlers = list(self._subs.get(event_type, [])) + list(self._wildcards)
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:  # a bad listener must not kill the mind
                self.errors.append((event_type, repr(handler), str(exc)))
        return event

    def recent(self, event_type=None, n=20):
        """Newest-first-filtered event log: optionally one type, up to n."""
        items = list(self.log)
        if event_type is not None:
            items = [e for e in items if e["type"] == event_type]
        return items[-n:]

    def counts(self):
        """How many logged events of each type."""
        out = {}
        for event in self.log:
            out[event["type"]] = out.get(event["type"], 0) + 1
        return out

    def clear(self):
        self.log.clear()

    def to_dict(self):
        return {
            "seq": self._seq,
            "enabled": self.enabled,
            "max_log": self.max_log,
            "log": [dict(e) for e in self.log],
        }

    @classmethod
    def from_dict(cls, data, now=None):
        bus = cls(
            max_log=data.get("max_log", 200),
            enabled=data.get("enabled", True),
            now=now,
        )
        bus._seq = data.get("seq", 0)
        for event in data.get("log", []):
            bus.log.append(dict(event))
        return bus
