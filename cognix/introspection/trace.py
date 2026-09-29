"""execution tracing: nested spans with durations, point events, ascii timelines."""
import time
from typing import Any, Dict, List, Optional


class Span:
    """one timed unit of work; used as a context manager via Tracer.start."""

    def __init__(self, tracer: "Tracer", name: str, attrs: Dict[str, Any]):
        self.tracer = tracer
        self.name = name
        self.attrs = dict(attrs)
        self.started: Optional[float] = None
        self.ended: Optional[float] = None
        self.children: List["Span"] = []

    @property
    def is_running(self) -> bool:
        return self.started is not None and self.ended is None

    @property
    def duration(self) -> Optional[float]:
        # seconds, or None while still running
        if self.started is None or self.ended is None:
            return None
        return self.ended - self.started

    def __enter__(self) -> "Span":
        self.started = time.perf_counter()
        self.tracer._push(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.ended = time.perf_counter()
        self.tracer._pop(self)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "started": self.started,
            "ended": self.ended,
            "duration": self.duration,
            "attrs": dict(self.attrs),
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, tracer: "Tracer", data: Dict[str, Any]) -> "Span":
        span = cls(tracer, data.get("name", ""), data.get("attrs", {}))
        span.started = data.get("started")
        span.ended = data.get("ended")
        span.children = [cls.from_dict(tracer, child) for child in data.get("children", [])]
        return span

    def __repr__(self) -> str:
        return "Span(name={!r})".format(self.name)


class Tracer:
    """records nested spans and point events."""

    def __init__(self) -> None:
        self._stack: List[Span] = []
        self._roots: List[Span] = []
        self._events: List[Dict[str, Any]] = []

    def start(self, name: str, **attrs: Any) -> Span:
        # returns a span usable as a context manager; nesting via an internal stack
        return Span(self, name, attrs)

    def _push(self, span: Span) -> None:
        if self._stack:
            self._stack[-1].children.append(span)
        else:
            self._roots.append(span)
        self._stack.append(span)

    def _pop(self, span: Span) -> None:
        if self._stack and self._stack[-1] is span:
            self._stack.pop()

    def event(self, name: str, **attrs: Any) -> None:
        # record a point-in-time event at the current nesting depth
        self._events.append(
            {
                "time": time.perf_counter(),
                "name": name,
                "attrs": dict(attrs),
                "depth": len(self._stack),
            }
        )

    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def active(self) -> Optional[Span]:
        # innermost currently open span, or None
        return self._stack[-1] if self._stack else None

    def roots(self) -> List[Span]:
        return list(self._roots)

    def clear(self) -> None:
        self._stack = []
        self._roots = []
        self._events = []

    def timeline(self) -> str:
        # ascii timeline: indented span names with durations
        lines: List[str] = []
        for span in self._roots:
            self._render(span, 0, lines)
        return "\n".join(lines)

    def _render(self, span: Span, depth: int, lines: List[str]) -> None:
        indent = "  " * depth
        duration = span.duration
        if duration is None:
            stamp = "[running]"
        else:
            stamp = "[{:.1f}ms]".format(duration * 1000.0)
        attrs = ""
        if span.attrs:
            attrs = " " + " ".join("{}={!r}".format(k, v) for k, v in span.attrs.items())
        lines.append("{}{}{}{}".format(indent, span.name, attrs, " " + stamp))
        for child in span.children:
            self._render(child, depth + 1, lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spans": [span.to_dict() for span in self._roots],
            "events": [dict(event) for event in self._events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tracer":
        tracer = cls()
        tracer._roots = [Span.from_dict(tracer, s) for s in data.get("spans", [])]
        tracer._events = [dict(event) for event in data.get("events", [])]
        return tracer
