"""Post-run reflection: lessons learned and replan signals."""

import itertools
import time

REPLAN_KINDS = ("bad_args", "tool_error", "precondition")
SCOPES = ("tool", "planning", "general")


class Lesson:
    """One takeaway from a run, with confidence and scope."""

    _ids = itertools.count(1)

    def __init__(self, text, confidence=0.5, scope="general", created=None,
                 id=None):
        if scope not in SCOPES:
            raise ValueError("unknown lesson scope: %r" % (scope,))
        self.id = id if id is not None else "lesson-%d" % next(Lesson._ids)
        self.text = text
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.scope = scope
        self.created = time.time() if created is None else created

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "confidence": self.confidence,
            "scope": self.scope,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            text=d["text"],
            confidence=d.get("confidence", 0.5),
            scope=d.get("scope", "general"),
            created=d.get("created"),
            id=d.get("id"),
        )

    def __repr__(self):
        return "Lesson(%r, %.2f, %r)" % (self.id, self.confidence, self.scope)


class LessonBook:
    """Collects lessons across runs; queryable by scope and confidence."""

    def __init__(self, lessons=None):
        self.lessons = list(lessons) if lessons else []

    def add(self, lesson):
        """Store one lesson; return it."""
        self.lessons.append(lesson)
        return lesson

    def extend(self, lessons):
        """Store several lessons."""
        self.lessons.extend(lessons)

    def absorb(self, outcome):
        """Add every lesson from a reflect() outcome dict."""
        self.extend(outcome.get("lessons", []))

    def query(self, scope=None, min_confidence=0.0):
        """Lessons filtered by scope and confidence, best first."""
        hits = [l for l in self.lessons
                if (scope is None or l.scope == scope)
                and l.confidence >= min_confidence]
        return sorted(hits, key=lambda l: (-l.confidence, l.created))

    def to_dict(self):
        return {"lessons": [l.to_dict() for l in self.lessons]}

    @classmethod
    def from_dict(cls, d):
        return cls([Lesson.from_dict(l) for l in d.get("lessons", [])])

    def __len__(self):
        return len(self.lessons)

    def __repr__(self):
        return "LessonBook(%d lessons)" % len(self)


def _lesson_for_failure(result):
    tool = result.tool or "unknown tool"
    if result.error_kind == "bad_args":
        return Lesson("tool %s failed with bad args, validate args against "
                      "the schema first" % tool, 0.7, "tool")
    if result.error_kind == "tool_error":
        return Lesson("tool %s raised an error, add a fallback step or retry "
                      "with adjusted args" % tool, 0.6, "tool")
    if result.error_kind == "precondition":
        return Lesson("step %s failed on a precondition, gather inputs earlier "
                      "in the plan" % result.step_id, 0.65, "planning")
    return Lesson("tool %s is not registered, pick a registered tool or a "
                  "think step" % tool, 0.8, "planning")


def _has_output(result):
    return result.ok and bool(str(result.output).strip())


def reflect(trace, goal):
    """Summarize a run: success, lessons, and whether replanning is needed."""
    failed = [r for r in trace.results if not r.ok]
    success = bool(trace.success) and any(_has_output(r) for r in trace.results)
    kind = failed[0].error_kind if failed else None

    lessons = [_lesson_for_failure(r) for r in failed]
    if success:
        lessons.append(Lesson(
            "plan for goal %s succeeded in %d steps"
            % (goal.id, len(trace.results)), 0.5, "planning"))
    if any(r.attempts > 1 for r in trace.results):
        lessons.append(Lesson(
            "some steps needed retries, prefer idempotent tools with small "
            "backoff", 0.4, "general"))
    if not trace.results:
        lessons.append(Lesson("plan had no steps, nothing to learn from",
                              0.3, "general"))

    if success:
        summary = "goal %s succeeded in %d steps" % (goal.id, len(trace.results))
    elif failed:
        first = failed[0]
        summary = ("goal %s failed (%s) at step %s: %s"
                   % (goal.id, kind, first.step_id, first.error))
    else:
        summary = "goal %s produced no usable output" % goal.id

    return {
        "success": success,
        "summary": summary,
        "lessons": lessons,
        "replan_needed": (not success) and kind in REPLAN_KINDS,
        "failure_kind": kind,
    }
