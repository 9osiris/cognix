"""state inspection: summaries and dotted-path queries over plain state dicts."""
from typing import Any, Dict


def _sized(value: Any) -> int:
    # len() when the value supports it, else 0
    try:
        return len(value)
    except TypeError:
        return 0


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _count_section(section: Any, key: str) -> int:
    # sections serialize as {"items": [...], ...}; count the inner list
    if isinstance(section, dict) and isinstance(section.get(key), list):
        return len(section[key])
    return _sized(section)


def summarize(state: Dict[str, Any]) -> Dict[str, Any]:
    """count the moving parts of a cognitive state dict.

    accepts the nested shape (memory.working, memory.episodic,
    memory.semantic, goals as a list) and the flat runtime shape
    (top-level working/episodic/semantic keys, goals as {"goals": [...]}).
    """
    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    memory = _as_dict(state.get("memory"))
    working = memory.get("working", state.get("working"))
    episodic = memory.get("episodic", state.get("episodic"))
    semantic = _as_dict(memory.get("semantic", state.get("semantic")))
    goals = state.get("goals", [])
    if isinstance(goals, dict):
        goals = goals.get("goals", [])
    by_status: Dict[str, int] = {}
    if isinstance(goals, list):
        for goal in goals:
            if isinstance(goal, dict):
                status = goal.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1
    goal_total = _sized(goals) if isinstance(goals, (list, dict)) else 0
    beliefs = state.get("beliefs", [])
    if isinstance(beliefs, dict):
        beliefs = beliefs.get("beliefs", beliefs)
    return {
        "working_items": _count_section(working, "items"),
        "episodes": _count_section(episodic, "episodes"),
        "concepts": _count_section(semantic, "concepts"),
        "relations": _count_section(semantic, "relations"),
        "beliefs": _sized(beliefs),
        "goals": {
            "total": goal_total,
            "by_status": by_status,
        },
        "tools": _sized(state.get("tools")),
    }


def summary(state: Dict[str, Any]) -> str:
    """one human-readable line describing a state dict."""
    counts = summarize(state)
    goals = counts["goals"]
    breakdown = ", ".join(
        "{}={}".format(status, num) for status, num in sorted(goals["by_status"].items())
    )
    goal_part = "goals={}".format(goals["total"])
    if breakdown:
        goal_part += "({})".format(breakdown)
    return (
        "working={working_items} episodes={episodes} concepts={concepts} "
        "relations={relations} beliefs={beliefs} {goals} tools={tools}".format(
            working_items=counts["working_items"],
            episodes=counts["episodes"],
            concepts=counts["concepts"],
            relations=counts["relations"],
            beliefs=counts["beliefs"],
            goals=goal_part,
            tools=counts["tools"],
        )
    )


def query_state(state: Any, path: str) -> Any:
    """dotted path lookup over plain dicts and lists; raises KeyError with help."""
    if not isinstance(path, str) or not path:
        raise KeyError("path must be a non-empty string")
    current = state
    walked: list = []
    for segment in path.split("."):
        walked.append(segment)
        here = ".".join(walked[:-1]) or "<root>"
        if isinstance(current, dict):
            if segment not in current:
                raise KeyError(
                    "no key {!r} under {!r}; available keys: {}".format(
                        segment, here, sorted(current.keys())
                    )
                )
            current = current[segment]
        elif isinstance(current, (list, tuple)):
            try:
                index = int(segment)
            except ValueError:
                raise KeyError(
                    "cannot index a list with {!r} under {!r}; use an integer 0-{}".format(
                        segment, here, len(current) - 1
                    )
                )
            if not 0 <= index < len(current):
                raise KeyError(
                    "index {} out of range under {!r}; length is {}".format(index, here, len(current))
                )
            current = current[index]
        else:
            raise KeyError(
                "cannot descend into {} at {!r}; path {!r} ends at a leaf".format(
                    type(current).__name__, here, segment
                )
            )
    return current
