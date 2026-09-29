"""Goal tracking for the cognix agent loop."""

import itertools
import time

STATUSES = ("proposed", "active", "suspended", "done", "failed")
TERMINAL_STATUSES = ("done", "failed")


class Goal:
    """One unit of work with priority, status, and an optional deadline."""

    _ids = itertools.count(1)

    def __init__(self, description, priority=0.5, status="active",
                 parent_id=None, created=None, deadline=None,
                 metadata=None, outcome="", id=None):
        if status not in STATUSES:
            raise ValueError("unknown goal status: %r" % (status,))
        self.id = id if id is not None else "goal-%d" % next(Goal._ids)
        self.description = description
        self.priority = max(0.0, min(1.0, float(priority)))
        self.status = status
        self.parent_id = parent_id
        self.created = time.time() if created is None else created
        self.deadline = deadline
        self.metadata = dict(metadata) if metadata else {}
        self.outcome = outcome

    @property
    def terminal(self):
        """True when the goal is done or failed."""
        return self.status in TERMINAL_STATUSES

    @property
    def is_overdue(self):
        """True when a deadline is set, passed, and the goal is not terminal."""
        return (self.deadline is not None and self.deadline < time.time()
                and not self.terminal)

    def time_left(self, now=None):
        """Seconds until the deadline; None when no deadline is set."""
        if self.deadline is None:
            return None
        if callable(now):
            return self.deadline - now()
        return self.deadline - (time.time() if now is None else now)

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "parent_id": self.parent_id,
            "created": self.created,
            "deadline": self.deadline,
            "metadata": self.metadata,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            description=d["description"],
            priority=d.get("priority", 0.5),
            status=d.get("status", "active"),
            parent_id=d.get("parent_id"),
            created=d.get("created"),
            deadline=d.get("deadline"),
            metadata=d.get("metadata"),
            outcome=d.get("outcome", ""),
            id=d.get("id"),
        )

    def __repr__(self):
        return "Goal(%r, %r, priority=%.2f)" % (self.id, self.status, self.priority)


class GoalStack:
    """Owns goals; suspends a parent while its children run, resumes after."""

    def __init__(self, now=None):
        self._goals = {}
        self._now = now or time.time

    def _t(self, now):
        if callable(now):
            return now()
        if now is not None:
            return now
        return self._now()

    def push(self, description, priority=0.5, parent_id=None, deadline=None,
             metadata=None):
        """Add a goal and return it."""
        goal = Goal(description, priority=priority, parent_id=parent_id,
                    created=self._now(), deadline=deadline, metadata=metadata)
        self._goals[goal.id] = goal
        return goal

    def get(self, goal_id):
        """Return the goal or None."""
        return self._goals.get(goal_id)

    def active(self):
        """Active goals sorted by priority desc, oldest first on ties."""
        return sorted(
            (g for g in self._goals.values() if g.status == "active"),
            key=lambda g: (-g.priority, g.created),
        )

    def top(self):
        """Highest priority active goal, or None."""
        items = self.active()
        return items[0] if items else None

    def by_status(self, status):
        """All goals currently in the given status."""
        return [g for g in self._goals.values() if g.status == status]

    def stats(self):
        """Counts per status plus the overdue total."""
        counts = {s: 0 for s in STATUSES}
        for goal in self._goals.values():
            counts[goal.status] += 1
        counts["overdue"] = len(self.overdue())
        return counts

    def prune_terminal(self):
        """Drop done/failed goals; return how many were removed."""
        doomed = [gid for gid, g in self._goals.items() if g.terminal]
        for gid in doomed:
            del self._goals[gid]
        return len(doomed)

    def complete(self, goal_id, outcome=""):
        """Mark done; resume the parent if all its children are terminal."""
        goal = self._goals.get(goal_id)
        if goal is None or goal.terminal:
            return False
        goal.status = "done"
        goal.outcome = outcome
        self._resume_parent_if_finished(goal)
        return True

    def fail(self, goal_id, reason=""):
        """Mark failed; resume the parent if all its children are terminal."""
        goal = self._goals.get(goal_id)
        if goal is None or goal.terminal:
            return False
        goal.status = "failed"
        goal.outcome = reason
        self._resume_parent_if_finished(goal)
        return True

    def suspend(self, goal_id):
        """Pause an active goal; False when missing or not active."""
        goal = self._goals.get(goal_id)
        if goal is None or goal.status != "active":
            return False
        goal.status = "suspended"
        return True

    def resume(self, goal_id):
        """Reactivate a suspended goal; False when missing or not suspended."""
        goal = self._goals.get(goal_id)
        if goal is None or goal.status != "suspended":
            return False
        goal.status = "active"
        return True

    def decompose(self, goal_id, subdescriptions, priorities=None):
        """Split a goal into child goals linked by parent_id.

        The parent suspends while any child is non-terminal and resumes
        automatically once every child is done or failed.
        """
        parent = self._goals.get(goal_id)
        if parent is None or parent.terminal or not subdescriptions:
            return []
        if priorities is None:
            priorities = [max(0.0, parent.priority - 0.05 * i)
                          for i in range(len(subdescriptions))]
        else:
            priorities = list(priorities)
            while len(priorities) < len(subdescriptions):
                priorities.append(parent.priority)
        children = []
        for desc, prio in zip(subdescriptions, priorities):
            children.append(self.push(desc, priority=prio, parent_id=parent.id))
        parent.status = "suspended"
        return children

    def overdue(self, now=None):
        """Non-terminal goals whose deadline has passed."""
        t = self._t(now)
        return [g for g in self._goals.values()
                if g.deadline is not None and g.deadline < t and not g.terminal]

    def children(self, goal_id):
        """Direct child goals of the given goal."""
        return [g for g in self._goals.values() if g.parent_id == goal_id]

    def remove(self, goal_id):
        """Drop a goal; returns False when missing."""
        if goal_id not in self._goals:
            return False
        del self._goals[goal_id]
        return True

    def _resume_parent_if_finished(self, child):
        if not child.parent_id:
            return
        parent = self._goals.get(child.parent_id)
        if parent is None or parent.status != "suspended":
            return
        kids = self.children(parent.id)
        if kids and all(k.terminal for k in kids):
            parent.status = "active"

    def _fix_counter(self):
        top = 0
        for gid in self._goals:
            try:
                top = max(top, int(str(gid).rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        Goal._ids = itertools.count(top + 1)

    def to_dict(self):
        return {"goals": [g.to_dict() for g in self._goals.values()]}

    @classmethod
    def from_dict(cls, d, now=None):
        stack = cls(now=now)
        for gd in d.get("goals", []):
            goal = Goal.from_dict(gd)
            stack._goals[goal.id] = goal
        stack._fix_counter()
        return stack

    def __len__(self):
        return len(self._goals)

    def __repr__(self):
        return "GoalStack(%d goals, %d active)" % (len(self), len(self.active()))
