"""Metacognitive control: strategy choice, confidence, difficulty."""

import re

from .planner import split_clauses

_TARGET_CUES = ("have", "ensure", "make sure")

_KNOWN_WORDS = frozenset((
    "a about after again all also always am an and any are around as at be "
    "because been before between both but by can cannot could did do does "
    "doing done down each even every few file for from get had has have he "
    "her here hers herself him his how i if in into is it its just know like "
    "make me more most my new no not now of off on once one only or other "
    "our out over own run same see should so some such take than that the "
    "their them then there these they this those through to too under up use "
    "using was we were what when where which while who will with would you "
    "your report config server backup summary text data code test plan goal "
    "step tool time day"
).split())


class MetaCognition:
    """Decides how to think about a goal before planning starts."""

    def __init__(self, tools=None):
        self._tools = tools
        # strategy -> [successes, attempts]; learns what works over time
        self._ledger = {}

    def record_outcome(self, strategy, succeeded):
        """Log whether a strategy worked; feeds future choices."""
        wins, tries = self._ledger.get(strategy, (0, 0))
        self._ledger[strategy] = (
            wins + (1 if succeeded else 0),
            tries + 1,
        )

    def strategy_win_rate(self, strategy):
        """Observed success rate, 0.5 when never tried."""
        wins, tries = self._ledger.get(strategy, (0, 0))
        if tries == 0:
            return 0.5
        return wins / tries

    def best_strategy(self):
        """Highest win rate among tried strategies, None when untried."""
        if not self._ledger:
            return None
        return max(self._ledger, key=self.strategy_win_rate)

    def ledger(self):
        """Copy of the raw {strategy: (wins, tries)} ledger."""
        return dict(self._ledger)

    def ledger_to_dict(self):
        """Serialize the ledger for persistence."""
        return {s: {"wins": w, "tries": t}
                for s, (w, t) in self._ledger.items()}

    def ledger_from_dict(self, data):
        """Restore the ledger from persisted state."""
        self._ledger = {
            s: (int(v.get("wins", 0)), int(v.get("tries", 0)))
            for s, v in (data or {}).items()
        }

    def choose_strategy(self, goal, beliefs, past_failures=0):
        """Pick a planning strategy for the goal.

        Reactive after repeated failures or for simple goals, means_ends
        when the goal names a target state, decompose otherwise. When
        the ledger shows a clear winner (>= 0.75 win rate over 3+ tries)
        and the heuristics are undecided, the winner is preferred.
        """
        desc = goal.description.lower()
        if past_failures >= 2:
            return "reactive"
        if any(cue in desc for cue in _TARGET_CUES):
            return "means_ends"
        if len(split_clauses(goal.description)) <= 1:
            return "reactive"
        best = self.best_strategy()
        if best is not None:
            wins, tries = self._ledger[best]
            if tries >= 3 and self.strategy_win_rate(best) >= 0.75:
                return best
        return "decompose"

    def confidence_in_plan(self, plan, beliefs):
        """Fraction of steps with known tools times mean required-arg coverage."""
        if not plan.steps:
            return 0.0
        names = set(self._tools.names()) if self._tools else set()
        schemas = {}
        if self._tools:
            schemas = {s["name"]: s for s in self._tools.schemas()}
        known = 0
        coverages = []
        for step in plan.steps:
            if step.tool == "think":
                known += 1
                coverages.append(1.0)
                continue
            if step.tool in names:
                known += 1
            params = schemas.get(step.tool, {}).get("params", {})
            required = [p for p, d in params.items() if d.get("required")]
            if not required:
                coverages.append(1.0)
            else:
                hit = sum(1 for p in required if p in (step.args or {}))
                coverages.append(hit / len(required))
        tool_frac = known / len(plan.steps)
        arg_frac = sum(coverages) / len(coverages)
        return round(tool_frac * arg_frac, 3)

    def should_ask_for_help(self, goal, failures=0):
        """Escalate to the user after repeated failures."""
        return failures >= 3

    def next_strategy(self, current, failure_kind):
        """Strategy to try after a failure of the given kind.

        Missing tools go reactive (fewer tool assumptions), missing
        preconditions go means_ends (gather inputs first), bad args keeps
        the current strategy with fixed args, anything else flips between
        reactive and decompose.
        """
        if failure_kind == "tool_missing":
            return "reactive"
        if failure_kind == "precondition":
            return "means_ends"
        if failure_kind == "bad_args":
            return current
        return "decompose" if current == "reactive" else "reactive"

    def advise(self, goal, beliefs, past_failures=0):
        """One metacognitive decision bundle for a goal."""
        return {
            "strategy": self.choose_strategy(goal, beliefs, past_failures),
            "difficulty": self.estimate_difficulty(goal),
            "ask_for_help": self.should_ask_for_help(goal, past_failures),
            "past_failures": past_failures,
        }

    def estimate_difficulty(self, goal):
        """0..1 difficulty from length, clause count, and unknown words."""
        desc = goal.description or ""
        words = re.findall(r"[a-z0-9]+", desc.lower())
        length = min(1.0, len(desc) / 200.0)
        clauses = max(1, len(split_clauses(desc)))
        clause_score = min(1.0, (clauses - 1) / 3.0)
        unknown = sum(1 for w in words if w not in _KNOWN_WORDS)
        unknown_score = (unknown / len(words)) if words else 0.0
        score = 0.3 * length + 0.4 * clause_score + 0.3 * unknown_score
        return round(min(1.0, max(0.0, score)), 3)
