"""Eval harness: scripted scenarios that score the cognitive runtime."""

import time
import traceback


class CheckResult:
    def __init__(self, name, passed, detail=""):
        self.name = name
        self.passed = passed
        self.detail = detail


class Scenario:
    """A scripted eval: actions drive a runtime, checks score the outcome."""

    def __init__(self, name, description, actions, checks):
        self.name = name
        self.description = description
        self.actions = actions
        self.checks = checks


def run_action(runtime, action):
    # actions are tuples like ("observe", text) or ("goal", desc, priority)
    kind = action[0]
    if kind == "observe":
        return runtime.observe(action[1], source=action[2] if len(action) > 2 else "eval")
    if kind == "goal":
        priority = action[2] if len(action) > 2 else 0.5
        goal = runtime.add_goal(action[1], priority=priority)
        return {"goal_id": goal.id}
    if kind == "run":
        goal_id = action[1] if len(action) > 1 else None
        return runtime.run_goal(goal_id)
    if kind == "consolidate":
        return runtime.consolidate()
    if kind == "dream":
        return runtime.dream()
    if kind == "subscribe":
        # subscribe a collector so a scenario can check live delivery
        captured = getattr(runtime, "_captured_events", None)
        if captured is None:
            captured = runtime._captured_events = []
        runtime.events.subscribe(action[1], captured.append)
        return {"subscribed": action[1]}
    if kind == "cycle":
        return runtime.cycle(action[1] if len(action) > 1 else 1.0)
    if kind == "rehearse_all":
        for item in runtime.working.top(20):
            runtime.working.rehearse(item.id)
        return {"rehearsed": True}
    raise ValueError("unknown action: %r" % (kind,))


class ScenarioResult:
    def __init__(self, scenario):
        self.scenario = scenario
        self.checks = []
        self.error = None
        self.duration = 0.0

    @property
    def passed(self):
        return self.error is None and all(c.passed for c in self.checks)

    @property
    def score(self):
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)


def run_scenario(scenario, make_runtime):
    result = ScenarioResult(scenario)
    started = time.time()
    try:
        runtime = make_runtime()
        for action in scenario.actions:
            run_action(runtime, action)
        for name, func in scenario.checks:
            try:
                passed, detail = func(runtime)
            except Exception as exc:
                passed, detail = False, "check raised: %s" % exc
            result.checks.append(CheckResult(name, bool(passed), str(detail)))
        result.runtime = runtime
    except Exception:
        result.error = traceback.format_exc(limit=3)
    result.duration = time.time() - started
    return result


class EvalReport:
    def __init__(self, results):
        self.results = results

    @property
    def mean_score(self):
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def passed(self):
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self):
        return len(self.results)

    def to_dict(self):
        return {
            "mean_score": self.mean_score,
            "passed": self.passed,
            "total": self.total,
            "scenarios": [
                {"name": r.scenario.name, "score": r.score,
                 "passed": r.passed, "duration": round(r.duration, 2),
                 "checks": [{"name": c.name, "passed": c.passed,
                             "detail": c.detail} for c in r.checks],
                 "error": r.error}
                for r in self.results
            ],
        }


def run_all(scenarios, make_runtime):
    return EvalReport([run_scenario(s, make_runtime) for s in scenarios])


def format_report(report):
    lines = []
    lines.append("cognix eval report")
    lines.append("scenarios: %d passed %d/%d, mean score %.2f" % (
        report.total, report.passed, report.total, report.mean_score))
    lines.append("")
    for result in report.results:
        mark = "PASS" if result.passed else "FAIL"
        lines.append("[%s] %s (score %.2f, %.1fs)" % (
            mark, result.scenario.name, result.score, result.duration))
        for check in result.checks:
            cmark = "ok" if check.passed else "FAIL"
            line = "  %s %s" % (cmark, check.name)
            if check.detail:
                line += " -- %s" % check.detail[:100]
            lines.append(line)
        if result.error:
            lines.append("  error: " + result.error.splitlines()[-1][:120])
    return "\n".join(lines)
