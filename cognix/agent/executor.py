"""Step execution with retries, timeouts, and execution traces."""

import concurrent.futures
import re
import time

_OUTPUT_REF_RE = re.compile(r"<output of ([\w\-]+)>")

ERROR_KINDS = ("tool_missing", "bad_args", "tool_error", "precondition")


def classify_error(error):
    """Map a tool error message to one of the planner's failure kinds."""
    msg = str(error).lower()
    if ("missing" in msg or "required" in msg or "unexpected arg" in msg
            or "invalid" in msg):
        return "bad_args"
    if ("precondition" in msg or "not found" in msg or "no such" in msg
            or "does not exist" in msg):
        return "precondition"
    return "tool_error"


class StepResult:
    """Outcome of running one plan step."""

    def __init__(self, step_id, ok, output=None, error="", duration=0.0,
                 attempts=1, error_kind=None, tool=""):
        self.step_id = step_id
        self.ok = bool(ok)
        self.output = output
        self.error = error
        self.duration = duration
        self.attempts = attempts
        self.error_kind = error_kind
        self.tool = tool

    def to_dict(self):
        return {
            "step_id": self.step_id,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "duration": self.duration,
            "attempts": self.attempts,
            "error_kind": self.error_kind,
            "tool": self.tool,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            step_id=d["step_id"],
            ok=d.get("ok", False),
            output=d.get("output"),
            error=d.get("error", ""),
            duration=d.get("duration", 0.0),
            attempts=d.get("attempts", 1),
            error_kind=d.get("error_kind"),
            tool=d.get("tool", ""),
        )

    def __repr__(self):
        state = "ok" if self.ok else "fail(%s)" % self.error_kind
        return "StepResult(%r, %s)" % (self.step_id, state)


class ExecutionTrace:
    """Full record of one plan run."""

    def __init__(self, plan_id, results=None, started=None, ended=None,
                 success=False):
        self.plan_id = plan_id
        self.results = list(results) if results else []
        self.started = time.time() if started is None else started
        self.ended = ended
        self.success = success

    @property
    def duration(self):
        """Seconds elapsed; uses now when the run is still open."""
        end = self.ended if self.ended is not None else time.time()
        return max(0.0, end - self.started)

    def failed(self):
        """Results that did not succeed."""
        return [r for r in self.results if not r.ok]

    def summary(self):
        """One-line human readable run report."""
        total = len(self.results)
        ok = sum(1 for r in self.results if r.ok)
        text = "%d/%d steps ok in %.2fs" % (ok, total, self.duration)
        bad = self.failed()
        if bad:
            text += "; failed: " + ", ".join(
                "%s (%s)" % (r.step_id, r.error_kind) for r in bad)
        return text

    def to_dict(self):
        return {
            "plan_id": self.plan_id,
            "results": [r.to_dict() for r in self.results],
            "started": self.started,
            "ended": self.ended,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            plan_id=d["plan_id"],
            results=[StepResult.from_dict(r) for r in d.get("results", [])],
            started=d.get("started"),
            ended=d.get("ended"),
            success=d.get("success", False),
        )

    def __repr__(self):
        return "ExecutionTrace(%r, %d results, success=%r)" % (
            self.plan_id, len(self.results), self.success)


class Executor:
    """Runs plan steps against a tool registry with retries and timeouts."""

    def __init__(self, tools, max_retries=2, step_timeout=30.0,
                 backoff_base=0.01):
        self.tools = tools
        self.max_retries = max_retries
        self.step_timeout = step_timeout
        self.backoff_base = backoff_base

    def run_step(self, step):
        """Run one step; retry failures with exponential backoff."""
        start = time.time()
        if step.tool == "think":
            note = "thought: " + step.description
            return StepResult(step.id, True, note, "",
                              time.time() - start, 1, None, "think")
        if self.tools.get(step.tool) is None:
            return StepResult(step.id, False, None,
                              "tool %r is not registered" % step.tool,
                              time.time() - start, 1, "tool_missing", step.tool)
        retries = (step.max_retries if step.max_retries is not None
                   else self.max_retries)
        attempts = 0
        error, kind = "", "tool_error"
        while attempts <= retries:
            attempts += 1
            error, kind, done, output = self._attempt(step)
            if done:
                return StepResult(step.id, True, output, "",
                                  time.time() - start, attempts, None, step.tool)
            if attempts <= retries:
                delay = self.backoff_base * (2 ** (attempts - 1))
                time.sleep(min(0.5, delay))
        return StepResult(step.id, False, None, error,
                          time.time() - start, attempts, kind, step.tool)

    def _attempt(self, step):
        """One call attempt; returns (error, kind, done, output)."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.tools.call, step.tool, dict(step.args))
                res = future.result(timeout=self.step_timeout)
        except concurrent.futures.TimeoutError:
            return ("step timed out after %ss" % self.step_timeout,
                    "tool_error", False, None)
        except Exception as exc:
            msg = str(exc) or repr(exc)
            return msg, classify_error(msg), False, None
        if isinstance(res, dict) and res.get("ok"):
            return "", "tool_error", True, res.get("result")
        if isinstance(res, dict):
            msg = str(res.get("error", "unknown tool error"))
        else:
            msg = "bad tool result: %r" % (res,)
        return msg, classify_error(msg), False, None

    def run_plan(self, plan, on_step=None):
        """Run steps in order; stop on tool_missing, else keep going."""
        plan.status = "executing"
        trace = ExecutionTrace(plan.id)
        outputs = {}
        for step in plan.steps:
            resolved = step.copy()
            resolved.args = {k: self._resolve_refs(v, outputs)
                             for k, v in step.args.items()}
            result = self.run_step(resolved)
            trace.results.append(result)
            if result.ok:
                outputs[step.id] = result.output
            if on_step is not None:
                on_step(result)
            if not result.ok and result.error_kind == "tool_missing":
                break
        trace.ended = time.time()
        trace.success = bool(trace.results) and all(r.ok for r in trace.results)
        plan.status = "done" if trace.success else "failed"
        return trace

    def _resolve_refs(self, value, outputs):
        # swap "<output of step-3>" for the recorded output of that step
        if not isinstance(value, str):
            return value
        def swap(match):
            out = outputs.get(match.group(1))
            return str(out) if out is not None else match.group(0)
        return _OUTPUT_REF_RE.sub(swap, value)

    def dry_run(self, plan):
        """Walk the steps without calling tools; report per-step readiness.

        Returns a list of (step_id, ready, note) tuples.
        """
        names = set(self.tools.names())
        report = []
        for step in plan.steps:
            if step.tool == "think":
                report.append((step.id, True, "think step, no tool needed"))
            elif step.tool in names:
                report.append((step.id, True,
                               "tool %r is registered" % step.tool))
            else:
                report.append((step.id, False,
                               "tool %r is not registered" % step.tool))
        return report
