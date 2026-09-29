"""Plan generation, tool matching, and replanning."""

import itertools
import re
import time

STRATEGIES = ("auto", "decompose", "reactive", "means_ends")
PLAN_STATUSES = ("draft", "ready", "executing", "done", "failed")
MAX_REPLAN_ATTEMPTS = 3

_CLAUSE_PATTERNS = (
    r"\s*\d+[.)]\s+",
    r"\s*\bthen\b\s*",
    r"\s*;\s*",
)
_SENTENCE_RE = re.compile(r"[.!?]+\s+|\n+")
_AND_RE = re.compile(r"\s+and\s+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:/[\w\-.]+)+/?|[\w][\w\-.]*\.[\w]{1,5}")
_MATH_RE = re.compile(r"\d[\d\s+\-*/().^%]*\d")
_NAME_RE = re.compile(r"(?:called|named)\s+[\"']?([\w][\w\-]*)[\"']?", re.IGNORECASE)
_OUTPUT_REF_RE = re.compile(r"<output of ([\w\-]+)>")
_CONTENT_PARAMS = ("content", "text", "data", "body", "input_text", "document")
_PATH_PARAMS = ("path", "file", "filename", "filepath")
_EXPR_PARAMS = ("expression", "formula")
_NAME_PARAMS = ("name", "title", "label")
_TEXT_PARAMS = ("text", "content", "input", "query", "question", "prompt",
                "description")


def split_clauses(text, limit=5):
    """Split goal text into 2-5 clauses on cue words and punctuation."""
    chunks = [text.strip()]
    for pattern in _CLAUSE_PATTERNS:
        if len(chunks) > 1:
            break
        chunks = [c for c in re.split(pattern, chunks[0], flags=re.IGNORECASE)
                  if c.strip()]
    if len(chunks) == 1:
        chunks = [c for c in _SENTENCE_RE.split(chunks[0]) if c.strip()]
    if len(chunks) == 1:
        chunks = [c for c in _AND_RE.split(chunks[0]) if c.strip()]
    if len(chunks) > limit:
        chunks = chunks[:limit - 1] + [" ".join(chunks[limit - 1:])]
    return [c.strip(" .") for c in chunks if c.strip(" .")]


def _score_tool(text, schema):
    """Keyword overlap between text and a tool schema; higher is better."""
    text = text.lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    score = 0
    for token in re.findall(r"[a-z0-9]+", schema["name"].lower()):
        if token in words or token in text:
            score += 3
    desc = schema.get("description", "").lower()
    for word in words:
        if word in desc:
            score += 1
    for pname, pdef in schema.get("params", {}).items():
        for token in re.findall(r"[a-z0-9]+", pname.lower()):
            if token in words:
                score += 2
        pdesc = str(pdef.get("description", "")).lower()
        for word in words:
            if word in pdesc:
                score += 1
    return score


def _looks_like_content(pname):
    return pname in _CONTENT_PARAMS


def _default_for(ptype):
    return {"string": "", "number": 0, "integer": 0, "boolean": False,
            "array": [], "object": {}}.get(ptype, "")


class PlanStep:
    """One executable plan step."""

    _ids = itertools.count(1)

    def __init__(self, description, tool="think", args=None, precondition="",
                 expected="", max_retries=2, id=None):
        self.id = id if id is not None else "step-%d" % next(PlanStep._ids)
        self.description = description
        self.tool = tool
        self.args = dict(args) if args else {}
        self.precondition = precondition
        self.expected = expected
        self.max_retries = max_retries

    def copy(self):
        """Independent copy with the same id."""
        return PlanStep.from_dict(self.to_dict())

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "tool": self.tool,
            "args": self.args,
            "precondition": self.precondition,
            "expected": self.expected,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            description=d["description"],
            tool=d.get("tool", "think"),
            args=d.get("args"),
            precondition=d.get("precondition", ""),
            expected=d.get("expected", ""),
            max_retries=d.get("max_retries", 2),
            id=d.get("id"),
        )

    def __repr__(self):
        return "PlanStep(%r, tool=%r)" % (self.id, self.tool)


class Plan:
    """An ordered list of steps aimed at one goal."""

    _ids = itertools.count(1)

    def __init__(self, goal_id, steps=None, status="draft", strategy="auto",
                 created=None, attempts=1, id=None):
        if status not in PLAN_STATUSES:
            raise ValueError("unknown plan status: %r" % (status,))
        self.id = id if id is not None else "plan-%d" % next(Plan._ids)
        self.goal_id = goal_id
        self.steps = list(steps) if steps else []
        self.status = status
        self.strategy = strategy
        self.created = time.time() if created is None else created
        self.attempts = attempts

    def step(self, step_id):
        """Return the step with the given id or None."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def to_dict(self):
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
            "strategy": self.strategy,
            "created": self.created,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, d):
        plan = cls(
            goal_id=d["goal_id"],
            steps=[PlanStep.from_dict(s) for s in d.get("steps", [])],
            status=d.get("status", "draft"),
            strategy=d.get("strategy", "auto"),
            created=d.get("created"),
            attempts=d.get("attempts", 1),
            id=d.get("id"),
        )
        return plan

    def __repr__(self):
        return "Plan(%r, %d steps, %r)" % (self.id, len(self.steps), self.status)


class Planner:
    """Turns goals into tool-using plans and repairs them on failure."""

    def __init__(self, tools):
        self.tools = tools

    def plan(self, goal, beliefs, strategy="auto"):
        """Build a plan; auto picks decompose for multi-clause goals."""
        if strategy == "auto":
            strategy = ("decompose" if len(split_clauses(goal.description)) > 1
                        else "reactive")
        if strategy == "decompose":
            return self._plan_decompose(goal, beliefs)
        if strategy == "reactive":
            return self._plan_reactive(goal, beliefs)
        if strategy == "means_ends":
            return self._plan_means_ends(goal, beliefs)
        raise ValueError("unknown strategy: %r" % (strategy,))

    def _plan_decompose(self, goal, beliefs):
        steps = []
        for clause in split_clauses(goal.description):
            tool, _ = self._best_tool(clause)
            args = self._fill_args(tool, clause, beliefs)
            steps.append(PlanStep(clause, tool=tool, args=args,
                                  expected="done: %s" % clause))
        self._wire_dataflow(steps)
        return Plan(goal.id, steps, status="ready", strategy="decompose")

    def _wire_dataflow(self, steps):
        # later steps can consume the prior step's output as content
        for i in range(1, len(steps)):
            params = self._schema(steps[i].tool).get("params", {})
            for pname, pdef in params.items():
                if not pdef.get("required"):
                    continue
                if not (pname in _TEXT_PARAMS or _looks_like_content(pname)):
                    continue
                current = steps[i].args.get(pname)
                if current is None or current == steps[i].description:
                    steps[i].args[pname] = "<output of %s>" % steps[i - 1].id

    def _plan_reactive(self, goal, beliefs):
        tool, _ = self._best_tool(goal.description)
        args = self._fill_args(tool, goal.description, beliefs)
        step = PlanStep(goal.description, tool=tool, args=args,
                        expected="%s complete" % goal.description[:60])
        return Plan(goal.id, [step], status="ready", strategy="reactive")

    def _plan_means_ends(self, goal, beliefs):
        """Work backwards: prepend gather steps for missing content inputs."""
        desc = goal.description
        tool, _ = self._best_tool(desc)
        schema = self._schema(tool)
        required = [p for p, d in schema.get("params", {}).items()
                    if d.get("required")]
        args = self._fill_args(tool, desc, beliefs, skip=_CONTENT_PARAMS)
        steps = []
        missing = [p for p in required
                   if _looks_like_content(p) and p not in args]
        path = _PATH_RE.search(desc)
        if missing and path:
            fpath = path.group(0)
            gather = self._gather_step(fpath)
            steps.append(gather)
            for pname in missing:
                args[pname] = "<output of %s>" % gather.id
        steps.append(PlanStep(desc, tool=tool, args=args,
                              expected="%s complete" % desc[:60]))
        return Plan(goal.id, steps, status="ready", strategy="means_ends")

    def _gather_step(self, fpath):
        reader = self._find_reader()
        if reader:
            return PlanStep("gather content from %s" % fpath, tool=reader,
                            args={"path": fpath},
                            expected="content of %s available" % fpath)
        return PlanStep("figure out how to get content from %s" % fpath,
                        tool="think", expected="source for content identified")

    def replan(self, plan, failure, beliefs):
        """Repair a plan after a step failure; gives up after 3 attempts."""
        attempts = plan.attempts + 1
        steps = [s.copy() for s in plan.steps]
        if attempts > MAX_REPLAN_ATTEMPTS:
            return Plan(plan.goal_id, steps, status="failed",
                        strategy=plan.strategy, attempts=attempts)
        kind = failure.get("kind", "tool_error")
        handler = {
            "tool_missing": self._fix_tool_missing,
            "bad_args": self._fix_bad_args,
            "precondition": self._fix_precondition,
        }.get(kind, self._fix_tool_error)
        steps = handler(steps, failure.get("step_id"), failure, beliefs)
        return Plan(plan.goal_id, steps, status="ready",
                    strategy=plan.strategy, attempts=attempts)

    def validate(self, plan):
        """Return a list of issues: unknown tools, missing args, empty plan."""
        issues = []
        if not plan.steps:
            issues.append("plan has no steps")
        names = set(self.tools.names())
        for step in plan.steps:
            if step.tool != "think" and step.tool not in names:
                issues.append("step %s: unknown tool %r" % (step.id, step.tool))
                continue
            schema = self._schema(step.tool)
            for pname, pdef in schema.get("params", {}).items():
                if pdef.get("required") and pname not in step.args:
                    issues.append("step %s: missing required arg %r for tool %r"
                                  % (step.id, pname, step.tool))
        return issues

    def explain(self, plan):
        """Human readable multi-line description of a plan."""
        lines = ["plan %s for goal %s (%s, attempt %d)" % (
            plan.id, plan.goal_id, plan.strategy, plan.attempts)]
        for i, step in enumerate(plan.steps, 1):
            lines.append("  %d. [%s] %s" % (i, step.tool, step.description))
            if step.args:
                lines.append("     args: " + ", ".join(
                    "%s=%r" % (k, v) for k, v in step.args.items()))
            if step.precondition:
                lines.append("     needs: %s" % step.precondition)
            if step.expected:
                lines.append("     expect: %s" % step.expected)
        return "\n".join(lines)

    def _fix_tool_missing(self, steps, step_id, failure, beliefs):
        for step in steps:
            if step.id == step_id:
                tool, _ = self._best_tool(step.description, exclude=(step.tool,))
                step.tool = tool
                step.args = self._fill_args(tool, step.description, beliefs)
        return steps

    def _fix_bad_args(self, steps, step_id, failure, beliefs):
        for step in steps:
            if step.id == step_id:
                params = self._schema(step.tool).get("params", {})
                fixed = {k: v for k, v in step.args.items() if k in params}
                for pname, pdef in params.items():
                    if pdef.get("required") and pname not in fixed:
                        filled = self._extract_arg(pname, pdef, step.description)
                        fixed[pname] = (filled if filled is not None
                                        else _default_for(pdef.get("type", "string")))
                step.args = fixed
        return steps

    def _fix_precondition(self, steps, step_id, failure, beliefs):
        idx = next((i for i, s in enumerate(steps) if s.id == step_id), 0)
        err = str(failure.get("error", ""))
        match = _PATH_RE.search(err)
        if match:
            steps.insert(idx, self._gather_step(match.group(0)))
        else:
            steps.insert(idx, PlanStep(
                "gather missing precondition: %s" % err[:80], tool="think",
                expected="precondition satisfied"))
        return steps

    def _fix_tool_error(self, steps, step_id, failure, beliefs):
        idx = next((i for i, s in enumerate(steps) if s.id == step_id),
                   len(steps) - 1)
        err = str(failure.get("error", ""))[:80]
        steps.insert(idx + 1, PlanStep(
            "fallback reasoning after tool error: %s" % err, tool="think",
            expected="recovery approach chosen"))
        steps[idx].max_retries = (steps[idx].max_retries or 0) + 1
        return steps

    def _best_tool(self, text, exclude=()):
        best, best_score = "think", 0
        for schema in self.tools.schemas():
            name = schema["name"]
            if name in exclude:
                continue
            score = _score_tool(text, schema)
            if score > best_score:
                best, best_score = name, score
        return best, best_score

    def _schema(self, tool):
        for schema in self.tools.schemas():
            if schema["name"] == tool:
                return schema
        return {"name": tool, "params": {}}

    def _find_reader(self):
        """First registered tool whose name or description mentions reading."""
        for schema in self.tools.schemas():
            blob = (schema["name"] + " " + schema.get("description", "")).lower()
            if "read" in blob:
                return schema["name"]
        return None

    def _fill_args(self, tool, clause, beliefs, skip=()):
        """Fill required args from clause text, falling back to beliefs."""
        schema = self._schema(tool)
        args = {}
        for pname, pdef in schema.get("params", {}).items():
            if not pdef.get("required") or pname in skip:
                continue
            value = self._extract_arg(pname, pdef, clause)
            if value is None and beliefs is not None:
                value = self._belief_arg(pname, beliefs)
            if value is not None:
                args[pname] = value
        return args

    def _extract_arg(self, pname, pdef, clause):
        if pdef.get("type", "string") != "string":
            return None
        if pname in _NAME_PARAMS:
            match = _NAME_RE.search(clause)
            if match:
                return match.group(1)
            quoted = re.search(r"[\"']([\w][\w\-]*)[\"']", clause)
            return quoted.group(1) if quoted else None
        if pname in _PATH_PARAMS:
            match = _PATH_RE.search(clause)
            return match.group(0) if match else None
        if pname in _EXPR_PARAMS:
            match = _MATH_RE.search(clause)
            return match.group(0).strip() if match else None
        if pname in _TEXT_PARAMS:
            return clause
        return None

    def _belief_arg(self, pname, beliefs):
        try:
            hits = beliefs.query(pname)
        except Exception:
            return None
        for hit in hits or []:
            prop = getattr(hit, "proposition", "")
            match = _PATH_RE.search(str(prop))
            return match.group(0) if match else str(prop)
        return None
