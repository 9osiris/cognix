"""Batch mode: run a .cx script of cognitive operations.

Script syntax, one command per line, # comments:
  observe: some text here
  goal: 0.8 do the thing
  run
  run: <goal_id>
  consolidate
  cycle: 2.0
  rehearse
  replay
  recall: query text
  curiosity: some text here      # novelty score, or boredom state when empty
  expand                         # expand the last goal into subgoals
  expand: <goal_id>
  runall                         # run every active goal in priority order
  assert: belief <substring>        # a belief mentioning substring must exist
  assert: goal-done <goal_id>       # goal must have status done
  assert: note <name> <substring>    # note must contain substring
  save: path.json
  print: some literal text
"""

import shlex


BARE_COMMANDS = {"run", "consolidate", "rehearse", "cycle", "replay",
                 "expand", "runall", "curiosity"}


def parse_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    lowered = line.lower()
    if lowered in BARE_COMMANDS:
        return (lowered, "")
    if ":" not in line:
        return ("observe", line)
    kind, rest = line.split(":", 1)
    return (kind.strip().lower(), rest.strip())


def run_batch_file(runtime, path):
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    return run_batch(runtime, lines)


def run_batch(runtime, lines):
    results = []
    last_goal_id = None
    for lineno, raw in enumerate(lines, 1):
        parsed = parse_line(raw)
        if parsed is None:
            continue
        kind, rest = parsed
        label = "line %d (%s)" % (lineno, kind)
        try:
            if kind == "observe":
                report = runtime.observe(rest)
                results.append((label, True, "salience=%.2f" % report.get("salience", 0)))
            elif kind == "goal":
                priority, _, text = rest.partition(" ")
                try:
                    priority = float(priority)
                except ValueError:
                    text, priority = rest, 0.5
                goal = runtime.add_goal(text.strip(), priority=priority)
                last_goal_id = goal.id
                results.append((label, True, "goal %s" % goal.id))
            elif kind == "run":
                result = runtime.run_goal(rest or last_goal_id)
                results.append((label, bool(result.get("ok")),
                                result.get("summary") or result.get("error", "")))
            elif kind == "consolidate":
                stats = runtime.consolidate()
                results.append((label, True, str(stats)))
            elif kind == "cycle":
                runtime.cycle(float(rest or 1.0))
                results.append((label, True, "ticked"))
            elif kind == "rehearse":
                for item in runtime.working.top(20):
                    runtime.working.rehearse(item.id)
                results.append((label, True, "rehearsed"))
            elif kind == "replay":
                stats = runtime.consolidate()["replay"]
                results.append((label, True, "replayed=%d" % stats["replayed"]))
            elif kind == "curiosity":
                if rest:
                    score = runtime.curiosity.novelty(rest)
                    results.append((label, True, "novelty=%.2f" % score))
                else:
                    results.append((label, True, "boredom=%.2f" % runtime.curiosity.boredom()))
            elif kind == "expand":
                target = rest or last_goal_id
                children = runtime.expand_goal(target) if target else []
                results.append((label, True, "%d subgoals" % len(children)))
            elif kind == "runall":
                done = runtime.run_all_goals()
                ok = all(r.get("ok") for r in done)
                results.append((label, ok, "%d goals run" % len(done)))
            elif kind == "recall":
                hits = runtime.recall(rest, k=5)
                results.append((label, True, "%d hits" % len(hits)))
            elif kind == "save":
                path = runtime.save(rest)
                results.append((label, True, path))
            elif kind == "print":
                print(rest)
                results.append((label, True, "printed"))
            elif kind == "assert":
                ok, detail = run_assert(runtime, rest, last_goal_id)
                results.append((label, ok, detail))
            else:
                results.append((label, False, "unknown command %r" % kind))
        except Exception as exc:
            results.append((label, False, "raised: %s" % exc))
    return results


def run_assert(runtime, rest, last_goal_id):
    parts = shlex.split(rest)
    if not parts:
        return False, "empty assert"
    kind = parts[0]
    if kind == "belief":
        needle = " ".join(parts[1:]).lower()
        found = [b for b in runtime.beliefs.strongest(200)
                 if needle in b.proposition.lower()]
        return (len(found) > 0, "%d matching beliefs" % len(found))
    if kind == "goal-done":
        goal_id = parts[1] if len(parts) > 1 else last_goal_id
        goal = runtime.goals.get(goal_id) if goal_id else None
        ok = goal is not None and goal.status == "done"
        return ok, "status=%s" % (goal.status if goal else "missing")
    if kind == "note":
        name = parts[1] if len(parts) > 1 else ""
        needle = " ".join(parts[2:]) if len(parts) > 2 else ""
        result = runtime.tools.call("note_read", {"name": name})
        ok = result.get("ok") and needle in str(result.get("result", ""))
        return ok, "note check"
    if kind == "episodes":
        minimum = int(parts[1]) if len(parts) > 1 else 1
        count = runtime.episodic.count()
        return count >= minimum, "episodes=%d" % count
    return False, "unknown assert kind %r" % kind
