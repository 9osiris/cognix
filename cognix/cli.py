"""Command line interface for cognix."""

import argparse
import json
import os
import sys
import tempfile

from .config import Config
from .evals.harness import format_report, run_all
from .evals.metrics import summarize as summarize_metrics
from .evals.scenarios import build_scenarios
from .introspection.inspect import summarize, summary
from .runtime import CognitiveRuntime


def _make_runtime(args):
    config = Config.defaults()
    if args.config:
        config = config.merge(Config.load(args.config).to_dict())
    workspace = args.workspace or tempfile.mkdtemp(prefix="cognix_")
    return CognitiveRuntime(config=config, workspace_dir=workspace)


def cmd_run(args):
    rt = _make_runtime(args)
    if args.observe:
        for text in args.observe:
            report = rt.observe(text)
            print("observed salience=%.2f attended=%s" % (
                report.get("salience", 0.0), report.get("attended")))
    goal = rt.add_goal(args.goal, priority=args.priority)
    print("goal: %s" % goal.description)
    result = rt.run_goal(goal.id)
    print("ok=%s attempts=%d" % (result.get("ok"), result.get("attempts", 0)))
    if result.get("summary"):
        print("summary: %s" % result["summary"])
    if result.get("error"):
        print("error: %s" % result["error"])
    if args.save:
        path = rt.save(args.save)
        print("saved to %s" % path)
    return 0 if result.get("ok") else 1


def cmd_runall(args):
    rt = _make_runtime(args)
    for description in args.goals:
        rt.add_goal(description, priority=args.priority)
    results = rt.run_all_goals()
    ok = True
    for result in results:
        print("goal %s ok=%s" % (result.get("goal_id"), result.get("ok")))
        ok = ok and bool(result.get("ok"))
    return 0 if ok else 1


def cmd_repl(args):
    from .repl import REPL
    rt = _make_runtime(args)
    if args.load:
        rt = CognitiveRuntime.load(args.load, workspace_dir=rt.workspace_dir)
        print("loaded %s" % args.load)
    repl = REPL(rt)
    repl.loop()
    return 0


def cmd_eval(args):
    scenarios = build_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            print("unknown scenario: %s" % args.scenario)
            return 1
    workspace = tempfile.mkdtemp(prefix="cognix_eval_")

    def make_runtime():
        return CognitiveRuntime(workspace_dir=workspace)

    report = run_all(scenarios, make_runtime)
    print(format_report(report))
    if args.json:
        print(json.dumps(summarize_metrics(report), indent=2))
    return 0 if report.passed == report.total else 1


def cmd_batch(args):
    from .batch import run_batch_file
    rt = _make_runtime(args)
    results = run_batch_file(rt, args.script)
    failed = 0
    for name, ok, detail in results:
        print("%s %s %s" % ("ok" if ok else "FAIL", name, detail))
        if not ok:
            failed += 1
    return 1 if failed else 0


def cmd_events(args):
    if args.load:
        rt = CognitiveRuntime.load(args.load)
    else:
        rt = _make_runtime(args)
    events = rt.events.recent(args.type, n=args.n)
    if not events:
        print("no events")
        return 0
    for event in events:
        bits = ["%s #%d" % (event["type"], event["seq"])]
        for key, value in event["payload"].items():
            bits.append("%s=%s" % (key, value))
        print(" ".join(bits))
    return 0


def cmd_inspect(args):
    rt = CognitiveRuntime.load(args.state)
    state = rt.state_dict()
    print(summary(state))
    info = summarize(state)
    for key, value in info.items():
        print("  %s: %s" % (key, value))
    return 0


def cmd_dream(args):
    if args.load:
        rt = CognitiveRuntime.load(args.load)
    else:
        rt = _make_runtime(args)
    stats = rt.dream()
    print("dreams=%d insights=%d links=%d relations=%d" % (
        stats["dreams"], stats["insights"],
        stats["links_added"], stats["relations_proposed"]))
    for ep_id in stats["dream_ids"]:
        ep = rt.episodic.get(ep_id)
        if ep is not None:
            print("  - %s" % ep.summary[:100])
    if args.save:
        path = rt.save(args.save)
        print("saved to %s" % path)
    return 0


def cmd_sleep(args):
    if args.load:
        rt = CognitiveRuntime.load(args.load)
    else:
        rt = _make_runtime(args)
    stats = rt.sleep()
    dream_stats = stats.get("dream", {})
    print("sleep: episodes_added=%d dreams=%d insights=%d" % (
        stats.get("episodes_added", 0),
        dream_stats.get("dreams", 0),
        dream_stats.get("insights", 0)))
    for ep_id in dream_stats.get("dream_ids", []):
        ep = rt.episodic.get(ep_id)
        if ep is not None:
            print("  - %s" % ep.summary[:100])
    if args.save:
        path = rt.save(args.save)
        print("saved to %s" % path)
    return 0


def cmd_init(args):
    target = args.dir or "cognix_workspace"
    os.makedirs(target, exist_ok=True)
    os.makedirs(os.path.join(target, "plugins"), exist_ok=True)
    os.makedirs(os.path.join(target, "notes"), exist_ok=True)
    cfg = Config.defaults()
    cfg.set("tools.workspace_dir", os.path.abspath(target))
    cfg.set("tools.plugin_dir", os.path.abspath(os.path.join(target, "plugins")))
    cfg_path = os.path.join(target, "cognix.json")
    cfg.save(cfg_path)
    print("initialized workspace at %s" % os.path.abspath(target))
    print("config: %s" % cfg_path)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="cognix",
                                     description="cognitive architecture runtime")
    parser.add_argument("--config", default=None, help="path to cognix.json")
    parser.add_argument("--workspace", default=None, help="workspace dir for tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a goal")
    p_run.add_argument("goal", help="goal description")
    p_run.add_argument("--priority", type=float, default=0.7)
    p_run.add_argument("--observe", action="append", default=[],
                       help="seed an observation first (repeatable)")
    p_run.add_argument("--save", default=None, help="save state to path after")
    p_run.set_defaults(func=cmd_run)

    p_runall = sub.add_parser("runall", help="run several goals in priority order")
    p_runall.add_argument("goals", nargs="+", help="goal descriptions")
    p_runall.add_argument("--priority", type=float, default=0.7)
    p_runall.set_defaults(func=cmd_runall)

    p_repl = sub.add_parser("repl", help="interactive session")
    p_repl.add_argument("--load", default=None, help="load state file first")
    p_repl.set_defaults(func=cmd_repl)

    p_eval = sub.add_parser("eval", help="run the eval suite")
    p_eval.add_argument("--scenario", default=None, help="run one scenario")
    p_eval.add_argument("--json", action="store_true", help="print metrics json")
    p_eval.set_defaults(func=cmd_eval)

    p_batch = sub.add_parser("batch", help="run a .cx batch script")
    p_batch.add_argument("script", help="path to batch script")
    p_batch.set_defaults(func=cmd_batch)

    p_inspect = sub.add_parser("inspect", help="summarize a saved state file")
    p_inspect.add_argument("state", help="path to state json")
    p_inspect.set_defaults(func=cmd_inspect)

    p_init = sub.add_parser("init", help="create a workspace")
    p_init.add_argument("dir", nargs="?", default=None)
    p_init.set_defaults(func=cmd_init)

    p_dream = sub.add_parser("dream", help="run a dream cycle")
    p_dream.add_argument("--load", default=None,
                         help="load state file first")
    p_dream.add_argument("--save", default=None,
                         help="save state to path after")
    p_dream.set_defaults(func=cmd_dream)

    p_sleep = sub.add_parser("sleep", help="run a full sleep cycle")
    p_sleep.add_argument("--load", default=None,
                         help="load state file first")
    p_sleep.add_argument("--save", default=None,
                         help="save state to path after")
    p_sleep.set_defaults(func=cmd_sleep)

    p_events = sub.add_parser("events", help="show the event log")
    p_events.add_argument("--load", default=None,
                          help="load state file first")
    p_events.add_argument("--type", default=None,
                          help="only show this event type")
    p_events.add_argument("-n", type=int, default=20,
                          help="how many recent events to show")
    p_events.set_defaults(func=cmd_events)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
