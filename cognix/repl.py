"""Interactive REPL for driving a cognitive runtime."""

import shlex

BANNER = """cognix repl. type /help for commands, /quit to exit."""


class REPL:
    def __init__(self, runtime, prompt="cognix> "):
        self.rt = runtime
        self.prompt = prompt
        self.commands = {
            "/help": (self.cmd_help, "show this help"),
            "/quit": (self.cmd_quit, "exit the repl"),
            "/observe": (self.cmd_observe, "/observe <text> - feed an observation"),
            "/goal": (self.cmd_goal, "/goal <priority> <text> - add a goal"),
            "/goals": (self.cmd_goals, "list goals by status"),
            "/run": (self.cmd_run, "/run [goal_id] - plan and execute a goal"),
            "/recall": (self.cmd_recall, "/recall <query> - associative recall"),
            "/beliefs": (self.cmd_beliefs, "list strongest beliefs"),
            "/working": (self.cmd_working, "show working memory"),
            "/episodes": (self.cmd_episodes, "show recent episodes"),
            "/concepts": (self.cmd_concepts, "show semantic concepts"),
            "/consolidate": (self.cmd_consolidate, "run consolidation now"),
            "/trace": (self.cmd_trace, "show the execution trace timeline"),
            "/save": (self.cmd_save, "/save <path> - save cognitive state"),
            "/load": (self.cmd_load, "/load <path> - load cognitive state"),
            "/tools": (self.cmd_tools, "list registered tools"),
            "/call": (self.cmd_call, "/call <tool> k=v ... - call a tool directly"),
            "/arousal": (self.cmd_arousal, "show arousal level"),
            "/curiosity": (self.cmd_curiosity, "/curiosity [text] - novelty of text or boredom state"),
            "/replay": (self.cmd_replay, "run an offline replay pass now"),
            "/ledger": (self.cmd_ledger, "show the strategy win/loss ledger"),
            "/expand": (self.cmd_expand, "/expand <goal_id> - split a goal into subgoals"),
            "/runall": (self.cmd_runall, "run every active goal in priority order"),
        }
        self.running = False

    # main loop

    def loop(self):
        print(BANNER)
        self.running = True
        while self.running:
            try:
                line = input(self.prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                self.dispatch(line)
            else:
                self.cmd_observe(line)

    def dispatch(self, line):
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            print("parse error: %s" % exc)
            return
        cmd, args = parts[0], parts[1:]
        handler = self.commands.get(cmd)
        if handler is None:
            print("unknown command %s (try /help)" % cmd)
            return
        try:
            handler[0](args)
        except Exception as exc:
            print("error: %s" % exc)

    # commands

    def cmd_help(self, args):
        for name, (_, help_text) in sorted(self.commands.items()):
            print("%-14s %s" % (name, help_text))
        print("bare text is treated as an observation.")

    def cmd_quit(self, args):
        self.running = False

    def cmd_observe(self, args):
        text = " ".join(args) if isinstance(args, list) else args
        report = self.rt.observe(text)
        print("salience=%.2f attended=%s signals=%s" % (
            report.get("salience", 0.0), report.get("attended"),
            ",".join(s["name"] for s in report.get("signals", []))))
        for belief in report.get("beliefs_added", [])[:5]:
            print("  belief: %s" % belief)

    def cmd_goal(self, args):
        if not args:
            print("usage: /goal <priority> <text>")
            return
        try:
            priority = float(args[0])
            text = " ".join(args[1:])
        except ValueError:
            priority, text = 0.5, " ".join(args)
        goal = self.rt.add_goal(text, priority=priority)
        print("goal %s: %s" % (goal.id, goal.description))

    def cmd_goals(self, args):
        for status in ("active", "proposed", "suspended", "done", "failed"):
            goals = self.rt.goals.by_status(status)
            for goal in goals:
                print("[%s] %s (p=%.2f) %s" % (
                    status, goal.id, goal.priority, goal.description[:70]))

    def cmd_run(self, args):
        goal_id = args[0] if args else None
        result = self.rt.run_goal(goal_id)
        print("ok=%s attempts=%s" % (result.get("ok"), result.get("attempts")))
        if result.get("summary"):
            print(result["summary"])
        if result.get("error"):
            print("error: %s" % result["error"])

    def cmd_recall(self, args):
        hits = self.rt.recall(" ".join(args), k=5)
        if not hits:
            print("nothing recalled")
            return
        for hit in hits:
            item = hit["item"]
            text = getattr(item, "summary", None) or getattr(item, "content", None)
            if text is None and hasattr(item, "proposition"):
                text = item.proposition
            print("[%s %.2f] %s" % (hit["store"], hit["score"], str(text)[:90]))

    def cmd_beliefs(self, args):
        for belief in self.rt.beliefs.strongest(15):
            print("%.2f %s" % (belief.confidence, belief.proposition[:90]))

    def cmd_working(self, args):
        for item in self.rt.working.top(10):
            print("%.2f [%s] %s" % (item.activation, item.kind, item.content[:80]))

    def cmd_episodes(self, args):
        for episode in self.rt.episodic.sequence(limit=10):
            print("%s %.2f %s" % (episode.id[:8], episode.salience,
                                  episode.summary[:80]))

    def cmd_concepts(self, args):
        for concept in self.rt.semantic.concepts()[:20]:
            print("%s (conf %.2f)" % (concept.name, concept.confidence))

    def cmd_consolidate(self, args):
        stats = self.rt.consolidate()
        print(", ".join("%s=%s" % kv for kv in sorted(stats.items())))

    def cmd_trace(self, args):
        print(self.rt.trace_timeline())

    def cmd_save(self, args):
        path = args[0] if args else "cognix_state.json"
        print("saved to %s" % self.rt.save(path))

    def cmd_load(self, args):
        if not args:
            print("usage: /load <path>")
            return
        from .runtime import CognitiveRuntime
        self.rt = CognitiveRuntime.load(args[0], workspace_dir=self.rt.workspace_dir)
        print("loaded %s" % args[0])

    def cmd_tools(self, args):
        print(self.rt.tools.help_text())

    def cmd_call(self, args):
        if not args:
            print("usage: /call <tool> k=v ...")
            return
        tool_args = {}
        for token in args[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                tool_args[key] = value
        result = self.rt.tools.call(args[0], tool_args)
        print(result)

    def cmd_arousal(self, args):
        print("arousal=%.2f depth=%d" % (
            self.rt.arousal.level(), self.rt.arousal.processing_depth()))

    def cmd_curiosity(self, args):
        tracker = self.rt.curiosity
        if args:
            text = " ".join(args)
            print("novelty=%.2f bonus=%.2f" % (
                tracker.novelty(text), tracker.bonus(text)))
        else:
            print("boredom=%.2f bored=%s familiar=%s" % (
                tracker.boredom(), tracker.is_bored(),
                ",".join(tracker.top_familiar(5)) or "none"))

    def cmd_replay(self, args):
        from .memory.replay import replay
        stats = replay(self.rt.episodic, self.rt.semantic, self.rt.working,
                       policy=self.rt.replay_policy, now=self.rt._now)
        print(", ".join("%s=%s" % kv for kv in sorted(stats.items())))

    def cmd_ledger(self, args):
        ledger = self.rt.meta.ledger()
        if not ledger:
            print("no strategy outcomes recorded yet")
            return
        for strategy, (wins, tries) in sorted(ledger.items()):
            print("%s: %d/%d (%.2f)" % (
                strategy, wins, tries,
                self.rt.meta.strategy_win_rate(strategy)))

    def cmd_expand(self, args):
        if not args:
            print("usage: /expand <goal_id>")
            return
        children = self.rt.expand_goal(args[0])
        if not children:
            print("nothing to expand (needs 2+ clauses)")
            return
        for child in children:
            print("  %s: %s" % (child.id, child.description[:60]))

    def cmd_runall(self, args):
        results = self.rt.run_all_goals()
        for result in results:
            print("goal %s ok=%s" % (result.get("goal_id"), result.get("ok")))
