# cognix

A cognitive architecture runtime in Python. Layered memory, salience-driven
attention, a planner-executor agent loop with reflection, real tools, a
built-in eval suite, and full state persistence. Standard library only,
no dependencies.

This is not a chatbot wrapper. It is a working model of how a mind-like
system can perceive, attend, remember, plan, act, and learn from the
outcome, all in one process you can inspect.

## architecture

```
                        +------------------+
                        |   observations   |
                        +--------+---------+
                                 |
                    +------------v------------+
                    | perception pipeline   |  parse, entities,
                    | entities/relations    |  relations, intent
                    +------------+----------+
                                 |
                    +------------v------------+
                    | salience scoring      |  novelty, relevance,
                    | novelty/relevance/    |  surprise, urgency
                    | surprise/urgency      |
                    +------------+----------+
                                 |
                    +------------v------------+
                    | focus + arousal       |  attention bottleneck,
                    | (bandwidth, depth)    |  processing depth
                    +------------+----------+
                                 |
        +------------------------+------------------------+
        |                        |                        |
+-------v-------+        +-------v-------+        +-------v-------+
| working       |        | belief store  |        | episodic      |
| memory        |------->| assert/revise |        | memory        |
| activation,   | beliefs| contradictions|        | record/recall |
| decay, chunks +-------^-------+        +-------+-------+
+-------+-------+                |                        |
        | consolidate            | consolidate            |
        v                        v                        v
+---------------------------------------------------------------+
| semantic memory: concepts, relations, spreading activation    |
+---------------------------------------------------------------+
        |
+-------v-------------------------------------------------------+
| offline replay: re-select salient episodes, link related ones,|
| strengthen semantic relations, rehearse matching working items |
+---------------------------------------------------------------+
        |
+-------v-------------------------------------------------------+
| agent loop                                                    |
|  goals -> metacognition picks strategy -> planner -> executor  |
|  -> reflection -> lessons as beliefs -> replan or done        |
+-------+-------------------------------+-----------------------+
        |                               |
+-------v-------+               +-------v-------+
| tool registry |               | eval harness  |
| calc, notes,  |               | 18 scenarios, |
| files, shell, |               | metrics,      |
| http, plugins |               | reports       |
+---------------+               +---------------+
        |
+-------v-------+
| persistence   |  versioned json, atomic writes, save/load
+---------------+
```

One cycle: an observation is parsed, scored for salience, and either
attended or dropped. Attended content enters working memory (activation
decays over time) and becomes beliefs. Salient events become episodes.
Consolidation moves rehearsed working items into episodic memory and
abstracts repeated episodes into semantic concepts and relations.
Offline replay then re-selects the most salient recent episodes, links
related ones, strengthens the semantic relations they share, and
rehearses matching working-memory items.

Goals go through metacognition (which picks a planning strategy),
planning (hierarchical decomposition, reactive, or means-ends),
execution (real tool calls with retries), and reflection (lessons become
beliefs, failures trigger replanning).

## quickstart

```bash
# run a goal
python -m cognix.cli run "calculate 12 * 8 and save the result in a note called math"

# interactive session
python -m cognix.cli repl

# batch script
python -m cognix.cli batch examples/demo.cx

# several goals in priority order
python -m cognix.cli runall "calculate 12 * 8" "write the result to the file answer.txt"

# run the eval suite
python -m cognix.cli eval

# save / inspect state
python -m cognix.cli run "calculate 2+2" --save state.json
python -m cognix.cli inspect state.json
```

Or from Python:

```python
from cognix import CognitiveRuntime

rt = CognitiveRuntime(workspace_dir="./work")
rt.observe("the deployment server is called mercury-04")
goal = rt.add_goal("calculate 12 * 8 and save the result in a note called math", priority=0.9)
result = rt.run_goal(goal.id)
print(result["ok"], result["summary"])
print(rt.recall("deployment server"))
rt.save("state.json")
```

## batch scripts

`.cx` files drive the runtime line by line:

```
# demo.cx
observe: the demo server is called atlas
goal: 0.8 calculate 5 * 5 and save the result in a note called demo
run
assert: goal-done
assert: note demo 25
```

## memory systems

- **working memory**: capacity-limited buffer (default 7 items). Items carry
  activation that decays exponentially; rehearse to keep them alive, pin to
  protect them, chunk to compress related items.
- **episodic memory**: timestamped episodes with tags, salience, and links.
  Recall strategies: recency, keyword (tf-idf-ish), temporal, mixed.
  Old low-salience episodes are pruned by a forgetting curve.
- **semantic memory**: concept/relation graph. Repeated observations
  consolidate into concepts; retrieval uses spreading activation.
- **consolidation**: the background process that moves rehearsed working
  items into episodes and abstracts repeated episodes into semantic
  knowledge. Runs automatically every N cycles and on demand.
- **associative recall**: one cue searches all three stores and ranks hits.
- **forgetting**: Ebbinghaus retention curves with rehearsal strengthening.
- **offline replay**: after each consolidation, the most salient recent
  episodes are re-selected (configurable budget), related episodes are
  linked, their shared semantic relations are strengthened, and matching
  working-memory items are rehearsed. Salient memories get stronger
  instead of quietly decaying.
- **dream cycle**: seeded recombination of salient but unlinked episodes
  into dream traces (tagged `dream`, linked with `dream-related-to`).
  Insights become beliefs at most 0.5 confidence, clearly labeled by
  source. CLI: `python -m cognix dream [--load state.json --save state.json]`.
  Config keys: `memory.dream_budget`, `memory.dreams_per_cycle`,
  `memory.dream_min_salience`, `memory.dream_seed`.

## attention

Every observation is scored on novelty (vs recent episodes), relevance
(vs active goals), surprise (vs current beliefs), and urgency (markers
like "urgent", deadlines). The focus bottleneck admits only the top few
above threshold. Arousal rises with salience and sets processing depth.

**curiosity** adds intrinsic motivation on top of salience: a tracker
records exposure to each topic, so repeated topics lose novelty while
fresh ones keep a small attention bonus. Topics nobody has touched in a
while read as boring; overexposed topics push the boredom metric up and
steer the system toward underexplored material. State persists across
save/load.

## agent loop

- **goals**: lifecycle (proposed, active, suspended, done, failed),
  priorities, deadlines, hierarchical decomposition with parent
  suspend/resume. `expand_goal(goal_id)` splits a multi-clause goal into
  subgoals via the planner's clause splitter; `run_all_goals()` runs
  every active goal in priority order, settling resumed parents from
  their children's outcomes instead of re-executing them.
- **planner**: three strategies. `decompose` splits multi-clause goals and
  maps clauses to tools by keyword. `reactive` fires one best tool.
  `means_ends` works backwards from the desired end state, prepending
  gather steps. `auto` picks per goal. Replanning handles bad args,
  missing tools, tool errors, and unmet preconditions, giving up after
  3 attempts. Compound goals chain data between steps: a later step can
  reference `<output of step-1>`.
- **executor**: dispatches tool calls with retries and backoff, records a
  full trace with timings.
- **reflection**: classifies outcomes, extracts lessons (stored as beliefs),
  and decides whether replanning is worthwhile.
- **metacognition**: strategy selection, plan confidence estimates,
  difficulty estimates. A **strategy ledger** records every run outcome
  per planning strategy, so `auto` learns which strategy actually wins
  over time instead of guessing from heuristics alone.

## tools

Built in: `calc` (safe arithmetic), `note_write/read/list/delete`,
`file_read/write/append/list/delete/search` (jailed to the workspace),
`shell`, `http_get`, `now_iso`, `echo`, `word_count`, `json_parse`.
Drop a `.py` file exposing `register_tools()` into the plugins dir to
add your own.

## evals

`python -m cognix.cli eval` runs 18 scripted scenarios: memory recall,
salience filtering, belief revision, tool-using planning, consolidation
and abstraction, clean failure paths, persistence roundtrips,
long-horizon multi-goal runs, associative recall, file tool safety,
dataflow chaining, replay strengthening, curiosity novelty,
metacognitive learning, belief contradiction, forgetting curves, and
hierarchical goals. Each scenario has checks; the harness reports
pass/fail, scores, and aggregate metrics.

## persistence

`rt.save(path)` writes versioned JSON atomically; `CognitiveRuntime.load`
restores everything: memories, beliefs, goals, arousal, curiosity,
replay policy, strategy ledger, trace, config. The CLI `init` command
scaffolds a workspace with config.

## design notes

- Everything is explicit state, no hidden globals, so the whole mind can
  be snapshotted, diffed, and restored.
- Salience is the gatekeeper: nothing reaches memory or planning without
  passing attention first.
- Reflection closes the loop: the system writes its own lessons into its
  beliefs, so it genuinely changes behavior over time.
- No network calls, no model APIs, no dependencies. It runs anywhere
  Python 3.8+ runs.

## layout

```
cognix/
  runtime.py        the cognitive cycle
  config.py         layered configuration
  persistence.py    versioned state save/load
  plugins.py        plugin loading
  cli.py repl.py batch.py
  memory/           working, episodic, semantic, consolidation,
                    associative, forgetting, replay, dream
  attention/        salience, focus, arousal
  perception/       pipeline, beliefs, schemas
  agent/            goals, planner, executor, reflection, metacognition
  tools/            registry, builtin tools
  evals/            harness, scenarios, metrics
  introspection/    tracer, state inspection
tests/              per-subsystem suites plus integration tests
```
