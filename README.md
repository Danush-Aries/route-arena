# route-arena

**A typed, dependency-light LLM router.** Given a prompt, route-arena picks the
best model across a **cost / quality / latency Pareto frontier** — using three
selectable strategies — then lets you A/B two strategies live in an *arena*.
Runs fully offline in **mock mode by default** (zero paid API keys); real
OpenAI / Anthropic APIs are optional via environment variables. No LangChain,
no heavyweight framework — the core has **zero required runtime dependencies**.

```bash
pip install -e .        # core only, mock mode, zero deps
route-arena bench       # see the leaderboard + routing benchmark
```

---

## Why this exists

Serving every request with `gpt-4o` or `claude-sonnet` is simple but wasteful.
Most prompts are handled just as well by a model that is **10–40x cheaper** and
lower latency. A router turns "one model for everything" into "the right model
per request", which is one of the highest-leverage cost levers in production
LLM infra.

route-arena makes that decision **explainable and measurable**:

- **Per-category quality** comes from a committed offline eval (`results.json`),
  not vibes — a model that is great at code may be mediocre at summarization.
- **Cost** is estimated from real per-token list pricing and per-category output
  length priors.
- **Latency** is a rolling p50 that starts from a prior and adapts to live
  observations fed back from the arena.
- Every routing decision ships with a one-line **rationale**.

---

## Results (offline eval, mock mode — deterministic)

Quality is an ELO-style score (`1000 + 600 * mean_score`) measured by the eval
harness. Prices are USD per 1M input/output tokens.

| model         | code | summarization | reasoning | extraction | avg  | in/out $/M    |
|---------------|-----:|--------------:|----------:|-----------:|-----:|---------------|
| claude-sonnet | 1541 | 1536          | 1600      | 1554       | 1558 | $3.00 / $15.00 |
| gpt-4o        | 1430 | 1460          | 1600      | 1600       | 1523 | $2.50 / $10.00 |
| claude-haiku  | 1344 | 1350          | 1527      | 1561       | 1445 | $0.80 / $4.00  |
| gpt-4o-mini   | 1344 | 1380          | 1468      | 1561       | 1438 | $0.15 / $0.60  |
| llama-3-70b   | 1381 | 1360          | 1393      | 1561       | 1424 | $0.59 / $0.79  |

### The Pareto frontier (cost vs quality)

Higher and to the left is better. The **balanced** strategy only ever picks a
model on the non-dominated frontier — never one that is strictly worse on every
axis.

```
 quality
  1560 |  * claude-sonnet
       |
  1520 |        * gpt-4o
       |
  1445 |  * claude-haiku
  1438 |     * gpt-4o-mini
  1424 |  * llama-3-70b
       +--------------------------------------------- cost ($/1M out)
          0.6   0.8        4          10        15
       cheap  <----------------------------->  premium
```

`gpt-4o-mini` and `llama-3-70b` anchor the cheap end; `claude-sonnet` anchors
the quality end; `gpt-4o` is the premium generalist. The router's job is to
walk this frontier per prompt.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -e ".[pretty]" for rich tables

route-arena bench                                    # leaderboard + benchmark
route-arena route "write a python function to reverse a linked list"
route-arena arena "write a python function to reverse a linked list"
```

Everything above works with **no API keys**. To go live, `cp .env.example .env`,
add real keys, and export them — matching models then call the real APIs and the
rest stay mocked.

### `route` — what each strategy picks

```
Prompt: write a python function to reverse a linked list
Category: code   mode: mock (offline, no API keys)

  cost_optimal  -> claude-sonnet  | q=  1541 | $0.00529 |  1600ms
      cheapest model clearing the quality floor (quality=1541, $0.00529)
  quality_first -> claude-sonnet  | q=  1541 | $0.00529 |  1600ms
      top quality=1541 (best available is claude-sonnet); cost $0.00529.
  balanced      -> llama-3-70b    | q=  1381 | $0.00028 |   900ms
      best weighted blend on the Pareto frontier (quality=1381, $0.00028, 900ms)
```

### `arena` — live A/B of two strategies

```
Arena  mode: mock
Prompt: write a python function to reverse a linked list
Category: code

  Side A balanced -> llama-3-70b
    projected: q=1381  $0.00028  900ms   realized: 900ms  score=n/a
  Side B quality_first -> claude-sonnet
    projected: q=1541  $0.00529  1600ms  realized: 1600ms  score=n/a

Verdict: Tradeoff: Side B (claude-sonnet) buys +161 quality for +$0.00500
($0.000031/point). Side A (llama-3-70b) is the budget pick.
```

Over the four benchmark prompts, `balanced` costs **~74% less** than
`quality_first` while staying on the Pareto frontier.

---

## How routing decisions are made

Every prompt flows through the same pipeline (`route_arena/router/`):

1. **Classify** the prompt into one of `code | summarization | reasoning |
   extraction` via inspectable keyword scoring (`classify_prompt`). Swappable
   for an embedding classifier without changing the router contract.
2. **Estimate tokens**: input from a ~4-chars/token heuristic, output from a
   per-category prior. This drives the cost estimate.
3. **Build candidates**: for each of the 5 models, compute
   `cost_usd` (from list pricing), `latency_ms` (rolling p50, falling back to a
   nominal prior), and `quality` (per-category ELO from `results.json`).
4. **Normalize** all three objectives to `[0, 1]` (higher = better).
5. **Compute the Pareto frontier**: drop any model that is dominated (no better
   on any axis) so no strategy can pick a strictly-worse model.
6. **Apply the strategy**:

   | strategy        | rule |
   |-----------------|------|
   | `cost_optimal`  | cheapest model that clears a relative **quality floor** (80% of the worst→best quality span) — never trades quality away for pennies |
   | `quality_first` | highest quality; ties broken by cost, then latency |
   | `balanced`      | best **weighted blend** (`quality 0.5 / cost 0.3 / latency 0.2`) *restricted to the Pareto frontier* |

7. **Explain**: attach a human-readable rationale to the `RoutingDecision`.

The **arena** then actually invokes the chosen model (mock or live), scores the
realized output against a reference when one exists, feeds the observed latency
back into the rolling history, and prints a cost-per-quality-point verdict.

---

## Architecture

```
route_arena/
├── models.py            # ModelSpec registry: pricing, latency priors, context
├── router/
│   ├── scoring.py       # classify, token/cost estimate, normalize, Pareto, latency history
│   └── policy.py        # Router + the 3 strategies + RoutingDecision
├── eval/
│   ├── prompts.py       # ~30 labeled prompts across 4 categories
│   ├── scorer.py        # deterministic ROUGE-1-style + exact/containment scorer
│   ├── harness.py       # runs the eval -> results.json (the quality table)
│   └── results.json     # committed, deterministic per-model quality table
├── mock.py              # offline deterministic model backend (skill profiles)
├── providers.py         # live OpenAI/Anthropic when keys set, mock otherwise
├── arena.py             # live A/B of two strategies on one prompt
└── cli.py               # `route-arena` : route / bench / arena
```

**Design choices**

- **Mock-first**: the whole system is runnable and testable with zero setup.
  Per-model "skill profiles" in `mock.py` are the *only* place competence is
  encoded; the eval measures it back out and the router consumes those measured
  scores — exactly mirroring a real deployment where the mock is swapped for
  live calls.
- **Pure & deterministic core**: scoring, Pareto logic, and the eval are pure
  functions, so `results.json` is reproducible and offline.
- **Dependency-light**: core needs nothing; `rich` is an optional nicety and the
  provider SDKs are lazily imported optional extras.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 40 passing: scoring, 3 strategies, eval scorer,
                   # Pareto dominance, and a CLI smoke test
```

## Regenerating the quality table

```bash
python -m route_arena.eval.harness   # deterministically rewrites eval/results.json
```

## Live mode (optional)

```bash
cp .env.example .env         # add OPENAI_API_KEY and/or ANTHROPIC_API_KEY
set -a && source .env && set +a
route-arena arena "..."      # matching models now hit the real APIs
```

No keys are ever committed — only `.env.example`. `.env` is gitignored.

---

## License

MIT © 2026 Dhanush Shankar
