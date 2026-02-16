# Twitter / X Launch

## What works on Twitter (research-backed)
- **Threads get 63% more impressions** and 54% more engagement than single tweets (Buffer data)
- **GIFs get 55% more engagement** than any other content type — record a terminal GIF of an agent running
- Strong hook in first line (the "scroll-stopper")
- Concrete numbers beat vague claims — Ruff's "10-100x faster" drove massive shares
- Comparison images/screenshots drive retweets
- Anti-complexity / anti-boilerplate messaging goes viral in AI/dev Twitter
- Tag relevant accounts, use 1-2 hashtags max per tweet

## Pre-Launch Asset: Terminal GIF
Record a short (<15 seconds) GIF showing:
- A simple agent definition → running → streaming output
- Keep terminal clean, no extra UI chrome
- Export at 500px width for optimal quality
- Attach to Tweet 1 (the hook) — this is the highest-impact visual

---

## Launch Thread

### Tweet 1 (Hook)

```
Your AI agent framework takes 1,117ms just to import.

I built PuffinFlow — a LangGraph alternative with a Rust core.

1ms import. 2.2x lower latency. Same features. Simpler API.

pip install puffinflow

Open source: [link]

🧵 Here's why and how:
```

### Tweet 2 (The Problem)

```
LangGraph in 2026:

- 1,117ms cold import (over a second before your code runs)
- TypedDict boilerplate for every state
- StateGraph ceremony for every workflow
- add_node() + add_edge() for every connection
- Annotation types for reducers
- Breaking API changes across versions

All of this to run a state machine.
```

### Tweet 3 (The Comparison)

```
PuffinFlow vs the field:

                  Import    Latency    Throughput
PuffinFlow        1ms       1.2ms      1,088 wf/sec
LangGraph         1,117ms   2.6ms      622 wf/sec
CrewAI            ~800ms    ~3ms       ~400 wf/sec
Raw asyncio       0.1ms     0.5ms      2,000+ wf/sec

PuffinFlow: LangGraph features at near-raw-asyncio speed.
```

### Tweet 4 (How It Works)

```
Define an AI agent in 12 lines:

from puffinflow import Agent, state, Command

class Assistant(Agent):
    @state()
    async def think(self, ctx):
        result = await call_llm(ctx.get_variable("input"))
        return Command(update={"result": result}, goto="respond")

    @state()
    async def respond(self, ctx):
        return None  # Done

No TypedDict. No StateGraph. No add_node. No add_edge.
```

### Tweet 5 (Production Features)

```
PuffinFlow in production:

- Circuit breakers (auto-trip on failure)
- Retries with exponential backoff + jitter
- Bulkhead isolation between states
- OpenTelemetry distributed tracing
- Prometheus metrics out of the box
- Checkpointing for recovery
- Multi-agent coordination

Your agent framework should handle failure, not just the happy path.
```

### Tweet 6 (The Rust Angle)

```
Why a Rust core?

Same pattern behind Ruff, uv, Polars, and Pydantic v2.

- Rust state machine → 2.2x lower latency
- PyO3 bindings → pure Python API
- No GIL contention in the hot path
- Fallback pure-Python mode if needed

You write Python. Rust does the heavy lifting.
```

### Tweet 7 (CTA)

```
PuffinFlow is MIT licensed and free forever.

- States, commands, send, reducers
- Streaming (token + event level)
- Persistent memory (SQLite)
- Subgraph composition
- Multi-agent teams, pools, orchestrators
- Circuit breakers, retries, observability

pip install puffinflow

GitHub: [link]

Star it if you're tired of LangGraph boilerplate.
```

---

## Standalone Tweets (for later / different angles)

### The Import Tweet

```
LangGraph: 1,117ms to import
PuffinFlow: 1ms to import

That's 1,000x faster — before your agent even starts running.

Every script. Every test. Every notebook restart.

pip install puffinflow

[link]
```

### The Boilerplate Tweet

```
LangGraph agent:
1. Define TypedDict state class
2. Create StateGraph(State)
3. Define node functions
4. graph.add_node() for each
5. graph.add_edge() for each connection
6. graph.compile()
7. Run

PuffinFlow agent:
1. class MyAgent(Agent)
2. @state() on your methods
3. Run

[link]
```

### The Latency Tweet

```
Sequential 5-step workflow:

LangGraph: 2.6ms per run, 622 workflows/sec
PuffinFlow: 1.2ms per run, 1,088 workflows/sec

2.2x faster. 1.8x higher throughput.

When you're running agents in production at scale, this adds up.

[link]
```

### The Pattern Tweet

```
The Rust-powered Python pattern:

Ruff (linting) → 10-100x faster than flake8
uv (packages) → 10-100x faster than pip
Polars (DataFrames) → 10-30x faster than pandas
Pydantic v2 → 5-50x faster than v1
PuffinFlow (agents) → 2.2x faster than LangGraph

Rust where it matters. Python where it's convenient.

[link]
```

### The Migration Tweet

```
Already using LangGraph?

PuffinFlow uses the same concepts:
- StateGraph → Agent
- add_node → @state() decorator
- add_edge → return "next_state"
- Command → Command (same API!)
- Send → Send (same API!)
- MemorySaver → MemoryStore/SqliteStore

Same ideas, simpler syntax, Rust speed.

[link]
```

### The Production Tweet

```
What happens when your AI agent fails in production?

LangGraph: 🤷
PuffinFlow:
- CircuitBreaker auto-trips after N failures
- RetryPolicy with exponential backoff + jitter
- Bulkhead isolates failing states
- Dead letter queue captures failures
- Checkpointing enables recovery

pip install puffinflow

[link]
```

---

## Hashtags (use 1-2 per tweet, not all)
#python #ai #llm #agents #opensource #rust #langchain

## Accounts to tag/mention
- @LangChainAI (only if engagement takes off — don't provoke early)
- @charliermarsh (Ruff/uv creator — fellow Rust-for-Python advocate)
- @paborea (Polars — fellow Rust-core Python project)
- @samuel_colvin (Pydantic v2 — Rust core Python success story)
- Developers who've publicly complained about LangGraph complexity

## Timing
- Best: Tuesday-Thursday, 9-11 AM EST or 1-3 PM EST
- AI/dev Twitter is most active during US work hours
- Space tweets 5-10 minutes apart in the thread
