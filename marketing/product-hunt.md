# Product Hunt Launch

## Research-Backed Strategy

Based on Corbado's #1 Dev Tool of the Week case study and Cursor's #1 Product of the Year 2024:

- **Thursday** is the busiest PH day — maximum exposure
- **First 4 hours are critical** — PH hides upvote counts during this window. Mobilize your network immediately.
- **Expect 2+ hour approval delay** for your launch page — keep pushing the direct link during this period
- **Recruit an experienced "hunter"** with AI/dev tool connections (increases visibility)
- **Each screenshot should convey ONE distinct feature** — no text overload
- **Maker's comment** should tell the personal founding story with clear formatting
- **Respond to every comment all day** — make it a dedicated launch day
- **Pre-schedule social media posts** to cross-promote throughout the day

### Pre-Launch (start 2-4 weeks before):
- [ ] Build PH following by engaging with other products (upvoting, commenting thoughtfully)
- [ ] Find and recruit a hunter with AI/dev tool community connections
- [ ] Prepare all assets (screenshots, description, maker comment)
- [ ] Pre-schedule Twitter/LinkedIn posts for launch day
- [ ] Brief your network to upvote and comment (genuinely — no astroturfing)

---

## Listing

**Name:** PuffinFlow

**Tagline:** The fast LangGraph alternative. Rust core. Python simplicity.

**Alternative taglines:**
- AI agent workflows that start in 1ms, not 1,117ms. Rust core, Python API.
- Build AI agents 2.2x faster. Rust-powered. Python-simple. Production-ready.

**Description:**
```
PuffinFlow is a high-performance AI agent and workflow orchestration
framework with a Rust core and pure Python API.

It replaces LangGraph with 2.2x lower latency, 1000x faster imports,
and a dramatically simpler API — no TypedDict, no StateGraph ceremony.

Define agents as async state machines with decorated functions. States
are auto-discovered. Routing is just a return value. Parallel fan-out,
reducers, streaming, persistent memory, and subgraph composition
all work out of the box.

Production features: circuit breakers, retries with backoff, bulkheads,
resource management, OpenTelemetry tracing, Prometheus metrics,
checkpointing, multi-agent coordination.

pip install puffinflow
```

**Topics:** Developer Tools, Artificial Intelligence, Open Source, Python, Machine Learning

---

## Screenshots (one feature per image)

1. **Code: Agent definition** — show how simple the @state() decorator API is (~12 lines for a complete agent)
2. **Terminal: Benchmark output** — show the comparison table: PuffinFlow vs LangGraph latency/throughput
3. **Code: Streaming** — show real-time token streaming from an agent run
4. **Comparison table** — PuffinFlow vs LangGraph vs CrewAI vs AutoGen feature/performance matrix

---

## Maker Comment

```
Hey Product Hunt! 👋

I'm the maker of PuffinFlow. I built it because my AI agent framework
(LangGraph) was taking over a second just to import — before running a
single line of my code. And the TypedDict/StateGraph boilerplate was
making simple workflows feel unnecessarily complex.

The numbers that drove me to build this:

  LangGraph: 1,117ms import, 2.6ms per workflow, 622 workflows/sec
  PuffinFlow: 1ms import, 1.2ms per workflow, 1,088 workflows/sec

**What PuffinFlow does:**
- Async state machines with @state() decorators — auto-discovered, no registration
- Commands for combined routing + state updates
- Send API for parallel fan-out (map-reduce patterns)
- Reducers for safe parallel merging
- Real-time streaming (token-level and event-level)
- Persistent memory (in-memory and SQLite stores)
- Subgraph composition for modular agent pipelines
- Multi-agent coordination (teams, pools, orchestrators)

**Production features:**
- Circuit breakers, retries with exponential backoff
- Bulkhead isolation, resource management
- OpenTelemetry tracing, Prometheus metrics
- Checkpointing and recovery
- Deadlock detection

**How it compares to LangGraph:**
- Agent definition: `class MyAgent(Agent)` vs `StateGraph(TypedDict)`
- State registration: `@state()` decorator on methods vs `graph.add_node()`
- Routing: `return Command(goto="next")` or `return "next"` vs `add_edge()` / `add_conditional_edges()`
- Same concepts: commands, send, reducers, streaming, memory
- 2.2x lower latency, 1.8x higher throughput

The Rust core (via PyO3) handles the state machine execution hot path.
You write pure Python. Same pattern as Ruff, uv, Polars, and Pydantic v2.

MIT licensed. Free forever. The core framework will always be open source.

Install:
  pip install puffinflow

Would love your feedback! Happy to answer any questions.
```

---

## Launch Day Checklist

### Morning (12:01 AM PST — PH resets at midnight)
- [ ] Publish listing (will take 1-2 hours for approval)
- [ ] Post maker comment immediately after approval
- [ ] Share PH link on Twitter with the launch thread
- [ ] Share PH link in LinkedIn first comment
- [ ] DM close contacts to check it out (no astroturfing — ask for genuine feedback)

### First 4 hours (critical window)
- [ ] Respond to every PH comment within 30 minutes
- [ ] Post Twitter update: "We just launched on Product Hunt..." with link
- [ ] Monitor upvotes and ranking

### Rest of the day
- [ ] Continue responding to all comments
- [ ] Post 2-3 Twitter updates with interesting comments/stats from PH
- [ ] Thank supporters

### Day after
- [ ] Post final PH results on Twitter/LinkedIn
- [ ] Thank the community
- [ ] Update README with PH badge if ranked well
