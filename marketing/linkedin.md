# LinkedIn Launch

## Research-Backed Strategy
- **Carousel PDFs get 6.60% engagement** — highest of all LinkedIn formats (Socialinsider data)
- Vertical slides outperform square by 20%, horizontal by 35%
- Sweet spot: 12-13 slides, 25-50 words per slide
- **Links in post body get penalized** by the algorithm — put the GitHub link in the FIRST COMMENT instead
- First 60 minutes are critical — strong early engagement gets amplified
- 15+ word comments count as "meaningful engagement"
- Best timing: Tuesday-Wednesday, 8-9 AM local time
- Limit to 3-5 hashtags max

---

## Option A: Carousel Post (RECOMMENDED — highest engagement format)

### Create a PDF carousel (12 slides):

**Slide 1 (Cover):**
```
Your AI agent framework takes
1,117ms just to import.

There's a better way.
```

**Slide 2 (The Problem):**
```
LangGraph in 2026:
- 1,117ms cold import time
- TypedDict boilerplate for every agent
- StateGraph ceremony for every workflow
- Breaking API changes across versions
- Annotation-buried reducers
- Complex add_node/add_edge wiring
```

**Slide 3 (The Contrast):**
```
PuffinFlow:
- 1ms import (1000x faster)
- @state() decorators (auto-discovered)
- return "next" for routing
- Same concepts, stable API
- Explicit reducers
- Rust core for speed
```

**Slide 4 (The Comparison Table):**
```
                Import   Latency   Throughput
PuffinFlow      1ms      1.2ms     1,088 wf/s
LangGraph       1,117ms  2.6ms     622 wf/s

2.2x lower latency
1.8x higher throughput
1000x faster import
```

**Slide 5 (What It Does):**
```
Everything LangGraph does.
None of the boilerplate.

- State machines with @state() decorators
- Commands for routing + updates
- Send for parallel fan-out
- Reducers for safe merging
- Streaming (token + event)
- Persistent memory (SQLite)
```

**Slide 6 (Simple Code):**
```
class Assistant(Agent):
    @state()
    async def think(self, ctx):
        result = await call_llm(ctx.get_variable("input"))
        return Command(
            update={"result": result},
            goto="respond"
        )

That's it. No TypedDict. No StateGraph.
```

**Slide 7 (Production Ready):**
```
Built for production:

- Circuit breakers
- Retries with exponential backoff
- Bulkhead isolation
- Resource management
- OpenTelemetry tracing
- Prometheus metrics
- Checkpointing + recovery
```

**Slide 8 (Multi-Agent):**
```
Multi-agent coordination:

- AgentTeam: messaging + event buses
- AgentPool: dynamic scaling + work queues
- AgentOrchestrator: staged execution
- Coordination primitives
- Deadlock detection

Build complex AI systems from simple agents.
```

**Slide 9 (The Rust Core):**
```
Why is it fast?

Rust core via PyO3.
Same pattern as:

Ruff → 10-100x faster linting
uv → 10-100x faster packages
Polars → 10-30x faster DataFrames
Pydantic v2 → 5-50x faster validation

Rust where it matters. Python where you write.
```

**Slide 10 (Migration):**
```
Already using LangGraph?

Same concepts, new syntax:

StateGraph → Agent
add_node → @state()
add_edge → return "next"
Command → Command (same!)
Send → Send (same!)
MemorySaver → SqliteStore
```

**Slide 11 (Easy Install):**
```
pip install puffinflow

That's it.

Python 3.9+
Async-first
MIT licensed
Works with any LLM
```

**Slide 12 (CTA):**
```
PuffinFlow
The Fast LangGraph Alternative

Rust core. Python simplicity.

Link in the comments.
```

### Post text (accompanies the carousel):

```
Your engineering team's AI agents are waiting 1,117ms just to import their framework — before a single line of agent code runs.

LangGraph pioneered the space. But TypedDict boilerplate, StateGraph ceremony, and 1-second import times are slowing teams down.

I built PuffinFlow — an open source AI agent framework with a Rust core. 2.2x lower latency, 1000x faster imports, same concepts (states, commands, send, reducers, streaming, memory), dramatically simpler API.

Same pattern as Ruff, uv, Polars, and Pydantic v2: Rust where performance matters, Python where you write code.

Production-ready: circuit breakers, retries, observability, multi-agent coordination. MIT licensed. Free forever.

Link in the comments.

#ai #python #opensource
```

### First comment (post IMMEDIATELY — this is where the link goes):

```
GitHub repo: [link]
Full benchmarks: [link]/BENCHMARKS.md

pip install puffinflow

Happy to answer questions about the architecture or Rust/PyO3 integration.
```

---

## Option B: Text Post (simpler, still effective)

```
Your engineering team's AI agents are waiting over a second just to import their framework.

I measured LangGraph's cold import time: 1,117ms. Over one second before your agent code even starts.

Then there's the TypedDict boilerplate, the StateGraph ceremony, the add_node/add_edge wiring. For what should be an async state machine.

I built PuffinFlow to fix this.

PuffinFlow is an open source AI agent framework with a Rust core and pure Python API.

The difference:
- 1ms import (not 1,117ms)
- 1.2ms per workflow (not 2.6ms)
- 1,088 workflows/sec (not 622)
- class MyAgent(Agent) + @state() (not TypedDict + StateGraph)
- return Command(goto="next") or return "next" for routing (not add_edge)
- Same concepts: commands, send, reducers, streaming, memory

Production features: circuit breakers, retries, bulkheads, OpenTelemetry tracing, Prometheus metrics, checkpointing, multi-agent coordination.

Built with a Rust core via PyO3 (same pattern as Ruff, uv, Polars, Pydantic v2). MIT licensed. Free forever.

Link in the comments.

#ai #python #devtools #opensource
```

### First comment:

```
GitHub: [link]
Benchmarks: [link]/BENCHMARKS.md

pip install puffinflow
```
