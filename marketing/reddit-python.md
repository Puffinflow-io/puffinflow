# Reddit r/Python Launch

## What works on r/Python
- Data-driven posts with benchmarks outperform opinion pieces
- "Rust core for Python tool" pattern is well-understood and respected (Ruff, uv, Polars)
- "I built X because Y was too complex/slow" narrative resonates
- Code examples are essential — show don't tell
- Comparison tables get saved and shared
- Titles that state a concrete claim get more engagement than vague ones

---

## Post Type: Text post (self-posts work better on r/Python for project showcases)

## Title

```
PuffinFlow: A fast LangGraph alternative — Rust core, 2.2x lower latency, 1000x faster import, same features, simpler API. pip install puffinflow.
```

### Alternative titles:

```
I replaced LangGraph's 1,117ms import and TypedDict boilerplate with a Rust-core framework — PuffinFlow, 2.2x faster AI agent workflows
```

```
PuffinFlow: AI agent framework with a Rust core. 1ms import (vs LangGraph's 1,117ms), @state() decorators, no TypedDict ceremony. Benchmarked.
```

```
Built a LangGraph alternative with a Rust core (PyO3). Same concepts — states, commands, send, reducers, streaming — 2.2x lower latency, dramatically simpler API.
```

---

## Comment (post immediately after submission)

```
Author here. Some context on why this exists:

I was spending more time fighting LangGraph's boilerplate than building
actual agent logic. The breaking point was when I profiled my test suite
and realized 1,117ms of every run was just importing the framework.

I built PuffinFlow from scratch with a Rust core (via PyO3) and a pure
Python API. Here's how it compares:

| | PuffinFlow | LangGraph |
|---|---|---|
| Cold import | 1ms | 1,117ms |
| 5-step workflow latency | 1.2ms | 2.6ms |
| Throughput | 1,088 wf/sec | 622 wf/sec |
| Agent definition | Agent("name") | StateGraph(TypedDict) |
| State registration | @state() decorator | graph.add_node() |
| Routing | return "next" | add_edge() / add_conditional_edges() |
| Commands | Command(update=, goto=) | Command(update=, goto=) — same! |
| Send (fan-out) | Send("state", payload) | Send("node", payload) — same! |
| Reducers | agent.add_reducer() | Annotated[list, operator.add] |
| Memory | SqliteStore (async) | MemorySaver (sync) |
| Streaming | agent.stream() | graph.stream() |
| Subgraphs | agent.add_subgraph() | Complex setup |

A simple agent in PuffinFlow:

    from puffinflow import Agent, state, Command

    class Assistant(Agent):
        @state()
        async def think(self, ctx):
            result = await call_llm(ctx.get_variable("input"))
            return Command(update={"result": result}, goto="respond")

        @state()
        async def respond(self, ctx):
            print(ctx.get_variable("result"))
            return None  # Done

    agent = Assistant("assistant")
    result = await agent.run(initial_context={"variables": {"input": "Hello!"}})

No TypedDict. No StateGraph. No add_node. No add_edge. No compile().
States are auto-discovered from the decorator. Routing is a return value.

For production: circuit breakers, retries with exponential backoff,
bulkhead isolation, resource management, OpenTelemetry tracing,
Prometheus metrics, checkpointing, multi-agent coordination.

The Rust core follows the same pattern as Ruff, uv, Polars, and
Pydantic v2 — Rust handles the state machine hot path, you write
pure Python.

Install:

    pip install puffinflow

    # With all extras:
    pip install puffinflow[all]

GitHub: [link]
Benchmarks: [link]/BENCHMARKS.md

Happy to answer questions. Benchmarks are fully reproducible.
```

---

## Cross-post to (wait 24-48h between each):
1. r/Python (main launch)
2. r/MachineLearning (2 days later)
3. r/LocalLLaMA (3 days later)
4. r/LangChain (4 days later)

## Timing
- Best: Monday-Wednesday, 9-11 AM EST
- r/Python peaks during US work hours
