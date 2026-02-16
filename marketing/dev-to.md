# Dev.to Article

## What works on Dev.to
- Long-form "Why I Built X" articles with personal narrative
- Tutorial-style with code examples
- Comparison/benchmark articles get bookmarked heavily
- Tags drive discovery: #python #ai #rust #opensource
- Cover image matters for click-through

---

## Article

**Title:** I Built a LangGraph Alternative with a Rust Core. It's 2.2x Faster and Imports in 1ms.

**Tags:** python, ai, rust, opensource

**Cover image idea:** Terminal screenshot showing benchmark comparison — PuffinFlow vs LangGraph

---

### Content

```markdown
## The Problem

I import my agent framework dozens of times a day — every script run, every test, every notebook restart. With LangGraph, that's:

- 1,117ms cold import time — **over a second** before my code even starts
- TypedDict boilerplate for every single state schema
- `StateGraph(State)` ceremony for every workflow
- `add_node()` + `add_edge()` for every connection
- Reducers buried in `Annotation` type hints
- Breaking API changes across versions that burn hours

I write state machines for a living. My state machine framework shouldn't be the bottleneck.

Multiple "I replaced LangGraph with 50 lines of Python" posts have gone viral. The frustration is real. But raw asyncio doesn't give you parallel fan-out, reducers, persistent memory, streaming, circuit breakers, or multi-agent coordination.

## The Solution

I built **PuffinFlow** — a complete AI agent and workflow framework with a Rust core and pure Python API.

Here's what a simple agent looks like:

```python
from puffinflow import Agent, state, Command

class Assistant(Agent):
    @state()
    async def think(self, ctx):
        prompt = ctx.get_variable("input", "Hello")
        result = await call_llm(prompt)
        return Command(update={"result": result}, goto="respond")

    @state()
    async def respond(self, ctx):
        print(ctx.get_variable("result"))
        return None  # Done

agent = Assistant("assistant")
result = await agent.run(initial_context={"variables": {"input": "Hello"}})
```

That's it. No TypedDict. No StateGraph. No add_node. No add_edge. States are auto-discovered from the `@state()` decorator. Routing is just a return value.

## The Numbers

I measured everything. Real numbers from reproducible benchmarks.

| | PuffinFlow | LangGraph | Difference |
|---|---|---|---|
| **Cold import** | 1ms | 1,117ms | 1000x faster |
| **5-step workflow latency** | 1.2ms | 2.6ms | 2.2x faster |
| **Throughput** | 1,088 wf/sec | 622 wf/sec | 1.8x higher |
| **API complexity** | @state() decorator | TypedDict + StateGraph + add_node + add_edge | Much simpler |
| **Rust core** | Yes (PyO3) | No (pure Python) | Performance ceiling |

## Why a Rust Core?

PuffinFlow follows the same pattern as the most successful Python performance projects:

- **Ruff** (linting) → 10-100x faster than flake8, written in Rust
- **uv** (packages) → 10-100x faster than pip, written in Rust
- **Polars** (DataFrames) → 10-30x faster than pandas, written in Rust
- **Pydantic v2** (validation) → 5-50x faster than v1, Rust core

The pattern: Rust handles the compute-intensive hot path (state machine execution, transition resolution). Python handles everything you write (state functions, LLM calls, business logic). PyO3 bridges the two seamlessly.

You write pure Python. Rust does the heavy lifting.

## What PuffinFlow Actually Does

This isn't a toy. It's a complete agent/workflow framework:

**Core agent features:**
```python
# Commands — combine routing + state updates
return Command(update={"result": value}, goto="next_state")

# Send — parallel fan-out (map-reduce)
return [Send("process_doc", {"doc": doc}) for doc in documents]

# Reducers — safe parallel merging
agent.add_reducer("summaries", add_reducer)  # list concat, number add, dict merge

# Streaming — real-time output
async for event in agent.stream():
    if event.event_type == "token":
        print(event.data["token"], end="")

# Memory — persistent across runs
store = MemoryStore()  # or SqliteStore("agent.db") for persistence
await store.put(("users", "alice"), "prefs", {"theme": "dark"})
```

**Production reliability:**
```python
# CPU-intensive states with resource declarations
@cpu_intensive(cpu=4.0, memory=1024.0)
async def heavy_compute(self, ctx):
    ...

# Memory-intensive states
@memory_intensive(memory=2048.0, cpu=2.0)
async def process_data(self, ctx):
    ...

# Circuit breakers and bulkheads via reliability module
from puffinflow import CircuitBreaker, Bulkhead
breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))
```

**Multi-agent coordination:**
```python
# Agent teams with messaging
from puffinflow import AgentTeam
team = AgentTeam("research-team")
team.add_agent(researcher_agent)
team.add_agent(writer_agent)
result = await team.execute()

# Agent pools with dynamic scaling
from puffinflow import AgentPool
pool = AgentPool(max_agents=10)
```

**Observability:**
```python
# Built-in Prometheus metrics and OpenTelemetry tracing
from puffinflow.core.observability import ObservableAgent
# Wraps any Agent with automatic metrics and tracing
```

## API Comparison: LangGraph vs PuffinFlow

| Concept | LangGraph | PuffinFlow |
|---------|-----------|-----------|
| Graph definition | `StateGraph(TypedDict)` | `class MyAgent(Agent)` |
| State registration | `graph.add_node("name", fn)` | `@state()` on class methods |
| Routing | `add_edge()` / `add_conditional_edges()` | `return Command(goto="next")` or `return "next"` |
| Commands | `Command(update=, goto=)` | `Command(update=, goto=)` — same! |
| Parallel dispatch | `Send("node", payload)` | `Send("state", payload)` — same! |
| Reducers | Buried in `Annotated[list, operator.add]` | `agent.add_reducer("key", add_reducer)` |
| Memory | `MemorySaver` (sync) | `MemoryStore` / `SqliteStore` (async, namespaced) |
| Streaming | `graph.stream()` | `agent.stream()` — async generator with event types |
| Subgraphs | Complex setup | `agent.add_subgraph()` with input/output mapping |

## The Git-Native Part

Your agent definitions are just Python files:

```
agents/
  assistant.py      # Main agent with states
  researcher.py     # Research sub-agent
  reviewer.py       # Review sub-agent
  config.py         # Settings and feature flags
  tests/
    test_assistant.py
    test_researcher.py
```

```bash
git add agents/
git commit -m "add research agent pipeline"
git push
```

They diff cleanly. They merge cleanly. They show up in PRs. Your agent logic is reviewed alongside your application code.

## Try It

```bash
# Install
pip install puffinflow

# Or with all extras
pip install puffinflow[all]

# Run an example
python examples/basic_agent.py

# Run benchmarks yourself
python -m pytest tests/benchmarks/ -v
```

**GitHub:** [link]

---

The project has 57 Python modules, a Rust core via PyO3, comprehensive test suite, and MIT license. Free forever.

If you're tired of TypedDict boilerplate and 1-second import times, give it a try.
```
