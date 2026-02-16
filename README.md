<p align="center">
  <h1 align="center">PuffinFlow</h1>
  <p align="center"><strong>The fast LangGraph alternative. Rust core. Python simplicity.</strong></p>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/puffinflow"><img src="https://badge.fury.io/py/puffinflow.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/puffinflow/"><img src="https://img.shields.io/pypi/pyversions/puffinflow.svg" alt="Python versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <b>2.2x lower latency</b> &middot; <b>1.8x higher throughput</b> &middot; <b>1000x faster import</b> &middot; <b>Same features, simpler code</b>
</p>

---

## Why PuffinFlow?

LangGraph is the go-to framework for building AI agent workflows. But it's slow, complex, and the API fights you at every step. PuffinFlow gives you the same capabilities — Command, Send, reducers, streaming, persistent memory, subgraphs — with a Rust-backed core that's measurably faster.

```
LangGraph sequential 5-step:    2.6 ms
PuffinFlow sequential 5-step:   1.2 ms  (2.2x faster)

LangGraph throughput:            622 wf/sec
PuffinFlow throughput:         1,088 wf/sec  (1.8x higher)

LangGraph cold import:         1,117 ms
PuffinFlow cold import:            1 ms  (1000x faster)
```

Run the benchmarks yourself: `python benchmarks/benchmark.py`

## Quick Start

```bash
pip install puffinflow
```

```python
from puffinflow import Agent, state, Command

class MyAgent(Agent):
    @state()
    async def think(self, ctx):
        question = ctx.get_variable("question")
        answer = await call_llm(question)
        return Command(update={"answer": answer}, goto="respond")

    @state()
    async def respond(self, ctx):
        ctx.set_output("result", ctx.get_variable("answer"))
        return None  # done

agent = MyAgent("my-agent")
result = await agent.run(initial_context={"variables": {"question": "What is PuffinFlow?"}})
print(result.outputs["result"])
```

## LangGraph vs PuffinFlow

Every LangGraph concept maps directly. If you know LangGraph, you know PuffinFlow.

| LangGraph | PuffinFlow | Notes |
|-----------|------------|-------|
| `StateGraph(State)` | `Agent("name")` | No schema class needed |
| `graph.add_node("name", fn)` | `agent.add_state("name", fn)` or `@state()` | Decorator auto-discovers states |
| `graph.add_edge("a", "b")` | `return "b"` from state `a` | Routing is just a return value |
| `graph.add_conditional_edges(...)` | `return "x" if cond else "y"` | No edge DSL needed |
| `Command(update=, goto=)` | `Command(update=, goto=)` | Same API |
| `Send("node", payload)` | `Send("state", payload)` | Same API |
| `Annotation.reducer` | `agent.add_reducer("key", add_reducer)` | Explicit, not buried in type hints |
| `MemorySaver` | `MemoryStore` / `SqliteStore` | Async, namespaced KV store |
| `graph.stream()` | `agent.stream()` | Async generator with event types |
| Subgraphs | `agent.add_subgraph(...)` | Input/output mapping built in |

### Side-by-Side: Research Agent

**LangGraph** (35 lines of boilerplate)

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    topic: str
    sources: Annotated[list, add_messages]
    summary: str

def research(state: State):
    results = search_web(state["topic"])
    return {"sources": results}

def summarize(state: State):
    summary = call_llm(f"Summarize: {state['sources']}")
    return {"summary": summary}

def route(state: State):
    if len(state["sources"]) > 3:
        return "summarize"
    return "research"

graph = StateGraph(State)
graph.add_node("research", research)
graph.add_node("summarize", summarize)
graph.add_edge(START, "research")
graph.add_conditional_edges("research", route)
graph.add_edge("summarize", END)
app = graph.compile()
result = app.invoke({"topic": "quantum computing"})
```

**PuffinFlow** (16 lines, same result)

```python
from puffinflow import Agent, state, Command

class Researcher(Agent):
    @state()
    async def research(self, ctx):
        results = await search_web(ctx.get_variable("topic"))
        if len(results) > 3:
            return Command(update={"sources": results}, goto="summarize")
        return Command(update={"sources": results}, goto="research")

    @state()
    async def summarize(self, ctx):
        summary = await call_llm(f"Summarize: {ctx.get_variable('sources')}")
        ctx.set_output("summary", summary)
        return None

agent = Researcher("researcher")
result = await agent.run(initial_context={"variables": {"topic": "quantum computing"}})
```

No `StateGraph`, no `TypedDict`, no `add_edge`, no `START`/`END` constants, no `compile()`. Just Python.

## Features

### Command Pattern — Unified State Updates and Routing

States return a `Command` that combines data writes and routing in one value:

```python
@state()
async def decide(self, ctx):
    result = await analyze(ctx.get_variable("input"))
    return Command(
        update={"analysis": result, "confidence": 0.95},
        goto="act" if result.confident else "gather_more"
    )
```

### Send API — Dynamic Fan-Out

Dispatch different payloads to parallel branches of the same state. True map-reduce:

```python
from puffinflow import Send

@state()
async def scatter(self, ctx):
    documents = ctx.get_variable("documents")
    return [Send("process_doc", {"doc": doc}) for doc in documents]

@state()
async def process_doc(self, ctx):
    doc = ctx.get_variable("doc")
    summary = await summarize(doc)
    return Command(update={"summaries": [summary]})
```

### Reducers — Safe Parallel Merging

When parallel branches write the same key, reducers merge correctly instead of clobbering:

```python
from puffinflow import add_reducer

agent = MyAgent("agent")
agent.add_reducer("summaries", add_reducer)  # list concat

# Now both branches writing to "summaries" get merged: [summary1, summary2, ...]
```

Built-in reducers: `add_reducer` (list concat, number add, dict merge), `append_reducer`, `replace_reducer`. Or write your own:

```python
agent.add_reducer("scores", lambda old, new: max(old, new))
```

### Streaming — Real-Time Output

Stream events as they happen. Tokens, state transitions, custom events:

```python
async for event in agent.stream():
    if event.event_type == "token":
        print(event.data["token"], end="", flush=True)
    elif event.event_type == "node_complete":
        print(f"\n[{event.state_name} done]")
```

Emit tokens from inside a state:

```python
@state()
async def generate(self, ctx):
    full_text = ""
    for chunk in llm.stream("Write a poem"):
        ctx.emit_token(chunk)
        full_text += chunk
    return Command(update={"poem": full_text})
```

### Store API — Persistent Agent Memory

Key-value store that survives across runs. Namespace-scoped. Async:

```python
from puffinflow import MemoryStore

store = MemoryStore()  # or SqliteStore("agent.db") for persistence
agent = MyAgent("agent", store=store)

@state()
async def remember(self, ctx):
    # Save user preferences
    await ctx.store.put(("users", "alice"), "prefs", {"theme": "dark"})
    return "recall"

@state()
async def recall(self, ctx):
    item = await ctx.store.get(("users", "alice"), "prefs")
    # item.value == {"theme": "dark"}
```

### Subgraph Composition — Modular Agent Pipelines

Compose agents into larger pipelines. Each child agent is a black box:

```python
researcher = ResearchAgent("research")
writer = WriterAgent("writer")

class Pipeline(Agent):
    def __init__(self):
        super().__init__("pipeline")

        self.add_subgraph("research", researcher,
            input_map={"topic": "query"},
            output_map={"findings": "research_results"})

        self.add_subgraph("write", writer,
            input_map={"research_results": "content"},
            output_map={"draft": "article"},
            dependencies=["research"])

result = await Pipeline().run(
    initial_context={"variables": {"topic": "AI agents"}}
)
print(result.variables["article"])
```

### Plus Everything You Need for Production

- **Resource management** — Declare CPU/memory/GPU per state: `@state(cpu=4.0, memory=2048.0)`
- **Retry policies** — Exponential backoff with jitter, dead letter queues
- **Circuit breakers** — Three-state failure protection per agent
- **Bulkheads** — Concurrency isolation between states
- **Checkpointing** — Save/restore agent state mid-execution
- **Multi-agent teams** — `AgentTeam`, `AgentPool`, `AgentOrchestrator`
- **Observability** — OpenTelemetry tracing, Prometheus metrics, alerting

## Performance

Orchestration overhead measured with identical `sum(i*i for i in range(5000))` workloads. Only framework overhead differs. Median of 20 runs.

**Lightweight frameworks (async/graph-based)**

| Test | PuffinFlow | LangGraph | LlamaIndex |
|------|-----------|-----------|------------|
| Sequential 3-step | **0.7 ms** | 1.7 ms | 2.2 ms |
| Sequential 5-step | **1.2 ms** | 2.6 ms | 3.4 ms |
| Per-step overhead | **0.2 ms** | 0.5 ms | 0.6 ms |
| Fan-out (3+1 agg) | **1.2 ms** | 3.4 ms | 2.1 ms |
| Throughput (wf/sec) | **1,088** | 622 | 394 |
| Peak memory (500 wf) | 2.50 MB | 4.93 MB | **0.67 MB** |
| Import time | **1 ms** | 1,117 ms | 1,852 ms |

Per-step overhead = `(5-step − 3-step) / 2`. Import time = cold-start subprocess, Python startup subtracted.

```bash
pip install puffinflow langgraph llama-index-core prefect dagster
python benchmarks/benchmark.py
```

## Install

```bash
pip install puffinflow                                    # Core (includes Rust engine)
pip install puffinflow[performance]                       # + profiling/benchmark tools
pip install puffinflow[observability]                     # + OpenTelemetry/Prometheus
pip install puffinflow[all]                               # Everything
pip install "puffinflow[dev]"                             # + test/lint tools
```

## Examples

See [`examples/`](./examples/) for runnable code:

- [`basic_agent.py`](./examples/basic_agent.py) — State decorators, context management, resource allocation
- [`advanced_workflows.py`](./examples/advanced_workflows.py) — Conditional branching, dynamic workflows, error recovery
- [`coordination_examples.py`](./examples/coordination_examples.py) — Multi-agent teams, parallel execution, messaging
- [`reliability_patterns.py`](./examples/reliability_patterns.py) — Circuit breakers, retries, bulkheads
- [`resource_management.py`](./examples/resource_management.py) — CPU/memory pools, quotas, allocation strategies
- [`observability_demo.py`](./examples/observability_demo.py) — Tracing, metrics, alerting

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License. Free for commercial and personal use.

---

<p align="center">
  <a href="./examples/">Examples</a> &middot;
  <a href="./benchmarks/">Benchmarks</a> &middot;
  <a href="https://github.com/m-ahmed-elbeskeri/puffinflow-main/issues">Issues</a> &middot;
  <a href="https://github.com/m-ahmed-elbeskeri/puffinflow-main/discussions">Discussions</a>
</p>
