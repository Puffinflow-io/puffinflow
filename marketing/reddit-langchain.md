# Reddit r/LangChain Launch

## What works on r/LangChain
- This is the competitor's community — be respectful and genuinely helpful
- Migration stories and comparison posts do well
- "I built X because LangGraph didn't do Y" is understood here
- Concrete code comparisons (side-by-side) get engagement
- This audience knows LangGraph deeply — technical depth is required
- Don't trash LangGraph. Acknowledge its strengths. Position as an alternative, not a replacement.

---

## Title

```
PuffinFlow: I built a LangGraph alternative — same concepts (states, commands, send, reducers, streaming, memory), simpler API, 2.2x faster. Here's what I learned.
```

### Alternative:

```
Built an alternative to LangGraph with a Rust core — same concepts, simpler syntax. Sharing a detailed comparison and what I learned.
```

---

## Comment

```
Hey r/LangChain,

First off — LangGraph is a great project that validated the entire space
of graph-based agent workflows. PuffinFlow wouldn't exist without the
concepts LangGraph pioneered. This isn't a "LangGraph bad" post — it's
a "here's a different approach" post.

I built PuffinFlow because I wanted the same conceptual model —
state machines with commands, parallel fan-out, reducers, streaming,
memory — but with less boilerplate and better performance.

**Side-by-side comparison:**

**Defining an agent:**

LangGraph:
```python
from typing import TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    input: str
    result: str

graph = StateGraph(State)

def think(state: State) -> dict:
    return {"result": process(state["input"])}

def respond(state: State) -> dict:
    print(state["result"])
    return {}

graph.add_node("think", think)
graph.add_node("respond", respond)
graph.add_edge("think", "respond")
graph.set_entry_point("think")
graph.set_finish_point("respond")

app = graph.compile()
app.invoke({"input": "Hello"})
```

PuffinFlow:
```python
from puffinflow import Agent, state, Command

class Assistant(Agent):
    @state()
    async def think(self, ctx):
        result = process(ctx.get_variable("input"))
        return Command(update={"result": result}, goto="respond")

    @state()
    async def respond(self, ctx):
        print(ctx.get_variable("result"))
        return None

agent = Assistant("assistant")
result = await agent.run(initial_context={"variables": {"input": "Hello"}})
```

Same outcome. Less ceremony.

**Commands (identical concept, similar API):**

LangGraph:
```python
from langgraph.types import Command
return Command(update={"result": value}, goto="next")
```

PuffinFlow:
```python
from puffinflow import Command
return Command(update={"result": value}, goto="next")
```

Almost identical — this was intentional. If you know LangGraph's
Command, you know PuffinFlow's Command.

**Send / fan-out (identical concept):**

LangGraph:
```python
from langgraph.types import Send
return [Send("process", {"item": x}) for x in items]
```

PuffinFlow:
```python
from puffinflow import Send
return [Send("process", {"item": x}) for x in items]
```

Same pattern. Same mental model.

**Reducers (different API, same concept):**

LangGraph:
```python
from typing import Annotated
import operator

class State(TypedDict):
    results: Annotated[list, operator.add]
```

PuffinFlow:
```python
from puffinflow import add_reducer
agent.add_reducer("results", add_reducer)
```

Explicit rather than buried in type annotations.

**Memory:**

LangGraph:
```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

PuffinFlow:
```python
from puffinflow import MemoryStore
store = MemoryStore()  # or SqliteStore("agent.db") for persistence
# Async operations with namespace scoping
await store.put(("users", "alice"), "prefs", {"theme": "dark"})
```

Async-first, namespace-scoped, with SQLite persistence built in.

**Performance numbers:**

| | PuffinFlow | LangGraph |
|---|---|---|
| Cold import | 1ms | 1,117ms |
| 5-step latency | 1.2ms | 2.6ms |
| Throughput | 1,088 wf/sec | 622 wf/sec |

The Rust core (via PyO3) handles the state machine execution. Same
pattern as Ruff, uv, Polars, and Pydantic v2.

**What PuffinFlow adds beyond LangGraph's core:**
- Circuit breakers, retries with backoff, bulkhead isolation
- Resource management (CPU, memory, GPU allocation per state)
- OpenTelemetry tracing and Prometheus metrics built-in
- Multi-agent coordination (AgentTeam, AgentPool, AgentOrchestrator)
- Deadlock detection
- Checkpointing for mid-execution recovery

**What LangGraph does that PuffinFlow doesn't (yet):**
- LangSmith integration (observability platform)
- LangGraph Platform (hosted execution)
- Broader ecosystem integration (LangChain chains, tools, retrievers)
- Larger community and more tutorials/examples

**Migration path:**

If you're considering trying PuffinFlow alongside LangGraph:

    pip install puffinflow

The concepts map directly:
- StateGraph(TypedDict) → class MyAgent(Agent)
- add_node → @state() on class methods
- add_edge → return Command(goto="next") or return "next"
- Command → Command (same!)
- Send → Send (same!)
- MemorySaver → MemoryStore / SqliteStore (async, namespaced)
- Annotation reducers → agent.add_reducer()

GitHub: [link]
Benchmarks: [link]/BENCHMARKS.md

MIT licensed. Happy to answer detailed comparison questions or help
with migration from specific LangGraph patterns.
```

---

## Engagement tips
- Be EXTRA respectful here — this is LangGraph's home turf
- Acknowledge LangGraph's strengths genuinely (ecosystem, community, LangSmith)
- Focus on "different approach" not "better approach"
- If someone is hostile, one polite response and move on
- Share specific migration examples when asked
- Be honest about what PuffinFlow doesn't do yet
