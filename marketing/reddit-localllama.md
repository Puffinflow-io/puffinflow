# Reddit r/LocalLLaMA Launch

## What works on r/LocalLLaMA
- Practical tools that work with local models
- Performance and resource efficiency are highly valued
- "No cloud dependency" is a major selling point
- Integration with popular local inference (Ollama, vLLM, llama.cpp)
- This community wants fast, lightweight tools that don't waste GPU resources
- Streaming support is important for local LLM UX

---

## Title

```
PuffinFlow: Fast, lightweight agent framework for local LLMs. Rust core, async Python API, streaming, memory persistence. 2.2x lower latency than LangGraph.
```

### Alternative:

```
Built an agent framework optimized for local LLMs — 1ms import (vs LangGraph's 1,117ms), streaming support, persistent memory, works with Ollama/vLLM/any backend.
```

---

## Comment

```
Hey r/LocalLLaMA,

I built PuffinFlow for people running local LLMs who want an agent
framework that doesn't waste resources.

**The problem with existing frameworks:**

When you're running Ollama or vLLM locally, you want your agent
framework to be as light as possible — your GPU RAM should go to the
model, not the orchestration layer. LangGraph takes 1,117ms just to
import and adds measurable overhead per workflow step.

**PuffinFlow's approach:**
- Rust core handles the state machine hot path → minimal CPU overhead
- 1ms import time (1000x faster than LangGraph)
- 1.2ms per workflow step (2.2x faster than LangGraph)
- Async-first — doesn't block while waiting for inference
- Token-level streaming — see output as your local model generates it

**Works with any LLM backend:**

PuffinFlow is LLM-agnostic. It's a workflow framework, not an LLM
wrapper. Use whatever client you want inside your state functions:

    from puffinflow import Agent, state, Command
    import httpx  # or openai, or requests, or whatever

    class LocalAssistant(Agent):
        @state()
        async def generate(self, ctx):
            # Call your local Ollama instance
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3", "prompt": ctx.get_variable("input")}
                )
            return Command(
                update={"output": response.json()["response"]},
                goto="process"
            )

        @state()
        async def process(self, ctx):
            # Do something with the output
            return None  # Done

    agent = LocalAssistant("local-assistant")

**Token-level streaming with local models:**

    async for event in agent.stream():
        if event.event_type == "token":
            print(event.data["token"], end="", flush=True)

**Persistent memory across conversations:**

    from puffinflow import MemoryStore

    store = MemoryStore()  # or SqliteStore("agent.db") for persistence

    # Store user preferences, conversation history, etc.
    await store.put(("users", "user_123"), "preferences", {"model": "llama3"})
    await store.put(("users", "user_123"), "history", [...])

    # Retrieve across sessions
    item = await store.get(("users", "user_123"), "preferences")
    prefs = item.value

**Multi-agent setups for complex local workflows:**

    from puffinflow import AgentTeam

    team = AgentTeam("research")
    team.add_agent(searcher_agent)    # Searches documents
    team.add_agent(summarizer_agent)  # Summarizes findings
    team.add_agent(writer_agent)      # Writes final output

    result = await team.execute()

**Production reliability features:**
- Circuit breakers (auto-trip if your local model stops responding)
- Retries with backoff (handle Ollama restart gracefully)
- Checkpointing (save/restore mid-workflow if you need to swap models)
- Resource management (CPU/GPU allocation per state)

**Performance:**

| | PuffinFlow | LangGraph |
|---|---|---|
| Import time | 1ms | 1,117ms |
| Per-step latency | 1.2ms | 2.6ms |
| Throughput | 1,088 wf/sec | 622 wf/sec |
| Framework overhead | Negligible | Measurable |

The framework overhead is negligible compared to inference time, which
means your GPU spends its cycles on generation, not orchestration.

Install:

    pip install puffinflow

GitHub: [link]

MIT licensed. Happy to answer questions about integration with
specific local LLM setups.
```
