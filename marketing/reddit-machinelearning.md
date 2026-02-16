# Reddit r/MachineLearning Launch

## What works on r/MachineLearning
- [P] tag (Project) is required — this community has strict flair rules
- Technical depth is expected — this audience is research-oriented
- Benchmarks and reproducibility are valued
- Architecture decisions and trade-offs get deep engagement
- "Why this approach vs alternatives" framing works well
- This audience cares about correctness and rigor more than hype

---

## Title

```
[P] PuffinFlow: Open-source AI agent framework — 2.2x faster than LangGraph, Rust core, pure Python API. Agents, workflows, streaming, memory, multi-agent coordination.
```

### Alternative:

```
[P] PuffinFlow: High-performance AI workflow orchestration — Rust core (PyO3), async-first, production reliability features. Benchmarked against LangGraph.
```

---

## Comment

```
Hey r/MachineLearning,

Sharing an open-source project: PuffinFlow is an AI agent and workflow
orchestration framework with a Rust core and pure Python API.

**What problem does it solve?**

Building AI agents with LangGraph means TypedDict boilerplate for every
state schema, StateGraph ceremony for every workflow, and over a second
of import time before your code runs. For research iteration — where you
restart scripts dozens of times per session — this overhead compounds.

PuffinFlow provides the same conceptual model (state machines with
commands, parallel fan-out, reducers, streaming, persistent memory) with
a simpler API and significantly better performance.

**Benchmark results** (reproducible, see BENCHMARKS.md):

| Metric | PuffinFlow | LangGraph | Improvement |
|--------|-----------|-----------|-------------|
| Cold import time | 1ms | 1,117ms | 1000x |
| Sequential 5-step latency | 1.2ms | 2.6ms | 2.2x |
| Throughput | 1,088 wf/sec | 622 wf/sec | 1.8x |

**Architecture:**

The core state machine is implemented in Rust (via PyO3) and compiled
to a Python extension module. This handles transition resolution, state
dispatching, and the execution hot path. Everything you write — state
functions, LLM calls, data processing — is pure Python async.

Fallback: if the Rust extension isn't available, a pure-Python
implementation provides identical behavior (useful for environments
where compiled extensions aren't possible).

**Key features for ML/AI workflows:**

1. **Agent states as async functions** — @state() decorator, auto-discovered
2. **Commands** — combined routing + state updates in one return value
3. **Send API** — parallel fan-out for map-reduce patterns (e.g., process
   N documents in parallel, merge results with reducers)
4. **Reducers** — safe parallel merging (add, append, replace, custom)
5. **Streaming** — token-level and event-level, async generators
6. **Memory** — async KV store (in-memory + SQLite) with namespace scoping
7. **Subgraph composition** — compose agents into larger pipelines
8. **Multi-agent coordination** — teams, pools, orchestrators

**Production reliability:**
- Circuit breakers (3-state: closed → open → half-open)
- Retries with exponential backoff + jitter
- Bulkhead isolation between states
- Resource management (CPU, memory, GPU allocation per state)
- OpenTelemetry tracing, Prometheus metrics
- Checkpointing for mid-execution recovery

**LLM-agnostic:** PuffinFlow is a workflow framework, not an LLM wrapper.
Use any LLM client inside your state functions — OpenAI, Anthropic,
Hugging Face, local models via vLLM/Ollama, whatever your stack uses.

Install:

    pip install puffinflow
    pip install puffinflow[all]  # with observability, performance extras

GitHub: [link]
Benchmarks: [link]/BENCHMARKS.md

MIT licensed. Happy to answer architecture questions or discuss
design decisions.
```

---

## Engagement tips
- This community values rigor — back every claim with data
- Be prepared to discuss benchmark methodology in detail
- Acknowledge limitations honestly
- Compare to specific alternatives with specific trade-offs
- Don't oversell — r/ML detects and punishes hype
