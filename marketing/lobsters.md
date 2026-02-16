# Lobste.rs Launch

## What works on Lobste.rs
- More technical than HN, less hype-tolerant
- Pure technical substance wins — no marketing speak
- "Here's what I built and here are the numbers" approach
- Rust + Python projects get genuine interest from this community
- Smaller community but higher signal-to-noise engagement
- Must be invited or have an account to post

---

## Title

```
PuffinFlow: AI workflow orchestration framework — Rust core (PyO3), Python API, async-first
```

### Alternative:

```
PuffinFlow — async agent framework: Rust state machine core, Python API, 2.2x faster than LangGraph
```

**Tags:** python, rust, ai, ml

---

## Description (if self-post)

```
PuffinFlow is an AI agent and workflow orchestration framework. Async
state machines where each state is a decorated Python function.
Commands for routing + updates, Send for parallel fan-out, reducers
for safe merging, streaming, persistent memory (SQLite).

Core state machine implemented in Rust via PyO3. Fallback pure-Python
mode available. 57 Python modules, comprehensive test suite.

Performance vs LangGraph:
- Cold import: 1ms vs 1,117ms (1000x)
- Sequential 5-step latency: 1.2ms vs 2.6ms (2.2x)
- Throughput: 1,088 vs 622 workflows/sec (1.8x)

Production features: circuit breakers (3-state), retries with exponential
backoff + jitter, bulkhead isolation, resource management (CPU/memory/GPU
allocation per state), OpenTelemetry tracing, Prometheus metrics,
checkpointing, multi-agent coordination (teams, pools, orchestrators),
deadlock detection.

Same conceptual model as LangGraph (states, commands, send, reducers,
streaming, memory, subgraphs) — different implementation approach.

Repo: [link]
Benchmarks: [link]/BENCHMARKS.md
```
