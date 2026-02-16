# Reddit r/opensource Launch

## What works on r/opensource
- Licensing and governance transparency
- Community-first development approach
- Clear differentiation from proprietary alternatives
- No vendor lock-in angle
- "Why I made this open source" narrative
- Active maintenance signals (tests, CI, code quality)

---

## Title

```
PuffinFlow: Open-source AI agent framework. Rust core for speed, Python for simplicity. MIT licensed, async-first, production-ready.
```

---

## Comment

```
PuffinFlow is an open-source AI agent and workflow orchestration framework.

**What it does:**
- Define AI agents as async state machines with decorated Python functions
- States are auto-discovered — no manual registration
- Commands for combined routing + state updates
- Parallel fan-out (Send) with safe merging (reducers)
- Real-time streaming (token and event level)
- Persistent memory (in-memory + SQLite)
- Subgraph composition for modular agent pipelines
- Multi-agent coordination (teams, pools, orchestrators)

**Production features:**
- Circuit breakers, retries with backoff, bulkhead isolation
- Resource management (CPU, memory, GPU allocation per state)
- OpenTelemetry tracing, Prometheus metrics
- Checkpointing and recovery
- Deadlock detection

**Why it exists:**

LangGraph is the dominant AI agent framework, but:
- 1,117ms cold import time (over 1 second)
- TypedDict boilerplate for every state schema
- StateGraph ceremony for every workflow
- Breaking API changes across versions

PuffinFlow uses the same concepts (states, commands, send, reducers,
streaming, memory) with a simpler API and a Rust core for performance:

| | PuffinFlow | LangGraph |
|---|---|---|
| Import | 1ms | 1,117ms |
| Latency | 1.2ms | 2.6ms |
| Throughput | 1,088 wf/sec | 622 wf/sec |
| License | MIT | MIT |

**Why open source (MIT):**

- AI agent frameworks should be community infrastructure, not vendor lock-in
- The core framework is free forever — no feature gating on the open source version
- MIT license — use it however you want, commercially or otherwise
- No CLA required for contributions
- No "open core" bait-and-switch planned for the framework itself

**Code quality:**
- 57 Python modules with comprehensive test suite
- Black formatting, Ruff linting, MyPy strict type checking
- 85% minimum code coverage
- GitHub Actions CI with pre-commit hooks
- Security scanning (TruffleHog, Bandit)
- Contributing guidelines, code of conduct, security policy

**Tech stack:**
- Python 3.9+ with async/await
- Rust core via PyO3 (with pure-Python fallback)
- Pydantic v2 for validation
- Optional: OpenTelemetry, Prometheus, structlog

Install:

    pip install puffinflow

GitHub: [link]

Happy to answer questions about the architecture, licensing decisions,
or contribution process.
```
