# PuffinFlow Benchmarks

Orchestration overhead measured across workflow frameworks using identical workloads. Only framework overhead differs.

---

## Methodology

**Workload:** `sum(i*i for i in range(5000))` — a pure CPU computation used identically across all frameworks. This isolates framework orchestration overhead from application logic.

**Setup:**
- Each test runs 20 times (median reported)
- 3 warmup iterations before measurement
- Throughput measured over 3-second window
- Memory measured with `tracemalloc` across 500 sequential workflow executions
- Import time measured in isolated subprocess (Python startup time subtracted)
- All frameworks use async execution where supported

**Environment:**
- Python 3.12+
- PuffinFlow (Rust core via PyO3)
- LangGraph (latest)
- LlamaIndex Workflows (latest)

**Reproduce:**
```bash
pip install puffinflow langgraph llama-index-core
python benchmarks/benchmark.py
```

For JSON output:
```bash
python benchmarks/benchmark.py --json
```

---

## Results

### Lightweight Frameworks (async/graph-based)

| Test | PuffinFlow | LangGraph | LlamaIndex Workflows |
|------|-----------|-----------|---------------------|
| Sequential 3-step | **0.7 ms** | 1.7 ms | 1.9 ms |
| Sequential 5-step | **1.4 ms** | 2.3 ms | 2.9 ms |
| Per-step overhead | **0.3 ms** | 0.3 ms | 0.5 ms |
| Fan-out (3 branches + 1 aggregate) | **1.2 ms** | 3.2 ms | 1.6 ms |
| Throughput (workflows/sec) | **1,145** | 705 | 405 |
| Peak memory (500 workflows) | 2.64 MB | 4.93 MB | **0.67 MB** |
| Import time (cold start) | **247 ms** | 891 ms | 1,585 ms |

Per-step overhead = `(5-step − 3-step) / 2`. Import time measures `from pkg import ...` with real symbols (e.g. `from puffinflow import Agent, state`) in a cold subprocess, Python startup subtracted.

---

## Key Takeaways

### Latency: ~1.6x faster

PuffinFlow's Rust core executes the state machine hot path — transition resolution, state dispatching, context management — in compiled Rust via PyO3. This delivers:

- **~1.6x lower latency** on sequential workflows (1.4ms vs 2.3ms for 5 steps)
- **0.3ms per-step overhead** — matching LangGraph's per-step cost
- Near-identical performance to raw asyncio for simple chains

### Throughput: ~1.6x higher

Running workflows back-to-back over a 3-second window:

- **PuffinFlow: 1,145 workflows/sec**
- LangGraph: 705 workflows/sec
- LlamaIndex: 405 workflows/sec

### Import Time: ~3.6x faster

Cold-start import time matters for scripts, tests, notebooks, and CI/CD. The benchmark measures real-world imports with actual symbols (e.g. `from puffinflow import Agent, state`), not bare `import pkg`:

- **PuffinFlow: 247ms** (lazy imports — only loads what you use)
- LangGraph: 891ms
- LlamaIndex: 1,585ms

PuffinFlow uses a lazy import system (`__getattr__`-based) at every level of the package. Submodules and heavy dependencies like structlog are only imported when first accessed, keeping the import path lean.

### Memory: Competitive

PuffinFlow uses 2.64 MB for 500 sequential workflow executions. LlamaIndex is lower at 0.67 MB (lightweight event-driven model). LangGraph uses 4.93 MB. PuffinFlow's memory usage is well-controlled thanks to the Rust core's explicit memory management.

---

## Fan-Out Performance

The fan-out benchmark measures a common pattern: one start node dispatching to 3 parallel branches, then aggregating results.

| Framework | Fan-out latency |
|-----------|----------------|
| **PuffinFlow** | **1.2 ms** |
| LlamaIndex | 1.6 ms |
| LangGraph | 3.2 ms |

PuffinFlow's `Send` API and reducer system handle parallel dispatch efficiently. The Rust core resolves transitions and merges without Python-level coordination overhead.

---

## What These Numbers Mean in Practice

### For scripts and tests

If you import your agent framework 50 times/day (script runs, test runs, notebook restarts):

| Framework | Daily import overhead |
|-----------|---------------------|
| PuffinFlow | 12.4 seconds |
| LangGraph | 44.6 seconds |
| LlamaIndex | 79.3 seconds |

### For CI/CD

In CI pipelines, cold-start import time adds directly to build time. PuffinFlow's 247ms import is ~3.6x faster than LangGraph and ~6.4x faster than LlamaIndex.

### For production throughput

At 1,145 workflows/sec, PuffinFlow can handle high-frequency agent invocations without becoming a bottleneck. The framework overhead is negligible compared to actual LLM inference time (typically 100ms-10s per call).

---

## Benchmark Architecture

### PuffinFlow benchmark agents

```python
from puffinflow import Agent, state

class SeqAgent(Agent):
    def __init__(self):
        super().__init__("bench-seq")
        for i in range(n_steps):
            name = f"step{i}"
            next_name = f"step{i + 1}" if i < n_steps - 1 else None

            async def make_fn(ctx, _next=next_name):
                workload()
                return _next

            make_fn.__name__ = name
            decorated = state()(make_fn)
            self.add_state(name, decorated)
```

### LangGraph benchmark graphs

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

class St(TypedDict):
    value: int

builder = StateGraph(St)
for name in names:
    builder.add_node(name, lambda s: {"value": workload()})
builder.set_entry_point(names[0])
for i in range(len(names) - 1):
    builder.add_edge(names[i], names[i + 1])
builder.set_finish_point(names[-1])
graph = builder.compile()
```

Both frameworks execute the same `workload()` function in each step. Only orchestration overhead differs.

---

## Running Your Own Benchmarks

```bash
# Install all frameworks
pip install puffinflow langgraph llama-index-core

# Run comparison
python benchmarks/benchmark.py

# JSON output (for programmatic analysis)
python benchmarks/benchmark.py --json
```

The benchmark source is at [`benchmarks/benchmark.py`](./benchmarks/benchmark.py). It's designed to be transparent and reproducible. Pull requests to improve methodology are welcome.

---

## Notes

- **LlamaIndex memory advantage**: LlamaIndex Workflows use a lightweight event-driven model that allocates less per-workflow overhead. This is a genuine strength of their architecture.
- **Rust core fallback**: If the Rust extension (`_rust_core`) is not available, PuffinFlow falls back to a pure-Python implementation. Performance numbers above are with the Rust core enabled.
- **Real-world caveat**: In production AI agent workflows, LLM inference time (100ms-10s) dominates total latency. Framework overhead matters most for: (1) high-frequency orchestration, (2) complex multi-step pipelines, (3) CI/CD and testing, (4) import-heavy development workflows.
