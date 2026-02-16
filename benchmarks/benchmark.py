"""
PuffinFlow Multi-Framework Benchmark Suite

Measures orchestration overhead across workflow frameworks using identical
workloads: sum(i*i for i in range(5000)). Only framework overhead differs.

Frameworks tested:
  - PuffinFlow (async agent state machine)
  - LangGraph (async graph-based)
  - LlamaIndex Workflows (async event-driven)
  - Prefect (server/materialization-based)
  - Dagster (server/materialization-based)

Usage:
    python benchmarks/benchmark.py            # Human-readable table
    python benchmarks/benchmark.py --json     # JSON output
"""

import argparse
import asyncio
import gc
import json
import operator
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Annotated, Any, Dict, List, Optional

WORKLOAD_SIZE = 5000
NUM_RUNS = 20
THROUGHPUT_DURATION = 3.0  # seconds
MEMORY_WORKFLOWS = 500


def workload() -> int:
    """Standard workload used across all frameworks."""
    return sum(i * i for i in range(WORKLOAD_SIZE))


# ---------------------------------------------------------------------------
# PuffinFlow benchmarks
# ---------------------------------------------------------------------------

async def _puffinflow_sequential(n_steps: int) -> float:
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
                # Use bare @state() — no resource args — enables fast path
                decorated = state()(make_fn)
                self.add_state(name, decorated)

    agent = SeqAgent()
    t0 = time.perf_counter()
    await agent.run()
    return time.perf_counter() - t0


async def _puffinflow_fanout() -> float:
    from puffinflow import Agent, state

    class FanOutAgent(Agent):
        def __init__(self):
            super().__init__("bench-fanout")

            @state()
            async def start(ctx):
                workload()
                return ["branch1", "branch2", "branch3"]

            @state()
            async def branch1(ctx):
                workload()
                return "aggregate"

            @state()
            async def branch2(ctx):
                workload()
                return "aggregate"

            @state()
            async def branch3(ctx):
                workload()
                return "aggregate"

            @state()
            async def aggregate(ctx):
                workload()
                return None

            self.add_state("start", start)
            self.add_state("branch1", branch1)
            self.add_state("branch2", branch2)
            self.add_state("branch3", branch3)
            self.add_state("aggregate", aggregate)

    agent = FanOutAgent()
    t0 = time.perf_counter()
    await agent.run()
    return time.perf_counter() - t0


async def _puffinflow_throughput() -> float:
    from puffinflow import Agent, state

    class ThroughputAgent(Agent):
        def __init__(self):
            super().__init__("bench-tp")

            @state()
            async def step0(ctx):
                workload()
                return "step1"

            @state()
            async def step1(ctx):
                workload()
                return "step2"

            @state()
            async def step2(ctx):
                workload()
                return None

            self.add_state("step0", step0)
            self.add_state("step1", step1)
            self.add_state("step2", step2)

    count = 0
    deadline = time.perf_counter() + THROUGHPUT_DURATION
    while time.perf_counter() < deadline:
        agent = ThroughputAgent()
        await agent.run()
        count += 1
    return count / THROUGHPUT_DURATION


async def _puffinflow_memory() -> float:
    from puffinflow import Agent, state

    class MemAgent(Agent):
        def __init__(self, idx: int):
            super().__init__(f"mem-{idx}")

            @state()
            async def step0(ctx):
                workload()
                return None

            self.add_state("step0", step0)

    gc.collect()
    tracemalloc.start()
    agents = []
    for i in range(MEMORY_WORKFLOWS):
        a = MemAgent(i)
        await a.run()
        agents.append(a)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del agents
    gc.collect()
    return peak / (1024 * 1024)  # MB


# ---------------------------------------------------------------------------
# LangGraph benchmarks
# ---------------------------------------------------------------------------

async def _langgraph_sequential(n_steps: int) -> float:
    from langgraph.graph import StateGraph
    from typing_extensions import TypedDict

    class St(TypedDict):
        value: int

    builder = StateGraph(St)
    names = [f"step{i}" for i in range(n_steps)]
    for name in names:
        builder.add_node(name, lambda s: {"value": workload()})
    builder.set_entry_point(names[0])
    for i in range(len(names) - 1):
        builder.add_edge(names[i], names[i + 1])
    builder.set_finish_point(names[-1])
    graph = builder.compile()

    t0 = time.perf_counter()
    graph.invoke({"value": 0})
    return time.perf_counter() - t0


async def _langgraph_fanout() -> float:
    from langgraph.graph import StateGraph
    from typing_extensions import TypedDict

    class St(TypedDict):
        values: Annotated[List[int], operator.add]

    builder = StateGraph(St)
    builder.add_node("start", lambda s: {"values": [workload()]})
    for b in ["branch1", "branch2", "branch3"]:
        builder.add_node(b, lambda s: {"values": [workload()]})
    builder.add_node("aggregate", lambda s: {"values": [workload()]})
    builder.set_entry_point("start")
    for b in ["branch1", "branch2", "branch3"]:
        builder.add_edge("start", b)
        builder.add_edge(b, "aggregate")
    builder.set_finish_point("aggregate")
    graph = builder.compile()

    t0 = time.perf_counter()
    graph.invoke({"values": []})
    return time.perf_counter() - t0


async def _langgraph_throughput() -> float:
    from langgraph.graph import StateGraph
    from typing_extensions import TypedDict

    class St(TypedDict):
        value: int

    builder = StateGraph(St)
    for name in ["step0", "step1", "step2"]:
        builder.add_node(name, lambda s: {"value": workload()})
    builder.set_entry_point("step0")
    builder.add_edge("step0", "step1")
    builder.add_edge("step1", "step2")
    builder.set_finish_point("step2")
    graph = builder.compile()

    count = 0
    deadline = time.perf_counter() + THROUGHPUT_DURATION
    while time.perf_counter() < deadline:
        graph.invoke({"value": 0})
        count += 1
    return count / THROUGHPUT_DURATION


async def _langgraph_memory() -> float:
    from langgraph.graph import StateGraph
    from typing_extensions import TypedDict

    class St(TypedDict):
        value: int

    gc.collect()
    tracemalloc.start()
    graphs = []
    for _ in range(MEMORY_WORKFLOWS):
        builder = StateGraph(St)
        builder.add_node("step0", lambda s: {"value": workload()})
        builder.set_entry_point("step0")
        builder.set_finish_point("step0")
        g = builder.compile()
        g.invoke({"value": 0})
        graphs.append(g)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del graphs
    gc.collect()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# LlamaIndex Workflows benchmarks
# ---------------------------------------------------------------------------

async def _llamaindex_sequential(n_steps: int) -> float:
    from llama_index.core.workflow import (
        Event,
        StartEvent,
        StopEvent,
        Workflow,
        step,
    )

    # Build event classes for chaining steps
    events = []
    for i in range(n_steps - 1):
        ev = type(f"Step{i}Done", (Event,), {})
        events.append(ev)

    if n_steps == 1:

        class SeqWorkflow1(Workflow):
            @step
            async def only_step(self, ev: StartEvent) -> StopEvent:
                workload()
                return StopEvent()

        wf = SeqWorkflow1()

    elif n_steps == 3:
        Ev0 = events[0]
        Ev1 = events[1]

        class SeqWorkflow3(Workflow):
            @step
            async def s0(self, ev: StartEvent) -> Ev0:
                workload()
                return Ev0()

            @step
            async def s1(self, ev: Ev0) -> Ev1:
                workload()
                return Ev1()

            @step
            async def s2(self, ev: Ev1) -> StopEvent:
                workload()
                return StopEvent()

        wf = SeqWorkflow3()

    elif n_steps == 5:
        Ev0 = events[0]
        Ev1 = events[1]
        Ev2 = events[2]
        Ev3 = events[3]

        class SeqWorkflow5(Workflow):
            @step
            async def s0(self, ev: StartEvent) -> Ev0:
                workload()
                return Ev0()

            @step
            async def s1(self, ev: Ev0) -> Ev1:
                workload()
                return Ev1()

            @step
            async def s2(self, ev: Ev1) -> Ev2:
                workload()
                return Ev2()

            @step
            async def s3(self, ev: Ev2) -> Ev3:
                workload()
                return Ev3()

            @step
            async def s4(self, ev: Ev3) -> StopEvent:
                workload()
                return StopEvent()

        wf = SeqWorkflow5()
    else:
        raise ValueError(f"Unsupported n_steps={n_steps} for LlamaIndex benchmark")

    t0 = time.perf_counter()
    await wf.run()
    return time.perf_counter() - t0


async def _llamaindex_fanout() -> float:
    from llama_index.core.workflow import (
        Context,
        Event,
        StartEvent,
        StopEvent,
        Workflow,
        step,
    )

    class BranchDone(Event):
        pass

    class FanOutWorkflow(Workflow):
        @step
        async def start(self, ctx: Context, ev: StartEvent) -> BranchDone:
            workload()
            ctx.send_event(BranchDone())
            ctx.send_event(BranchDone())
            ctx.send_event(BranchDone())
            return None

        @step(num_workers=3)
        async def branch(self, ctx: Context, ev: BranchDone) -> StopEvent:
            workload()
            return StopEvent()

    wf = FanOutWorkflow()
    t0 = time.perf_counter()
    await wf.run()
    return time.perf_counter() - t0


async def _llamaindex_throughput() -> float:
    from llama_index.core.workflow import (
        Event,
        StartEvent,
        StopEvent,
        Workflow,
        step,
    )

    class S1Done(Event):
        pass

    class S2Done(Event):
        pass

    class TPWorkflow(Workflow):
        @step
        async def s0(self, ev: StartEvent) -> S1Done:
            workload()
            return S1Done()

        @step
        async def s1(self, ev: S1Done) -> S2Done:
            workload()
            return S2Done()

        @step
        async def s2(self, ev: S2Done) -> StopEvent:
            workload()
            return StopEvent()

    count = 0
    deadline = time.perf_counter() + THROUGHPUT_DURATION
    while time.perf_counter() < deadline:
        wf = TPWorkflow()
        await wf.run()
        count += 1
    return count / THROUGHPUT_DURATION


async def _llamaindex_memory() -> float:
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    class MemWorkflow(Workflow):
        @step
        async def only(self, ev: StartEvent) -> StopEvent:
            workload()
            return StopEvent()

    gc.collect()
    tracemalloc.start()
    wfs = []
    for _ in range(MEMORY_WORKFLOWS):
        wf = MemWorkflow()
        await wf.run()
        wfs.append(wf)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del wfs
    gc.collect()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Prefect benchmarks
# ---------------------------------------------------------------------------

async def _prefect_sequential(n_steps: int) -> float:
    from prefect import flow, task

    @task
    def work_task():
        return workload()

    @flow(log_prints=False)
    def seq_flow():
        for _ in range(n_steps):
            work_task()

    t0 = time.perf_counter()
    seq_flow()
    return time.perf_counter() - t0


async def _prefect_fanout() -> float:
    from prefect import flow, task

    @task
    def work_task():
        return workload()

    @flow(log_prints=False)
    def fanout_flow():
        work_task()  # start
        work_task()  # branch 1
        work_task()  # branch 2
        work_task()  # branch 3
        work_task()  # aggregate

    t0 = time.perf_counter()
    fanout_flow()
    return time.perf_counter() - t0


async def _prefect_throughput() -> float:
    from prefect import flow, task

    @task
    def work_task():
        return workload()

    @flow(log_prints=False)
    def tp_flow():
        work_task()
        work_task()
        work_task()

    count = 0
    deadline = time.perf_counter() + THROUGHPUT_DURATION
    while time.perf_counter() < deadline:
        tp_flow()
        count += 1
    return count / THROUGHPUT_DURATION


async def _prefect_memory() -> float:
    from prefect import flow, task

    @task
    def work_task():
        return workload()

    @flow(log_prints=False)
    def mem_flow():
        work_task()

    gc.collect()
    tracemalloc.start()
    results = []
    for _ in range(MEMORY_WORKFLOWS):
        r = mem_flow()
        results.append(r)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del results
    gc.collect()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Dagster benchmarks
# ---------------------------------------------------------------------------

async def _dagster_sequential(n_steps: int) -> float:
    from dagster import In, job, op

    ops = []
    for i in range(n_steps):
        if i == 0:
            o = op(name=f"dstep{i}")(lambda: workload())
        else:
            o = op(name=f"dstep{i}", ins={"inp": In()})(lambda inp: workload())
        ops.append(o)

    @job
    def seq_job():
        result = ops[0]()
        for o in ops[1:]:
            result = o(result)

    t0 = time.perf_counter()
    seq_job.execute_in_process(raise_on_error=True)
    return time.perf_counter() - t0


async def _dagster_fanout() -> float:
    from dagster import In, job, op

    @op
    def d_start():
        return workload()

    @op(ins={"inp": In()})
    def d_branch1(inp):
        return workload()

    @op(ins={"inp": In()})
    def d_branch2(inp):
        return workload()

    @op(ins={"inp": In()})
    def d_branch3(inp):
        return workload()

    @op(ins={"a": In(), "b": In(), "c": In()})
    def d_aggregate(a, b, c):
        return workload()

    @job
    def fanout_job():
        s = d_start()
        d_aggregate(d_branch1(s), d_branch2(s), d_branch3(s))

    t0 = time.perf_counter()
    fanout_job.execute_in_process(raise_on_error=True)
    return time.perf_counter() - t0


async def _dagster_throughput() -> float:
    from dagster import In, job, op

    @op
    def dt_step0():
        return workload()

    @op(ins={"inp": In()})
    def dt_step1(inp):
        return workload()

    @op(ins={"inp": In()})
    def dt_step2(inp):
        return workload()

    @job
    def tp_job():
        dt_step2(dt_step1(dt_step0()))

    count = 0
    deadline = time.perf_counter() + THROUGHPUT_DURATION
    while time.perf_counter() < deadline:
        tp_job.execute_in_process(raise_on_error=True)
        count += 1
    return count / THROUGHPUT_DURATION


async def _dagster_memory() -> float:
    from dagster import job, op

    @op
    def dm_step():
        return workload()

    @job
    def mem_job():
        dm_step()

    gc.collect()
    tracemalloc.start()
    results = []
    for _ in range(MEMORY_WORKFLOWS):
        r = mem_job.execute_in_process(raise_on_error=True)
        results.append(r)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del results
    gc.collect()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Import time measurement
# ---------------------------------------------------------------------------

def _measure_import_time(module_name: str) -> float:
    """Measure cold-start import time in a subprocess, subtracting Python startup."""
    # Script that prints elapsed time for importing the module
    # Use __import__ with fromlist to handle submodule imports (e.g. langgraph.graph)
    if "." in module_name:
        import_stmt = f"__import__('{module_name}', fromlist=['_'])"
    else:
        import_stmt = f"import {module_name}"
    script = (
        "import time; t0 = time.perf_counter(); "
        f"{import_stmt}; "
        "print(time.perf_counter() - t0)"
    )

    baseline_script = "import time; t0 = time.perf_counter(); print(time.perf_counter() - t0)"

    def _median_time(code: str) -> float:
        times = []
        for _ in range(5):
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return float("inf")
            try:
                times.append(float(result.stdout.strip()))
            except ValueError:
                return float("inf")
        return statistics.median(times)

    baseline = _median_time(baseline_script)
    import_t = _median_time(script)
    return max(0, import_t - baseline)


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

async def _run_timed(fn, *args, runs: int = NUM_RUNS) -> float:
    """Run fn multiple times and return median duration in seconds."""
    times = []
    # Warmup
    for _ in range(min(3, runs)):
        await fn(*args)
    gc.collect()
    for _ in range(runs):
        t = await fn(*args)
        times.append(t)
    return statistics.median(times)


def _fmt_ms(seconds: float) -> str:
    ms = seconds * 1000
    if ms >= 1000:
        return f"{ms:,.0f} ms"
    if ms >= 10:
        return f"{ms:.0f} ms"
    return f"{ms:.1f} ms"


def _fmt_throughput(wf_per_sec: float) -> str:
    if wf_per_sec >= 100:
        return f"{wf_per_sec:,.0f}"
    if wf_per_sec >= 1:
        return f"{wf_per_sec:.1f}"
    return f"{wf_per_sec:.1f}"


def _fmt_mb(mb: float) -> str:
    if mb < 0.01:
        return "~0 MB"
    return f"{mb:.2f} MB"


def _fmt_import(seconds: float) -> str:
    ms = seconds * 1000
    return f"{ms:,.0f} ms"


# ---------------------------------------------------------------------------
# Framework definitions
# ---------------------------------------------------------------------------

LIGHTWEIGHT_FRAMEWORKS = {
    "PuffinFlow": {
        "seq3": lambda: _puffinflow_sequential(3),
        "seq5": lambda: _puffinflow_sequential(5),
        "fanout": _puffinflow_fanout,
        "throughput": _puffinflow_throughput,
        "memory": _puffinflow_memory,
        "import_module": "puffinflow",
    },
    "LangGraph": {
        "seq3": lambda: _langgraph_sequential(3),
        "seq5": lambda: _langgraph_sequential(5),
        "fanout": _langgraph_fanout,
        "throughput": _langgraph_throughput,
        "memory": _langgraph_memory,
        "import_module": "langgraph.graph",
    },
    "LlamaIndex Workflows": {
        "seq3": lambda: _llamaindex_sequential(3),
        "seq5": lambda: _llamaindex_sequential(5),
        "fanout": _llamaindex_fanout,
        "throughput": _llamaindex_throughput,
        "memory": _llamaindex_memory,
        "import_module": "llama_index.core.workflow",
    },
}

HEAVY_FRAMEWORKS: Dict[str, Any] = {
    # Prefect and Dagster removed — not comparing against server-based frameworks
}


async def _bench_framework(name: str, fns: dict) -> Dict[str, Any]:
    """Benchmark a single framework, returning raw results."""
    print(f"  Benchmarking {name}...")
    results: Dict[str, Any] = {"name": name}

    try:
        results["seq3"] = await _run_timed(fns["seq3"])
        print(f"    Sequential 3-step: {_fmt_ms(results['seq3'])}")
    except Exception as e:
        print(f"    Sequential 3-step: FAILED ({e})")
        results["seq3"] = None

    try:
        results["seq5"] = await _run_timed(fns["seq5"])
        print(f"    Sequential 5-step: {_fmt_ms(results['seq5'])}")
    except Exception as e:
        print(f"    Sequential 5-step: FAILED ({e})")
        results["seq5"] = None

    if results["seq3"] is not None and results["seq5"] is not None:
        results["per_step"] = (results["seq5"] - results["seq3"]) / 2
        print(f"    Per-step overhead: {_fmt_ms(results['per_step'])}")
    else:
        results["per_step"] = None

    try:
        results["fanout"] = await _run_timed(fns["fanout"])
        print(f"    Fan-out (3+1 agg): {_fmt_ms(results['fanout'])}")
    except Exception as e:
        print(f"    Fan-out: FAILED ({e})")
        results["fanout"] = None

    try:
        # Throughput: single run (already measures over THROUGHPUT_DURATION)
        results["throughput"] = await fns["throughput"]()
        print(f"    Throughput: {_fmt_throughput(results['throughput'])} wf/sec")
    except Exception as e:
        print(f"    Throughput: FAILED ({e})")
        results["throughput"] = None

    try:
        results["memory"] = await fns["memory"]()
        print(f"    Peak memory ({MEMORY_WORKFLOWS} wf): {_fmt_mb(results['memory'])}")
    except Exception as e:
        print(f"    Peak memory: FAILED ({e})")
        results["memory"] = None

    try:
        results["import_time"] = _measure_import_time(fns["import_module"])
        print(f"    Import time: {_fmt_import(results['import_time'])}")
    except Exception as e:
        print(f"    Import time: FAILED ({e})")
        results["import_time"] = None

    return results


def _check_available(frameworks: dict) -> dict:
    """Filter to only frameworks whose imports succeed."""
    available = {}
    for name, fns in frameworks.items():
        mod = fns["import_module"]
        try:
            __import__(mod)
            available[name] = fns
        except ImportError:
            print(f"  Skipping {name} (not installed)")
    return available


def _print_table(title: str, all_results: List[dict]) -> None:
    """Print a markdown-style comparison table."""
    if not all_results:
        return

    names = [r["name"] for r in all_results]
    header = "| Test | " + " | ".join(names) + " |"
    sep = "|------|" + "|".join(["--------"] * len(names)) + "|"

    print(f"\n{title}\n")
    print(header)
    print(sep)

    rows = [
        ("Sequential 3-step", "seq3", _fmt_ms),
        ("Sequential 5-step", "seq5", _fmt_ms),
        ("Per-step overhead", "per_step", _fmt_ms),
        ("Fan-out (3+1 agg)", "fanout", _fmt_ms),
        ("Throughput (wf/sec)", "throughput", _fmt_throughput),
        (f"Peak memory ({MEMORY_WORKFLOWS} wf)", "memory", _fmt_mb),
        ("Import time (cold start)", "import_time", _fmt_import),
    ]

    for label, key, fmt in rows:
        vals = []
        for r in all_results:
            v = r.get(key)
            vals.append(fmt(v) if v is not None else "N/A")
        print(f"| {label} | " + " | ".join(vals) + " |")


def _build_json(lightweight_results: list, heavy_results: list) -> dict:
    """Build structured JSON output."""
    return {
        "benchmark_config": {
            "workload": f"sum(i*i for i in range({WORKLOAD_SIZE}))",
            "num_runs": NUM_RUNS,
            "throughput_duration_sec": THROUGHPUT_DURATION,
            "memory_workflows": MEMORY_WORKFLOWS,
        },
        "lightweight_frameworks": lightweight_results,
        "heavy_frameworks": heavy_results,
    }


async def run_benchmarks(output_json: bool = False) -> dict:
    """Run the full benchmark suite."""
    print("=" * 60)
    print("PuffinFlow Multi-Framework Benchmark Suite")
    print(f"Workload: sum(i*i for i in range({WORKLOAD_SIZE}))")
    print(f"Runs: {NUM_RUNS} (median reported)")
    print("=" * 60)

    # Lightweight frameworks
    print("\n--- Lightweight Frameworks (async/graph-based) ---")
    light_avail = _check_available(LIGHTWEIGHT_FRAMEWORKS)
    light_results = []
    for name, fns in light_avail.items():
        r = await _bench_framework(name, fns)
        light_results.append(r)

    # Heavy frameworks
    print("\n--- Heavy Frameworks (server/materialization-based) ---")
    heavy_avail = _check_available(HEAVY_FRAMEWORKS)
    heavy_results = []
    for name, fns in heavy_avail.items():
        r = await _bench_framework(name, fns)
        heavy_results.append(r)

    # Output
    if output_json:
        data = _build_json(light_results, heavy_results)
        print("\n" + json.dumps(data, indent=2, default=str))
    else:
        _print_table(
            "**Lightweight frameworks (async/graph-based)**", light_results
        )
        _print_table(
            "**Heavy frameworks (server/materialization-based)**", heavy_results
        )

    print("\n" + "=" * 60)
    print("Done.")
    return _build_json(light_results, heavy_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="PuffinFlow benchmark suite")
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON"
    )
    args = parser.parse_args()
    asyncio.run(run_benchmarks(output_json=args.json))


if __name__ == "__main__":
    main()
