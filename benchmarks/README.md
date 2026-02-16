# PuffinFlow Benchmarks

Compares orchestration overhead across five frameworks using identical workloads:
**PuffinFlow**, **LangGraph**, **LlamaIndex Workflows**, **Prefect**, and **Dagster**.

## Install

```bash
pip install puffinflow                          # required
pip install langgraph llama-index-core prefect dagster  # optional comparisons
pip install psutil                              # optional, for memory measurement
```

Frameworks that aren't installed are skipped automatically.

## Run

```bash
python benchmarks/benchmark.py            # ASCII table
python benchmarks/benchmark.py --json     # machine-readable JSON
python benchmarks/benchmark.py -n 10000   # larger workload per step
python benchmarks/benchmark.py --runs 50  # more runs for stable medians
```

## What it measures

Each benchmark step calls the same `compute_work(n)` function. The benchmark
measures total wall-clock time including framework overhead (graph compilation,
state transitions, context management) — not just the compute work itself.
