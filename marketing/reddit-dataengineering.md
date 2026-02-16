# Reddit r/dataengineering Launch

## What works on r/dataengineering
- Workflow orchestration is core to this audience
- Comparisons against Prefect, Dagster, Airflow get attention
- Performance and reliability at scale
- "Production-ready" features: retries, circuit breakers, observability
- This audience values boring, reliable infrastructure over hype
- Reproducibility and testing narratives resonate

---

## Title

```
PuffinFlow: Workflow orchestration with a Rust core. Async-first, streaming, circuit breakers, observability. A simpler alternative to complex DAG frameworks.
```

### Alternative:

```
Built a lightweight workflow orchestration framework with a Rust core — async state machines, retries, circuit breakers, OpenTelemetry, Prometheus. pip install puffinflow.
```

---

## Comment

```
Hey r/dataengineering,

PuffinFlow is a workflow orchestration framework that takes a different
approach from traditional DAG-based tools.

**The approach:**

Instead of defining DAGs with tasks and dependencies, PuffinFlow uses
async state machines. Each state is a Python async function. Transitions
are explicit return values. This makes the execution model simple and
predictable.

    from puffinflow import Agent, state, Command

    class ETLPipeline(Agent):
        @state()
        async def extract(self, ctx):
            data = await fetch_from_source()
            return Command(update={"raw_data": data}, goto="transform")

        @state()
        async def transform(self, ctx):
            clean = process(ctx.get_variable("raw_data"))
            return Command(update={"clean_data": clean}, goto="load")

        @state()
        async def load(self, ctx):
            await write_to_warehouse(ctx.get_variable("clean_data"))
            return None  # Done

    pipeline = ETLPipeline("etl-pipeline")
    await pipeline.run()

**Why state machines over DAGs?**

- Explicit routing: you see exactly where execution goes next
- Dynamic branching: routing can depend on runtime data
- Parallel fan-out: Send API dispatches to parallel branches, reducers merge results
- Simpler mental model: "where am I, what do I do, where do I go next"

**Production reliability features:**

This is where PuffinFlow shines for data engineering:

1. **Circuit breakers** — auto-trip after N failures, half-open recovery
   ```python
   from puffinflow import CircuitBreaker, CircuitBreakerConfig
   breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))
   ```

2. **Retries with exponential backoff** — built-in retry policies
   with jitter and dead-letter queue support

3. **Bulkhead isolation** — one failing state doesn't exhaust resources
   for other states
   ```python
   from puffinflow import Bulkhead, BulkheadConfig
   bulkhead = Bulkhead(BulkheadConfig(max_concurrent=10))
   ```

4. **Resource management** — declare CPU, memory, I/O needs per state
   ```python
   from puffinflow import io_intensive

   @io_intensive()
   async def bulk_read(self, ctx):
       ...
   ```

5. **Checkpointing** — save/restore mid-execution for recovery
   ```python
   from puffinflow import AgentCheckpoint
   # Save and restore agent state mid-execution
   ```

**Observability:**

- OpenTelemetry distributed tracing (see every state transition as a span)
- Prometheus metrics (execution time, throughput, error rates)
- Webhook + email alerting
- Structured logging with context

**Performance:**

The Rust core (via PyO3) handles state machine execution:

| | PuffinFlow | LangGraph |
|---|---|---|
| Import time | 1ms | 1,117ms |
| Step latency | 1.2ms | 2.6ms |
| Throughput | 1,088 wf/sec | 622 wf/sec |

The framework overhead is negligible — your workflow time is dominated
by actual work (I/O, computation), not orchestration overhead.

**Parallel fan-out for batch processing:**

    @state()
    async def process_batch(self, ctx):
        items = ctx.get_variable("batch")
        return [Send("process_item", {"item": x}) for x in items]

    @state()
    async def process_item(self, ctx):
        result = transform(ctx.get_variable("item"))
        return Command(update={"results": [result]}, goto="aggregate")

    # Reducer merges parallel results
    agent.add_reducer("results", add_reducer)

**What PuffinFlow is NOT:**

- Not a full Airflow/Prefect/Dagster replacement for scheduled pipeline orchestration
- No built-in scheduler (yet)
- No web UI (yet)
- Best suited for: real-time workflows, event-driven processing, API orchestration, AI agent workflows

If you need cron scheduling and a web dashboard, stick with Prefect/Dagster.
If you need fast, reliable, code-first workflow execution with production
reliability features, PuffinFlow is worth looking at.

Install:

    pip install puffinflow[observability]

GitHub: [link]

MIT licensed. Happy to answer questions about specific use cases.
```
