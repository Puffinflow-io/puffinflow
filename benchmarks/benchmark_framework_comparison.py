#!/usr/bin/env python3
"""
Fair Agent Framework Benchmark Suite
Compares LlamaIndex, LangGraph, and PuffinFlow using equivalent implementations
and appropriate execution models for each framework.

METHODOLOGY:
- Each framework tested using its intended execution model
- Equivalent functionality with consistent import counting
- Real-world workloads instead of artificial compute
- Framework overhead measured separately from execution model differences
- Native concurrency patterns for each framework
"""

import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import psutil

# Framework availability flags
PUFFINFLOW_AVAILABLE = False
LANGGRAPH_AVAILABLE = False
LLAMAINDEX_AVAILABLE = False

# Try importing each framework
try:
    from puffinflow import Agent, state

    PUFFINFLOW_AVAILABLE = True
except ImportError:
    PUFFINFLOW_AVAILABLE = False

try:
    import langgraph  # noqa: F401

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

try:
    from llama_index.core import Settings
    from llama_index.core.llms import MockLLM
    from llama_index.core.workflow import Context as LlamaContext
    from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Fair benchmark results across all metrics."""

    framework: str
    execution_model: str  # "async", "sync", "mixed"

    # Code Efficiency (lines of equivalent functionality)
    code_simple_loc: int  # Simple 3-step workflow
    code_complex_loc: int  # Complex workflow with error handling
    code_typed_loc: int  # Type-safe implementation

    # Execution Speed (milliseconds) - native execution model
    speed_simple_ms: float  # Simple workflow
    speed_complex_ms: float  # Complex workflow
    speed_io_heavy_ms: float  # I/O heavy workflow

    # Framework Overhead (percentage of total time spent on orchestration)
    overhead_simple_percent: float  # Simple workflow overhead
    overhead_complex_percent: float  # Complex workflow overhead

    # Concurrency (tasks per second) - native concurrency model
    concurrency_low_tps: float  # 10 concurrent tasks
    concurrency_high_tps: float  # 100 concurrent tasks

    # Memory Efficiency (MB per task)
    memory_simple_mb: float  # Simple task memory footprint
    memory_complex_mb: float  # Complex task memory footprint


class RealWorldWorkloads:
    """Real-world workloads that are equivalent across frameworks."""

    @staticmethod
    async def mock_llm_call(prompt: str, complexity: int = 200) -> dict[str, Any]:
        """Simulate LLM processing with actual computation."""
        # Simulate text processing and token generation
        tokens = prompt.split()
        processed_tokens = []

        # More CPU-intensive text processing simulation
        for _ in range(complexity):
            for token in tokens:
                # Simulate token embedding calculation with matrix operations
                embedding = 0
                for i, c in enumerate(token):
                    # Simulate complex mathematical operations
                    embedding += (ord(c) ** 2) * (i + 1) * 3.14159
                    embedding = int(embedding) % 100000
                processed_tokens.append(embedding)

                # Add more computational work
                for j in range(10):
                    embedding = (embedding * 31 + j) % 1000000

        # Simulate expensive response generation with string operations
        response_parts = []
        for i in range(len(tokens) * 5):
            part = f"token_{i}_" + str(sum(processed_tokens) % 1000)
            response_parts.append(part)

        response = f"Mock response to: {prompt[:50]}..." + "_".join(response_parts)

        return {
            "response": response,
            "tokens": len(processed_tokens),
            "embedding_sum": sum(processed_tokens) % 10000,
        }

    @staticmethod
    def mock_llm_call_sync(prompt: str, complexity: int = 200) -> dict[str, Any]:
        """Synchronous version of mock LLM processing."""
        # Same CPU-intensive work as async version
        tokens = prompt.split()
        processed_tokens = []

        # More CPU-intensive text processing simulation
        for _ in range(complexity):
            for token in tokens:
                # Simulate token embedding calculation with matrix operations
                embedding = 0
                for i, c in enumerate(token):
                    # Simulate complex mathematical operations
                    embedding += (ord(c) ** 2) * (i + 1) * 3.14159
                    embedding = int(embedding) % 100000
                processed_tokens.append(embedding)

                # Add more computational work
                for j in range(10):
                    embedding = (embedding * 31 + j) % 1000000

        # Simulate expensive response generation with string operations
        response_parts = []
        for i in range(len(tokens) * 5):
            part = f"token_{i}_" + str(sum(processed_tokens) % 1000)
            response_parts.append(part)

        response = f"Mock response to: {prompt[:50]}..." + "_".join(response_parts)

        return {
            "response": response,
            "tokens": len(processed_tokens),
            "embedding_sum": sum(processed_tokens) % 10000,
        }

    @staticmethod
    async def mock_vector_search(
        query: str, complexity: int = 50
    ) -> list[dict[str, Any]]:
        """Simulate vector database search with actual computation."""
        # Simulate vector similarity calculation with higher dimensionality
        query_vector = [
            ord(c) % 100 for c in query.ljust(512)[:512]
        ]  # 512-dimensional vectors
        results = []

        # CPU-intensive similarity calculations with more documents
        for doc_id in range(10):  # More documents to search
            doc_vector = [
                (i * doc_id + ord(c)) % 100
                for i, c in enumerate(query.ljust(512)[:512])
            ]

            # Simulate multiple expensive similarity calculations
            best_similarity = 0.0
            for _ in range(complexity):
                # Cosine similarity calculation
                dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
                magnitude_q = sum(a * a for a in query_vector) ** 0.5
                magnitude_d = sum(b * b for b in doc_vector) ** 0.5
                if magnitude_q > 0 and magnitude_d > 0:
                    similarity = dot_product / (magnitude_q * magnitude_d)
                    best_similarity = max(best_similarity, abs(similarity))

                # Additional vector operations
                (sum((a - b) ** 2 for a, b in zip(query_vector, doc_vector)) ** 0.5)
                sum(abs(a - b) for a, b in zip(query_vector, doc_vector))

            results.append(
                {
                    "doc_id": f"doc_{doc_id}",
                    "score": abs(best_similarity) % 1.0,
                    "content": f"Result {doc_id} for {query}",
                }
            )

        return sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    @staticmethod
    def mock_vector_search_sync(
        query: str, complexity: int = 50
    ) -> list[dict[str, Any]]:
        """Synchronous vector search with same computation."""
        # Simulate vector similarity calculation with higher dimensionality
        query_vector = [
            ord(c) % 100 for c in query.ljust(512)[:512]
        ]  # 512-dimensional vectors
        results = []

        # CPU-intensive similarity calculations with more documents
        for doc_id in range(10):  # More documents to search
            doc_vector = [
                (i * doc_id + ord(c)) % 100
                for i, c in enumerate(query.ljust(512)[:512])
            ]

            # Simulate multiple expensive similarity calculations
            best_similarity = 0.0
            for _ in range(complexity):
                # Cosine similarity calculation
                dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
                magnitude_q = sum(a * a for a in query_vector) ** 0.5
                magnitude_d = sum(b * b for b in doc_vector) ** 0.5
                if magnitude_q > 0 and magnitude_d > 0:
                    similarity = dot_product / (magnitude_q * magnitude_d)
                    best_similarity = max(best_similarity, abs(similarity))

                # Additional vector operations
                (sum((a - b) ** 2 for a, b in zip(query_vector, doc_vector)) ** 0.5)
                sum(abs(a - b) for a, b in zip(query_vector, doc_vector))

            results.append(
                {
                    "doc_id": f"doc_{doc_id}",
                    "score": abs(best_similarity) % 1.0,
                    "content": f"Result {doc_id} for {query}",
                }
            )

        return sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    @staticmethod
    async def mock_api_call(
        endpoint: str, data: dict, complexity: int = 30
    ) -> dict[str, Any]:
        """Simulate external API call with data processing."""
        # Simulate JSON serialization/deserialization overhead
        str(data)
        processed_data = {}

        # More CPU-intensive data transformation
        for iteration in range(complexity):
            for key, value in data.items():
                # Simulate complex data validation and transformation
                key_str = str(key)
                value_str = str(value)

                # Hash calculations with multiple passes
                key_hash = 0
                for i, c in enumerate(key_str):
                    key_hash += (ord(c) ** 2) * (i + 1) * iteration
                    key_hash = key_hash % 1000000

                value_hash = 0
                for i, c in enumerate(value_str):
                    value_hash += (ord(c) ** 3) * (i + 1) * iteration
                    value_hash = value_hash % 1000000

                # Complex transformation operations
                combined_hash = (key_hash * value_hash) % 1000000
                for _ in range(5):
                    combined_hash = (combined_hash * 31 + iteration) % 1000000

                processed_data[f"processed_{key}_{iteration}"] = combined_hash

        return {
            "status": "success",
            "endpoint": endpoint,
            "data": processed_data,
            "processed_items": len(processed_data),
        }

    @staticmethod
    def mock_api_call_sync(
        endpoint: str, data: dict, complexity: int = 30
    ) -> dict[str, Any]:
        """Synchronous API call with same computation."""
        # Same CPU-intensive processing as async version
        str(data)
        processed_data = {}

        # More CPU-intensive data transformation
        for iteration in range(complexity):
            for key, value in data.items():
                # Simulate complex data validation and transformation
                key_str = str(key)
                value_str = str(value)

                # Hash calculations with multiple passes
                key_hash = 0
                for i, c in enumerate(key_str):
                    key_hash += (ord(c) ** 2) * (i + 1) * iteration
                    key_hash = key_hash % 1000000

                value_hash = 0
                for i, c in enumerate(value_str):
                    value_hash += (ord(c) ** 3) * (i + 1) * iteration
                    value_hash = value_hash % 1000000

                # Complex transformation operations
                combined_hash = (key_hash * value_hash) % 1000000
                for _ in range(5):
                    combined_hash = (combined_hash * 31 + iteration) % 1000000

                processed_data[f"processed_{key}_{iteration}"] = combined_hash

        return {
            "status": "success",
            "endpoint": endpoint,
            "data": processed_data,
            "processed_items": len(processed_data),
        }


# =============================================================================
# PUFFINFLOW IMPLEMENTATION (Async Native)
# =============================================================================


class PuffinFlowBenchmark:
    """PuffinFlow benchmark using native async execution."""

    def setup_framework(self):
        if not PUFFINFLOW_AVAILABLE:
            return {"error": "PuffinFlow not available"}
        return {
            "framework": "puffinflow",
            "execution_model": "async",
            "status": "ready",
        }

    def get_code_efficiency_metrics(self):
        """Measure equivalent functionality lines of code."""

        # Simple: 3-step workflow with imports
        simple_code = """
from puffinflow import Agent, state

agent = Agent("simple-workflow")

@state
async def search_step(context):
    query = context.get_variable("query")
    results = await RealWorldWorkloads.mock_vector_search(query)
    context.set_variable("search_results", results)
    return "llm_step"

@state
async def llm_step(context):
    results = context.get_variable("search_results")
    prompt = f"Analyze: {results}"
    response = await RealWorldWorkloads.mock_llm_call(prompt)
    context.set_variable("llm_response", response)
    return "api_step"

@state
async def api_step(context):
    response = context.get_variable("llm_response")
    result = await RealWorldWorkloads.mock_api_call("/save", {"data": response})
    context.set_variable("final_result", result)
    return None

result = await agent.run({"query": "test query"})
"""

        # Complex: Error handling, retries, validation
        complex_code = """
from puffinflow import Agent, state

agent = Agent("complex-workflow")

@state
async def search_with_retry(context):
    attempt = context.get_variable("attempt", 0)
    query = context.get_variable("query")

    try:
        results = await RealWorldWorkloads.mock_vector_search(query)
        if not results:
            raise ValueError("No results found")
        context.set_variable("search_results", results)
        return "validate_results"
    except Exception as e:
        if attempt < 3:
            context.set_variable("attempt", attempt + 1)
            context.set_variable("last_error", str(e))
            return "search_with_retry"
        return "error_handler"

@state
async def validate_results(context):
    results = context.get_variable("search_results")
    if len(results) >= 2 and all(r.get("score", 0) > 0.5 for r in results):
        return "llm_step"
    else:
        context.set_variable("validation_error", "Quality check failed")
        return "error_handler"

@state
async def llm_step(context):
    results = context.get_variable("search_results")
    prompt = f"Analyze: {results}"
    response = await RealWorldWorkloads.mock_llm_call(prompt)
    context.set_variable("llm_response", response)
    return "api_step"

@state
async def api_step(context):
    response = context.get_variable("llm_response")
    result = await RealWorldWorkloads.mock_api_call("/save", {"data": response})
    context.set_variable("final_result", result)
    return None

@state
async def error_handler(context):
    error = context.get_variable("last_error", context.get_variable("validation_error", "Unknown"))
    context.set_variable("status", "failed")
    context.set_variable("error", error)
    return None

result = await agent.run({"query": "test query"})
"""

        # Typed: With type hints and validation
        typed_code = """
from puffinflow import Agent, state
from typing import Dict, List, Any, Optional

agent = Agent("typed-workflow")

@state
async def search_step(context) -> Optional[str]:
    query: str = context.get_variable("query")
    results: List[Dict[str, Any]] = await RealWorldWorkloads.mock_vector_search(query)
    context.set_variable("search_results", results)
    return "llm_step"

@state
async def llm_step(context) -> Optional[str]:
    results: List[Dict[str, Any]] = context.get_variable("search_results")
    prompt: str = f"Analyze: {results}"
    response: Dict[str, Any] = await RealWorldWorkloads.mock_llm_call(prompt)
    context.set_variable("llm_response", response)
    return "api_step"

@state
async def api_step(context) -> None:
    response: Dict[str, Any] = context.get_variable("llm_response")
    result: Dict[str, Any] = await RealWorldWorkloads.mock_api_call("/save", {"data": response})
    context.set_variable("final_result", result)
    return None

result = await agent.run({"query": "test query"})
"""

        return {
            "simple": len(
                [line for line in simple_code.strip().split("\n") if line.strip()]
            ),
            "complex": len(
                [line for line in complex_code.strip().split("\n") if line.strip()]
            ),
            "typed": len(
                [line for line in typed_code.strip().split("\n") if line.strip()]
            ),
        }

    async def test_execution_speed(self, complexity: str):
        """Test execution speed using real workloads."""

        if complexity == "simple":
            return await self._test_simple_workflow()
        elif complexity == "complex":
            return await self._test_complex_workflow()
        else:  # io_heavy
            return await self._test_io_heavy_workflow()

    async def _test_simple_workflow(self):
        """Simple 3-step workflow."""
        agent = Agent("simple-speed-test")

        @state
        async def search_step(context):
            query = context.get_variable("query")
            results = await RealWorldWorkloads.mock_vector_search(query)
            context.set_variable("search_results", results)
            return "llm_step"

        @state
        async def llm_step(context):
            results = context.get_variable("search_results")
            prompt = f"Analyze: {results}"
            response = await RealWorldWorkloads.mock_llm_call(prompt)
            context.set_variable("llm_response", response)
            return "api_step"

        @state
        async def api_step(context):
            response = context.get_variable("llm_response")
            result = await RealWorldWorkloads.mock_api_call("/save", {"data": response})
            context.set_variable("final_result", result)
            return None

        agent.add_state("search_step", search_step)
        agent.add_state("llm_step", llm_step)
        agent.add_state("api_step", api_step)

        start_time = time.perf_counter()
        await agent.run({"query": "test query"})
        execution_time = (time.perf_counter() - start_time) * 1000
        return execution_time

    async def _test_complex_workflow(self):
        """Complex workflow with error handling."""
        agent = Agent("complex-speed-test")

        @state
        async def search_with_retry(context):
            query = context.get_variable("query")
            results = await RealWorldWorkloads.mock_vector_search(query)
            context.set_variable("search_results", results)
            return "validate_results"

        @state
        async def validate_results(context):
            return "llm_step"  # Always pass for speed test

        @state
        async def llm_step(context):
            results = context.get_variable("search_results")
            prompt = f"Analyze: {results}"
            response = await RealWorldWorkloads.mock_llm_call(prompt)
            context.set_variable("llm_response", response)
            return "api_step"

        @state
        async def api_step(context):
            response = context.get_variable("llm_response")
            result = await RealWorldWorkloads.mock_api_call("/save", {"data": response})
            context.set_variable("final_result", result)
            return None

        agent.add_state("search_with_retry", search_with_retry)
        agent.add_state("validate_results", validate_results)
        agent.add_state("llm_step", llm_step)
        agent.add_state("api_step", api_step)

        start_time = time.perf_counter()
        await agent.run({"query": "test query"})
        execution_time = (time.perf_counter() - start_time) * 1000
        return execution_time

    async def _test_io_heavy_workflow(self):
        """I/O heavy workflow with multiple API calls."""
        agent = Agent("io-heavy-speed-test")

        @state
        async def parallel_search(context):
            query = context.get_variable("query")
            tasks = [
                RealWorldWorkloads.mock_vector_search(f"{query}_db1"),
                RealWorldWorkloads.mock_vector_search(f"{query}_db2"),
                RealWorldWorkloads.mock_vector_search(f"{query}_db3"),
            ]
            results = await asyncio.gather(*tasks)
            context.set_variable("search_results", results)
            return "llm_step"

        @state
        async def llm_step(context):
            results = context.get_variable("search_results")
            tasks = [
                RealWorldWorkloads.mock_llm_call(f"Analyze DB1: {results[0]}"),
                RealWorldWorkloads.mock_llm_call(f"Analyze DB2: {results[1]}"),
            ]
            responses = await asyncio.gather(*tasks)
            context.set_variable("llm_responses", responses)
            return "api_step"

        @state
        async def api_step(context):
            responses = context.get_variable("llm_responses")
            tasks = [
                RealWorldWorkloads.mock_api_call("/save", {"data": responses[0]}),
                RealWorldWorkloads.mock_api_call("/notify", {"data": responses[1]}),
            ]
            results = await asyncio.gather(*tasks)
            context.set_variable("final_results", results)
            return None

        agent.add_state("parallel_search", parallel_search)
        agent.add_state("llm_step", llm_step)
        agent.add_state("api_step", api_step)

        start_time = time.perf_counter()
        await agent.run({"query": "test query"})
        execution_time = (time.perf_counter() - start_time) * 1000
        return execution_time

    async def test_framework_overhead(self, complexity: str):
        """Measure actual framework orchestration overhead."""

        # First, measure raw compute time
        if complexity == "simple":
            start_compute = time.perf_counter()
            await RealWorldWorkloads.mock_vector_search("query")
            await RealWorldWorkloads.mock_llm_call("prompt")
            await RealWorldWorkloads.mock_api_call("/save", {"data": "test"})
            pure_compute_time = (time.perf_counter() - start_compute) * 1000
        else:  # complex
            start_compute = time.perf_counter()
            await RealWorldWorkloads.mock_vector_search("query")
            await RealWorldWorkloads.mock_llm_call("prompt")
            await RealWorldWorkloads.mock_api_call("/save", {"data": "test"})
            pure_compute_time = (time.perf_counter() - start_compute) * 1000

        # Then measure with framework
        framework_time = await self.test_execution_speed(complexity)

        # Calculate overhead (properly handle near-zero times)
        if pure_compute_time < 0.001:  # If pure compute is less than 1ms
            pure_compute_time = 0.001  # Set minimum baseline

        if framework_time < 0.001:  # If framework time is less than 1ms
            framework_time = 0.001  # Set minimum

        framework_overhead = framework_time - pure_compute_time
        overhead_percent = (framework_overhead / pure_compute_time) * 100

        return max(0, overhead_percent)  # Ensure non-negative

    async def test_concurrency(self, task_count: int):
        """Test native async concurrency."""

        async def single_task(task_id: int):
            agent = Agent(f"concurrent-{task_id}")

            @state
            async def task_step(context):
                query = f"query_{task_id}"
                result = await RealWorldWorkloads.mock_vector_search(query)
                context.set_variable("result", result)
                return None

            agent.add_state("task_step", task_step)

            try:
                await agent.run({"task_id": task_id})
                return True
            except Exception:
                return False

        start_time = time.perf_counter()
        tasks = [single_task(i) for i in range(task_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time

        successful = sum(1 for r in results if r is True)
        return successful / total_time if total_time > 0 else 0

    async def test_memory_efficiency(self, complexity: str):
        """Test memory usage for different complexity levels."""
        gc.collect()
        baseline_memory = self._get_memory_mb()

        # Hold reference to prevent garbage collection
        memory_holder = []

        if complexity == "simple":
            agent = Agent("memory-test")

            @state
            async def memory_step(context):
                # Allocate and hold memory
                data = [f"memory_test_{i}" * 100 for i in range(5000)]  # ~4MB
                context.set_variable("memory_data", data)
                memory_holder.append(data)  # Prevent GC
                return None

            agent.add_state("memory_step", memory_step)
            # Just run the specific state
            await agent.run_state("memory_step")
        else:  # complex
            agent = Agent("complex-memory-test")

            @state
            async def complex_memory_step(context):
                # Allocate larger memory for complex test
                data = [f"complex_memory_{i}" * 200 for i in range(8000)]  # ~12.8MB
                context.set_variable("complex_data", data)
                memory_holder.append(data)  # Prevent GC
                return None

            agent.add_state("complex_memory_step", complex_memory_step)
            # Just run the specific state
            await agent.run_state("complex_memory_step")

        peak_memory = self._get_memory_mb()
        memory_used = peak_memory - baseline_memory

        # Clean up
        memory_holder.clear()
        gc.collect()

        return max(0, memory_used)

    def _get_memory_mb(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    async def run_full_benchmark(self, context):
        """Run complete benchmark suite."""

        # Code Efficiency
        code_metrics = self.get_code_efficiency_metrics()

        # Execution Speed
        speed_simple = await self.test_execution_speed("simple")
        speed_complex = await self.test_execution_speed("complex")
        speed_io_heavy = await self.test_execution_speed("io_heavy")

        # Framework Overhead
        overhead_simple = await self.test_framework_overhead("simple")
        overhead_complex = await self.test_framework_overhead("complex")

        # Concurrency
        concurrency_low = await self.test_concurrency(10)
        concurrency_high = await self.test_concurrency(100)

        # Memory Efficiency
        memory_simple = await self.test_memory_efficiency("simple")
        memory_complex = await self.test_memory_efficiency("complex")

        return BenchmarkResult(
            framework="PuffinFlow",
            execution_model="async",
            code_simple_loc=code_metrics["simple"],
            code_complex_loc=code_metrics["complex"],
            code_typed_loc=code_metrics["typed"],
            speed_simple_ms=speed_simple,
            speed_complex_ms=speed_complex,
            speed_io_heavy_ms=speed_io_heavy,
            overhead_simple_percent=overhead_simple,
            overhead_complex_percent=overhead_complex,
            concurrency_low_tps=concurrency_low,
            concurrency_high_tps=concurrency_high,
            memory_simple_mb=memory_simple,
            memory_complex_mb=memory_complex,
        )


# =============================================================================
# LANGGRAPH IMPLEMENTATION (Sync Native)
# =============================================================================


class LangGraphBenchmark:
    """LangGraph benchmark using native synchronous execution."""

    def setup_framework(self):
        if not LANGGRAPH_AVAILABLE:
            return {"error": "LangGraph not available"}

        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        return {
            "framework": "langgraph",
            "execution_model": "sync",
            "status": "ready",
            "StateGraph": StateGraph,
            "START": START,
            "END": END,
            "TypedDict": TypedDict,
        }

    def get_code_efficiency_metrics(self):
        """Measure equivalent functionality lines of code."""

        # Simple: 3-step workflow with imports
        simple_code = """
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class WorkflowState(TypedDict):
    query: str
    search_results: list
    llm_response: dict
    final_result: dict

def search_step(state: WorkflowState) -> WorkflowState:
    query = state["query"]
    results = RealWorldWorkloads.mock_vector_search_sync(query)
    return {"search_results": results}

def llm_step(state: WorkflowState) -> WorkflowState:
    results = state["search_results"]
    prompt = f"Analyze: {results}"
    response = RealWorldWorkloads.mock_llm_call_sync(prompt)
    return {"llm_response": response}

def api_step(state: WorkflowState) -> WorkflowState:
    response = state["llm_response"]
    result = RealWorldWorkloads.mock_api_call_sync("/save", {"data": response})
    return {"final_result": result}

workflow = StateGraph(WorkflowState)
workflow.add_node("search_step", search_step)
workflow.add_node("llm_step", llm_step)
workflow.add_node("api_step", api_step)
workflow.add_edge(START, "search_step")
workflow.add_edge("search_step", "llm_step")
workflow.add_edge("llm_step", "api_step")
workflow.add_edge("api_step", END)

app = workflow.compile()
result = app.invoke({"query": "test query", "search_results": [], "llm_response": {}, "final_result": {}})
"""

        # Complex: Error handling, retries, validation
        complex_code = """
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class ComplexState(TypedDict):
    query: str
    attempt: int
    search_results: list
    llm_response: dict
    final_result: dict
    error: str
    status: str

def search_with_retry(state: ComplexState) -> ComplexState:
    attempt = state.get("attempt", 0)
    query = state["query"]

    try:
        results = RealWorldWorkloads.mock_vector_search_sync(query)
        if not results:
            raise ValueError("No results found")
        return {"search_results": results, "attempt": attempt}
    except Exception as e:
        if attempt < 3:
            return {"attempt": attempt + 1, "error": str(e)}
        return {"error": str(e), "status": "failed"}

def validate_results(state: ComplexState) -> ComplexState:
    results = state["search_results"]
    if len(results) >= 2 and all(r.get("score", 0) > 0.5 for r in results):
        return state
    else:
        return {"error": "Quality check failed", "status": "failed"}

def llm_step(state: ComplexState) -> ComplexState:
    results = state["search_results"]
    prompt = f"Analyze: {results}"
    response = RealWorldWorkloads.mock_llm_call_sync(prompt)
    return {"llm_response": response}

def api_step(state: ComplexState) -> ComplexState:
    response = state["llm_response"]
    result = RealWorldWorkloads.mock_api_call_sync("/save", {"data": response})
    return {"final_result": result, "status": "completed"}

def error_handler(state: ComplexState) -> ComplexState:
    return {"status": "failed"}

def should_retry(state: ComplexState) -> str:
    if state.get("error") and state.get("attempt", 0) < 3:
        return "search_with_retry"
    elif state.get("error"):
        return "error_handler"
    return "validate_results"

def should_continue(state: ComplexState) -> str:
    if state.get("error"):
        return "error_handler"
    return "llm_step"

workflow = StateGraph(ComplexState)
workflow.add_node("search_with_retry", search_with_retry)
workflow.add_node("validate_results", validate_results)
workflow.add_node("llm_step", llm_step)
workflow.add_node("api_step", api_step)
workflow.add_node("error_handler", error_handler)

workflow.add_edge(START, "search_with_retry")
workflow.add_conditional_edges("search_with_retry", should_retry)
workflow.add_conditional_edges("validate_results", should_continue)
workflow.add_edge("llm_step", "api_step")
workflow.add_edge("api_step", END)
workflow.add_edge("error_handler", END)

app = workflow.compile()
result = app.invoke({"query": "test query", "attempt": 0, "search_results": [], "llm_response": {}, "final_result": {}, "error": "", "status": ""})
"""

        # Typed: Already type-safe by default
        typed_code = simple_code  # LangGraph is inherently typed

        return {
            "simple": len(
                [line for line in simple_code.strip().split("\n") if line.strip()]
            ),
            "complex": len(
                [line for line in complex_code.strip().split("\n") if line.strip()]
            ),
            "typed": len(
                [line for line in typed_code.strip().split("\n") if line.strip()]
            ),
        }

    def test_execution_speed(self, complexity: str, context):
        """Test execution speed using native synchronous execution."""

        if complexity == "simple":
            return self._test_simple_workflow(context)
        elif complexity == "complex":
            return self._test_complex_workflow(context)
        else:  # io_heavy
            return self._test_io_heavy_workflow(context)

    def _test_simple_workflow(self, context):
        """Simple 3-step workflow using sync execution."""
        StateGraph = context["StateGraph"]
        START = context["START"]
        END = context["END"]
        TypedDict = context["TypedDict"]

        class WorkflowState(TypedDict):
            query: str
            search_results: list
            llm_response: dict
            final_result: dict

        def search_step(state: WorkflowState) -> WorkflowState:
            query = state["query"]
            results = RealWorldWorkloads.mock_vector_search_sync(query)
            return {"search_results": results}

        def llm_step(state: WorkflowState) -> WorkflowState:
            results = state["search_results"]
            prompt = f"Analyze: {results}"
            response = RealWorldWorkloads.mock_llm_call_sync(prompt)
            return {"llm_response": response}

        def api_step(state: WorkflowState) -> WorkflowState:
            response = state["llm_response"]
            result = RealWorldWorkloads.mock_api_call_sync("/save", {"data": response})
            return {"final_result": result}

        workflow = StateGraph(WorkflowState)
        workflow.add_node("search_step", search_step)
        workflow.add_node("llm_step", llm_step)
        workflow.add_node("api_step", api_step)
        workflow.add_edge(START, "search_step")
        workflow.add_edge("search_step", "llm_step")
        workflow.add_edge("llm_step", "api_step")
        workflow.add_edge("api_step", END)

        app = workflow.compile()

        start_time = time.perf_counter()
        app.invoke(
            {
                "query": "test query",
                "search_results": [],
                "llm_response": {},
                "final_result": {},
            }
        )
        return (time.perf_counter() - start_time) * 1000

    def _test_complex_workflow(self, context):
        """Complex workflow with error handling."""
        StateGraph = context["StateGraph"]
        START = context["START"]
        END = context["END"]
        TypedDict = context["TypedDict"]

        class ComplexState(TypedDict):
            query: str
            search_results: list
            llm_response: dict
            final_result: dict

        def search_step(state: ComplexState) -> ComplexState:
            query = state["query"]
            results = RealWorldWorkloads.mock_vector_search_sync(query)
            return {"search_results": results}

        def llm_step(state: ComplexState) -> ComplexState:
            results = state["search_results"]
            prompt = f"Analyze: {results}"
            response = RealWorldWorkloads.mock_llm_call_sync(prompt)
            return {"llm_response": response}

        def api_step(state: ComplexState) -> ComplexState:
            response = state["llm_response"]
            result = RealWorldWorkloads.mock_api_call_sync("/save", {"data": response})
            return {"final_result": result}

        workflow = StateGraph(ComplexState)
        workflow.add_node("search_step", search_step)
        workflow.add_node("llm_step", llm_step)
        workflow.add_node("api_step", api_step)
        workflow.add_edge(START, "search_step")
        workflow.add_edge("search_step", "llm_step")
        workflow.add_edge("llm_step", "api_step")
        workflow.add_edge("api_step", END)

        app = workflow.compile()

        start_time = time.perf_counter()
        app.invoke(
            {
                "query": "test query",
                "search_results": [],
                "llm_response": {},
                "final_result": {},
            }
        )
        return (time.perf_counter() - start_time) * 1000

    def _test_io_heavy_workflow(self, context):
        """I/O heavy workflow using thread pool for parallelism."""
        StateGraph = context["StateGraph"]
        START = context["START"]
        END = context["END"]
        TypedDict = context["TypedDict"]

        class IOHeavyState(TypedDict):
            query: str
            search_results: list
            llm_responses: list
            final_results: list

        def parallel_search(state: IOHeavyState) -> IOHeavyState:
            query = state["query"]
            # Use thread pool for parallel execution in sync context
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        RealWorldWorkloads.mock_vector_search_sync, f"{query}_db1"
                    ),
                    executor.submit(
                        RealWorldWorkloads.mock_vector_search_sync, f"{query}_db2"
                    ),
                    executor.submit(
                        RealWorldWorkloads.mock_vector_search_sync, f"{query}_db3"
                    ),
                ]
                results = [future.result() for future in futures]
            return {"search_results": results}

        def llm_step(state: IOHeavyState) -> IOHeavyState:
            results = state["search_results"]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        RealWorldWorkloads.mock_llm_call_sync,
                        f"Analyze DB1: {results[0]}",
                    ),
                    executor.submit(
                        RealWorldWorkloads.mock_llm_call_sync,
                        f"Analyze DB2: {results[1]}",
                    ),
                ]
                responses = [future.result() for future in futures]
            return {"llm_responses": responses}

        def api_step(state: IOHeavyState) -> IOHeavyState:
            responses = state["llm_responses"]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        RealWorldWorkloads.mock_api_call_sync,
                        "/save",
                        {"data": responses[0]},
                    ),
                    executor.submit(
                        RealWorldWorkloads.mock_api_call_sync,
                        "/notify",
                        {"data": responses[1]},
                    ),
                ]
                results = [future.result() for future in futures]
            return {"final_results": results}

        workflow = StateGraph(IOHeavyState)
        workflow.add_node("parallel_search", parallel_search)
        workflow.add_node("llm_step", llm_step)
        workflow.add_node("api_step", api_step)
        workflow.add_edge(START, "parallel_search")
        workflow.add_edge("parallel_search", "llm_step")
        workflow.add_edge("llm_step", "api_step")
        workflow.add_edge("api_step", END)

        app = workflow.compile()

        start_time = time.perf_counter()
        app.invoke(
            {
                "query": "test query",
                "search_results": [],
                "llm_responses": [],
                "final_results": [],
            }
        )
        return (time.perf_counter() - start_time) * 1000

    def test_framework_overhead(self, complexity: str, context):
        """Measure framework orchestration overhead."""

        # Measure raw compute time
        if complexity == "simple":
            start_compute = time.perf_counter()
            RealWorldWorkloads.mock_vector_search_sync("query")
            RealWorldWorkloads.mock_llm_call_sync("prompt")
            RealWorldWorkloads.mock_api_call_sync("/save", {"data": "test"})
            pure_compute_time = (time.perf_counter() - start_compute) * 1000
        else:  # complex
            start_compute = time.perf_counter()
            RealWorldWorkloads.mock_vector_search_sync("query")
            RealWorldWorkloads.mock_llm_call_sync("prompt")
            RealWorldWorkloads.mock_api_call_sync("/save", {"data": "test"})
            pure_compute_time = (time.perf_counter() - start_compute) * 1000

        # Measure with framework
        framework_time = self.test_execution_speed(complexity, context)

        # Calculate overhead (properly handle near-zero times)
        if pure_compute_time < 0.001:  # If pure compute is less than 1ms
            pure_compute_time = 0.001  # Set minimum baseline

        if framework_time < 0.001:  # If framework time is less than 1ms
            framework_time = 0.001  # Set minimum

        framework_overhead = framework_time - pure_compute_time
        overhead_percent = (framework_overhead / pure_compute_time) * 100

        return max(0, overhead_percent)  # Ensure non-negative

    def test_concurrency(self, task_count: int, context):
        """Test thread-based concurrency."""

        StateGraph = context["StateGraph"]
        START = context["START"]
        END = context["END"]
        TypedDict = context["TypedDict"]

        class TaskState(TypedDict):
            task_id: int
            result: list

        def single_task_workflow(task_id: int):
            def task_step(state: TaskState) -> TaskState:
                query = f"query_{task_id}"
                result = RealWorldWorkloads.mock_vector_search_sync(query)
                return {"result": result}

            workflow = StateGraph(TaskState)
            workflow.add_node("task_step", task_step)
            workflow.add_edge(START, "task_step")
            workflow.add_edge("task_step", END)

            app = workflow.compile()

            try:
                app.invoke({"task_id": task_id, "result": []})
                return True
            except Exception:
                return False

        start_time = time.perf_counter()

        # Use thread pool for concurrency
        with ThreadPoolExecutor(max_workers=min(task_count, 50)) as executor:
            futures = [
                executor.submit(single_task_workflow, i) for i in range(task_count)
            ]
            results = [future.result() for future in as_completed(futures)]

        total_time = time.perf_counter() - start_time
        successful = sum(1 for r in results if r is True)

        return successful / total_time if total_time > 0 else 0

    def test_memory_efficiency(self, complexity: str, context):
        """Test memory usage."""
        gc.collect()
        baseline_memory = self._get_memory_mb()

        # Hold reference to prevent garbage collection
        memory_holder = []

        StateGraph = context["StateGraph"]
        START = context["START"]
        END = context["END"]
        TypedDict = context["TypedDict"]

        class MemoryState(TypedDict):
            data: list
            result: str

        if complexity == "simple":

            def memory_step(state: MemoryState) -> MemoryState:
                # Allocate and hold memory
                data = [f"memory_test_{i}" * 100 for i in range(5000)]  # ~4MB
                memory_holder.append(data)  # Prevent GC
                return {"data": data, "result": "memory_test"}
        else:  # complex

            def memory_step(state: MemoryState) -> MemoryState:
                # Allocate larger memory for complex test
                data = [f"complex_memory_{i}" * 200 for i in range(8000)]  # ~12.8MB
                memory_holder.append(data)  # Prevent GC
                return {"data": data, "result": "complex_memory_test"}

        workflow = StateGraph(MemoryState)
        workflow.add_node("memory_step", memory_step)
        workflow.add_edge(START, "memory_step")
        workflow.add_edge("memory_step", END)

        app = workflow.compile()
        app.invoke({"data": [], "result": ""})

        peak_memory = self._get_memory_mb()
        memory_used = peak_memory - baseline_memory

        # Clean up
        memory_holder.clear()
        gc.collect()

        return max(0, memory_used)

    def _get_memory_mb(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def run_full_benchmark(self, context):
        """Run complete benchmark suite."""

        # Code Efficiency
        code_metrics = self.get_code_efficiency_metrics()

        # Execution Speed
        speed_simple = self.test_execution_speed("simple", context)
        speed_complex = self.test_execution_speed("complex", context)
        speed_io_heavy = self.test_execution_speed("io_heavy", context)

        # Framework Overhead
        overhead_simple = self.test_framework_overhead("simple", context)
        overhead_complex = self.test_framework_overhead("complex", context)

        # Concurrency
        concurrency_low = self.test_concurrency(10, context)
        concurrency_high = self.test_concurrency(100, context)

        # Memory Efficiency
        memory_simple = self.test_memory_efficiency("simple", context)
        memory_complex = self.test_memory_efficiency("complex", context)

        return BenchmarkResult(
            framework="LangGraph",
            execution_model="sync",
            code_simple_loc=code_metrics["simple"],
            code_complex_loc=code_metrics["complex"],
            code_typed_loc=code_metrics["typed"],
            speed_simple_ms=speed_simple,
            speed_complex_ms=speed_complex,
            speed_io_heavy_ms=speed_io_heavy,
            overhead_simple_percent=overhead_simple,
            overhead_complex_percent=overhead_complex,
            concurrency_low_tps=concurrency_low,
            concurrency_high_tps=concurrency_high,
            memory_simple_mb=memory_simple,
            memory_complex_mb=memory_complex,
        )


# =============================================================================
# LLAMAINDEX IMPLEMENTATION (Async Native)
# =============================================================================


class LlamaIndexBenchmark:
    """LlamaIndex benchmark using native async execution."""

    def setup_framework(self):
        if not LLAMAINDEX_AVAILABLE:
            return {"error": "LlamaIndex not available"}

        # Configure LlamaIndex settings
        Settings.llm = MockLLM()

        return {
            "framework": "llamaindex",
            "execution_model": "async",
            "status": "ready",
            "Workflow": Workflow,
            "StartEvent": StartEvent,
            "StopEvent": StopEvent,
            "step": step,
            "LlamaContext": LlamaContext,
            "Event": Event,
        }

    def get_code_efficiency_metrics(self):
        """Measure equivalent functionality lines of code."""

        # Simple: 3-step workflow with imports
        simple_code = """
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Context, Event

class SearchEvent(Event):
    results: list

class LLMEvent(Event):
    response: dict

class SimpleWorkflow(Workflow):
    @step
    async def search_step(self, ctx: Context, ev: StartEvent) -> SearchEvent:
        query = ev.query
        results = await RealWorldWorkloads.mock_vector_search(query)
        await ctx.store.set("search_results", results)
        return SearchEvent(results=results)

    @step
    async def llm_step(self, ctx: Context, ev: SearchEvent) -> LLMEvent:
        prompt = f"Analyze: {ev.results}"
        response = await RealWorldWorkloads.mock_llm_call(prompt)
        await ctx.store.set("llm_response", response)
        return LLMEvent(response=response)

    @step
    async def api_step(self, ctx: Context, ev: LLMEvent) -> StopEvent:
        result = await RealWorldWorkloads.mock_api_call("/save", {"data": ev.response})
        await ctx.store.set("final_result", result)
        return StopEvent(result=result)

workflow = SimpleWorkflow()
result = await workflow.run(query="test query")
"""

        # Complex: Error handling, retries, validation
        complex_code = """
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Context, Event

class SearchEvent(Event):
    results: list

class RetryEvent(Event):
    query: str
    attempt: int

class LLMEvent(Event):
    response: dict

class ErrorEvent(Event):
    error: str

class ComplexWorkflow(Workflow):
    @step
    async def search_with_retry(self, ctx: Context, ev: StartEvent | RetryEvent) -> SearchEvent | ErrorEvent:
        query = ev.query if hasattr(ev, 'query') else "default"
        attempt = ev.attempt if hasattr(ev, 'attempt') else 0

        try:
            results = await RealWorldWorkloads.mock_vector_search(query)
            if not results:
                raise ValueError("No results found")
            await ctx.store.set("search_results", results)
            return SearchEvent(results=results)
        except Exception as e:
            if attempt < 3:
                return RetryEvent(query=query, attempt=attempt + 1)
            return ErrorEvent(error=str(e))

    @step
    async def validate_results(self, ctx: Context, ev: SearchEvent) -> LLMEvent | ErrorEvent:
        if len(ev.results) >= 2 and all(r.get("score", 0) > 0.5 for r in ev.results):
            return LLMEvent(response={})  # Continue to LLM
        else:
            return ErrorEvent(error="Quality check failed")

    @step
    async def llm_step(self, ctx: Context, ev: LLMEvent) -> StopEvent:
        results = await ctx.store.get("search_results")
        prompt = f"Analyze: {results}"
        response = await RealWorldWorkloads.mock_llm_call(prompt)
        result = await RealWorldWorkloads.mock_api_call("/save", {"data": response})
        await ctx.store.set("final_result", result)
        return StopEvent(result=result)

    @step
    async def error_handler(self, ctx: Context, ev: ErrorEvent) -> StopEvent:
        await ctx.store.set("error", ev.error)
        await ctx.store.set("status", "failed")
        return StopEvent(result={"status": "failed", "error": ev.error})

    @step
    async def retry_handler(self, ctx: Context, ev: RetryEvent) -> SearchEvent | ErrorEvent:
        return await self.search_with_retry(ctx, ev)

workflow = ComplexWorkflow()
result = await workflow.run(query="test query")
"""

        # Typed: With type annotations
        typed_code = """
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Context, Event
from typing import List, Dict, Any

class SearchEvent(Event):
    results: List[Dict[str, Any]]

class LLMEvent(Event):
    response: Dict[str, Any]

class TypedWorkflow(Workflow):
    @step
    async def search_step(self, ctx: Context, ev: StartEvent) -> SearchEvent:
        query: str = ev.query
        results: List[Dict[str, Any]] = await RealWorldWorkloads.mock_vector_search(query)
        await ctx.store.set("search_results", results)
        return SearchEvent(results=results)

    @step
    async def llm_step(self, ctx: Context, ev: SearchEvent) -> LLMEvent:
        prompt: str = f"Analyze: {ev.results}"
        response: Dict[str, Any] = await RealWorldWorkloads.mock_llm_call(prompt)
        await ctx.store.set("llm_response", response)
        return LLMEvent(response=response)

    @step
    async def api_step(self, ctx: Context, ev: LLMEvent) -> StopEvent:
        result: Dict[str, Any] = await RealWorldWorkloads.mock_api_call("/save", {"data": ev.response})
        await ctx.store.set("final_result", result)
        return StopEvent(result=result)

workflow = TypedWorkflow()
result = await workflow.run(query="test query")
"""

        return {
            "simple": len(
                [line for line in simple_code.strip().split("\n") if line.strip()]
            ),
            "complex": len(
                [line for line in complex_code.strip().split("\n") if line.strip()]
            ),
            "typed": len(
                [line for line in typed_code.strip().split("\n") if line.strip()]
            ),
        }

    async def test_execution_speed(self, complexity: str, context):
        """Test execution speed using real workloads."""

        if complexity == "simple":
            return await self._test_simple_workflow(context)
        elif complexity == "complex":
            return await self._test_complex_workflow(context)
        else:  # io_heavy
            return await self._test_io_heavy_workflow(context)

    async def _test_simple_workflow(self, context):
        """Simple 3-step workflow."""
        Workflow = context["Workflow"]
        StartEvent = context["StartEvent"]
        StopEvent = context["StopEvent"]
        step = context["step"]
        LlamaContext = context["LlamaContext"]
        Event = context["Event"]

        class SearchEvent(Event):
            results: list

        class LLMEvent(Event):
            response: dict

        class SimpleWorkflow(Workflow):
            @step
            async def search_step(
                self, ctx: LlamaContext, ev: StartEvent
            ) -> SearchEvent:
                query = getattr(ev, "query", "test query")
                results = await RealWorldWorkloads.mock_vector_search(query)
                await ctx.store.set("search_results", results)
                return SearchEvent(results=results)

            @step
            async def llm_step(self, ctx: LlamaContext, ev: SearchEvent) -> LLMEvent:
                prompt = f"Analyze: {ev.results}"
                response = await RealWorldWorkloads.mock_llm_call(prompt)
                await ctx.store.set("llm_response", response)
                return LLMEvent(response=response)

            @step
            async def api_step(self, ctx: LlamaContext, ev: LLMEvent) -> StopEvent:
                result = await RealWorldWorkloads.mock_api_call(
                    "/save", {"data": ev.response}
                )
                await ctx.store.set("final_result", result)
                return StopEvent(result=result)

        workflow = SimpleWorkflow()

        start_time = time.perf_counter()
        await workflow.run()
        return (time.perf_counter() - start_time) * 1000

    async def _test_complex_workflow(self, context):
        """Complex workflow with error handling."""
        Workflow = context["Workflow"]
        StartEvent = context["StartEvent"]
        StopEvent = context["StopEvent"]
        step = context["step"]
        LlamaContext = context["LlamaContext"]
        Event = context["Event"]

        class SearchEvent(Event):
            results: list

        class LLMEvent(Event):
            response: dict

        class ComplexWorkflow(Workflow):
            @step
            async def search_step(
                self, ctx: LlamaContext, ev: StartEvent
            ) -> SearchEvent:
                query = getattr(ev, "query", "test query")
                results = await RealWorldWorkloads.mock_vector_search(query)
                await ctx.store.set("search_results", results)
                return SearchEvent(results=results)

            @step
            async def validate_step(
                self, ctx: LlamaContext, ev: SearchEvent
            ) -> LLMEvent:
                # Always pass for speed test
                return LLMEvent(response={})

            @step
            async def llm_step(self, ctx: LlamaContext, ev: LLMEvent) -> StopEvent:
                results = await ctx.store.get("search_results")
                prompt = f"Analyze: {results}"
                response = await RealWorldWorkloads.mock_llm_call(prompt)
                result = await RealWorldWorkloads.mock_api_call(
                    "/save", {"data": response}
                )
                await ctx.store.set("final_result", result)
                return StopEvent(result=result)

        workflow = ComplexWorkflow()

        start_time = time.perf_counter()
        await workflow.run()
        return (time.perf_counter() - start_time) * 1000

    async def _test_io_heavy_workflow(self, context):
        """I/O heavy workflow with parallel operations."""
        Workflow = context["Workflow"]
        StartEvent = context["StartEvent"]
        StopEvent = context["StopEvent"]
        step = context["step"]
        LlamaContext = context["LlamaContext"]
        Event = context["Event"]

        class SearchEvent(Event):
            results: list

        class LLMEvent(Event):
            responses: list

        class IOHeavyWorkflow(Workflow):
            @step
            async def parallel_search(
                self, ctx: LlamaContext, ev: StartEvent
            ) -> SearchEvent:
                query = getattr(ev, "query", "test query")
                tasks = [
                    RealWorldWorkloads.mock_vector_search(f"{query}_db1"),
                    RealWorldWorkloads.mock_vector_search(f"{query}_db2"),
                    RealWorldWorkloads.mock_vector_search(f"{query}_db3"),
                ]
                results = await asyncio.gather(*tasks)
                await ctx.store.set("search_results", results)
                return SearchEvent(results=results)

            @step
            async def llm_step(self, ctx: LlamaContext, ev: SearchEvent) -> LLMEvent:
                tasks = [
                    RealWorldWorkloads.mock_llm_call(f"Analyze DB1: {ev.results[0]}"),
                    RealWorldWorkloads.mock_llm_call(f"Analyze DB2: {ev.results[1]}"),
                ]
                responses = await asyncio.gather(*tasks)
                await ctx.store.set("llm_responses", responses)
                return LLMEvent(responses=responses)

            @step
            async def api_step(self, ctx: LlamaContext, ev: LLMEvent) -> StopEvent:
                tasks = [
                    RealWorldWorkloads.mock_api_call(
                        "/save", {"data": ev.responses[0]}
                    ),
                    RealWorldWorkloads.mock_api_call(
                        "/notify", {"data": ev.responses[1]}
                    ),
                ]
                results = await asyncio.gather(*tasks)
                await ctx.store.set("final_results", results)
                return StopEvent(result=results)

        workflow = IOHeavyWorkflow()

        start_time = time.perf_counter()
        await workflow.run()
        return (time.perf_counter() - start_time) * 1000

    async def test_framework_overhead(self, complexity: str, context):
        """Measure framework orchestration overhead."""

        # Measure raw compute time
        if complexity == "simple":
            start_compute = time.perf_counter()
            await RealWorldWorkloads.mock_vector_search("query")
            await RealWorldWorkloads.mock_llm_call("prompt")
            await RealWorldWorkloads.mock_api_call("/save", {"data": "test"})
            pure_compute_time = (time.perf_counter() - start_compute) * 1000
        else:  # complex
            start_compute = time.perf_counter()
            await RealWorldWorkloads.mock_vector_search("query")
            await RealWorldWorkloads.mock_llm_call("prompt")
            await RealWorldWorkloads.mock_api_call("/save", {"data": "test"})
            pure_compute_time = (time.perf_counter() - start_compute) * 1000

        # Measure with framework
        framework_time = await self.test_execution_speed(complexity, context)

        # Calculate overhead (properly handle near-zero times)
        if pure_compute_time < 0.001:  # If pure compute is less than 1ms
            pure_compute_time = 0.001  # Set minimum baseline

        if framework_time < 0.001:  # If framework time is less than 1ms
            framework_time = 0.001  # Set minimum

        framework_overhead = framework_time - pure_compute_time
        overhead_percent = (framework_overhead / pure_compute_time) * 100

        return max(0, overhead_percent)  # Ensure non-negative

    async def test_concurrency(self, task_count: int, context):
        """Test native async concurrency."""

        Workflow = context["Workflow"]
        StartEvent = context["StartEvent"]
        StopEvent = context["StopEvent"]
        step = context["step"]
        LlamaContext = context["LlamaContext"]

        async def single_task(task_id: int):
            class TaskWorkflow(Workflow):
                def __init__(self, task_id):
                    super().__init__()
                    self.task_id = task_id

                @step
                async def task_step(
                    self, ctx: LlamaContext, ev: StartEvent
                ) -> StopEvent:
                    query = f"query_{self.task_id}"
                    result = await RealWorldWorkloads.mock_vector_search(query)
                    await ctx.store.set("result", result)
                    return StopEvent(result=result)

            workflow = TaskWorkflow(task_id)

            try:
                await workflow.run()
                return True
            except Exception:
                return False

        start_time = time.perf_counter()
        tasks = [single_task(i) for i in range(task_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time

        successful = sum(1 for r in results if r is True)
        return successful / total_time if total_time > 0 else 0

    async def test_memory_efficiency(self, complexity: str, context):
        """Test memory usage."""
        gc.collect()
        baseline_memory = self._get_memory_mb()

        # Hold reference to prevent garbage collection
        memory_holder = []

        Workflow = context["Workflow"]
        StartEvent = context["StartEvent"]
        StopEvent = context["StopEvent"]
        step = context["step"]
        LlamaContext = context["LlamaContext"]
        Event = context["Event"]

        class MemoryEvent(Event):
            data: list

        class MemoryWorkflow(Workflow):
            @step
            async def memory_step(self, ctx: LlamaContext, ev: StartEvent) -> StopEvent:
                if complexity == "simple":
                    # Allocate and hold memory
                    data = [f"memory_test_{i}" * 100 for i in range(5000)]  # ~4MB
                else:  # complex
                    # Allocate larger memory for complex test
                    data = [f"complex_memory_{i}" * 200 for i in range(8000)]  # ~12.8MB

                memory_holder.append(data)  # Prevent GC
                await ctx.store.set("memory_data", data)
                return StopEvent(result="memory_test")

        workflow = MemoryWorkflow()
        await workflow.run()

        peak_memory = self._get_memory_mb()
        memory_used = peak_memory - baseline_memory

        # Clean up
        memory_holder.clear()
        gc.collect()

        return max(0, memory_used)

    def _get_memory_mb(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    async def run_full_benchmark(self, context):
        """Run complete benchmark suite."""

        # Code Efficiency
        code_metrics = self.get_code_efficiency_metrics()

        # Execution Speed
        speed_simple = await self.test_execution_speed("simple", context)
        speed_complex = await self.test_execution_speed("complex", context)
        speed_io_heavy = await self.test_execution_speed("io_heavy", context)

        # Framework Overhead
        overhead_simple = await self.test_framework_overhead("simple", context)
        overhead_complex = await self.test_framework_overhead("complex", context)

        # Concurrency
        concurrency_low = await self.test_concurrency(10, context)
        concurrency_high = await self.test_concurrency(100, context)

        # Memory Efficiency
        memory_simple = await self.test_memory_efficiency("simple", context)
        memory_complex = await self.test_memory_efficiency("complex", context)

        return BenchmarkResult(
            framework="LlamaIndex",
            execution_model="async",
            code_simple_loc=code_metrics["simple"],
            code_complex_loc=code_metrics["complex"],
            code_typed_loc=code_metrics["typed"],
            speed_simple_ms=speed_simple,
            speed_complex_ms=speed_complex,
            speed_io_heavy_ms=speed_io_heavy,
            overhead_simple_percent=overhead_simple,
            overhead_complex_percent=overhead_complex,
            concurrency_low_tps=concurrency_low,
            concurrency_high_tps=concurrency_high,
            memory_simple_mb=memory_simple,
            memory_complex_mb=memory_complex,
        )


# =============================================================================
# BENCHMARK RUNNER AND RESULTS
# =============================================================================


class FairBenchmarkRunner:
    """Fair benchmark runner that respects each framework's execution model."""

    def __init__(self):
        self.process = psutil.Process()
        self.results: list[BenchmarkResult] = []

    async def run_framework_benchmark(
        self, framework_name: str, benchmark_class
    ) -> BenchmarkResult:
        """Run complete benchmark for a framework using its native execution model."""

        print(f"🔧 Running fair benchmark for {framework_name}...")

        benchmark = benchmark_class()

        # Setup framework
        setup_start = time.perf_counter()
        context = benchmark.setup_framework()
        (time.perf_counter() - setup_start) * 1000

        if context.get("error"):
            print(f"  ❌ {framework_name}: Setup failed - {context['error']}")
            return self._create_failed_result(framework_name)

        try:
            # Run benchmark using appropriate execution model
            if context["execution_model"] == "async":
                result = await benchmark.run_full_benchmark(context)
            else:  # sync
                result = benchmark.run_full_benchmark(context)

            self.results.append(result)

            print(
                f"  ✅ {framework_name} ({context['execution_model']}): Benchmark completed"
            )
            print(
                f"     Code LOC: {result.code_simple_loc}/{result.code_complex_loc}/{result.code_typed_loc}"
            )
            print(
                f"     Speed: {result.speed_simple_ms:.1f}/{result.speed_complex_ms:.1f}/{result.speed_io_heavy_ms:.1f}ms"
            )
            print(
                f"     Overhead: {result.overhead_simple_percent:.1f}/{result.overhead_complex_percent:.1f}%"
            )

            return result

        except Exception as e:
            print(f"  ❌ {framework_name}: Benchmark failed - {e!s}")
            import traceback

            traceback.print_exc()
            return self._create_failed_result(framework_name)

    def _create_failed_result(self, framework: str) -> BenchmarkResult:
        """Create a failed result."""
        return BenchmarkResult(
            framework=framework,
            execution_model="unknown",
            code_simple_loc=999,
            code_complex_loc=999,
            code_typed_loc=999,
            speed_simple_ms=float("inf"),
            speed_complex_ms=float("inf"),
            speed_io_heavy_ms=float("inf"),
            overhead_simple_percent=100.0,
            overhead_complex_percent=100.0,
            concurrency_low_tps=0.0,
            concurrency_high_tps=0.0,
            memory_simple_mb=float("inf"),
            memory_complex_mb=float("inf"),
        )


def print_fair_benchmark_results(results: list[BenchmarkResult]):
    """Print fair benchmark results with execution model awareness."""

    print("\n" + "=" * 120)
    print("🏆 FAIR AGENT FRAMEWORK BENCHMARK RESULTS")
    print("=" * 120)
    print("Each framework tested using its intended execution model:")
    print("• PuffinFlow: Native async with state management")
    print("• LangGraph: Native sync with typed state graphs")
    print("• LlamaIndex: Native async with event-driven workflows")
    print("• Real workloads: Mock LLM calls, vector search, API calls")
    print("• Equivalent functionality measured consistently")
    print("=" * 120)

    if not results:
        print("No benchmark results available.")
        return

    # Filter successful results
    successful_results = [r for r in results if r.code_simple_loc < 999]

    if not successful_results:
        print("No frameworks completed successfully.")
        return

    # 1. EXECUTION MODEL SUMMARY
    print("\n🔄 EXECUTION MODELS")
    print("-" * 60)
    print(f"{'Framework':<12} {'Execution Model':<15} {'Concurrency Type'}")
    print("-" * 60)
    for result in successful_results:
        concurrency_type = (
            "async/await" if result.execution_model == "async" else "threads"
        )
        print(f"{result.framework:<12} {result.execution_model:<15} {concurrency_type}")

    # 2. CODE EFFICIENCY ANALYSIS
    print("\n📝 CODE EFFICIENCY (Lines of Code - Lower is Better)")
    print("-" * 80)
    print(
        f"{'Framework':<12} {'Simple':<8} {'Complex':<9} {'Typed':<8} {'Avg':<8} {'Best In'}"
    )
    print("-" * 80)

    for result in successful_results:
        avg_loc = (
            result.code_simple_loc + result.code_complex_loc + result.code_typed_loc
        ) / 3
        best_in = []

        if result.code_simple_loc == min(r.code_simple_loc for r in successful_results):
            best_in.append("Simple")
        if result.code_complex_loc == min(
            r.code_complex_loc for r in successful_results
        ):
            best_in.append("Complex")
        if result.code_typed_loc == min(r.code_typed_loc for r in successful_results):
            best_in.append("Typed")

        best_str = ", ".join(best_in) if best_in else "-"

        print(
            f"{result.framework:<12} {result.code_simple_loc:<8} {result.code_complex_loc:<9} "
            f"{result.code_typed_loc:<8} {avg_loc:<8.1f} {best_str}"
        )

    # 3. EXECUTION SPEED ANALYSIS
    print("\n⚡ EXECUTION SPEED (Milliseconds - Lower is Better)")
    print("-" * 80)
    print(
        f"{'Framework':<12} {'Model':<6} {'Simple':<8} {'Complex':<9} {'I/O Heavy':<10} {'Best In'}"
    )
    print("-" * 80)

    for result in successful_results:
        best_in = []

        if result.speed_simple_ms == min(r.speed_simple_ms for r in successful_results):
            best_in.append("Simple")
        if result.speed_complex_ms == min(
            r.speed_complex_ms for r in successful_results
        ):
            best_in.append("Complex")
        if result.speed_io_heavy_ms == min(
            r.speed_io_heavy_ms for r in successful_results
        ):
            best_in.append("I/O")

        best_str = ", ".join(best_in) if best_in else "-"

        print(
            f"{result.framework:<12} {result.execution_model:<6} {result.speed_simple_ms:<8.1f} "
            f"{result.speed_complex_ms:<9.1f} {result.speed_io_heavy_ms:<10.1f} {best_str}"
        )

    # 4. FRAMEWORK OVERHEAD ANALYSIS
    print("\n🔧 FRAMEWORK OVERHEAD (Percentage - Lower is Better)")
    print("-" * 70)
    print(
        f"{'Framework':<12} {'Simple (%)':<11} {'Complex (%)':<12} {'Avg (%)':<9} {'Best In'}"
    )
    print("-" * 70)

    for result in successful_results:
        avg_overhead = (
            result.overhead_simple_percent + result.overhead_complex_percent
        ) / 2
        best_in = []

        if result.overhead_simple_percent == min(
            r.overhead_simple_percent for r in successful_results
        ):
            best_in.append("Simple")
        if result.overhead_complex_percent == min(
            r.overhead_complex_percent for r in successful_results
        ):
            best_in.append("Complex")

        best_str = ", ".join(best_in) if best_in else "-"

        print(
            f"{result.framework:<12} {result.overhead_simple_percent:<11.1f} "
            f"{result.overhead_complex_percent:<12.1f} {avg_overhead:<9.1f} {best_str}"
        )

    # 5. CONCURRENCY ANALYSIS
    print("\n🚀 CONCURRENCY (Tasks per Second - Higher is Better)")
    print("-" * 80)
    print(
        f"{'Framework':<12} {'Model':<6} {'Low (10)':<9} {'High (100)':<11} {'Scaling':<10} {'Best In'}"
    )
    print("-" * 80)

    for result in successful_results:
        scaling_factor = (
            result.concurrency_high_tps / result.concurrency_low_tps
            if result.concurrency_low_tps > 0
            else 0
        )
        best_in = []

        if result.concurrency_low_tps == max(
            r.concurrency_low_tps for r in successful_results
        ):
            best_in.append("Low")
        if result.concurrency_high_tps == max(
            r.concurrency_high_tps for r in successful_results
        ):
            best_in.append("High")

        best_str = ", ".join(best_in) if best_in else "-"

        print(
            f"{result.framework:<12} {result.execution_model:<6} {result.concurrency_low_tps:<9.1f} "
            f"{result.concurrency_high_tps:<11.1f} {scaling_factor:<10.2f}x {best_str}"
        )

    # 6. MEMORY EFFICIENCY ANALYSIS
    print("\n💾 MEMORY EFFICIENCY (MB per Task - Lower is Better)")
    print("-" * 70)
    print(f"{'Framework':<12} {'Simple':<8} {'Complex':<9} {'Avg':<8} {'Best In'}")
    print("-" * 70)

    for result in successful_results:
        avg_memory = (result.memory_simple_mb + result.memory_complex_mb) / 2
        best_in = []

        valid_simple = [
            r.memory_simple_mb
            for r in successful_results
            if r.memory_simple_mb != float("inf")
        ]
        valid_complex = [
            r.memory_complex_mb
            for r in successful_results
            if r.memory_complex_mb != float("inf")
        ]

        if valid_simple and result.memory_simple_mb == min(valid_simple):
            best_in.append("Simple")
        if valid_complex and result.memory_complex_mb == min(valid_complex):
            best_in.append("Complex")

        best_str = ", ".join(best_in) if best_in else "-"

        simple_str = (
            f"{result.memory_simple_mb:.1f}"
            if result.memory_simple_mb != float("inf")
            else "∞"
        )
        complex_str = (
            f"{result.memory_complex_mb:.1f}"
            if result.memory_complex_mb != float("inf")
            else "∞"
        )
        avg_str = f"{avg_memory:.1f}" if avg_memory != float("inf") else "∞"

        print(
            f"{result.framework:<12} {simple_str:<8} {complex_str:<9} {avg_str:<8} {best_str}"
        )

    # 7. OVERALL PERFORMANCE CHAMPION
    print("\n🏆 OVERALL PERFORMANCE ANALYSIS")
    print("-" * 120)

    # Calculate weighted scores
    framework_scores = {}

    for result in successful_results:
        scores = {}

        # Code efficiency (lower is better - invert for scoring)
        total_code = (
            result.code_simple_loc + result.code_complex_loc + result.code_typed_loc
        )
        max_code = max(
            r.code_simple_loc + r.code_complex_loc + r.code_typed_loc
            for r in successful_results
        )
        scores["code"] = (max_code - total_code) / max_code * 100

        # Speed (lower is better - invert for scoring)
        total_speed = (
            result.speed_simple_ms + result.speed_complex_ms + result.speed_io_heavy_ms
        )
        max_speed = max(
            r.speed_simple_ms + r.speed_complex_ms + r.speed_io_heavy_ms
            for r in successful_results
        )
        scores["speed"] = (max_speed - total_speed) / max_speed * 100

        # Overhead (lower is better - invert for scoring)
        avg_overhead = (
            result.overhead_simple_percent + result.overhead_complex_percent
        ) / 2
        max_overhead = max(
            (r.overhead_simple_percent + r.overhead_complex_percent) / 2
            for r in successful_results
        )
        scores["overhead"] = (max_overhead - avg_overhead) / max_overhead * 100

        # Concurrency (higher is better)
        total_concurrency = result.concurrency_low_tps + result.concurrency_high_tps
        max_concurrency = max(
            r.concurrency_low_tps + r.concurrency_high_tps for r in successful_results
        )
        scores["concurrency"] = (
            (total_concurrency / max_concurrency) * 100 if max_concurrency > 0 else 0
        )

        # Memory (lower is better - handle infinity)
        valid_memory = []
        for r in successful_results:
            total_mem = r.memory_simple_mb + r.memory_complex_mb
            if total_mem != float("inf"):
                valid_memory.append(total_mem)

        if valid_memory:
            max_memory = max(valid_memory)
            result_memory = result.memory_simple_mb + result.memory_complex_mb
            if result_memory != float("inf") and max_memory > 0:
                scores["memory"] = (max_memory - result_memory) / max_memory * 100
            else:
                scores["memory"] = 0
        else:
            scores["memory"] = 50  # Neutral score if no valid data

        # Overall weighted score
        overall = (
            scores["code"] * 0.20  # 20% weight
            + scores["speed"] * 0.30  # 30% weight (execution is critical)
            + scores["overhead"] * 0.25  # 25% weight (framework efficiency)
            + scores["concurrency"] * 0.15  # 15% weight
            + scores["memory"] * 0.10  # 10% weight
        )

        framework_scores[result.framework] = {
            "overall": overall,
            "execution_model": result.execution_model,
            **scores,
        }

    # Sort by overall score
    sorted_frameworks = sorted(
        framework_scores.items(), key=lambda x: x[1]["overall"], reverse=True
    )

    print(
        f"{'Rank':<6} {'Framework':<12} {'Model':<6} {'Overall':<8} {'Code':<6} {'Speed':<6} {'Overhead':<9} {'Concur':<6} {'Memory'}"
    )
    print("-" * 120)

    for i, (framework, scores) in enumerate(sorted_frameworks):
        rank_symbol = (
            "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        )

        print(
            f"{rank_symbol:<6} {framework:<12} {scores['execution_model']:<6} {scores['overall']:<8.1f} "
            f"{scores['code']:<6.1f} {scores['speed']:<6.1f} {scores['overhead']:<9.1f} "
            f"{scores['concurrency']:<6.1f} {scores['memory']:.1f}"
        )

    # Performance insights
    champion_name, champion_scores = sorted_frameworks[0]
    print(f"\n🏆 PERFORMANCE CHAMPION: {champion_name}")
    print(f"   • Execution Model: {champion_scores['execution_model']}")
    print(f"   • Overall Score: {champion_scores['overall']:.1f}/100")
    print("   • Strongest in: ", end="")

    strengths = []
    if champion_scores["code"] >= 80:
        strengths.append("code efficiency")
    if champion_scores["speed"] >= 80:
        strengths.append("execution speed")
    if champion_scores["overhead"] >= 80:
        strengths.append("low overhead")
    if champion_scores["concurrency"] >= 80:
        strengths.append("concurrency")
    if champion_scores["memory"] >= 80:
        strengths.append("memory efficiency")

    print(", ".join(strengths) if strengths else "balanced performance")

    print("\n📊 KEY INSIGHTS:")
    print("   • Async frameworks (PuffinFlow, LlamaIndex) excel at I/O-heavy workloads")
    print(
        "   • Sync frameworks (LangGraph) may have lower overhead for CPU-bound tasks"
    )
    print("   • Framework choice should align with your workload characteristics")
    print("   • All frameworks are viable - choose based on your specific needs")


async def main():
    """Run the fair benchmark suite."""

    print("🔧 Fair Agent Framework Benchmark Suite")
    print("=" * 80)
    print("Testing frameworks using their intended execution models:")
    print("• PuffinFlow: Native async execution")
    print("• LangGraph: Native sync execution with thread-based concurrency")
    print("• LlamaIndex: Native async execution")
    print("• Real workloads: LLM calls, vector search, API calls")
    print("• Consistent measurement methodology")
    print("=" * 80)

    runner = FairBenchmarkRunner()

    # Framework benchmarks to run
    framework_benchmarks = [
        ("PuffinFlow", PuffinFlowBenchmark),
        ("LangGraph", LangGraphBenchmark),
        ("LlamaIndex", LlamaIndexBenchmark),
    ]

    # Run benchmarks
    for framework_name, benchmark_class in framework_benchmarks:
        try:
            await runner.run_framework_benchmark(framework_name, benchmark_class)
        except Exception as e:
            print(f"❌ {framework_name}: Benchmark suite failed - {e!s}")
            import traceback

            traceback.print_exc()

    # Print results
    print_fair_benchmark_results(runner.results)

    print("\n✅ Fair benchmark suite completed!")
    print(
        f"📊 Frameworks tested: {len([r for r in runner.results if r.code_simple_loc < 999])}"
    )
    print("📈 Each framework tested using its native execution model")
    print("🎯 Results reflect real-world performance characteristics")

    return runner.results


if __name__ == "__main__":
    results = asyncio.run(main())
