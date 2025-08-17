export const troubleshootingMarkdown = `# Troubleshooting Guide

This guide helps you resolve common issues when working with PuffinFlow. If you don't find your issue here, please check our [GitHub Issues](https://github.com/m-ahmed-elbeskeri/puffinflow/issues) or create a new one.

## Installation Issues

### pip install fails

**Problem:** \`pip install puffinflow\` fails with dependency conflicts or build errors.

**Solutions:**

1. **Update pip and setuptools:**
   \`\`\`bash
   pip install --upgrade pip setuptools wheel
   pip install puffinflow
   \`\`\`

2. **Use virtual environment:**
   \`\`\`bash
   python -m venv puffinflow-env
   source puffinflow-env/bin/activate  # On Windows: puffinflow-env\\Scripts\\activate
   pip install puffinflow
   \`\`\`

3. **Install with specific Python version:**
   \`\`\`bash
   python3.9 -m pip install puffinflow
   \`\`\`

4. **Clear pip cache:**
   \`\`\`bash
   pip cache purge
   pip install puffinflow
   \`\`\`

### Import errors

**Problem:** \`ImportError: cannot import name 'Agent' from 'puffinflow'\`

**Solutions:**

1. **Verify installation:**
   \`\`\`bash
   pip show puffinflow
   python -c "import puffinflow; print(puffinflow.__version__)"
   \`\`\`

2. **Check Python version compatibility:**
   \`\`\`bash
   python --version  # Should be 3.8+
   \`\`\`

3. **Reinstall PuffinFlow:**
   \`\`\`bash
   pip uninstall puffinflow
   pip install puffinflow
   \`\`\`

---

## Runtime Issues

### Agent won't start

**Problem:** Agent fails to start or hangs during initialization.

**Diagnosis:**
\`\`\`python
import asyncio
import logging
from puffinflow import Agent

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

agent = Agent("debug-agent")

@agent.state
async def test_state(context):
    print("Agent is working!")
    return None

# Test basic functionality
if __name__ == "__main__":
    asyncio.run(agent.run())
\`\`\`

**Common causes:**
1. **Missing async/await:** Ensure all state functions are async
2. **Blocking operations:** Use async versions of I/O operations
3. **Resource conflicts:** Check if required resources are available

### Context data not persisting

**Problem:** Data stored in context disappears between states.

**Solution:**
\`\`\`python
async def state_one(context):
    # ✅ Correct - data persists
    context.set_variable("data", {"key": "value"})

    # ❌ Incorrect - local variable, doesn't persist
    local_data = {"key": "value"}

    return "state_two"

async def state_two(context):
    # ✅ This works
    data = context.get_variable("data")

    # ❌ This fails - local_data doesn't exist
    # print(local_data)
\`\`\`

### States not executing in expected order

**Problem:** States execute out of order or skip states.

**Common causes:**
1. **Missing dependencies:** Use explicit dependencies
   \`\`\`python
   agent.add_state("dependent", func, dependencies=["prerequisite"])
   \`\`\`

2. **Incorrect return values:** Check state return values
   \`\`\`python
   async def router_state(context):
       # ✅ Correct - returns next state name
       return "next_state"

       # ❌ Incorrect - returns None, workflow ends
       # return None
   \`\`\`

---

## Performance Issues

### Agent running slowly

**Problem:** Agent operations are slower than expected.

**Diagnosis steps:**

1. **Enable performance monitoring:**
   \`\`\`python
   from puffinflow import Agent
   from puffinflow.observability import enable_monitoring

   enable_monitoring()
   agent = Agent("performance-test")
   \`\`\`

2. **Check resource allocation:**
   \`\`\`python
   @agent.state(cpu=2.0, memory=1024)  # Allocate adequate resources
   async def resource_intensive_task(context):
       # Your code here
       pass
   \`\`\`

3. **Profile async operations:**
   \`\`\`python
   import asyncio
   import time

   async def slow_state(context):
       start = time.time()

       # Your async operation
       await asyncio.sleep(0.1)  # Replace with actual work

       elapsed = time.time() - start
       print(f"Operation took {elapsed:.2f} seconds")
   \`\`\`

### Memory usage growing

**Problem:** Memory consumption increases over time.

**Solutions:**

1. **Use context cleanup:**
   \`\`\`python
   async def cleanup_state(context):
       # Clear large data when no longer needed
       context.clear_variable("large_dataset")

       # Use cached data with TTL
       context.set_cached("temp_data", data, ttl=300)  # 5 minutes
   \`\`\`

2. **Implement resource limits:**
   \`\`\`python
   @agent.state(memory=1024, timeout=60.0)  # 1GB limit, 60s timeout
   async def memory_intensive_task(context):
       # Process data in chunks
       for chunk in process_in_chunks(large_data):
           await process_chunk(chunk)
   \`\`\`

---

## Common Error Messages

### "State 'state_name' not found"

**Problem:** Agent tries to transition to a non-existent state.

**Solution:**
\`\`\`python
# ✅ Ensure state is registered
agent.add_state("target_state", target_function)

async def source_state(context):
    # ✅ Return registered state name
    return "target_state"

    # ❌ Don't return unregistered state names
    # return "nonexistent_state"
\`\`\`

### "Context variable 'key' not found"

**Problem:** Trying to access a variable that doesn't exist.

**Solution:**
\`\`\`python
async def safe_access(context):
    # ✅ Use get_variable with default
    value = context.get_variable("key", "default_value")

    # ✅ Check if variable exists
    if context.has_variable("key"):
        value = context.get_variable("key")

    # ❌ Direct access without checking
    # value = context.get_variable("key")  # May raise KeyError
\`\`\`

### "Resource allocation failed"

**Problem:** Insufficient resources available for state execution.

**Solutions:**

1. **Reduce resource requirements:**
   \`\`\`python
   @agent.state(cpu=1.0, memory=512)  # Reduce from higher values
   async def lightweight_task(context):
       pass
   \`\`\`

2. **Use priority scheduling:**
   \`\`\`python
   from puffinflow import Priority

   @agent.state(priority=Priority.HIGH)
   async def important_task(context):
       pass
   \`\`\`

3. **Implement backoff and retry:**
   \`\`\`python
   @agent.state(max_retries=3, retry_delay=1.0)
   async def retry_on_resource_failure(context):
       pass
   \`\`\`

---

## Development and Testing

### Testing agent workflows

**Problem:** How to test agent workflows effectively.

**Solution:**
\`\`\`python
import pytest
from puffinflow import Agent

@pytest.mark.asyncio
async def test_agent_workflow():
    agent = Agent("test-agent")

    @agent.state
    async def test_state(context):
        context.set_variable("test_result", "success")
        return None

    # Run agent with test data
    result = await agent.run(
        initial_context={"input": "test_data"}
    )

    # Assert expected outcomes
    assert result.get_variable("test_result") == "success"
\`\`\`

### Debugging state transitions

**Problem:** Hard to track state transitions during development.

**Solution:**
\`\`\`python
import logging
from puffinflow import Agent

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

agent = Agent("debug-agent")

@agent.state
async def debug_state(context):
    logger.debug(f"Executing state with context: {context.get_all_variables()}")

    # Your state logic
    result = await some_operation()

    logger.debug(f"State result: {result}")
    return "next_state"
\`\`\`

---

## Production Deployment

### Agent performance in production

**Problem:** Agent performs differently in production vs development.

**Production checklist:**

1. **Use production-ready configuration:**
   \`\`\`python
   from puffinflow import Agent
   from puffinflow.observability import configure_monitoring

   # Configure for production
   configure_monitoring(
       enable_metrics=True,
       enable_tracing=True,
       sample_rate=0.1  # 10% sampling to reduce overhead
   )

   agent = Agent("production-agent")
   \`\`\`

2. **Implement proper error handling:**
   \`\`\`python
   @agent.state(max_retries=3, timeout=30.0)
   async def production_state(context):
       try:
           result = await external_api_call()
           context.set_variable("result", result)
       except Exception as e:
           logger.error(f"Production error: {e}")
           context.set_variable("error", str(e))
           return "error_handler"
   \`\`\`

3. **Use health checks:**
   \`\`\`python
   async def health_check():
       agent = Agent("health-check")

       @agent.state
       async def ping(context):
           return None

       try:
           await asyncio.wait_for(agent.run(), timeout=5.0)
           return True
       except asyncio.TimeoutError:
           return False
   \`\`\`

### Monitoring and alerting

**Problem:** Need visibility into agent performance in production.

**Solution:**
\`\`\`python
from puffinflow.observability import MetricsCollector, AlertManager

# Set up monitoring
metrics = MetricsCollector()
alerts = AlertManager()

@agent.state
async def monitored_state(context):
    with metrics.timer("state_execution_time"):
        try:
            result = await business_logic()
            metrics.increment("successful_operations")
            return "success"
        except Exception as e:
            metrics.increment("failed_operations")
            alerts.send_alert(f"State failed: {e}")
            return "error"
\`\`\`

---

## Getting Help

### Community resources

- **GitHub Issues**: [Report bugs and request features](https://github.com/m-ahmed-elbeskeri/puffinflow/issues)
- **Discussions**: [Ask questions and share experiences](https://github.com/m-ahmed-elbeskeri/puffinflow/discussions)
- **Documentation**: [Complete guides and API reference](https://puffinflow.readthedocs.io/)

### Creating effective bug reports

When reporting issues, include:

1. **PuffinFlow version**: \`pip show puffinflow\`
2. **Python version**: \`python --version\`
3. **Operating system**: Windows/macOS/Linux
4. **Minimal reproduction code**
5. **Expected vs actual behavior**
6. **Error messages and stack traces**

**Example bug report:**
\`\`\`
**Environment:**
- PuffinFlow: 1.0.0
- Python: 3.9.7
- OS: Ubuntu 20.04

**Issue:** Agent hangs after state transition

**Reproduction:**
\`\`\`python
from puffinflow import Agent

agent = Agent("bug-reproduction")

@agent.state
async def hanging_state(context):
    # This hangs
    return "next_state"

asyncio.run(agent.run())
\`\`\`

**Expected:** Agent should complete
**Actual:** Agent hangs indefinitely
**Error:** No error message, just hangs
\`\`\`

---

## Advanced Troubleshooting

### Debugging async issues

**Problem:** Complex async behavior causing issues.

**Debugging tools:**
\`\`\`python
import asyncio
import traceback

async def debug_async_issue():
    try:
        # Your async code
        await problematic_function()
    except Exception as e:
        # Print full stack trace
        traceback.print_exc()

        # Get event loop info
        loop = asyncio.get_event_loop()
        print(f"Event loop: {loop}")
        print(f"Running: {loop.is_running()}")

        # Check for pending tasks
        tasks = asyncio.all_tasks(loop)
        print(f"Pending tasks: {len(tasks)}")
        for task in tasks:
            print(f"  {task}")
\`\`\`

### Performance profiling

**Problem:** Need to identify performance bottlenecks.

**Profiling approach:**
\`\`\`python
import cProfile
import pstats
import asyncio

def profile_agent():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run your agent
    asyncio.run(agent.run())

    profiler.disable()

    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
\`\`\`

This troubleshooting guide should help you resolve most common issues with PuffinFlow. For additional help, don't hesitate to reach out to the community!
`.trim();
