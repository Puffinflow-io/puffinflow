import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import asyncio
import time
from puffinflow import Agent, Context, state
from puffinflow.core.agent.result import AgentStatus
from puffinflow.core.resources.pool import ResourcePool
from puffinflow.core.resources.requirements import ResourceRequirements

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

class DebugResourceIntensiveAgent(Agent):
    def __init__(self, name: str, cpu_req: float = 1.0, memory_req: float = 256.0):
        super().__init__(name)
        self.set_variable('cpu_req', cpu_req)
        self.set_variable('memory_req', memory_req)
        self.add_state('start', self.start)
    
    @state(cpu=1.5, memory=256.0, io=10.0, network=5.0)
    async def start(self, context: Context):
        print(f">>> Inside start method - self: {self}, context: {context}")
        cpu_req = self.get_variable('cpu_req', 1.0)
        memory_req = self.get_variable('memory_req', 256.0)
        
        print(f">>> CPU requirement: {cpu_req}, Memory requirement: {memory_req}")
        
        # Simulate resource-intensive work
        start_time = time.time()
        await asyncio.sleep(0.2)
        duration = time.time() - start_time
        
        print(f">>> Setting outputs...")
        context.set_output('cpu_used', cpu_req)
        context.set_output('memory_used', memory_req)
        context.set_output('execution_time', duration)
        context.set_metric('resource_efficiency', 0.85)
        
        print(f">>> Context outputs: {context.outputs}")
        print(f">>> Returning None to indicate success")
        return None

async def debug_execution():
    print("=== Debugging State Execution in Detail ===")
    
    # Create a resource pool
    resource_pool = ResourcePool(
        total_cpu=4.0,
        total_memory=1024.0,
        total_io=100.0,
        total_network=100.0
    )
    
    # Create agent
    agent = DebugResourceIntensiveAgent('test-agent', cpu_req=1.5, memory_req=256.0)
    agent.resource_pool = resource_pool
    
    print(f"Agent created: {agent.name}")
    print(f"Agent states: {list(agent.states.keys())}")
    
    # Check state metadata
    start_state = agent.states.get('start')
    if start_state:
        print(f"Start state found: {start_state}")
        if hasattr(start_state, '__wrapped__'):
            print(f"Wrapped function: {start_state.__wrapped__}")
        if hasattr(start_state, 'resource_requirements'):
            print(f"Resource requirements: {start_state.resource_requirements}")
    
    # Override the _execute_state method to add debugging
    original_execute = agent._execute_state
        
        async def debug_execute_state(state_name, context):
            print(f"\n>>> _execute_state called with state_name: {state_name}")
            print(f">>> Context before execution: {context}")
            
            try:
                result = await original_execute(state_name, context)
                print(f">>> _execute_state result: {result}")
                return result
            except Exception as e:
                print(f">>> _execute_state exception: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        agent._execute_state = debug_execute_state
        
        # Run the agent
        print("\n>>> Starting agent execution...")
        result = await agent.run()
        
        print(f"\n>>> Final result:")
        print(f"Status: {result.status}")
        print(f"Status type: {type(result.status)}")
        print(f"Outputs: {result.outputs}")
        print(f"Metadata: {result.metadata}")
        print(f"Error: {result.error}")
        
        # Check if it's actually successful
        if result.status == AgentStatus.COMPLETED or result.status == AgentStatus.SUCCESS:
            print("\n✅ Agent completed successfully!")
        else:
            print(f"\n❌ Agent failed with status: {result.status}")
            
            # Check circuit breaker and bulkhead metrics
            if 'circuit_breaker_metrics' in result.metrics:
                cb = result.metrics['circuit_breaker_metrics']
                print(f"\nCircuit Breaker State: {cb.get('state')}")
                print(f"Failure Count: {cb.get('failure_count')}")
                print(f"Success Count: {cb.get('success_count')}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Exception during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the debug
asyncio.run(debug_execution())