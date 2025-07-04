import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Test the failing test directly
import asyncio
from puffinflow.core.resources.pool import ResourcePool

# Import the test classes
exec(open('tests/integration/test_resource_reliability.py').read())

async def debug_test():
    print('=== Debugging test_resource_pool_allocation ===')
    
    # Create a resource pool
    resource_pool = ResourcePool(
        total_cpu=4.0,
        total_memory=1024.0,
        total_io=100.0,
        total_network=100.0
    )
    
    # Test single agent execution using the fixed ResourceIntensiveAgent
    agent = ResourceIntensiveAgent('test-agent', cpu_req=1.5, memory_req=256.0)
    agent.resource_pool = resource_pool
    
    print(f'Agent created: {agent.name}')
    print(f'Agent initial_state: {agent.initial_state}')
    print(f'Agent states: {list(agent.states.keys())}')
    
    # Check if the state has requirements
    start_state = agent.states.get('start')
    if start_state:
        print(f'Start state exists: {start_state}')
        if hasattr(start_state, '_state_requirements'):
            print(f'State requirements: {start_state._state_requirements}')
        if hasattr(start_state, '__wrapped__'):
            print(f'Wrapped function: {start_state.__wrapped__}')
            if hasattr(start_state.__wrapped__, '_state_requirements'):
                print(f'Wrapped requirements: {start_state.__wrapped__._state_requirements}')
    
    # Run the agent
    try:
        result = await agent.run()
        
        # Check the result
        print(f'\nResult: {result}')
        print(f'Result status: {result.status}')
        print(f'Result status type: {type(result.status)}')
        print(f'Result outputs: {result.outputs}')
        print(f'Result errors: {getattr(result, "errors", None)}')
        
        status = result.status.name if hasattr(result.status, 'name') else str(result.status)
        print(f'Normalized status: {status}')
        
        if hasattr(result, 'error'):
            print(f'Error details: {result.error}')
            
        # Check context history
        if hasattr(result, 'context'):
            print(f'\nContext history:')
            if hasattr(result.context, '_history'):
                for entry in result.context._history:
                    print(f'  {entry}')
            
    except Exception as e:
        print(f'Exception during run: {e}')
        import traceback
        traceback.print_exc()

# Run the debug
asyncio.run(debug_test())