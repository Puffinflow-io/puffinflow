"""Test examples from the multi-agent documentation."""

import pytest

from puffinflow import Agent, state
from puffinflow.core.coordination import AgentTeam


@pytest.mark.asyncio
class TestMultiagentExamples:
    """Test examples from multiagent.ts documentation."""

    async def test_agent_team_creation_and_run(self):
        """Test creating an AgentTeam with two agents and running them."""
        # Create two agents with different tasks
        agent_a = Agent("data-fetcher")

        @state
        async def fetch(context):
            context.set_variable("data", [1, 2, 3])
            return None

        agent_a.add_state("fetch", fetch)

        agent_b = Agent("data-analyzer")

        @state
        async def analyze(context):
            context.set_variable("analysis", "complete")
            return None

        agent_b.add_state("analyze", analyze)

        # Create team and add agents
        team = AgentTeam("my-team")
        team.add_agent(agent_a)
        team.add_agent(agent_b)

        # Run the team in parallel
        result = await team.run(mode="parallel")

        assert result.status == "completed"
        assert result.team_name == "my-team"

        # Check individual agent results
        fetcher_result = result.get_agent_result("data-fetcher")
        assert fetcher_result is not None
        assert fetcher_result.get_variable("data") == [1, 2, 3]

        analyzer_result = result.get_agent_result("data-analyzer")
        assert analyzer_result is not None
        assert analyzer_result.get_variable("analysis") == "complete"

    async def test_two_agent_workflow(self):
        """Test two agents processing different data in a team."""
        # Agent 1: processes user data
        user_agent = Agent("user-processor")

        async def process_users(context):
            users = ["Alice", "Bob", "Charlie"]
            context.set_variable("user_count", len(users))
            context.set_variable("users", users)

        user_agent.add_state("process_users", process_users)

        # Agent 2: processes order data
        order_agent = Agent("order-processor")

        async def process_orders(context):
            orders = [{"id": 1, "total": 50}, {"id": 2, "total": 75}]
            context.set_variable("order_count", len(orders))
            context.set_variable("total_revenue", sum(o["total"] for o in orders))

        order_agent.add_state("process_orders", process_orders)

        # Run both agents in a team
        team = AgentTeam("data-pipeline")
        team.add_agent(user_agent)
        team.add_agent(order_agent)

        result = await team.run(mode="parallel")

        assert result.status == "completed"

        # Collect results from all agents
        user_counts = result.get_all_variables("user_count")
        assert 3 in user_counts

        order_counts = result.get_all_variables("order_count")
        assert 2 in order_counts

        revenues = result.get_all_variables("total_revenue")
        assert 125 in revenues

    async def test_team_sequential_execution(self):
        """Test running team agents sequentially."""
        agent_a = Agent("step-one")

        async def step_one_fn(context):
            context.set_variable("step", 1)

        agent_a.add_state("step_one_fn", step_one_fn)

        agent_b = Agent("step-two")

        async def step_two_fn(context):
            context.set_variable("step", 2)

        agent_b.add_state("step_two_fn", step_two_fn)

        team = AgentTeam("sequential-team")
        team.add_agent(agent_a)
        team.add_agent(agent_b)

        result = await team.run(mode="sequential")

        assert result.status == "completed"
        assert len(result.agent_results) == 2
