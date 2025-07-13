"""Test all examples from the introduction documentation."""

import pytest

from puffinflow import Agent, state


@pytest.mark.asyncio
class TestIntroductionExamples:
    """Test examples from introduction.ts documentation."""

    async def test_research_assistant_workflow(self):
        """Test the research assistant workflow from introduction."""
        agent = Agent("research-assistant")

        @state(cpu=1.0, memory=512)
        async def gather_info(context):
            query = context.get_variable("search_query")
            # Simulate web search
            results = [{"title": f"Article about {query}", "content": "..."}]
            context.set_variable("raw_results", results)
            return "analyze_results"

        @state(cpu=2.0, memory=1024)
        async def analyze_results(context):
            results = context.get_variable("raw_results")
            # Simulate LLM analysis
            analysis = f"Analysis of {len(results)} articles about the query"
            context.set_variable("analysis", analysis)
            return "generate_report"

        @state(cpu=1.0, memory=512)
        async def generate_report(context):
            analysis = context.get_variable("analysis")
            # Generate final report
            report = f"Report: {analysis}"
            context.set_variable("final_report", report)
            return None  # End of workflow

        # Add states to agent
        agent.add_state("gather_info", gather_info)
        agent.add_state("analyze_results", analyze_results)
        agent.add_state("generate_report", generate_report)

        # Run it with initial context
        result = await agent.run(initial_context={"search_query": "latest AI trends"})
        
        # Verify the workflow completed successfully
        final_report = result.get_variable("final_report")
        assert final_report is not None
        assert "latest AI trends" in final_report
        assert "Analysis of 1 articles" in final_report
        
        # Verify intermediate results
        raw_results = result.get_variable("raw_results")
        assert len(raw_results) == 1
        assert "Article about latest AI trends" == raw_results[0]["title"]
        
        analysis = result.get_variable("analysis")
        assert "Analysis of 1 articles about the query" == analysis