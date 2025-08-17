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
            query = context.get_variable("search_query")
            # Simulate LLM analysis
            analysis = f"Analysis of {len(results)} articles about {query}"
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
        assert raw_results[0]["title"] == "Article about latest AI trends"

        analysis = result.get_variable("analysis")
        assert analysis == "Analysis of 1 articles about latest AI trends"

    async def test_enhanced_research_assistant_workflow(self):
        """Test the enhanced research assistant workflow from introduction."""
        import asyncio

        agent = Agent("research-assistant")

        @state(cpu=1.0, memory=512, timeout=30.0, max_retries=2)
        async def gather_info(context):
            """Search for information on the web."""
            query = context.get_variable("search_query")

            # Simulate web search API call
            await asyncio.sleep(0.01)  # Reduced for faster tests
            results = [
                {
                    "title": f"Article about {query}",
                    "content": f"Detailed content about {query}...",
                },
                {
                    "title": f"{query} - Latest Research",
                    "content": f"Recent findings on {query}...",
                },
                {
                    "title": f"Industry Analysis: {query}",
                    "content": f"Market analysis of {query}...",
                },
            ]

            # Store results in context for next state
            context.set_variable("raw_results", results)
            context.set_variable("search_count", len(results))

            return "analyze_results"

        @state(cpu=2.0, memory=1024, timeout=60.0, rate_limit=5)  # Rate limit LLM calls
        async def analyze_results(context):
            """Use LLM to analyze the gathered information."""
            results = context.get_variable("raw_results")
            query = context.get_variable("search_query")

            # Simulate LLM API call (GPT-4, Claude, etc.)
            await asyncio.sleep(0.01)  # Reduced for faster tests

            # Create comprehensive analysis
            topics = [f"topic_{i+1}" for i in range(len(results))]
            analysis = {
                "summary": f"Comprehensive analysis of {len(results)} articles about {query}",
                "key_topics": topics,
                "sentiment": "neutral",
                "confidence": 0.92,
                "word_count": sum(len(r["content"]) for r in results),
                "sources_analyzed": len(results),
            }

            # Store analysis results
            context.set_variable("analysis", analysis)

            return "generate_report"

        @state(cpu=1.0, memory=512, timeout=45.0, max_retries=1)
        async def generate_report(context):
            """Generate the final research report."""
            query = context.get_variable("search_query")
            analysis = context.get_variable("analysis")
            raw_results = context.get_variable("raw_results")

            # Create structured report
            report = {
                "title": f"Research Report: {query.title()}",
                "query": query,
                "executive_summary": analysis["summary"],
                "key_findings": analysis["key_topics"],
                "sentiment_analysis": analysis["sentiment"],
                "confidence_score": analysis["confidence"],
                "methodology": {
                    "sources_searched": len(raw_results),
                    "sources_analyzed": analysis["sources_analyzed"],
                    "analysis_method": "LLM-powered content analysis",
                },
                "metadata": {
                    "generated_at": "2024-01-15T10:30:00Z",
                    "agent_id": agent.name,
                    "word_count": analysis["word_count"],
                },
            }

            # Store final report
            context.set_variable("final_report", report)
            context.set_output("research_report", report)  # Mark as workflow output

            return None  # End of workflow

        # Wire up the workflow
        agent.add_state("gather_info", gather_info)
        agent.add_state("analyze_results", analyze_results)
        agent.add_state("generate_report", generate_report)

        # Run the enhanced workflow
        result = await agent.run(
            initial_context={"search_query": "machine learning trends 2024"}
        )

        # Verify the workflow completed successfully
        final_report = result.get_variable("final_report")
        research_report = result.get_output("research_report")

        assert final_report is not None
        assert research_report is not None
        assert final_report == research_report  # Should be the same object

        # Verify report structure
        assert (
            research_report["title"] == "Research Report: Machine Learning Trends 2024"
        )
        assert research_report["query"] == "machine learning trends 2024"
        assert research_report["confidence_score"] == 0.92
        assert research_report["sentiment_analysis"] == "neutral"
        assert len(research_report["key_findings"]) == 3

        # Verify methodology
        methodology = research_report["methodology"]
        assert methodology["sources_searched"] == 3
        assert methodology["sources_analyzed"] == 3
        assert methodology["analysis_method"] == "LLM-powered content analysis"

        # Verify metadata
        metadata = research_report["metadata"]
        assert metadata["agent_id"] == "research-assistant"
        assert "word_count" in metadata

        # Verify intermediate results
        raw_results = result.get_variable("raw_results")
        assert len(raw_results) == 3
        assert "machine learning trends 2024" in raw_results[0]["title"]

        analysis = result.get_variable("analysis")
        assert analysis["confidence"] == 0.92
        assert analysis["sources_analyzed"] == 3
        assert len(analysis["key_topics"]) == 3

    async def test_concurrent_research_workflows(self):
        """Test running multiple research workflows concurrently."""
        import asyncio

        async def create_research_agent(agent_name: str):
            """Create a research agent with basic workflow."""
            agent = Agent(agent_name)

            @state(cpu=1.0, memory=512)
            async def gather_info(context):
                query = context.get_variable("search_query")
                await asyncio.sleep(0.01)  # Simulate network delay
                results = [{"title": f"Article about {query}", "content": "..."}]
                context.set_variable("raw_results", results)
                return "analyze_results"

            @state(cpu=2.0, memory=1024)
            async def analyze_results(context):
                results = context.get_variable("raw_results")
                query = context.get_variable("search_query")
                await asyncio.sleep(0.01)  # Simulate processing
                analysis = f"Analysis of {len(results)} articles about {query}"
                context.set_variable("analysis", analysis)
                return "generate_report"

            @state(cpu=1.0, memory=512)
            async def generate_report(context):
                analysis = context.get_variable("analysis")
                report = f"Report: {analysis}"
                context.set_variable("final_report", report)
                context.set_output("research_report", report)
                return None

            agent.add_state("gather_info", gather_info)
            agent.add_state(
                "analyze_results", analyze_results, dependencies=["gather_info"]
            )
            agent.add_state(
                "generate_report", generate_report, dependencies=["analyze_results"]
            )

            return agent

        async def run_research(query: str, agent_id: str):
            """Run a research workflow."""
            agent = await create_research_agent(f"research-{agent_id}")
            result = await agent.run(initial_context={"search_query": query})
            return result.get_output("research_report")

        # Run multiple research queries concurrently
        queries = [
            "machine learning trends 2024",
            "sustainable energy solutions",
            "remote work productivity tools",
        ]

        tasks = [run_research(query, str(i)) for i, query in enumerate(queries)]
        reports = await asyncio.gather(*tasks)

        # Verify all reports completed successfully
        assert len(reports) == 3
        for i, report in enumerate(reports):
            assert report is not None
            # The report format is "Report: Analysis of 1 articles about {query}"
            assert queries[i] in report
            assert "Analysis of 1 articles" in report
