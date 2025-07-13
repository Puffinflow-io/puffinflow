"""Test all examples from the getting started documentation."""

import asyncio
import json
import pytest

from puffinflow import Agent, state


@pytest.mark.asyncio
class TestGettingStartedExamples:
    """Test examples from getting-started.ts documentation."""

    async def test_first_workflow_example(self):
        """Test the first workflow example."""
        # 1. Create an agent
        agent = Agent("my-first-workflow")

        # 2. Define a state (just a regular async function)
        async def hello_world(context):
            print("Hello, Puffinflow! 🐧")
            print(f"Agent name: {agent.name}")
            context.set_variable("greeting", "Hello from PuffinFlow!")
            return None

        # 3. Add state and run it
        agent.add_state("hello_world", hello_world)

        result = await agent.run()
        assert result.get_variable("greeting") == "Hello from PuffinFlow!"

    async def test_plain_function_state(self):
        """Test plain function state definition."""
        agent = Agent("plain-function-test")

        async def process_data(context):
            context.set_variable("result", "Hello!")
            return None

        agent.add_state("process_data", process_data)
        result = await agent.run()
        assert result.get_variable("result") == "Hello!"

    async def test_decorated_state(self):
        """Test decorated state definition."""
        agent = Agent("decorated-state-test")

        @state
        async def process_data(context):
            context.set_variable("result", "Hello!")
            return None

        agent.add_state("process_data", process_data)
        result = await agent.run()
        assert result.get_variable("result") == "Hello!"

    async def test_data_sharing_between_states(self):
        """Test sharing data between states."""
        agent = Agent("data-pipeline")

        async def fetch_data(context):
            # Simulate fetching data from an API
            print("📊 Fetching user data...")

            # Store data in context
            context.set_variable("user_count", 1250)
            context.set_variable("revenue", 45000)
            print("✅ Data fetched successfully")

        async def calculate_metrics(context):
            # Get data from previous state
            users = context.get_variable("user_count")
            revenue = context.get_variable("revenue")

            # Calculate and store result
            revenue_per_user = revenue / users
            context.set_variable("revenue_per_user", revenue_per_user)

            print(f"💰 Revenue per user: ${revenue_per_user:.2f}")
            print("✅ Metrics calculated")

        async def send_report(context):
            # Use the calculated metric
            rpu = context.get_variable("revenue_per_user")

            print(f"📧 Sending report: RPU is ${rpu:.2f}")
            print("✅ Report sent!")

        # Add states to workflow with proper dependencies for sequential execution
        agent.add_state("fetch_data", fetch_data)
        agent.add_state("calculate_metrics", calculate_metrics, dependencies=["fetch_data"])
        agent.add_state("send_report", send_report, dependencies=["calculate_metrics"])

        # Run the complete pipeline
        result = await agent.run()
        
        # Verify results
        assert result.get_variable("user_count") == 1250
        assert result.get_variable("revenue") == 45000
        assert result.get_variable("revenue_per_user") == 36.0

    async def test_sequential_execution(self):
        """Test sequential execution example."""
        agent = Agent("sequential-workflow")
        execution_order = []

        async def step_one(context):
            execution_order.append("step_one")
            print("Step 1: Preparing data")
            context.set_variable("step1_done", True)

        async def step_two(context):
            execution_order.append("step_two")
            print("Step 2: Processing data")
            context.set_variable("step2_done", True)

        async def step_three(context):
            execution_order.append("step_three")
            print("Step 3: Finalizing")
            print("All steps complete!")

        # Runs in this exact order: step_one → step_two → step_three
        agent.add_state("step_one", step_one)
        agent.add_state("step_two", step_two, dependencies=["step_one"])
        agent.add_state("step_three", step_three, dependencies=["step_two"])

        result = await agent.run()
        
        # Verify execution order and results
        assert execution_order == ["step_one", "step_two", "step_three"]
        assert result.get_variable("step1_done") is True
        assert result.get_variable("step2_done") is True

    async def test_static_dependencies(self):
        """Test static dependencies example."""
        agent = Agent("dependencies-test")

        async def fetch_user_data(context):
            print("👥 Fetching user data...")
            await asyncio.sleep(0.1)  # Reduced for faster tests
            context.set_variable("user_count", 1250)

        async def fetch_sales_data(context):
            print("💰 Fetching sales data...")
            await asyncio.sleep(0.1)  # Reduced for faster tests
            context.set_variable("revenue", 45000)

        async def generate_report(context):
            print("📊 Generating report...")
            users = context.get_variable("user_count")
            revenue = context.get_variable("revenue")
            print(f"Users: {users}, Revenue: {revenue}")
            context.set_variable("report", f"Revenue per user: ${revenue/users:.2f}")
            print("Report generated and stored")

        # fetch_user_data and fetch_sales_data run in parallel
        # generate_report waits for BOTH to complete
        agent.add_state("fetch_user_data", fetch_user_data)
        agent.add_state("fetch_sales_data", fetch_sales_data)
        agent.add_state("generate_report", generate_report,
                       dependencies=["fetch_user_data", "fetch_sales_data"])

        result = await agent.run()
        
        print(f"Final variables: {result.variables}")
        print(f"Final outputs: {result.outputs}")
        
        # Verify all data is present
        assert result.get_variable("user_count") == 1250
        assert result.get_variable("revenue") == 45000
        report = result.get_variable("report")
        assert report is not None
        assert "36.00" in report

    async def test_dynamic_flow_control(self):
        """Test dynamic flow control example."""
        agent = Agent("dynamic-flow-test")

        async def check_user_type(context):
            print("🔍 Checking user type...")
            user_type = "premium"  # Could come from database
            context.set_variable("user_type", user_type)

            # Dynamic routing based on data
            if user_type == "premium":
                return "premium_flow"
            else:
                return "basic_flow"

        async def premium_flow(context):
            print("⭐ Premium user workflow")
            context.set_variable("features", ["advanced_analytics", "priority_support"])
            return "send_welcome"

        async def basic_flow(context):
            print("👋 Basic user workflow")
            context.set_variable("features", ["basic_analytics"])
            return "send_welcome"

        async def send_welcome(context):
            user_type = context.get_variable("user_type")
            features = context.get_variable("features")
            context.set_variable("welcome_message", f"Welcome {user_type} user! Features: {', '.join(features)}")

        # Add all states - only check_user_type should start as entry state
        agent.add_state("check_user_type", check_user_type)
        agent.add_state("premium_flow", premium_flow)
        agent.add_state("basic_flow", basic_flow)
        agent.add_state("send_welcome", send_welcome)

        result = await agent.run()
        
        # Verify premium flow was executed
        assert result.get_variable("user_type") == "premium"
        assert "advanced_analytics" in result.get_variable("features")
        assert "premium" in result.get_variable("welcome_message")

    async def test_parallel_execution(self):
        """Test parallel execution example."""
        agent = Agent("parallel-test")

        async def process_order(context):
            print("📦 Processing order...")
            context.set_variable("order_id", "ORD-123")

            # Run these three states in parallel
            return ["send_confirmation", "update_inventory", "charge_payment"]

        async def send_confirmation(context):
            order_id = context.get_variable("order_id")
            context.set_variable("confirmation_sent", f"Confirmation sent for {order_id}")

        async def update_inventory(context):
            context.set_variable("inventory_updated", "Inventory updated")

        async def charge_payment(context):
            order_id = context.get_variable("order_id")
            context.set_variable("payment_processed", f"Payment processed for {order_id}")

        agent.add_state("process_order", process_order)
        agent.add_state("send_confirmation", send_confirmation)
        agent.add_state("update_inventory", update_inventory)
        agent.add_state("charge_payment", charge_payment)

        result = await agent.run()
        
        # Verify all parallel operations completed
        assert result.get_variable("order_id") == "ORD-123"
        assert "ORD-123" in result.get_variable("confirmation_sent")
        assert result.get_variable("inventory_updated") == "Inventory updated"
        assert "ORD-123" in result.get_variable("payment_processed")

    async def test_complete_data_pipeline(self):
        """Test complete data pipeline example."""
        agent = Agent("data-pipeline")

        async def extract(context):
            data = {"sales": [100, 200, 150], "customers": ["Alice", "Bob", "Charlie"]}
            context.set_variable("raw_data", data)
            print("✅ Data extracted")

        async def transform(context):
            raw_data = context.get_variable("raw_data")
            total_sales = sum(raw_data["sales"])
            customer_count = len(raw_data["customers"])

            transformed = {
                "total_sales": total_sales,
                "customer_count": customer_count,
                "avg_sale": total_sales / customer_count
            }

            context.set_variable("processed_data", transformed)
            print("✅ Data transformed")

        async def load(context):
            processed_data = context.get_variable("processed_data")
            context.set_variable("load_result", f"Saved: {processed_data}")

        # Set up the pipeline - runs sequentially
        agent.add_state("extract", extract)
        agent.add_state("transform", transform, dependencies=["extract"])
        agent.add_state("load", load, dependencies=["transform"])

        result = await agent.run()
        
        # Verify pipeline results
        raw_data = result.get_variable("raw_data")
        processed_data = result.get_variable("processed_data")
        
        assert len(raw_data["sales"]) == 3
        assert len(raw_data["customers"]) == 3
        assert processed_data["total_sales"] == 450
        assert processed_data["customer_count"] == 3
        assert processed_data["avg_sale"] == 150.0

    async def test_state_decorator_with_resources(self):
        """Test state decorator with resource specifications."""
        agent = Agent("resource-test")

        @state(cpu=2.0, memory=1024, timeout=60.0)
        async def intensive_task(context):
            # This state gets 2 CPU units, 1GB memory, 60s timeout
            context.set_variable("task_completed", True)
            return None

        agent.add_state("intensive_task", intensive_task)
        result = await agent.run()
        
        assert result.get_variable("task_completed") is True

    async def test_ai_research_assistant_workflow(self):
        """Test complete AI research assistant workflow."""
        # Simulate external APIs
        async def search_web(query):
            """Simulate web search API"""
            await asyncio.sleep(0.1)  # Reduced for faster tests
            return [
                {"title": f"Article about {query}", "content": f"Detailed info on {query}..."},
                {"title": f"{query} trends", "content": f"Latest trends in {query}..."}
            ]

        async def call_llm(prompt):
            """Simulate LLM API call"""
            await asyncio.sleep(0.1)  # Reduced for faster tests
            return f"AI Analysis: {prompt[:50]}..."

        # Create the research agent
        research_agent = Agent("ai-research-assistant")

        async def validate_query(context):
            """Validate and prepare the search query"""
            query = context.get_variable("search_query", "")

            if not query or len(query) < 3:
                print("❌ Invalid query - too short")
                return None  # End workflow

            # Clean and prepare query
            clean_query = query.strip().lower()
            context.set_variable("clean_query", clean_query)

            print(f"✅ Query validated: '{clean_query}'")
            return "search_information"

        async def search_information(context):
            """Search for information on the web"""
            query = context.get_variable("clean_query")

            print(f"🔍 Searching for: {query}")
            results = await search_web(query)

            context.set_variable("search_results", results)
            print(f"✅ Found {len(results)} results")

            return "analyze_results"

        async def analyze_results(context):
            """Use LLM to analyze search results"""
            results = context.get_variable("search_results")
            query = context.get_variable("clean_query")

            print("🧠 Analyzing results with AI...")

            # Prepare prompt for LLM
            prompt = f"Analyze these search results for query '{query}': {json.dumps(results, indent=2)}"
            analysis = await call_llm(prompt)
            context.set_variable("analysis", analysis)

            print("✅ Analysis complete")
            return "generate_report"

        async def generate_report(context):
            """Generate final research report"""
            query = context.get_variable("search_query")
            analysis = context.get_variable("analysis")
            results = context.get_variable("search_results")

            print("📝 Generating final report...")

            # Create structured report
            report = {
                "query": query,
                "sources_found": len(results),
                "analysis": analysis,
                "generated_at": "2024-01-15 10:30:00",
                "confidence": "high"
            }

            context.set_variable("final_report", report)
            print("🎉 Research Report Generated!")
            return None  # End workflow

        # Wire up the workflow
        research_agent.add_state("validate_query", validate_query)
        research_agent.add_state("search_information", search_information)
        research_agent.add_state("analyze_results", analyze_results)
        research_agent.add_state("generate_report", generate_report)

        # Set initial context
        research_agent.set_variable("search_query", "machine learning trends 2024")
        
        # Run research workflow
        result = await research_agent.run()

        # Verify the workflow completed successfully
        final_report = result.get_variable("final_report")
        assert final_report is not None
        assert final_report["query"] == "machine learning trends 2024"
        assert final_report["sources_found"] == 2
        assert "AI Analysis" in final_report["analysis"]
        assert final_report["confidence"] == "high"

    async def test_invalid_query_workflow(self):
        """Test AI research assistant with invalid query."""
        research_agent = Agent("ai-research-assistant-invalid")

        async def validate_query(context):
            """Validate and prepare the search query"""
            query = context.get_variable("search_query", "")

            if not query or len(query) < 3:
                context.set_variable("error", "Invalid query - too short")
                return None  # End workflow

            clean_query = query.strip().lower()
            context.set_variable("clean_query", clean_query)
            return "search_information"

        research_agent.add_state("validate_query", validate_query)

        # Set initial context with invalid query
        research_agent.set_variable("search_query", "ai")  # Too short
        
        # Test with invalid query
        result = await research_agent.run()

        assert result.get_variable("error") == "Invalid query - too short"
        assert result.get_variable("clean_query") is None