"""Test all examples from the getting started documentation."""

import asyncio
import json

import pytest

from puffinflow import Agent, ExecutionMode, state


@pytest.mark.asyncio
class TestGettingStartedExamples:
    """Test examples from getting-started.ts documentation."""

    async def test_first_workflow_example(self):
        """Test the first workflow example from documentation."""
        # Create an agent
        agent = Agent("data-processor")

        @state
        async def fetch_data(context):
            """Step 1: Get some data"""
            data = {"users": ["Alice", "Bob", "Charlie"]}
            context.set_variable("raw_data", data)
            return "process_data"

        @state
        async def process_data(context):
            """Step 2: Transform the data"""
            raw_data = context.get_variable("raw_data")
            processed = [f"Hello, {user}!" for user in raw_data["users"]]
            context.set_variable("greetings", processed)
            return "save_results"

        @state
        async def save_results(context):
            """Step 3: Output results"""
            greetings = context.get_variable("greetings")
            # Store results for testing instead of printing
            context.set_variable("final_results", greetings)
            # Return None to end the workflow
            return None

        # Add states to agent
        agent.add_state("fetch_data", fetch_data)
        agent.add_state("process_data", process_data)
        agent.add_state("save_results", save_results)

        # Run the workflow
        result = await agent.run()

        # Verify the workflow worked correctly
        assert result.get_variable("final_results") == [
            "Hello, Alice!",
            "Hello, Bob!",
            "Hello, Charlie!",
        ]

    async def test_plain_function_state(self):
        """Test alternative without decorators from documentation."""

        async def my_function(context):
            context.set_variable("message", "Hello from Puffinflow!")
            return None

        agent = Agent("simple-workflow")
        agent.add_state("hello", my_function)

        result = await agent.run()
        assert result.get_variable("message") == "Hello from Puffinflow!"

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
        agent.add_state(
            "calculate_metrics", calculate_metrics, dependencies=["fetch_data"]
        )
        agent.add_state("send_report", send_report, dependencies=["calculate_metrics"])

        # Run the complete pipeline
        result = await agent.run()

        # Verify results
        assert result.get_variable("user_count") == 1250
        assert result.get_variable("revenue") == 45000
        assert result.get_variable("revenue_per_user") == 36.0

    async def test_sequential_execution(self):
        """Test sequential execution example from documentation."""
        agent = Agent("sequential-workflow")

        @state
        async def step_one(context):
            context.set_variable("step", 1)
            return "step_two"  # Explicitly control next step

        @state
        async def step_two(context):
            context.set_variable("step", 2)
            # End workflow

        agent.add_state("step_one", step_one)
        agent.add_state("step_two", step_two)

        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        # Verify the workflow worked correctly
        assert result.get_variable("step") == 2

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
        agent.add_state(
            "generate_report",
            generate_report,
            dependencies=["fetch_user_data", "fetch_sales_data"],
        )

        result = await agent.run(execution_mode=ExecutionMode.PARALLEL)

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
            context.set_variable(
                "welcome_message",
                f"Welcome {user_type} user! Features: {', '.join(features)}",
            )

        # Add only the entry state initially, then add conditional states
        # The framework should only execute states that are entry points or called via return values
        agent.add_state("check_user_type", check_user_type)
        agent.add_state("premium_flow", premium_flow)
        agent.add_state("basic_flow", basic_flow)
        agent.add_state("send_welcome", send_welcome)

        # Run with SEQUENTIAL mode to ensure proper flow control
        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        # Verify the premium flow was executed correctly
        assert result.get_variable("user_type") == "premium"
        features = result.get_variable("features")
        welcome_message = result.get_variable("welcome_message")

        # With sequential mode, only the premium flow should execute
        assert features == ["advanced_analytics", "priority_support"]
        assert welcome_message is not None
        assert "premium" in welcome_message

    async def test_parallel_execution(self):
        """Test parallel execution example from documentation."""
        agent = Agent("parallel-workflow")

        @state
        async def fetch_users(context):
            # This runs in parallel with fetch_orders
            context.set_variable("users", ["Alice", "Bob"])

        @state
        async def fetch_orders(context):
            # This runs in parallel with fetch_users
            context.set_variable("orders", [{"id": 1}, {"id": 2}])

        @state
        async def generate_report(context):
            # Waits for both parallel states to complete
            users = context.get_variable("users")
            orders = context.get_variable("orders")
            context.set_variable(
                "report", f"Report: {len(users)} users, {len(orders)} orders"
            )

        agent.add_state("fetch_users", fetch_users)
        agent.add_state("fetch_orders", fetch_orders)
        agent.add_state(
            "generate_report",
            generate_report,
            dependencies=["fetch_users", "fetch_orders"],
        )

        result = await agent.run(execution_mode=ExecutionMode.PARALLEL)

        # Verify all parallel operations completed
        assert result.get_variable("users") == ["Alice", "Bob"]
        assert result.get_variable("orders") == [{"id": 1}, {"id": 2}]
        assert result.get_variable("report") == "Report: 2 users, 2 orders"

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
                "avg_sale": total_sales / customer_count,
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
                {
                    "title": f"Article about {query}",
                    "content": f"Detailed info on {query}...",
                },
                {"title": f"{query} trends", "content": f"Latest trends in {query}..."},
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
                "confidence": "high",
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

    async def test_production_document_processor_workflow(self):
        """Test production-ready document processing workflow with error handling."""
        from puffinflow import Priority, state

        # Create production agent
        processor = Agent("document-processor")

        @state(
            cpu=2.0, memory=1024, priority=Priority.HIGH, max_retries=3, timeout=120.0
        )
        async def validate_document(context):
            """Validate uploaded document format and size."""
            try:
                file_path = context.get_variable("file_path")
                file_size = 5 * 1024 * 1024  # Simulate 5MB file

                # Validate file size (max 10MB)
                if file_size > 10 * 1024 * 1024:
                    context.set_variable("error", "File too large")
                    return "error_handler"

                # Validate file format
                if not file_path.lower().endswith((".pdf", ".docx", ".txt")):
                    context.set_variable("error", "Unsupported file format")
                    return "error_handler"

                context.set_variable("file_size", file_size)
                return "extract_content"

            except Exception as e:
                context.set_variable("error", str(e))
                return "error_handler"

        @state(
            cpu=4.0, memory=512, priority=Priority.NORMAL, max_retries=2, timeout=300.0
        )
        async def extract_content(context):
            """Extract text content from document."""
            try:
                file_path = context.get_variable("file_path")

                # Simulate content extraction
                await asyncio.sleep(0.1)  # Reduced for tests

                content = f"Extracted content from {file_path}"
                word_count = len(content.split())

                context.set_variable("content", content)
                context.set_variable("word_count", word_count)

                return "analyze_content"

            except Exception as e:
                context.set_variable("error", str(e))
                return "error_handler"

        @state(
            cpu=2.0, memory=1024, priority=Priority.NORMAL, max_retries=1, timeout=180.0
        )
        async def analyze_content(context):
            """Analyze content with AI/ML processing."""
            try:
                context.get_variable("content")
                word_count = context.get_variable("word_count")

                # Simulate AI analysis
                await asyncio.sleep(0.1)  # Reduced for tests

                analysis = {
                    "sentiment": "positive",
                    "topics": ["technology", "business"],
                    "summary": f"Document contains {word_count} words about technology and business.",
                    "confidence": 0.95,
                }

                context.set_variable("analysis", analysis)
                return "save_results"

            except Exception as e:
                context.set_variable("error", str(e))
                return "error_handler"

        @state(
            cpu=1.0, memory=512, priority=Priority.NORMAL, max_retries=2, timeout=60.0
        )
        async def save_results(context):
            """Save processing results to database."""
            try:
                analysis = context.get_variable("analysis")
                file_path = context.get_variable("file_path")

                # Simulate database save
                await asyncio.sleep(0.1)  # Reduced for tests

                result_id = f"doc_{hash(file_path) % 10000}"  # Simplified for tests
                results = {
                    "id": result_id,
                    "file_path": file_path,
                    "analysis": analysis,
                    "processed_at": "2024-01-15T10:30:00Z",
                }

                context.set_variable("results", results)
                return "send_notification"

            except Exception as e:
                context.set_variable("error", str(e))
                return "error_handler"

        @state(cpu=0.5, memory=256, priority=Priority.LOW, max_retries=3, timeout=30.0)
        async def send_notification(context):
            """Send completion notification."""
            try:
                results = context.get_variable("results")

                # Simulate notification
                await asyncio.sleep(0.1)  # Reduced for tests

                notification = {
                    "type": "success",
                    "message": f"Document {results['id']} processed successfully",
                    "timestamp": "2024-01-15T10:35:00Z",
                }

                context.set_variable("notification", notification)
                return None  # End workflow

            except Exception as e:
                context.set_variable("error", str(e))
                return "error_handler"

        @state(cpu=0.5, memory=256, priority=Priority.HIGH, max_retries=1, timeout=30.0)
        async def error_handler(context):
            """Handle errors and cleanup."""
            try:
                error = context.get_variable("error")
                file_path = context.get_variable("file_path", "unknown")

                # Send error notification
                error_notification = {
                    "type": "error",
                    "message": f"Document processing failed: {error}",
                    "file_path": file_path,
                    "timestamp": "2024-01-15T10:30:00Z",
                }

                context.set_variable("error_notification", error_notification)
                return None  # End workflow

            except Exception as e:
                context.set_variable("critical_error", str(e))
                return None

        # Add all states to processor
        processor.add_state("validate_document", validate_document)
        processor.add_state("extract_content", extract_content)
        processor.add_state("analyze_content", analyze_content)
        processor.add_state("save_results", save_results)
        processor.add_state("send_notification", send_notification)
        processor.add_state("error_handler", error_handler)

        # Test successful document processing
        processor.set_variable("file_path", "/path/to/document.pdf")

        result = await processor.run()

        # Verify successful processing
        results = result.get_variable("results")
        notification = result.get_variable("notification")

        assert results is not None
        assert results["file_path"] == "/path/to/document.pdf"
        assert results["analysis"]["sentiment"] == "positive"
        assert notification["type"] == "success"
        assert "processed successfully" in notification["message"]

    async def test_production_document_processor_error_handling(self):
        """Test production document processor with error scenarios."""
        from puffinflow import Priority, state

        processor = Agent("document-processor-error-test")

        @state(priority=Priority.HIGH, max_retries=3, timeout=120.0)
        async def validate_document(context):
            """Validate uploaded document - test error case."""
            file_path = context.get_variable("file_path")

            # Test unsupported file format
            if not file_path.lower().endswith((".pdf", ".docx", ".txt")):
                context.set_variable("error", "Unsupported file format")
                return "error_handler"

            return "extract_content"

        @state(priority=Priority.HIGH, max_retries=1, timeout=30.0)
        async def error_handler(context):
            """Handle errors and cleanup."""
            error = context.get_variable("error")
            file_path = context.get_variable("file_path", "unknown")

            error_notification = {
                "type": "error",
                "message": f"Document processing failed: {error}",
                "file_path": file_path,
            }

            context.set_variable("error_notification", error_notification)
            return None

        processor.add_state("validate_document", validate_document)
        processor.add_state("error_handler", error_handler)

        # Test with unsupported file format
        processor.set_variable("file_path", "/path/to/document.xyz")

        result = await processor.run()

        # Verify error handling
        error_notification = result.get_variable("error_notification")
        assert error_notification is not None
        assert error_notification["type"] == "error"
        assert "Unsupported file format" in error_notification["message"]
