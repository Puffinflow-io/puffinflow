Examples
========

This section provides practical examples of using PuffinFlow for real-world scenarios.

Data Processing Pipeline
------------------------

A complete ETL (Extract, Transform, Load) pipeline with error handling and monitoring:

.. code-block:: python

   import asyncio
   import aiofiles
   import aiohttp
   from puffinflow import Agent, Context, state, memory_intensive, io_intensive

   class ETLPipeline(Agent):
       """Complete ETL pipeline with PuffinFlow."""
       
       def __init__(self, source_url: str, output_file: str):
           super().__init__(enable_checkpointing=True)
           self.source_url = source_url
           self.output_file = output_file

       @state
       @io_intensive(network_bandwidth_mbps=100)
       async def extract_data(self, ctx: Context) -> None:
           """Extract data from external API."""
           async with aiohttp.ClientSession() as session:
               async with session.get(self.source_url) as response:
                   if response.status == 200:
                       ctx.raw_data = await response.json()
                       ctx.extraction_timestamp = datetime.utcnow()
                       print(f"Extracted {len(ctx.raw_data)} records")
                   else:
                       raise Exception(f"Failed to extract data: {response.status}")

       @state(depends_on=["extract_data"])
       @memory_intensive(memory_mb=2048)
       async def transform_data(self, ctx: Context) -> None:
           """Transform the extracted data."""
           transformed_records = []
           
           for record in ctx.raw_data:
               # Data cleaning and transformation
               cleaned_record = {
                   'id': record.get('id'),
                   'name': record.get('name', '').strip().title(),
                   'email': record.get('email', '').lower(),
                   'created_at': record.get('created_at'),
                   'processed_at': ctx.extraction_timestamp.isoformat()
               }
               
               # Data validation
               if cleaned_record['id'] and cleaned_record['email']:
                   transformed_records.append(cleaned_record)
           
           ctx.transformed_data = transformed_records
           print(f"Transformed {len(transformed_records)} valid records")

       @state(depends_on=["transform_data"])
       @io_intensive(disk_io_mbps=50)
       async def load_data(self, ctx: Context) -> None:
           """Load transformed data to file."""
           async with aiofiles.open(self.output_file, 'w') as f:
               import json
               await f.write(json.dumps(ctx.transformed_data, indent=2))
           
           ctx.records_loaded = len(ctx.transformed_data)
           print(f"Loaded {ctx.records_loaded} records to {self.output_file}")

   # Usage
   async def run_etl():
       pipeline = ETLPipeline(
           source_url="https://api.example.com/users",
           output_file="processed_users.json"
       )
       
       result = await pipeline.run()
       print(f"ETL Pipeline completed: {result.status}")

   asyncio.run(run_etl())

Web Scraping with Rate Limiting
--------------------------------

A web scraper that respects rate limits and handles failures gracefully:

.. code-block:: python

   import asyncio
   import aiohttp
   from puffinflow import Agent, Context, state, AgentPool
   from puffinflow.core.coordination import RateLimiter
   from puffinflow.core.reliability import CircuitBreaker, CircuitBreakerConfig

   class WebScraperAgent(Agent):
       """Web scraper with rate limiting and circuit breaker."""
       
       def __init__(self, base_url: str):
           super().__init__()
           self.base_url = base_url
           
           # Circuit breaker for external requests
           self.circuit_breaker = CircuitBreaker(
               CircuitBreakerConfig(
                   failure_threshold=3,
                   recovery_timeout=30,
                   expected_exception=aiohttp.ClientError
               )
           )

       @state
       async def scrape_page(self, ctx: Context) -> None:
           """Scrape a single page with protection."""
           url = f"{self.base_url}/{ctx.page_id}"
           
           try:
               async with self.circuit_breaker:
                   async with aiohttp.ClientSession() as session:
                       async with session.get(url, timeout=10) as response:
                           if response.status == 200:
                               content = await response.text()
                               ctx.scraped_content = self.parse_content(content)
                               ctx.success = True
                           else:
                               raise aiohttp.ClientResponseError(
                                   request_info=response.request_info,
                                   history=response.history,
                                   status=response.status
                               )
           except Exception as e:
               ctx.error = str(e)
               ctx.success = False
               print(f"Failed to scrape {url}: {e}")

       def parse_content(self, html_content: str) -> dict:
           """Parse HTML content (simplified)."""
           # In real implementation, use BeautifulSoup or similar
           return {
               'title': 'Extracted Title',
               'content_length': len(html_content),
               'links_found': html_content.count('<a href=')
           }

   async def run_web_scraping():
       """Run web scraping with rate limiting."""
       # Create rate limiter (10 requests per minute)
       rate_limiter = RateLimiter(max_calls=10, time_window=60)
       
       # Create agent pool with rate limiting
       pool = AgentPool(
           agent_class=WebScraperAgent,
           pool_size=5,
           rate_limiter=rate_limiter,
           agent_kwargs={'base_url': 'https://example.com/pages'}
       )
       
       # Create work items
       page_ids = [f"page_{i}" for i in range(1, 21)]
       contexts = [Context({'page_id': page_id}) for page_id in page_ids]
       
       # Process all pages
       results = await pool.process_contexts(contexts)
       
       # Analyze results
       successful = sum(1 for r in results if r.context.get('success', False))
       print(f"Successfully scraped {successful}/{len(results)} pages")

   asyncio.run(run_web_scraping())

Machine Learning Pipeline
--------------------------

A machine learning training pipeline with resource management:

.. code-block:: python

   import asyncio
   import numpy as np
   from puffinflow import Agent, Context, state, cpu_intensive, memory_intensive, gpu_accelerated

   class MLTrainingPipeline(Agent):
       """Machine learning training pipeline."""
       
       def __init__(self, model_config: dict):
           super().__init__(enable_checkpointing=True)
           self.model_config = model_config

       @state
       @io_intensive(disk_io_mbps=100)
       async def load_dataset(self, ctx: Context) -> None:
           """Load and prepare dataset."""
           # Simulate loading large dataset
           await asyncio.sleep(2)  # Simulate I/O time
           
           # Generate synthetic data for example
           ctx.X_train = np.random.randn(10000, 100)
           ctx.y_train = np.random.randint(0, 2, 10000)
           ctx.X_test = np.random.randn(2000, 100)
           ctx.y_test = np.random.randint(0, 2, 2000)
           
           print(f"Loaded dataset: {ctx.X_train.shape[0]} training samples")

       @state(depends_on=["load_dataset"])
       @memory_intensive(memory_mb=4096)
       async def preprocess_data(self, ctx: Context) -> None:
           """Preprocess the dataset."""
           # Feature scaling
           from sklearn.preprocessing import StandardScaler
           
           scaler = StandardScaler()
           ctx.X_train_scaled = scaler.fit_transform(ctx.X_train)
           ctx.X_test_scaled = scaler.transform(ctx.X_test)
           ctx.scaler = scaler
           
           print("Data preprocessing completed")

       @state(depends_on=["preprocess_data"])
       @gpu_accelerated(gpu_memory_mb=2048, cuda_cores=1024)
       async def train_model(self, ctx: Context) -> None:
           """Train the machine learning model."""
           from sklearn.ensemble import RandomForestClassifier
           
           # Create and train model
           model = RandomForestClassifier(**self.model_config)
           
           # Simulate training time
           await asyncio.sleep(5)
           model.fit(ctx.X_train_scaled, ctx.y_train)
           
           ctx.trained_model = model
           print("Model training completed")

       @state(depends_on=["train_model"])
       @cpu_intensive(cores=4)
       async def evaluate_model(self, ctx: Context) -> None:
           """Evaluate model performance."""
           from sklearn.metrics import accuracy_score, classification_report
           
           # Make predictions
           y_pred = ctx.trained_model.predict(ctx.X_test_scaled)
           
           # Calculate metrics
           accuracy = accuracy_score(ctx.y_test, y_pred)
           report = classification_report(ctx.y_test, y_pred)
           
           ctx.accuracy = accuracy
           ctx.classification_report = report
           
           print(f"Model accuracy: {accuracy:.4f}")

       @state(depends_on=["evaluate_model"])
       @io_intensive(disk_io_mbps=50)
       async def save_model(self, ctx: Context) -> None:
           """Save the trained model."""
           import joblib
           
           # Save model and scaler
           model_path = f"model_{ctx.accuracy:.4f}.joblib"
           scaler_path = f"scaler_{ctx.accuracy:.4f}.joblib"
           
           await asyncio.sleep(1)  # Simulate save time
           # joblib.dump(ctx.trained_model, model_path)
           # joblib.dump(ctx.scaler, scaler_path)
           
           ctx.model_path = model_path
           ctx.scaler_path = scaler_path
           
           print(f"Model saved to {model_path}")

   # Usage
   async def run_ml_pipeline():
       config = {
           'n_estimators': 100,
           'max_depth': 10,
           'random_state': 42
       }
       
       pipeline = MLTrainingPipeline(config)
       result = await pipeline.run()
       
       print(f"ML Pipeline completed: {result.status}")
       print(f"Final accuracy: {result.context.accuracy:.4f}")

   asyncio.run(run_ml_pipeline())

Microservices Orchestration
----------------------------

Orchestrate multiple microservices with fault tolerance:

.. code-block:: python

   import asyncio
   import aiohttp
   from puffinflow import Agent, Context, state, AgentTeam
   from puffinflow.core.reliability import CircuitBreaker, CircuitBreakerConfig

   class MicroserviceAgent(Agent):
       """Base agent for microservice calls."""
       
       def __init__(self, service_name: str, base_url: str):
           super().__init__()
           self.service_name = service_name
           self.base_url = base_url
           
           # Circuit breaker per service
           self.circuit_breaker = CircuitBreaker(
               CircuitBreakerConfig(
                   failure_threshold=3,
                   recovery_timeout=30,
                   expected_exception=aiohttp.ClientError
               )
           )

       async def call_service(self, endpoint: str, data: dict = None) -> dict:
           """Make a call to the microservice."""
           url = f"{self.base_url}/{endpoint}"
           
           async with self.circuit_breaker:
               async with aiohttp.ClientSession() as session:
                   if data:
                       async with session.post(url, json=data) as response:
                           return await response.json()
                   else:
                       async with session.get(url) as response:
                           return await response.json()

   class UserServiceAgent(MicroserviceAgent):
       """Agent for user service operations."""
       
       def __init__(self):
           super().__init__("user-service", "http://user-service:8080")

       @state
       async def get_user_profile(self, ctx: Context) -> None:
           """Get user profile from user service."""
           try:
               user_data = await self.call_service(f"users/{ctx.user_id}")
               ctx.user_profile = user_data
               ctx.user_service_success = True
           except Exception as e:
               ctx.user_service_error = str(e)
               ctx.user_service_success = False

   class OrderServiceAgent(MicroserviceAgent):
       """Agent for order service operations."""
       
       def __init__(self):
           super().__init__("order-service", "http://order-service:8080")

       @state
       async def get_user_orders(self, ctx: Context) -> None:
           """Get user orders from order service."""
           if not ctx.get('user_service_success', False):
               ctx.orders = []
               return
           
           try:
               orders_data = await self.call_service(f"orders/user/{ctx.user_id}")
               ctx.orders = orders_data
               ctx.order_service_success = True
           except Exception as e:
               ctx.order_service_error = str(e)
               ctx.order_service_success = False
               ctx.orders = []

   class PaymentServiceAgent(MicroserviceAgent):
       """Agent for payment service operations."""
       
       def __init__(self):
           super().__init__("payment-service", "http://payment-service:8080")

       @state
       async def get_payment_methods(self, ctx: Context) -> None:
           """Get user payment methods."""
           if not ctx.get('user_service_success', False):
               ctx.payment_methods = []
               return
           
           try:
               payment_data = await self.call_service(f"payments/user/{ctx.user_id}")
               ctx.payment_methods = payment_data
               ctx.payment_service_success = True
           except Exception as e:
               ctx.payment_service_error = str(e)
               ctx.payment_service_success = False
               ctx.payment_methods = []

   class AggregatorAgent(Agent):
       """Aggregate data from multiple services."""
       
       @state
       async def aggregate_user_data(self, ctx: Context) -> None:
           """Aggregate all user data."""
           ctx.user_dashboard = {
               'profile': ctx.get('user_profile', {}),
               'orders': ctx.get('orders', []),
               'payment_methods': ctx.get('payment_methods', []),
               'services_status': {
                   'user_service': ctx.get('user_service_success', False),
                   'order_service': ctx.get('order_service_success', False),
                   'payment_service': ctx.get('payment_service_success', False)
               }
           }
           
           print(f"User dashboard aggregated for user {ctx.user_id}")

   async def get_user_dashboard(user_id: str):
       """Get complete user dashboard by orchestrating microservices."""
       
       # Create service agents
       user_agent = UserServiceAgent()
       order_agent = OrderServiceAgent()
       payment_agent = PaymentServiceAgent()
       aggregator_agent = AggregatorAgent()
       
       # Create team with parallel execution for independent services
       team = AgentTeam([
           user_agent,      # Must run first
           [order_agent, payment_agent],  # Can run in parallel after user_agent
           aggregator_agent  # Runs after all services
       ])
       
       # Execute with shared context
       context = Context({'user_id': user_id})
       result = await team.run(context)
       
       return result.context.user_dashboard

   # Usage
   async def main():
       dashboard = await get_user_dashboard("user123")
       print("User Dashboard:", dashboard)

   asyncio.run(main())

File Processing Workflow
-------------------------

Process multiple files with parallel execution and progress tracking:

.. code-block:: python

   import asyncio
   import aiofiles
   from pathlib import Path
   from puffinflow import Agent, Context, state, run_agents_parallel

   class FileProcessorAgent(Agent):
       """Process individual files."""
       
       def __init__(self, file_path: Path):
           super().__init__()
           self.file_path = file_path

       @state
       async def read_file(self, ctx: Context) -> None:
           """Read file content."""
           async with aiofiles.open(self.file_path, 'r') as f:
               ctx.content = await f.read()
               ctx.original_size = len(ctx.content)

       @state(depends_on=["read_file"])
       async def process_content(self, ctx: Context) -> None:
           """Process file content."""
           # Example processing: count words, lines, characters
           lines = ctx.content.split('\n')
           words = ctx.content.split()
           
           ctx.stats = {
               'lines': len(lines),
               'words': len(words),
               'characters': len(ctx.content),
               'file_name': self.file_path.name
           }

       @state(depends_on=["process_content"])
       async def save_results(self, ctx: Context) -> None:
           """Save processing results."""
           output_path = self.file_path.with_suffix('.stats.json')
           
           import json
           async with aiofiles.open(output_path, 'w') as f:
               await f.write(json.dumps(ctx.stats, indent=2))
           
           ctx.output_path = output_path
           print(f"Processed {self.file_path.name}: {ctx.stats['words']} words")

   class BatchFileProcessor(Agent):
       """Coordinate batch file processing."""
       
       def __init__(self, input_directory: Path, max_concurrent: int = 5):
           super().__init__()
           self.input_directory = input_directory
           self.max_concurrent = max_concurrent

       @state
       async def discover_files(self, ctx: Context) -> None:
           """Discover files to process."""
           file_paths = list(self.input_directory.glob('*.txt'))
           ctx.file_paths = file_paths
           ctx.total_files = len(file_paths)
           print(f"Discovered {len(file_paths)} files to process")

       @state(depends_on=["discover_files"])
       async def process_files_batch(self, ctx: Context) -> None:
           """Process files in batches."""
           all_results = []
           
           # Process files in batches to control concurrency
           for i in range(0, len(ctx.file_paths), self.max_concurrent):
               batch = ctx.file_paths[i:i + self.max_concurrent]
               
               # Create agents for this batch
               batch_agents = [FileProcessorAgent(file_path) for file_path in batch]
               
               # Process batch in parallel
               batch_results = await run_agents_parallel(batch_agents)
               all_results.extend(batch_results)
               
               print(f"Completed batch {i//self.max_concurrent + 1}")
           
           ctx.processing_results = all_results
           ctx.successful_files = sum(1 for r in all_results if r.status == 'completed')

       @state(depends_on=["process_files_batch"])
       async def generate_summary(self, ctx: Context) -> None:
           """Generate processing summary."""
           total_words = sum(
               r.context.stats['words'] 
               for r in ctx.processing_results 
               if hasattr(r.context, 'stats')
           )
           
           total_lines = sum(
               r.context.stats['lines'] 
               for r in ctx.processing_results 
               if hasattr(r.context, 'stats')
           )
           
           ctx.summary = {
               'total_files_processed': ctx.successful_files,
               'total_files_discovered': ctx.total_files,
               'total_words': total_words,
               'total_lines': total_lines,
               'success_rate': ctx.successful_files / ctx.total_files if ctx.total_files > 0 else 0
           }
           
           print(f"Processing Summary: {ctx.summary}")

   # Usage
   async def run_batch_processing():
       input_dir = Path("./input_files")
       processor = BatchFileProcessor(input_dir, max_concurrent=3)
       
       result = await processor.run()
       print(f"Batch processing completed: {result.context.summary}")

   asyncio.run(run_batch_processing())

Real-time Data Streaming
-------------------------

Process real-time data streams with backpressure handling:

.. code-block:: python

   import asyncio
   from asyncio import Queue
   from puffinflow import Agent, Context, state, AgentPool

   class StreamProcessorAgent(Agent):
       """Process individual stream messages."""
       
       @state
       async def process_message(self, ctx: Context) -> None:
           """Process a single stream message."""
           message = ctx.message
           
           # Simulate processing time
           await asyncio.sleep(0.1)
           
           # Process message (example: transform and validate)
           processed = {
               'id': message.get('id'),
               'timestamp': message.get('timestamp'),
               'data': message.get('data', '').upper(),  # Transform
               'processed_at': asyncio.get_event_loop().time()
           }
           
           ctx.processed_message = processed

   class StreamCoordinator(Agent):
       """Coordinate stream processing with backpressure."""
       
       def __init__(self, max_queue_size: int = 1000, pool_size: int = 10):
           super().__init__()
           self.message_queue = Queue(maxsize=max_queue_size)
           self.processed_queue = Queue()
           self.pool_size = pool_size
           self.running = False

       async def start_processing(self):
           """Start the stream processing."""
           self.running = True
           
           # Create agent pool for processing
           pool = AgentPool(
               agent_class=StreamProcessorAgent,
               pool_size=self.pool_size
           )
           
           # Start processing task
           processing_task = asyncio.create_task(
               self.process_stream(pool)
           )
           
           return processing_task

       async def process_stream(self, pool: AgentPool):
           """Main stream processing loop."""
           while self.running:
               try:
                   # Get message with timeout to allow checking running flag
                   message = await asyncio.wait_for(
                       self.message_queue.get(), 
                       timeout=1.0
                   )
                   
                   # Create context for processing
                   context = Context({'message': message})
                   
                   # Process message using agent pool
                   result = await pool.process_single(context)
                   
                   # Put processed message in output queue
                   await self.processed_queue.put(result.context.processed_message)
                   
               except asyncio.TimeoutError:
                   continue  # Check running flag
               except Exception as e:
                   print(f"Error processing message: {e}")

       async def add_message(self, message: dict):
           """Add message to processing queue."""
           try:
               await asyncio.wait_for(
                   self.message_queue.put(message),
                   timeout=1.0
               )
           except asyncio.TimeoutError:
               print("Queue full, dropping message (backpressure)")

       async def get_processed_message(self):
           """Get processed message."""
           return await self.processed_queue.get()

       def stop_processing(self):
           """Stop stream processing."""
           self.running = False

   # Usage example
   async def simulate_stream():
       """Simulate real-time data stream."""
       coordinator = StreamCoordinator(max_queue_size=100, pool_size=5)
       
       # Start processing
       processing_task = await coordinator.start_processing()
       
       # Simulate message producer
       async def produce_messages():
           for i in range(50):
               message = {
                   'id': i,
                   'timestamp': asyncio.get_event_loop().time(),
                   'data': f'message_{i}'
               }
               await coordinator.add_message(message)
               await asyncio.sleep(0.05)  # 20 messages per second
       
       # Simulate message consumer
       async def consume_messages():
           processed_count = 0
           while processed_count < 50:
               try:
                   processed = await asyncio.wait_for(
                       coordinator.get_processed_message(),
                       timeout=5.0
                   )
                   processed_count += 1
                   print(f"Consumed: {processed['id']} - {processed['data']}")
               except asyncio.TimeoutError:
                   break
       
       # Run producer and consumer concurrently
       await asyncio.gather(
           produce_messages(),
           consume_messages()
       )
       
       # Stop processing
       coordinator.stop_processing()
       await processing_task

   asyncio.run(simulate_stream())

These examples demonstrate various real-world applications of PuffinFlow, showcasing its flexibility and power for different types of workflow orchestration scenarios.