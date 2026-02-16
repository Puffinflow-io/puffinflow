Examples and Use Cases
======================

This section provides comprehensive examples of using PuffinFlow for real-world scenarios, from basic data processing to advanced AI/ML workflows.

.. contents:: Table of Contents
   :depth: 2
   :local:

Basic Workflow Examples
=======================

These examples demonstrate fundamental PuffinFlow concepts and patterns.

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

AI/ML Workflow Examples
=======================

This section demonstrates advanced AI/ML workflows using PuffinFlow's sophisticated coordination and resource management capabilities.

These examples showcase how PuffinFlow's enterprise-grade features enable building production-ready AI systems with proper fault tolerance, resource management, and observability.

RAG (Retrieval Augmented Generation) System
--------------------------------------------

A complete RAG system with document ingestion, embedding generation, vector storage, and retrieval:

.. code-block:: python

   import asyncio
   import numpy as np
   from typing import List, Dict, Any
   from puffinflow import Agent, Context, state, gpu_accelerated, memory_intensive, AgentTeam
   from puffinflow.core.coordination import RateLimiter
   from puffinflow.core.reliability import CircuitBreaker, CircuitBreakerConfig

   class DocumentIngestionAgent(Agent):
       """Ingest and preprocess documents for RAG system."""

       def __init__(self, chunk_size: int = 1000, overlap: int = 200):
           super().__init__(enable_checkpointing=True)
           self.chunk_size = chunk_size
           self.overlap = overlap

       @state(profile="io_intensive")
       async def load_documents(self, ctx: Context) -> None:
           """Load documents from various sources."""
           document_paths = ctx.document_paths

           documents = []
           for path in document_paths:
               # Simulate document loading
               await asyncio.sleep(0.1)
               documents.append({
                   'path': path,
                   'content': f"Sample content from {path}",
                   'metadata': {'source': path, 'type': 'text'}
               })

           ctx.raw_documents = documents
           print(f"Loaded {len(documents)} documents")

       @state(depends_on=["load_documents"], profile="cpu_intensive")
       async def chunk_documents(self, ctx: Context) -> None:
           """Split documents into overlapping chunks."""
           chunks = []

           for doc in ctx.raw_documents:
               content = doc['content']

               # Simple chunking strategy
               for i in range(0, len(content), self.chunk_size - self.overlap):
                   chunk_text = content[i:i + self.chunk_size]
                   if len(chunk_text.strip()) > 50:  # Skip very small chunks
                       chunks.append({
                           'text': chunk_text,
                           'source': doc['path'],
                           'chunk_id': f"{doc['path']}_{i}",
                           'metadata': doc['metadata']
                       })

           ctx.document_chunks = chunks
           print(f"Created {len(chunks)} document chunks")

   class EmbeddingAgent(Agent):
       """Generate embeddings for document chunks."""

       def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
           super().__init__()
           self.model_name = model_name

           # Circuit breaker for embedding API calls
           self.circuit_breaker = CircuitBreaker(
               CircuitBreakerConfig(
                   failure_threshold=5,
                   recovery_timeout=60,
                   expected_exception=Exception
               )
           )

       @state(profile="gpu_accelerated")
       async def generate_embeddings(self, ctx: Context) -> None:
           """Generate embeddings for document chunks."""
           chunks = ctx.document_chunks
           embeddings = []

           try:
               async with self.circuit_breaker:
                   for chunk in chunks:
                       # Simulate embedding generation
                       await asyncio.sleep(0.01)

                       # Generate fake embedding (replace with actual model)
                       embedding = np.random.randn(384).tolist()  # 384-dim embedding

                       embeddings.append({
                           'chunk_id': chunk['chunk_id'],
                           'embedding': embedding,
                           'text': chunk['text'],
                           'metadata': chunk['metadata']
                       })

               ctx.embeddings = embeddings
               print(f"Generated embeddings for {len(embeddings)} chunks")

           except Exception as e:
               ctx.embedding_error = str(e)
               print(f"Embedding generation failed: {e}")

   class VectorStoreAgent(Agent):
       """Store and index embeddings in vector database."""

       def __init__(self, index_type: str = "faiss"):
           super().__init__()
           self.index_type = index_type
           self.vector_index = {}  # Simplified in-memory store

       @state(depends_on=["generate_embeddings"], profile="memory_intensive")
       async def index_embeddings(self, ctx: Context) -> None:
           """Index embeddings in vector store."""
           embeddings = ctx.embeddings

           # Simulate indexing process
           await asyncio.sleep(0.5)

           for emb_data in embeddings:
               self.vector_index[emb_data['chunk_id']] = {
                   'vector': np.array(emb_data['embedding']),
                   'text': emb_data['text'],
                   'metadata': emb_data['metadata']
               }

           ctx.index_size = len(self.vector_index)
           ctx.vector_store_ready = True
           print(f"Indexed {len(embeddings)} embeddings")

       @state(profile="cpu_intensive")
       async def search_similar(self, ctx: Context) -> None:
           """Search for similar documents."""
           query_embedding = np.array(ctx.query_embedding)
           top_k = ctx.get('top_k', 5)

           if not hasattr(self, 'vector_index') or not self.vector_index:
               ctx.search_results = []
               return

           # Simple cosine similarity search
           similarities = []
           for chunk_id, data in self.vector_index.items():
               similarity = np.dot(query_embedding, data['vector']) / (
                   np.linalg.norm(query_embedding) * np.linalg.norm(data['vector'])
               )
               similarities.append({
                   'chunk_id': chunk_id,
                   'similarity': float(similarity),
                   'text': data['text'],
                   'metadata': data['metadata']
               })

           # Sort by similarity and return top_k
           similarities.sort(key=lambda x: x['similarity'], reverse=True)
           ctx.search_results = similarities[:top_k]
           print(f"Found {len(ctx.search_results)} similar documents")

   class LLMGenerationAgent(Agent):
       """Generate responses using retrieved context."""

       def __init__(self, model_name: str = "gpt-3.5-turbo"):
           super().__init__()
           self.model_name = model_name

           # Rate limiter for LLM API calls
           self.rate_limiter = RateLimiter(max_calls=10, time_window=60)

           # Circuit breaker for LLM API
           self.circuit_breaker = CircuitBreaker(
               CircuitBreakerConfig(
                   failure_threshold=3,
                   recovery_timeout=30,
                   expected_exception=Exception
               )
           )

       @state(depends_on=["search_similar"], profile="external_service")
       async def generate_response(self, ctx: Context) -> None:
           """Generate response using retrieved context."""
           query = ctx.query
           search_results = ctx.search_results

           # Prepare context from search results
           context_text = "\n\n".join([
               f"Source: {result['metadata']['source']}\n{result['text']}"
               for result in search_results
           ])

           prompt = f"""
           Based on the following context, answer the question:

           Context:
           {context_text}

           Question: {query}

           Answer:
           """

           try:
               async with self.rate_limiter:
                   async with self.circuit_breaker:
                       # Simulate LLM API call
                       await asyncio.sleep(1.0)

                       # Generate fake response (replace with actual LLM call)
                       response = f"Based on the provided context, here's the answer to '{query}': [Generated response based on {len(search_results)} retrieved documents]"

                       ctx.generated_response = response
                       ctx.sources_used = [r['metadata']['source'] for r in search_results]
                       print(f"Generated response using {len(search_results)} sources")

           except Exception as e:
               ctx.generation_error = str(e)
               print(f"Response generation failed: {e}")

   # RAG System Orchestrator
   class RAGSystem(Agent):
       """Orchestrate the complete RAG pipeline."""

       def __init__(self):
           super().__init__()
           self.ingestion_agent = DocumentIngestionAgent()
           self.embedding_agent = EmbeddingAgent()
           self.vector_agent = VectorStoreAgent()
           self.llm_agent = LLMGenerationAgent()

       async def setup_knowledge_base(self, document_paths: List[str]) -> None:
           """Setup the knowledge base with documents."""
           # Create team for document processing pipeline
           processing_team = AgentTeam([
               self.ingestion_agent,
               self.embedding_agent,
               self.vector_agent
           ])

           # Setup context
           context = Context({'document_paths': document_paths})

           # Run processing pipeline
           result = await processing_team.run(context)
           print(f"Knowledge base setup completed: {result.status}")

           return result

       async def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
           """Query the RAG system."""
           # Generate query embedding
           query_embedding = np.random.randn(384).tolist()  # Replace with actual embedding

           # Create query context
           query_context = Context({
               'query': question,
               'query_embedding': query_embedding,
               'top_k': top_k
           })

           # Create retrieval and generation team
           query_team = AgentTeam([
               self.vector_agent,  # Search for similar documents
               self.llm_agent      # Generate response
           ])

           # Execute query
           result = await query_team.run(query_context)

           return {
               'question': question,
               'answer': result.context.get('generated_response'),
               'sources': result.context.get('sources_used', []),
               'retrieved_docs': len(result.context.get('search_results', [])),
               'status': result.status
           }

   # Usage Example
   async def run_rag_system():
       """Demonstrate the RAG system."""
       rag = RAGSystem()

       # Setup knowledge base
       documents = [
           "/docs/machine_learning.txt",
           "/docs/deep_learning.txt",
           "/docs/natural_language_processing.txt"
       ]

       await rag.setup_knowledge_base(documents)

       # Query the system
       questions = [
           "What is machine learning?",
           "How does deep learning work?",
           "What are the applications of NLP?"
       ]

       for question in questions:
           result = await rag.query(question)
           print(f"\nQ: {result['question']}")
           print(f"A: {result['answer']}")
           print(f"Sources: {result['sources']}")

   asyncio.run(run_rag_system())

Self-RAG with Reflection
------------------------

A self-improving RAG system that reflects on its responses and iteratively improves:

.. code-block:: python

   class SelfRAGAgent(Agent):
       """Self-improving RAG with reflection and iteration."""

       def __init__(self, max_iterations: int = 3):
           super().__init__(enable_checkpointing=True)
           self.max_iterations = max_iterations
           self.rag_system = RAGSystem()

       @state(profile="external_service")
       async def initial_response(self, ctx: Context) -> None:
           """Generate initial response."""
           query = ctx.query
           initial_result = await self.rag_system.query(query)

           ctx.responses = [initial_result]
           ctx.current_iteration = 0
           ctx.initial_answer = initial_result['answer']

       @state(depends_on=["initial_response"], profile="external_service")
       async def reflect_on_response(self, ctx: Context) -> None:
           """Reflect on the quality of the response."""
           current_response = ctx.responses[-1]
           query = ctx.query

           reflection_prompt = f"""
           Question: {query}
           Answer: {current_response['answer']}

           Please evaluate this answer on a scale of 1-10 for:
           1. Accuracy
           2. Completeness
           3. Relevance

           If the score is below 8, suggest specific improvements needed.
           """

           # Simulate reflection (replace with actual LLM call)
           await asyncio.sleep(0.5)

           # Fake reflection scores
           scores = {
               'accuracy': np.random.randint(6, 10),
               'completeness': np.random.randint(6, 10),
               'relevance': np.random.randint(6, 10)
           }

           avg_score = sum(scores.values()) / len(scores)

           ctx.reflection_scores = scores
           ctx.average_score = avg_score
           ctx.needs_improvement = avg_score < 8.0

           if ctx.needs_improvement:
               ctx.improvement_suggestions = [
                   "Need more specific examples",
                   "Should include recent developments",
                   "Requires better source attribution"
               ]

           print(f"Reflection scores: {scores}, Average: {avg_score:.1f}")

       @state(depends_on=["reflect_on_response"], profile="external_service")
       async def improve_query(self, ctx: Context) -> None:
           """Improve the query based on reflection."""
           if not ctx.needs_improvement or ctx.current_iteration >= self.max_iterations:
               return  # No improvement needed or max iterations reached

           original_query = ctx.query
           suggestions = ctx.improvement_suggestions

           # Generate improved query
           improved_query = f"{original_query} (Focus on: {', '.join(suggestions)})"

           # Query again with improved prompt
           improved_result = await self.rag_system.query(improved_query, top_k=8)

           ctx.responses.append(improved_result)
           ctx.current_iteration += 1

           print(f"Iteration {ctx.current_iteration}: Improved query and response")

           # Continue reflection loop
           await self.reflect_on_response(ctx)

       @state(profile="cpu_intensive")
       async def finalize_response(self, ctx: Context) -> None:
           """Finalize the best response."""
           responses = ctx.responses

           # Select best response based on reflection scores
           best_response = responses[-1]  # Latest is typically best after improvements

           ctx.final_answer = best_response['answer']
           ctx.final_sources = best_response['sources']
           ctx.total_iterations = len(responses)
           ctx.improvement_achieved = len(responses) > 1

           print(f"Finalized response after {ctx.total_iterations} iterations")

   # Usage
   async def run_self_rag():
       """Demonstrate self-improving RAG."""
       self_rag = SelfRAGAgent(max_iterations=3)

       context = Context({'query': "Explain the latest developments in transformer architecture"})
       result = await self_rag.run(context)

       print(f"Final Answer: {result.context.final_answer}")
       print(f"Iterations: {result.context.total_iterations}")
       print(f"Sources: {result.context.final_sources}")

   asyncio.run(run_self_rag())

Graph RAG System
----------------

A Graph RAG implementation that builds knowledge graphs from documents:

.. code-block:: python

   import json
   from typing import Dict, List, Tuple
   from dataclasses import dataclass

   @dataclass
   class Entity:
       """Represents an entity in the knowledge graph."""
       name: str
       type: str
       properties: Dict[str, Any]

   @dataclass
   class Relationship:
       """Represents a relationship between entities."""
       source: str
       target: str
       relation_type: str
       properties: Dict[str, Any]

   class EntityExtractionAgent(Agent):
       """Extract entities and relationships from text."""

       @state(profile="gpu_accelerated")
       async def extract_entities(self, ctx: Context) -> None:
           """Extract named entities from document chunks."""
           chunks = ctx.document_chunks
           entities = {}
           relationships = []

           for chunk in chunks:
               text = chunk['text']

               # Simulate NER and relation extraction
               await asyncio.sleep(0.1)

               # Fake entity extraction (replace with actual NER)
               chunk_entities = [
                   Entity("Python", "TECHNOLOGY", {"description": "Programming language"}),
                   Entity("Machine Learning", "CONCEPT", {"description": "AI technique"}),
                   Entity("TensorFlow", "FRAMEWORK", {"description": "ML framework"})
               ]

               chunk_relationships = [
                   Relationship("Python", "Machine Learning", "USED_FOR", {}),
                   Relationship("TensorFlow", "Machine Learning", "IMPLEMENTS", {})
               ]

               # Add to global collections
               for entity in chunk_entities:
                   entities[entity.name] = entity

               relationships.extend(chunk_relationships)

           ctx.entities = entities
           ctx.relationships = relationships
           print(f"Extracted {len(entities)} entities and {len(relationships)} relationships")

   class GraphBuilderAgent(Agent):
       """Build and maintain the knowledge graph."""

       def __init__(self):
           super().__init__()
           self.graph = {"nodes": {}, "edges": []}

       @state(depends_on=["extract_entities"], profile="memory_intensive")
       async def build_graph(self, ctx: Context) -> None:
           """Build knowledge graph from entities and relationships."""
           entities = ctx.entities
           relationships = ctx.relationships

           # Add nodes (entities)
           for entity_name, entity in entities.items():
               self.graph["nodes"][entity_name] = {
                   "type": entity.type,
                   "properties": entity.properties,
                   "connections": 0
               }

           # Add edges (relationships)
           for rel in relationships:
               if rel.source in self.graph["nodes"] and rel.target in self.graph["nodes"]:
                   edge = {
                       "source": rel.source,
                       "target": rel.target,
                       "relation": rel.relation_type,
                       "properties": rel.properties
                   }
                   self.graph["edges"].append(edge)

                   # Update connection counts
                   self.graph["nodes"][rel.source]["connections"] += 1
                   self.graph["nodes"][rel.target]["connections"] += 1

           ctx.knowledge_graph = self.graph
           ctx.graph_stats = {
               "nodes": len(self.graph["nodes"]),
               "edges": len(self.graph["edges"]),
               "avg_connections": sum(node["connections"] for node in self.graph["nodes"].values()) / len(self.graph["nodes"]) if self.graph["nodes"] else 0
           }

           print(f"Built knowledge graph: {ctx.graph_stats}")

       @state(profile="cpu_intensive")
       async def graph_search(self, ctx: Context) -> None:
           """Search the knowledge graph for relevant information."""
           query_entities = ctx.query_entities  # Entities extracted from query
           max_hops = ctx.get('max_hops', 2)

           relevant_subgraph = {"nodes": {}, "edges": []}
           visited = set()
           queue = [(entity, 0) for entity in query_entities if entity in self.graph["nodes"]]

           # BFS to find relevant subgraph
           while queue:
               current_entity, hops = queue.pop(0)

               if current_entity in visited or hops > max_hops:
                   continue

               visited.add(current_entity)
               relevant_subgraph["nodes"][current_entity] = self.graph["nodes"][current_entity]

               # Find connected entities
               for edge in self.graph["edges"]:
                   if edge["source"] == current_entity and edge["target"] not in visited:
                       queue.append((edge["target"], hops + 1))
                       relevant_subgraph["edges"].append(edge)
                   elif edge["target"] == current_entity and edge["source"] not in visited:
                       queue.append((edge["source"], hops + 1))
                       relevant_subgraph["edges"].append(edge)

           ctx.relevant_subgraph = relevant_subgraph
           print(f"Found relevant subgraph with {len(relevant_subgraph['nodes'])} nodes")

   class GraphRAGAgent(Agent):
       """Main Graph RAG orchestrator."""

       def __init__(self):
           super().__init__()
           self.entity_agent = EntityExtractionAgent()
           self.graph_agent = GraphBuilderAgent()
           self.llm_agent = LLMGenerationAgent()

       async def build_knowledge_graph(self, document_paths: List[str]) -> None:
           """Build knowledge graph from documents."""
           # First, ingest and chunk documents
           ingestion = DocumentIngestionAgent()
           context = Context({'document_paths': document_paths})
           await ingestion.run(context)

           # Build graph pipeline
           graph_team = AgentTeam([
               self.entity_agent,
               self.graph_agent
           ])

           result = await graph_team.run(context)
           self.knowledge_graph = result.context.knowledge_graph
           print("Knowledge graph construction completed")

       async def graph_rag_query(self, question: str) -> Dict[str, Any]:
           """Answer question using graph-enhanced RAG."""
           # Extract entities from question
           query_entities = ["Python", "Machine Learning"]  # Simplified entity extraction

           # Search graph for relevant information
           graph_context = Context({
               'query_entities': query_entities,
               'max_hops': 2
           })

           await self.graph_agent.graph_search(graph_context)

           # Prepare graph context for LLM
           subgraph = graph_context.relevant_subgraph
           graph_context_text = self.format_graph_context(subgraph)

           # Generate response using graph context
           llm_context = Context({
               'query': question,
               'graph_context': graph_context_text
           })

           await self.generate_graph_response(llm_context)

           return {
               'question': question,
               'answer': llm_context.generated_response,
               'graph_nodes_used': len(subgraph['nodes']),
               'graph_edges_used': len(subgraph['edges'])
           }

       def format_graph_context(self, subgraph: Dict) -> str:
           """Format graph information for LLM context."""
           context_parts = []

           # Add entity information
           context_parts.append("Relevant Entities:")
           for entity_name, entity_data in subgraph['nodes'].items():
               context_parts.append(f"- {entity_name} ({entity_data['type']}): {entity_data['properties'].get('description', '')}")

           # Add relationship information
           context_parts.append("\nRelationships:")
           for edge in subgraph['edges']:
               context_parts.append(f"- {edge['source']} {edge['relation']} {edge['target']}")

           return "\n".join(context_parts)

       @state(profile="external_service")
       async def generate_graph_response(self, ctx: Context) -> None:
           """Generate response using graph context."""
           query = ctx.query
           graph_context = ctx.graph_context

           prompt = f"""
           Based on the following knowledge graph information, answer the question:

           Knowledge Graph Context:
           {graph_context}

           Question: {query}

           Answer using the relationships and entities from the graph:
           """

           # Simulate LLM call
           await asyncio.sleep(1.0)
           response = f"Based on the knowledge graph, here's the answer to '{query}': [Graph-enhanced response using connected entities and relationships]"

           ctx.generated_response = response

   # Usage
   async def run_graph_rag():
       """Demonstrate Graph RAG system."""
       graph_rag = GraphRAGAgent()

       # Build knowledge graph
       documents = ["/docs/ai_concepts.txt", "/docs/ml_frameworks.txt"]
       await graph_rag.build_knowledge_graph(documents)

       # Query with graph enhancement
       result = await graph_rag.graph_rag_query("How is Python used in machine learning?")

       print(f"Question: {result['question']}")
       print(f"Answer: {result['answer']}")
       print(f"Graph nodes used: {result['graph_nodes_used']}")

   asyncio.run(run_graph_rag())

Prompt Routing and Model Selection
----------------------------------

Intelligent prompt routing system that selects the best model based on query characteristics:

.. code-block:: python

   import asyncio
   import hashlib
   from typing import Dict, List, Optional
   from puffinflow import Agent, Context, state, AgentTeam, Agents
   from puffinflow.core.coordination import RateLimiter

   class QueryAnalysisAgent(Agent):
       """Analyze incoming queries to determine routing strategy."""

       @state(profile="cpu_intensive")
       async def analyze_query(self, ctx: Context) -> None:
           """Analyze query characteristics for routing decisions."""
           query = ctx.query

           # Simulate query analysis
           await asyncio.sleep(0.1)

           # Extract features for routing decisions
           query_features = {
               'length': len(query.split()),
               'complexity': 'complex' if len(query.split()) > 20 else 'simple',
               'domain': self.detect_domain(query),
               'urgency': self.detect_urgency(query),
               'language': 'en',  # Simplified
               'has_code': 'def ' in query or 'import ' in query,
               'requires_reasoning': any(word in query.lower() for word in ['why', 'how', 'explain', 'analyze']),
               'requires_creativity': any(word in query.lower() for word in ['create', 'generate', 'write', 'design'])
           }

           ctx.query_features = query_features
           ctx.routing_score = self.calculate_routing_score(query_features)
           print(f"Query analysis: {query_features['domain']}, complexity: {query_features['complexity']}")

       def detect_domain(self, query: str) -> str:
           """Detect the domain of the query."""
           query_lower = query.lower()
           if any(word in query_lower for word in ['code', 'programming', 'function', 'algorithm']):
               return 'technical'
           elif any(word in query_lower for word in ['creative', 'story', 'poem', 'art']):
               return 'creative'
           elif any(word in query_lower for word in ['analyze', 'data', 'statistics', 'research']):
               return 'analytical'
           else:
               return 'general'

       def detect_urgency(self, query: str) -> str:
           """Detect urgency level of the query."""
           urgent_words = ['urgent', 'asap', 'immediately', 'emergency', 'quick']
           return 'high' if any(word in query.lower() for word in urgent_words) else 'normal'

       def calculate_routing_score(self, features: Dict) -> Dict[str, float]:
           """Calculate routing scores for different models."""
           scores = {
               'gpt-4': 0.0,
               'gpt-3.5-turbo': 0.0,
               'claude-3': 0.0,
               'local-llama': 0.0
           }

           # GPT-4 for complex reasoning and technical tasks
           if features['complexity'] == 'complex' or features['requires_reasoning']:
               scores['gpt-4'] += 0.8

           # GPT-3.5 for general tasks and speed
           if features['complexity'] == 'simple' and features['urgency'] == 'high':
               scores['gpt-3.5-turbo'] += 0.9

           # Claude for creative and analytical tasks
           if features['domain'] in ['creative', 'analytical']:
               scores['claude-3'] += 0.7

           # Local model for code and privacy-sensitive tasks
           if features['has_code'] or features['domain'] == 'technical':
               scores['local-llama'] += 0.6

           return scores

   class ModelRouterAgent(Agent):
       """Route queries to appropriate models based on analysis."""

       def __init__(self):
           super().__init__()
           self.model_agents = {
               'gpt-4': GPT4Agent(),
               'gpt-3.5-turbo': GPT35Agent(),
               'claude-3': ClaudeAgent(),
               'local-llama': LocalLlamaAgent()
           }

       @state(depends_on=["analyze_query"], profile="cpu_intensive")
       async def route_query(self, ctx: Context) -> None:
           """Route query to the best model based on analysis."""
           routing_scores = ctx.routing_score
           query = ctx.query

           # Select best model
           best_model = max(routing_scores.items(), key=lambda x: x[1])
           selected_model, confidence = best_model

           ctx.selected_model = selected_model
           ctx.routing_confidence = confidence

           print(f"Routing to {selected_model} with confidence {confidence:.2f}")

           # Execute with selected model
           model_agent = self.model_agents[selected_model]
           model_context = Context({
               'query': query,
               'model_name': selected_model,
               'priority': 'high' if ctx.query_features['urgency'] == 'high' else 'normal'
           })

           result = await model_agent.run(model_context)
           ctx.model_response = result.context.response
           ctx.model_metadata = result.context.metadata

   class GPT4Agent(Agent):
       """GPT-4 model agent with advanced capabilities."""

       def __init__(self):
           super().__init__()
           self.rate_limiter = RateLimiter(max_calls=3, time_window=60)

       @state(profile="external_service")
       async def generate_response(self, ctx: Context) -> None:
           """Generate response using GPT-4."""
           async with self.rate_limiter:
               await asyncio.sleep(2.0)  # Simulate longer processing time

               ctx.response = f"GPT-4 response to: {ctx.query}"
               ctx.metadata = {
                   'model': 'gpt-4',
                   'tokens_used': 150,
                   'processing_time': 2.0,
                   'cost': 0.03
               }

   class GPT35Agent(Agent):
       """GPT-3.5 model agent optimized for speed."""

       def __init__(self):
           super().__init__()
           self.rate_limiter = RateLimiter(max_calls=10, time_window=60)

       @state(profile="external_service")
       async def generate_response(self, ctx: Context) -> None:
           """Generate response using GPT-3.5."""
           async with self.rate_limiter:
               await asyncio.sleep(0.8)  # Faster processing

               ctx.response = f"GPT-3.5 response to: {ctx.query}"
               ctx.metadata = {
                   'model': 'gpt-3.5-turbo',
                   'tokens_used': 120,
                   'processing_time': 0.8,
                   'cost': 0.002
               }

   class ClaudeAgent(Agent):
       """Claude model agent for creative and analytical tasks."""

       def __init__(self):
           super().__init__()
           self.rate_limiter = RateLimiter(max_calls=5, time_window=60)

       @state(profile="external_service")
       async def generate_response(self, ctx: Context) -> None:
           """Generate response using Claude."""
           async with self.rate_limiter:
               await asyncio.sleep(1.5)

               ctx.response = f"Claude response to: {ctx.query}"
               ctx.metadata = {
                   'model': 'claude-3',
                   'tokens_used': 140,
                   'processing_time': 1.5,
                   'cost': 0.025
               }

   class LocalLlamaAgent(Agent):
       """Local Llama model for privacy-sensitive tasks."""

       @state(profile="gpu_accelerated")
       async def generate_response(self, ctx: Context) -> None:
           """Generate response using local Llama model."""
           await asyncio.sleep(3.0)  # Longer but private

           ctx.response = f"Local Llama response to: {ctx.query}"
           ctx.metadata = {
               'model': 'local-llama',
               'tokens_used': 100,
               'processing_time': 3.0,
               'cost': 0.0  # No API cost
           }

   class PromptRoutingSystem(Agent):
       """Main prompt routing orchestrator."""

       def __init__(self):
           super().__init__()
           self.query_analyzer = QueryAnalysisAgent()
           self.router = ModelRouterAgent()

       async def process_query(self, query: str) -> Dict:
           """Process query through the routing system."""
           # Create routing team
           routing_team = AgentTeam([
               self.query_analyzer,
               self.router
           ])

           context = Context({'query': query})
           result = await routing_team.run(context)

           return {
               'query': query,
               'selected_model': result.context.selected_model,
               'confidence': result.context.routing_confidence,
               'response': result.context.model_response,
               'metadata': result.context.model_metadata,
               'features': result.context.query_features
           }

   # Batch Processing with A/B Testing
   class ABTestRoutingAgent(Agent):
       """A/B test different routing strategies."""

       def __init__(self, test_ratio: float = 0.5):
           super().__init__()
           self.test_ratio = test_ratio
           self.routing_system = PromptRoutingSystem()

       @state(profile="batch")
       async def process_batch_with_ab_test(self, ctx: Context) -> None:
           """Process batch of queries with A/B testing."""
           queries = ctx.queries

           results_a = []  # Control group (original routing)
           results_b = []  # Test group (alternative routing)

           for i, query in enumerate(queries):
               is_test_group = (hash(query) % 100) < (self.test_ratio * 100)

               if is_test_group:
                   # Test group: force to GPT-4
                   result = await self.force_model_routing(query, 'gpt-4')
                   results_b.append(result)
               else:
                   # Control group: normal routing
                   result = await self.routing_system.process_query(query)
                   results_a.append(result)

           ctx.control_results = results_a
           ctx.test_results = results_b
           ctx.test_metrics = self.calculate_ab_metrics(results_a, results_b)

       async def force_model_routing(self, query: str, model: str) -> Dict:
           """Force routing to specific model for testing."""
           model_agents = {
               'gpt-4': GPT4Agent(),
               'gpt-3.5-turbo': GPT35Agent(),
               'claude-3': ClaudeAgent(),
               'local-llama': LocalLlamaAgent()
           }

           agent = model_agents[model]
           context = Context({'query': query})
           result = await agent.run(context)

           return {
               'query': query,
               'selected_model': model,
               'response': result.context.response,
               'metadata': result.context.metadata
           }

       def calculate_ab_metrics(self, control: List, test: List) -> Dict:
           """Calculate A/B test metrics."""
           control_cost = sum(r['metadata']['cost'] for r in control)
           test_cost = sum(r['metadata']['cost'] for r in test)

           control_time = sum(r['metadata']['processing_time'] for r in control)
           test_time = sum(r['metadata']['processing_time'] for r in test)

           return {
               'control_group_size': len(control),
               'test_group_size': len(test),
               'control_avg_cost': control_cost / len(control) if control else 0,
               'test_avg_cost': test_cost / len(test) if test else 0,
               'control_avg_time': control_time / len(control) if control else 0,
               'test_avg_time': test_time / len(test) if test else 0,
               'cost_difference': (test_cost / len(test) - control_cost / len(control)) if control and test else 0
           }

   # Usage Examples
   async def run_prompt_routing():
       """Demonstrate prompt routing system."""
       routing_system = PromptRoutingSystem()

       test_queries = [
           "Write a creative story about a robot",
           "Explain quantum computing in simple terms",
           "def fibonacci(n): # complete this function",
           "What's 2+2?",
           "Analyze the market trends for AI stocks"
       ]

       for query in test_queries:
           result = await routing_system.process_query(query)
           print(f"\nQuery: {result['query']}")
           print(f"Routed to: {result['selected_model']} (confidence: {result['confidence']:.2f})")
           print(f"Cost: ${result['metadata']['cost']:.3f}")

   async def run_ab_test():
       """Demonstrate A/B testing for routing strategies."""
       ab_tester = ABTestRoutingAgent(test_ratio=0.3)

       queries = [f"Test query {i} about various topics" for i in range(20)]
       context = Context({'queries': queries})

       await ab_tester.run(context)
       print("A/B Test Results:", context.test_metrics)

   asyncio.run(run_prompt_routing())

LLM Fine-tuning Pipeline
------------------------

A complete pipeline for fine-tuning language models with data preparation and evaluation:

.. code-block:: python

   import asyncio
   import json
   from pathlib import Path
   from typing import List, Dict, Any
   from puffinflow import Agent, Context, state, AgentTeam, gpu_accelerated

   class DataPreparationAgent(Agent):
       """Prepare training data for fine-tuning."""

       def __init__(self, task_type: str = "classification"):
           super().__init__(enable_checkpointing=True)
           self.task_type = task_type

       @state(profile="io_intensive")
       async def load_raw_data(self, ctx: Context) -> None:
           """Load raw training data from various sources."""
           data_sources = ctx.data_sources

           all_data = []
           for source in data_sources:
               # Simulate data loading
               await asyncio.sleep(0.2)

               # Mock data based on task type
               if self.task_type == "classification":
                   mock_data = [
                       {"text": f"Sample text {i}", "label": i % 3}
                       for i in range(1000)
                   ]
               elif self.task_type == "text_generation":
                   mock_data = [
                       {"input": f"Prompt {i}", "output": f"Generated response {i}"}
                       for i in range(1000)
                   ]
               else:
                   mock_data = [{"text": f"Sample {i}"} for i in range(1000)]

               all_data.extend(mock_data)

           ctx.raw_data = all_data
           print(f"Loaded {len(all_data)} raw training examples")

       @state(depends_on=["load_raw_data"], profile="cpu_intensive")
       async def clean_and_validate(self, ctx: Context) -> None:
           """Clean and validate the training data."""
           raw_data = ctx.raw_data

           cleaned_data = []
           validation_errors = []

           for i, item in enumerate(raw_data):
               try:
                   # Data validation and cleaning
                   if self.task_type == "classification":
                       if "text" in item and "label" in item and len(item["text"].strip()) > 0:
                           cleaned_item = {
                               "text": item["text"].strip(),
                               "label": int(item["label"])
                           }
                           cleaned_data.append(cleaned_item)
                   elif self.task_type == "text_generation":
                       if "input" in item and "output" in item:
                           cleaned_item = {
                               "input": item["input"].strip(),
                               "output": item["output"].strip()
                           }
                           cleaned_data.append(cleaned_item)

               except Exception as e:
                   validation_errors.append(f"Row {i}: {str(e)}")

           ctx.cleaned_data = cleaned_data
           ctx.validation_errors = validation_errors
           ctx.data_quality_score = len(cleaned_data) / len(raw_data) if raw_data else 0

           print(f"Cleaned data: {len(cleaned_data)} valid examples")
           print(f"Data quality score: {ctx.data_quality_score:.2f}")

       @state(depends_on=["clean_and_validate"], profile="cpu_intensive")
       async def split_data(self, ctx: Context) -> None:
           """Split data into train/validation/test sets."""
           cleaned_data = ctx.cleaned_data

           # Shuffle and split
           import random
           random.shuffle(cleaned_data)

           total_size = len(cleaned_data)
           train_size = int(0.8 * total_size)
           val_size = int(0.1 * total_size)

           ctx.train_data = cleaned_data[:train_size]
           ctx.val_data = cleaned_data[train_size:train_size + val_size]
           ctx.test_data = cleaned_data[train_size + val_size:]

           ctx.split_info = {
               "train_size": len(ctx.train_data),
               "val_size": len(ctx.val_data),
               "test_size": len(ctx.test_data)
           }

           print(f"Data split: {ctx.split_info}")

   class ModelConfigurationAgent(Agent):
       """Configure the model and training parameters."""

       @state(profile="cpu_intensive")
       async def setup_model_config(self, ctx: Context) -> None:
           """Setup model configuration for fine-tuning."""
           base_model = ctx.get('base_model', 'distilbert-base-uncased')
           task_type = ctx.get('task_type', 'classification')

           # Model configuration based on task
           if task_type == "classification":
               num_labels = len(set(item['label'] for item in ctx.train_data))
               model_config = {
                   "model_name": base_model,
                   "num_labels": num_labels,
                   "task_type": "classification",
                   "architecture": "AutoModelForSequenceClassification"
               }
           elif task_type == "text_generation":
               model_config = {
                   "model_name": base_model,
                   "task_type": "text_generation",
                   "architecture": "AutoModelForCausalLM",
                   "max_length": 512
               }

           # Training configuration
           training_config = {
               "learning_rate": 2e-5,
               "batch_size": 16,
               "num_epochs": 3,
               "warmup_steps": 500,
               "weight_decay": 0.01,
               "save_strategy": "epoch",
               "evaluation_strategy": "epoch",
               "logging_steps": 100
           }

           ctx.model_config = model_config
           ctx.training_config = training_config

           print(f"Model config: {model_config['model_name']} for {task_type}")

   class FineTuningAgent(Agent):
       """Execute the fine-tuning process."""

       def __init__(self):
           super().__init__(enable_checkpointing=True)

       @state(depends_on=["setup_model_config"], profile="gpu_accelerated")
       async def fine_tune_model(self, ctx: Context) -> None:
           """Fine-tune the model with prepared data."""
           model_config = ctx.model_config
           training_config = ctx.training_config
           train_data = ctx.train_data
           val_data = ctx.val_data

           print(f"Starting fine-tuning of {model_config['model_name']}")

           # Simulate fine-tuning process
           training_metrics = []

           for epoch in range(training_config['num_epochs']):
               # Simulate training epoch
               await asyncio.sleep(5.0)  # Simulate training time

               epoch_metrics = {
                   "epoch": epoch + 1,
                   "train_loss": 0.5 - (epoch * 0.1),  # Decreasing loss
                   "val_loss": 0.6 - (epoch * 0.08),
                   "train_accuracy": 0.7 + (epoch * 0.1),  # Increasing accuracy
                   "val_accuracy": 0.65 + (epoch * 0.08),
                   "learning_rate": training_config['learning_rate'] * (0.9 ** epoch)
               }

               training_metrics.append(epoch_metrics)
               print(f"Epoch {epoch + 1}: val_loss={epoch_metrics['val_loss']:.3f}, val_acc={epoch_metrics['val_accuracy']:.3f}")

           ctx.training_metrics = training_metrics
           ctx.model_path = f"./fine_tuned_{model_config['model_name']}_epoch_{training_config['num_epochs']}"
           ctx.training_completed = True

           print(f"Fine-tuning completed. Model saved to {ctx.model_path}")

   class ModelEvaluationAgent(Agent):
       """Evaluate the fine-tuned model."""

       @state(depends_on=["fine_tune_model"], profile="gpu_accelerated")
       async def evaluate_model(self, ctx: Context) -> None:
           """Evaluate the fine-tuned model on test data."""
           test_data = ctx.test_data
           model_path = ctx.model_path
           task_type = ctx.model_config['task_type']

           print(f"Evaluating model on {len(test_data)} test examples")

           # Simulate model evaluation
           await asyncio.sleep(2.0)

           if task_type == "classification":
               evaluation_metrics = {
                   "accuracy": 0.85,
                   "precision": 0.84,
                   "recall": 0.86,
                   "f1_score": 0.85,
                   "confusion_matrix": [[150, 10, 5], [8, 140, 12], [6, 14, 155]],
                   "classification_report": "Detailed classification metrics..."
               }
           elif task_type == "text_generation":
               evaluation_metrics = {
                   "perplexity": 15.2,
                   "bleu_score": 0.45,
                   "rouge_l": 0.52,
                   "coherence_score": 0.78,
                   "fluency_score": 0.82
               }

           ctx.evaluation_metrics = evaluation_metrics
           ctx.baseline_comparison = self.compare_with_baseline(evaluation_metrics, task_type)

           print(f"Evaluation completed: {evaluation_metrics}")

       def compare_with_baseline(self, metrics: Dict, task_type: str) -> Dict:
           """Compare with baseline model performance."""
           if task_type == "classification":
               baseline_accuracy = 0.75
               improvement = metrics['accuracy'] - baseline_accuracy
               return {
                   "baseline_accuracy": baseline_accuracy,
                   "fine_tuned_accuracy": metrics['accuracy'],
                   "improvement": improvement,
                   "relative_improvement": improvement / baseline_accuracy
               }
           elif task_type == "text_generation":
               baseline_perplexity = 25.0
               improvement = baseline_perplexity - metrics['perplexity']
               return {
                   "baseline_perplexity": baseline_perplexity,
                   "fine_tuned_perplexity": metrics['perplexity'],
                   "improvement": improvement,
                   "relative_improvement": improvement / baseline_perplexity
               }

           return {}

   class ModelDeploymentAgent(Agent):
       """Deploy the fine-tuned model."""

       @state(depends_on=["evaluate_model"], profile="io_intensive")
       async def prepare_deployment(self, ctx: Context) -> None:
           """Prepare model for deployment."""
           model_path = ctx.model_path
           evaluation_metrics = ctx.evaluation_metrics

           # Check if model meets deployment criteria
           deployment_threshold = ctx.get('deployment_threshold', 0.8)

           if ctx.model_config['task_type'] == "classification":
               meets_criteria = evaluation_metrics['accuracy'] >= deployment_threshold
           else:
               meets_criteria = evaluation_metrics['perplexity'] <= 20.0

           if meets_criteria:
               ctx.deployment_approved = True
               ctx.deployment_config = {
                   "model_path": model_path,
                   "serving_framework": "FastAPI",
                   "container_image": "pytorch/pytorch:latest",
                   "resource_requirements": {
                       "cpu": 2,
                       "memory": "8Gi",
                       "gpu": 1
                   },
                   "endpoints": {
                       "predict": "/predict",
                       "health": "/health",
                       "metrics": "/metrics"
                   }
               }
               print("Model approved for deployment")
           else:
               ctx.deployment_approved = False
               ctx.deployment_rejection_reason = f"Model performance below threshold ({deployment_threshold})"
               print(f"Deployment rejected: {ctx.deployment_rejection_reason}")

   class FineTuningPipeline(Agent):
       """Orchestrate the complete fine-tuning pipeline."""

       def __init__(self, task_type: str = "classification"):
           super().__init__()
           self.task_type = task_type

       async def run_fine_tuning(self, data_sources: List[str], base_model: str,
                                deployment_threshold: float = 0.8) -> Dict[str, Any]:
           """Run the complete fine-tuning pipeline."""

           # Create pipeline team
           pipeline_team = AgentTeam([
               DataPreparationAgent(self.task_type),
               ModelConfigurationAgent(),
               FineTuningAgent(),
               ModelEvaluationAgent(),
               ModelDeploymentAgent()
           ])

           # Setup context
           context = Context({
               'data_sources': data_sources,
               'base_model': base_model,
               'task_type': self.task_type,
               'deployment_threshold': deployment_threshold
           })

           # Execute pipeline
           result = await pipeline_team.run(context)

           return {
               'pipeline_status': result.status,
               'data_quality': result.context.data_quality_score,
               'training_metrics': result.context.training_metrics,
               'evaluation_metrics': result.context.evaluation_metrics,
               'baseline_comparison': result.context.baseline_comparison,
               'deployment_approved': result.context.deployment_approved,
               'model_path': result.context.model_path,
               'split_info': result.context.split_info
           }

   # Usage Examples
   async def run_classification_fine_tuning():
       """Demonstrate classification model fine-tuning."""
       pipeline = FineTuningPipeline(task_type="classification")

       result = await pipeline.run_fine_tuning(
           data_sources=["/data/classification_dataset.json"],
           base_model="distilbert-base-uncased",
           deployment_threshold=0.82
       )

       print("Fine-tuning Results:")
       print(f"Status: {result['pipeline_status']}")
       print(f"Data quality: {result['data_quality']:.2f}")
       print(f"Final accuracy: {result['evaluation_metrics']['accuracy']:.3f}")
       print(f"Deployment approved: {result['deployment_approved']}")

   async def run_text_generation_fine_tuning():
       """Demonstrate text generation model fine-tuning."""
       pipeline = FineTuningPipeline(task_type="text_generation")

       result = await pipeline.run_fine_tuning(
           data_sources=["/data/text_generation_dataset.json"],
           base_model="gpt2-medium"
       )

       print("Text Generation Fine-tuning Results:")
       print(f"Final perplexity: {result['evaluation_metrics']['perplexity']:.1f}")
       print(f"BLEU score: {result['evaluation_metrics']['bleu_score']:.3f}")

   asyncio.run(run_classification_fine_tuning())

Data Analysis and Research Workflows
------------------------------------

Comprehensive data analysis pipelines for research and business intelligence:

.. code-block:: python

   import asyncio
   import pandas as pd
   import numpy as np
   from typing import Dict, List, Any, Optional
   from puffinflow import Agent, Context, state, AgentTeam, memory_intensive, cpu_intensive

   class DataIngestionAgent(Agent):
       """Ingest data from multiple sources and formats."""

       @state(profile="io_intensive")
       async def ingest_data_sources(self, ctx: Context) -> None:
           """Ingest data from various sources."""
           data_sources = ctx.data_sources

           ingested_datasets = {}

           for source_name, source_config in data_sources.items():
               print(f"Ingesting data from {source_name}")

               # Simulate data ingestion based on source type
               if source_config['type'] == 'csv':
                   # Simulate CSV loading
                   await asyncio.sleep(0.2)
                   data = pd.DataFrame(np.random.randn(1000, 5),
                                     columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target'])
               elif source_config['type'] == 'database':
                   # Simulate database query
                   await asyncio.sleep(0.5)
                   data = pd.DataFrame(np.random.randn(2000, 6),
                                     columns=['id', 'timestamp', 'value', 'category', 'amount', 'status'])
               elif source_config['type'] == 'api':
                   # Simulate API call
                   await asyncio.sleep(0.3)
                   data = pd.DataFrame(np.random.randn(500, 4),
                                     columns=['metric_a', 'metric_b', 'metric_c', 'label'])
               else:
                   data = pd.DataFrame()

               ingested_datasets[source_name] = {
                   'data': data,
                   'shape': data.shape,
                   'columns': list(data.columns),
                   'source_type': source_config['type']
               }

           ctx.ingested_datasets = ingested_datasets
           ctx.total_records = sum(dataset['shape'][0] for dataset in ingested_datasets.values())

           print(f"Ingested {len(ingested_datasets)} datasets with {ctx.total_records} total records")

   class DataQualityAgent(Agent):
       """Assess and improve data quality."""

       @state(depends_on=["ingest_data_sources"], profile="cpu_intensive")
       async def assess_data_quality(self, ctx: Context) -> None:
           """Assess data quality across all datasets."""
           datasets = ctx.ingested_datasets

           quality_reports = {}

           for name, dataset in datasets.items():
               data = dataset['data']

               # Calculate quality metrics
               quality_metrics = {
                   'completeness': 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])),
                   'duplicate_rate': data.duplicated().sum() / len(data),
                   'outlier_rate': self.detect_outliers(data),
                   'consistency_score': self.check_consistency(data),
                   'missing_values': data.isnull().sum().to_dict(),
                   'data_types': data.dtypes.to_dict()
               }

               quality_reports[name] = quality_metrics

           ctx.quality_reports = quality_reports
           ctx.overall_quality_score = np.mean([report['completeness'] for report in quality_reports.values()])

           print(f"Data quality assessment completed. Overall score: {ctx.overall_quality_score:.2f}")

       def detect_outliers(self, data: pd.DataFrame) -> float:
           """Detect outliers using IQR method."""
           numeric_cols = data.select_dtypes(include=[np.number]).columns
           outlier_count = 0
           total_values = 0

           for col in numeric_cols:
               Q1 = data[col].quantile(0.25)
               Q3 = data[col].quantile(0.75)
               IQR = Q3 - Q1
               outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
               outlier_count += outliers
               total_values += len(data[col].dropna())

           return outlier_count / total_values if total_values > 0 else 0

       def check_consistency(self, data: pd.DataFrame) -> float:
           """Check data consistency (simplified)."""
           # Simplified consistency check
           return np.random.uniform(0.8, 0.95)

   class StatisticalAnalysisAgent(Agent):
       """Perform statistical analysis on the datasets."""

       @state(depends_on=["assess_data_quality"], profile="cpu_intensive")
       async def descriptive_statistics(self, ctx: Context) -> None:
           """Calculate descriptive statistics for all datasets."""
           datasets = ctx.ingested_datasets

           statistical_summaries = {}

           for name, dataset in datasets.items():
               data = dataset['data']

               # Descriptive statistics
               numeric_cols = data.select_dtypes(include=[np.number]).columns
               categorical_cols = data.select_dtypes(include=['object', 'category']).columns

               summary = {
                   'numeric_summary': data[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
                   'categorical_summary': {col: data[col].value_counts().to_dict() for col in categorical_cols},
                   'correlations': data[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {},
                   'distributions': self.analyze_distributions(data[numeric_cols]) if len(numeric_cols) > 0 else {}
               }

               statistical_summaries[name] = summary

           ctx.statistical_summaries = statistical_summaries
           print("Descriptive statistics calculated for all datasets")

       def analyze_distributions(self, numeric_data: pd.DataFrame) -> Dict:
           """Analyze distributions of numeric variables."""
           distributions = {}

           for col in numeric_data.columns:
               # Simple distribution analysis
               skewness = numeric_data[col].skew()
               kurtosis = numeric_data[col].kurtosis()

               distributions[col] = {
                   'skewness': skewness,
                   'kurtosis': kurtosis,
                   'distribution_type': 'normal' if abs(skewness) < 0.5 else 'skewed'
               }

           return distributions

   class HypothesisTestingAgent(Agent):
       """Perform hypothesis testing and statistical inference."""

       @state(depends_on=["descriptive_statistics"], profile="cpu_intensive")
       async def test_hypotheses(self, ctx: Context) -> None:
           """Perform various hypothesis tests."""
           datasets = ctx.ingested_datasets
           hypotheses = ctx.get('hypotheses', [])

           test_results = {}

           for hypothesis in hypotheses:
               test_name = hypothesis['name']
               test_type = hypothesis['type']
               dataset_name = hypothesis['dataset']

               if dataset_name in datasets:
                   data = datasets[dataset_name]['data']

                   if test_type == 'correlation':
                       result = await self.test_correlation(data, hypothesis['variables'])
                   elif test_type == 'mean_difference':
                       result = await self.test_mean_difference(data, hypothesis['groups'], hypothesis['variable'])
                   elif test_type == 'distribution':
                       result = await self.test_distribution(data, hypothesis['variable'])
                   else:
                       result = {'error': f'Unknown test type: {test_type}'}

                   test_results[test_name] = result

           ctx.hypothesis_test_results = test_results
           print(f"Completed {len(test_results)} hypothesis tests")

       async def test_correlation(self, data: pd.DataFrame, variables: List[str]) -> Dict:
           """Test correlation between variables."""
           if len(variables) >= 2 and all(var in data.columns for var in variables):
               correlation = data[variables[0]].corr(data[variables[1]])

               # Simulate p-value calculation
               p_value = np.random.uniform(0.001, 0.1)

               return {
                   'correlation': correlation,
                   'p_value': p_value,
                   'significant': p_value < 0.05,
                   'interpretation': 'Strong positive correlation' if correlation > 0.7 else 'Moderate correlation'
               }
           return {'error': 'Invalid variables for correlation test'}

       async def test_mean_difference(self, data: pd.DataFrame, groups: List[str], variable: str) -> Dict:
           """Test difference in means between groups."""
           if len(groups) == 2 and variable in data.columns:
               group1_data = data[data[groups[0]] == True][variable]
               group2_data = data[data[groups[0]] == False][variable]

               mean_diff = group1_data.mean() - group2_data.mean()
               p_value = np.random.uniform(0.001, 0.1)

               return {
                   'mean_difference': mean_diff,
                   'group1_mean': group1_data.mean(),
                   'group2_mean': group2_data.mean(),
                   'p_value': p_value,
                   'significant': p_value < 0.05
               }
           return {'error': 'Invalid groups or variable for mean difference test'}

       async def test_distribution(self, data: pd.DataFrame, variable: str) -> Dict:
           """Test if variable follows a specific distribution."""
           if variable in data.columns:
               # Simulate normality test
               p_value = np.random.uniform(0.001, 0.5)

               return {
                   'test_type': 'normality',
                   'p_value': p_value,
                   'is_normal': p_value > 0.05,
                   'recommendation': 'Use parametric tests' if p_value > 0.05 else 'Use non-parametric tests'
               }
           return {'error': f'Variable {variable} not found'}

   class PredictiveModelingAgent(Agent):
       """Build and evaluate predictive models."""

       @state(depends_on=["test_hypotheses"], profile="memory_intensive")
       async def build_predictive_models(self, ctx: Context) -> None:
           """Build predictive models for specified targets."""
           datasets = ctx.ingested_datasets
           modeling_configs = ctx.get('modeling_configs', [])

           model_results = {}

           for config in modeling_configs:
               dataset_name = config['dataset']
               target_variable = config['target']
               model_types = config.get('models', ['linear_regression', 'random_forest'])

               if dataset_name in datasets:
                   data = datasets[dataset_name]['data']
                   results = await self.train_and_evaluate_models(data, target_variable, model_types)
                   model_results[f"{dataset_name}_{target_variable}"] = results

           ctx.model_results = model_results
           print(f"Built and evaluated models for {len(model_results)} targets")

       async def train_and_evaluate_models(self, data: pd.DataFrame, target: str, model_types: List[str]) -> Dict:
           """Train and evaluate multiple model types."""
           if target not in data.columns:
               return {'error': f'Target variable {target} not found'}

           # Simulate model training and evaluation
           await asyncio.sleep(1.0)  # Simulate training time

           results = {}

           for model_type in model_types:
               # Simulate model performance metrics
               if model_type == 'linear_regression':
                   metrics = {
                       'r2_score': np.random.uniform(0.6, 0.9),
                       'mse': np.random.uniform(0.1, 0.5),
                       'mae': np.random.uniform(0.1, 0.3)
                   }
               elif model_type == 'random_forest':
                   metrics = {
                       'r2_score': np.random.uniform(0.7, 0.95),
                       'mse': np.random.uniform(0.05, 0.3),
                       'mae': np.random.uniform(0.05, 0.2),
                       'feature_importance': {f'feature_{i}': np.random.uniform(0, 1) for i in range(5)}
                   }
               else:
                   metrics = {'error': f'Unknown model type: {model_type}'}

               results[model_type] = metrics

           # Select best model
           best_model = max(results.keys(), key=lambda k: results[k].get('r2_score', 0))
           results['best_model'] = best_model

           return results

   class ReportGenerationAgent(Agent):
       """Generate comprehensive analysis reports."""

       @state(depends_on=["build_predictive_models"], profile="cpu_intensive")
       async def generate_analysis_report(self, ctx: Context) -> None:
           """Generate comprehensive analysis report."""

           report = {
               'executive_summary': self.create_executive_summary(ctx),
               'data_overview': self.create_data_overview(ctx),
               'quality_assessment': ctx.quality_reports,
               'statistical_findings': self.summarize_statistical_findings(ctx),
               'hypothesis_test_results': ctx.hypothesis_test_results,
               'predictive_models': self.summarize_model_results(ctx),
               'recommendations': self.generate_recommendations(ctx),
               'methodology': self.document_methodology(ctx)
           }

           ctx.analysis_report = report
           ctx.report_generated = True

           print("Comprehensive analysis report generated")

       def create_executive_summary(self, ctx: Context) -> Dict:
           """Create executive summary of the analysis."""
           return {
               'total_datasets': len(ctx.ingested_datasets),
               'total_records': ctx.total_records,
               'overall_quality_score': ctx.overall_quality_score,
               'key_findings': [
                   f"Analyzed {ctx.total_records} records across {len(ctx.ingested_datasets)} datasets",
                   f"Overall data quality score: {ctx.overall_quality_score:.2f}",
                   f"Completed {len(ctx.hypothesis_test_results)} hypothesis tests",
                   f"Built predictive models for {len(ctx.model_results)} targets"
               ],
               'business_impact': "Analysis provides insights for data-driven decision making"
           }

       def create_data_overview(self, ctx: Context) -> Dict:
           """Create overview of all datasets."""
           overview = {}

           for name, dataset in ctx.ingested_datasets.items():
               overview[name] = {
                   'shape': dataset['shape'],
                   'columns': dataset['columns'],
                   'source_type': dataset['source_type']
               }

           return overview

       def summarize_statistical_findings(self, ctx: Context) -> Dict:
           """Summarize key statistical findings."""
           findings = {}

           for dataset_name, summary in ctx.statistical_summaries.items():
               key_insights = []

               # Extract key insights from correlations
               if summary['correlations']:
                   strong_correlations = [(k, v) for k, corr_dict in summary['correlations'].items()
                                        for v_k, v in corr_dict.items() if abs(v) > 0.7 and k != v_k]
                   if strong_correlations:
                       key_insights.append(f"Found {len(strong_correlations)} strong correlations")

               findings[dataset_name] = {
                   'key_insights': key_insights,
                   'data_characteristics': summary['distributions']
               }

           return findings

       def summarize_model_results(self, ctx: Context) -> Dict:
           """Summarize predictive modeling results."""
           summary = {}

           for target, results in ctx.model_results.items():
               if 'best_model' in results:
                   best_model = results['best_model']
                   best_performance = results[best_model]

                   summary[target] = {
                       'best_model': best_model,
                       'performance': best_performance,
                       'model_comparison': {model: metrics.get('r2_score', 0)
                                          for model, metrics in results.items()
                                          if model != 'best_model'}
                   }

           return summary

       def generate_recommendations(self, ctx: Context) -> List[str]:
           """Generate actionable recommendations."""
           recommendations = []

           # Data quality recommendations
           if ctx.overall_quality_score < 0.8:
               recommendations.append("Improve data quality processes to achieve >80% quality score")

           # Model performance recommendations
           for target, results in ctx.model_results.items():
               if 'best_model' in results:
                   best_score = results[results['best_model']].get('r2_score', 0)
                   if best_score < 0.7:
                       recommendations.append(f"Consider feature engineering for {target} to improve model performance")

           return recommendations

       def document_methodology(self, ctx: Context) -> Dict:
           """Document the analysis methodology."""
           return {
               'data_sources': list(ctx.ingested_datasets.keys()),
               'quality_metrics': ['completeness', 'consistency', 'outlier_detection'],
               'statistical_tests': list(ctx.hypothesis_test_results.keys()),
               'modeling_approaches': ['linear_regression', 'random_forest'],
               'validation_method': 'train_test_split',
               'tools_used': ['pandas', 'numpy', 'scikit-learn', 'puffinflow']
           }

   class ResearchWorkflowOrchestrator(Agent):
       """Orchestrate the complete research and analysis workflow."""

       async def run_analysis_pipeline(self, data_sources: Dict, hypotheses: List[Dict],
                                     modeling_configs: List[Dict]) -> Dict[str, Any]:
           """Run the complete analysis pipeline."""

           # Create analysis team
           analysis_team = AgentTeam([
               DataIngestionAgent(),
               DataQualityAgent(),
               StatisticalAnalysisAgent(),
               HypothesisTestingAgent(),
               PredictiveModelingAgent(),
               ReportGenerationAgent()
           ])

           # Setup context
           context = Context({
               'data_sources': data_sources,
               'hypotheses': hypotheses,
               'modeling_configs': modeling_configs
           })

           # Execute pipeline
           result = await analysis_team.run(context)

           return {
               'pipeline_status': result.status,
               'analysis_report': result.context.analysis_report,
               'total_records_analyzed': result.context.total_records,
               'overall_quality_score': result.context.overall_quality_score,
               'hypothesis_results': len(result.context.hypothesis_test_results),
               'models_built': len(result.context.model_results),
               'execution_time': result.context.get('execution_time', 'N/A')
           }

   # Usage Example
   async def run_comprehensive_analysis():
       """Demonstrate comprehensive data analysis workflow."""

       # Define data sources
       data_sources = {
           'customer_data': {'type': 'csv', 'path': '/data/customers.csv'},
           'transaction_data': {'type': 'database', 'query': 'SELECT * FROM transactions'},
           'market_data': {'type': 'api', 'endpoint': 'https://api.market.com/data'}
       }

       # Define hypotheses to test
       hypotheses = [
           {
               'name': 'customer_satisfaction_correlation',
               'type': 'correlation',
               'dataset': 'customer_data',
               'variables': ['satisfaction_score', 'loyalty_score']
           },
           {
               'name': 'seasonal_sales_difference',
               'type': 'mean_difference',
               'dataset': 'transaction_data',
               'groups': ['is_holiday', 'is_regular'],
               'variable': 'sales_amount'
           }
       ]

       # Define modeling configurations
       modeling_configs = [
           {
               'dataset': 'customer_data',
               'target': 'target',
               'models': ['linear_regression', 'random_forest']
           }
       ]

       # Run analysis
       orchestrator = ResearchWorkflowOrchestrator()
       results = await orchestrator.run_analysis_pipeline(data_sources, hypotheses, modeling_configs)

       print("Analysis Pipeline Results:")
       print(f"Status: {results['pipeline_status']}")
       print(f"Records analyzed: {results['total_records_analyzed']}")
       print(f"Quality score: {results['overall_quality_score']:.2f}")
       print(f"Hypotheses tested: {results['hypothesis_results']}")
       print(f"Models built: {results['models_built']}")

   asyncio.run(run_comprehensive_analysis())

These comprehensive AI/ML workflow examples demonstrate PuffinFlow's advanced capabilities for building sophisticated AI workflows with proper resource management, fault tolerance, and coordination patterns. The framework's flexibility allows for easy customization and extension of these patterns for specific use cases.
