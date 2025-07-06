
export const ragRecipeMarkdown = `# Building a Production RAG System with Puffinflow

Learn how to build a robust Retrieval-Augmented Generation (RAG) system using Puffinflow, progressing from basic functionality to production-ready features with re-ranking, error handling, and multi-agent coordination.

## Overview

This tutorial demonstrates how Puffinflow's state management, resource control, and coordination features transform a simple RAG implementation into a scalable, resilient system. We'll build incrementally, adding one feature at a time.

**What you'll learn:**
- Basic RAG workflow orchestration
- Adding re-ranking for better retrieval quality
- Implementing rate limiting and retry policies
- Managing secrets and configuration
- Checkpointing for long-running processes
- Multi-agent coordination for parallel processing

## 1. Basic RAG Implementation

Let's start with a simple document processing and query system:

\`\`\`python
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from puffinflow import Agent
from puffinflow.decorators import state

# Using LangChain for RAG components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextFileLoader, PDFLoader

# Create our RAG agent
rag_agent = Agent("basic-rag-agent")

@state
async def initialize_system(context):
    """Initialize the RAG system components"""
    print("🔧 Initializing RAG system...")

    # Store API keys securely using context secrets
    context.set_secret("openai_api_key", "your-openai-api-key")
    context.set_secret("pinecone_api_key", "your-pinecone-api-key")

    # Initialize components with secrets
    openai_key = context.get_secret("openai_api_key")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    llm = OpenAI(openai_api_key=openai_key, temperature=0.1)

    # Store components in context for later use
    context.set_variable("embeddings", embeddings)
    context.set_variable("llm", llm)
    context.set_constant("index_name", "rag-knowledge-base")

    print("✅ RAG system initialized")

@state
async def load_documents(context):
    """Load documents from specified sources"""
    print("📚 Loading documents...")

    document_paths = [
        "docs/company_handbook.pdf",
        "docs/product_specifications.txt",
        "docs/user_manual.pdf"
    ]

    documents = []
    for doc_path in document_paths:
        if not Path(doc_path).exists():
            print(f"⚠️ Document not found: {doc_path}")
            continue

        print(f"   📄 Loading {doc_path}...")

        # Choose appropriate loader
        if doc_path.endswith('.pdf'):
            loader = PDFLoader(doc_path)
        else:
            loader = TextFileLoader(doc_path)

        doc_data = loader.load()
        documents.extend(doc_data)
        print(f"   ✅ Loaded {len(doc_data)} pages")

    context.set_variable("documents", documents)
    print(f"📖 Total documents loaded: {len(documents)}")

@state
async def create_embeddings(context):
    """Create vector embeddings for documents"""
    print("🔢 Creating vector embeddings...")

    documents = context.get_variable("documents", [])
    embeddings = context.get_variable("embeddings")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)

    # Create vector store
    vectorstore = Pinecone.from_documents(
        chunks,
        embeddings,
        index_name=context.get_constant("index_name")
    )

    context.set_variable("vectorstore", vectorstore)
    context.set_output("total_chunks", len(chunks))
    print(f"✅ Created {len(chunks)} embeddings")

@state
async def setup_qa_chain(context):
    """Set up the question-answering chain"""
    print("🔗 Setting up QA chain...")

    vectorstore = context.get_variable("vectorstore")
    llm = context.get_variable("llm")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    context.set_variable("qa_chain", qa_chain)
    print("✅ QA chain ready")

@state
async def test_system(context):
    """Test the RAG system with sample queries"""
    print("🧪 Testing RAG system...")

    qa_chain = context.get_variable("qa_chain")
    test_queries = [
        "What is the company's vacation policy?",
        "How do I reset my password?",
        "What are the system requirements?"
    ]

    for query in test_queries:
        print(f"   ❓ {query}")
        result = qa_chain({"query": query})
        answer = result["result"][:100] + "..." if len(result["result"]) > 100 else result["result"]
        print(f"   💭 {answer}")
        print(f"   📚 Sources: {len(result['source_documents'])}")

    print("✅ Testing complete")

# Build the workflow
rag_agent.add_state("initialize_system", initialize_system)
rag_agent.add_state("load_documents", load_documents, dependencies=["initialize_system"])
rag_agent.add_state("create_embeddings", create_embeddings, dependencies=["load_documents"])
rag_agent.add_state("setup_qa_chain", setup_qa_chain, dependencies=["create_embeddings"])
rag_agent.add_state("test_system", test_system, dependencies=["setup_qa_chain"])

async def main():
    print("🚀 Starting basic RAG system...")
    await rag_agent.run()

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

**Key concepts introduced:**
- \`context.set_secret()\` for secure API key storage
- \`context.set_constant()\` for immutable configuration
- \`context.set_output()\` for metrics tracking
- State dependencies for workflow ordering

## 2. Adding Re-ranking for Better Results

Now let's enhance our system with re-ranking to improve retrieval quality:

\`\`\`python
from sentence_transformers import CrossEncoder

@state
async def setup_reranker(context):
    """Initialize the re-ranking model"""
    print("🔄 Setting up re-ranker...")

    # Use a cross-encoder model for re-ranking
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    context.set_variable("reranker", reranker)

    print("✅ Re-ranker ready")

@state
async def setup_enhanced_qa_chain(context):
    """Set up QA chain with re-ranking"""
    print("🔗 Setting up enhanced QA chain with re-ranking...")

    vectorstore = context.get_variable("vectorstore")
    llm = context.get_variable("llm")
    reranker = context.get_variable("reranker")

    class RerankedRetriever:
        def __init__(self, vectorstore, reranker, k=10, final_k=4):
            self.vectorstore = vectorstore
            self.reranker = reranker
            self.k = k
            self.final_k = final_k

        def get_relevant_documents(self, query):
            # Get more documents than needed
            docs = self.vectorstore.similarity_search(query, k=self.k)

            # Re-rank using cross-encoder
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)

            # Sort by score and take top results
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return [doc for doc, score in scored_docs[:self.final_k]]

    # Create retriever with re-ranking
    retriever = RerankedRetriever(vectorstore, reranker)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    context.set_variable("qa_chain", qa_chain)
    print("✅ Enhanced QA chain with re-ranking ready")

@state
async def test_enhanced_system(context):
    """Test the enhanced system and compare results"""
    print("🧪 Testing enhanced RAG system...")

    qa_chain = context.get_variable("qa_chain")
    vectorstore = context.get_variable("vectorstore")

    test_query = "What is the company's vacation policy?"

    print(f"   ❓ Query: {test_query}")

    # Test basic retrieval
    basic_docs = vectorstore.similarity_search(test_query, k=4)
    print(f"   📄 Basic retrieval: {len(basic_docs)} documents")

    # Test enhanced retrieval with re-ranking
    result = qa_chain({"query": test_query})
    enhanced_docs = result["source_documents"]

    print(f"   🔄 Enhanced with re-ranking: {len(enhanced_docs)} documents")
    print(f"   💭 Answer: {result['result'][:120]}...")

    # Store results for comparison
    context.set_output("basic_retrieval_count", len(basic_docs))
    context.set_output("enhanced_retrieval_count", len(enhanced_docs))

    print("✅ Enhanced testing complete")

# Update the workflow
rag_agent_v2 = Agent("enhanced-rag-agent")

rag_agent_v2.add_state("initialize_system", initialize_system)
rag_agent_v2.add_state("load_documents", load_documents, dependencies=["initialize_system"])
rag_agent_v2.add_state("create_embeddings", create_embeddings, dependencies=["load_documents"])
rag_agent_v2.add_state("setup_reranker", setup_reranker, dependencies=["create_embeddings"])
rag_agent_v2.add_state("setup_enhanced_qa_chain", setup_enhanced_qa_chain,
                      dependencies=["setup_reranker"])
rag_agent_v2.add_state("test_enhanced_system", test_enhanced_system,
                      dependencies=["setup_enhanced_qa_chain"])
\`\`\`

**New concepts:**
- Custom retriever classes for enhanced functionality
- Output metrics for performance comparison
- Modular state design for easy enhancement

## 3. Adding Rate Limiting and Error Handling

Now let's make our system production-ready with rate limiting and robust error handling:

\`\`\`python
from puffinflow.core.agent.base import RetryPolicy

# Create custom retry policies
embedding_retry_policy = RetryPolicy(
    max_retries=3,
    initial_delay=2.0,
    exponential_base=2.0,
    jitter=True
)

# Production-ready RAG agent
rag_agent_v3 = Agent("production-rag-agent", retry_policy=embedding_retry_policy)

@state(rate_limit=10.0, burst_limit=20, max_retries=3, timeout=60.0)
async def create_embeddings_robust(context):
    """Create embeddings with rate limiting and error handling"""
    print("🔢 Creating embeddings with rate limiting...")

    documents = context.get_variable("documents", [])
    embeddings = context.get_variable("embeddings")

    if not documents:
        raise Exception("No documents to process")

    # Process in batches to respect rate limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []
    failed_docs = []

    batch_size = 5
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = len(documents) // batch_size + 1

        try:
            print(f"   🔄 Processing batch {batch_num}/{total_batches}")

            # Split documents in this batch
            batch_chunks = []
            for doc in batch:
                chunks = text_splitter.split_documents([doc])
                batch_chunks.extend(chunks)

            # Validate chunk content
            valid_chunks = []
            for chunk in batch_chunks:
                if len(chunk.page_content.strip()) >= 10:
                    valid_chunks.append(chunk)
                else:
                    print(f"   ⚠️ Skipping short chunk")

            all_chunks.extend(valid_chunks)
            print(f"   ✅ Batch {batch_num} complete: {len(valid_chunks)} chunks")

        except Exception as e:
            print(f"   ❌ Batch {batch_num} failed: {e}")
            failed_docs.extend([doc.metadata.get('source', 'unknown') for doc in batch])
            continue

    if not all_chunks:
        raise Exception("No valid chunks created")

    # Create vector store with error handling
    try:
        pinecone_key = context.get_secret("pinecone_api_key")

        vectorstore = Pinecone.from_documents(
            all_chunks,
            embeddings,
            index_name=context.get_constant("index_name")
        )

        context.set_variable("vectorstore", vectorstore)
        context.set_output("successful_chunks", len(all_chunks))
        context.set_output("failed_documents", len(failed_docs))

        success_rate = (len(all_chunks) / (len(all_chunks) + len(failed_docs))) * 100
        print(f"✅ Embeddings created: {len(all_chunks)} chunks ({success_rate:.1f}% success rate)")

    except Exception as e:
        print(f"❌ Vector store creation failed: {e}")
        raise

@state(rate_limit=3.0, max_retries=2, timeout=30.0)
async def test_with_error_handling(context):
    """Test system with comprehensive error handling"""
    print("🧪 Testing with error handling...")

    qa_chain = context.get_variable("qa_chain")

    test_queries = [
        "What is the company's vacation policy?",
        "How do I reset my password?",
        "What are the product specifications?",
        "Invalid query with special chars: @#$%^&*()"
    ]

    successful_queries = []
    failed_queries = []

    for query in test_queries:
        try:
            print(f"   ❓ Testing: {query}")

            # Validate query
            if len(query.strip()) < 3:
                print(f"   ⚠️ Query too short, skipping")
                continue

            result = qa_chain({"query": query})

            # Validate result quality
            if len(result["result"].strip()) < 10:
                print(f"   ⚠️ Poor quality answer received")
                failed_queries.append(query)
                continue

            successful_queries.append({
                "query": query,
                "answer_length": len(result["result"]),
                "sources": len(result["source_documents"])
            })

            print(f"   ✅ Success: {len(result['result'])} chars, {len(result['source_documents'])} sources")

        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            failed_queries.append(query)

    # Store comprehensive metrics
    context.set_output("successful_queries", len(successful_queries))
    context.set_output("failed_queries", len(failed_queries))
    context.set_output("success_rate", len(successful_queries) / len(test_queries) * 100)

    if successful_queries:
        avg_sources = sum(q["sources"] for q in successful_queries) / len(successful_queries)
        context.set_output("avg_sources_per_answer", avg_sources)

    print(f"✅ Testing complete: {len(successful_queries)}/{len(test_queries)} successful")

# Build robust workflow
rag_agent_v3.add_state("initialize_system", initialize_system)
rag_agent_v3.add_state("load_documents", load_documents, dependencies=["initialize_system"])
rag_agent_v3.add_state("create_embeddings_robust", create_embeddings_robust,
                      dependencies=["load_documents"])
rag_agent_v3.add_state("setup_reranker", setup_reranker, dependencies=["create_embeddings_robust"])
rag_agent_v3.add_state("setup_enhanced_qa_chain", setup_enhanced_qa_chain,
                      dependencies=["setup_reranker"])
rag_agent_v3.add_state("test_with_error_handling", test_with_error_handling,
                      dependencies=["setup_enhanced_qa_chain"])
\`\`\`

**Production features added:**
- \`@state(rate_limit=10.0)\` prevents API quota exhaustion
- \`max_retries=3\` with exponential backoff
- Input validation and quality checks
- Comprehensive error metrics
- Graceful degradation on failures

## 4. Checkpointing for Long Processes

For large document corpora, add checkpointing to resume interrupted processing:

\`\`\`python
import json

@state(checkpoint_interval=30.0)  # Auto-checkpoint every 30 seconds
async def process_large_corpus(context):
    """Process large document corpus with checkpointing"""
    print("📚 Processing large corpus with checkpointing...")

    # Check if resuming from checkpoint
    processing_state = context.get_variable("corpus_state")

    if processing_state:
        print("🔄 Resuming from checkpoint...")
        current_index = processing_state["current_index"]
        processed_count = processing_state["processed_count"]
        total_docs = processing_state["total_documents"]

        print(f"   📊 Resume point: {current_index}/{total_docs}")
        print(f"   ✅ Previously processed: {processed_count}")
    else:
        print("🆕 Starting fresh processing...")

        # Discover all documents
        document_dirs = ["docs/", "knowledge_base/", "manuals/"]
        all_documents = []

        for doc_dir in document_dirs:
            dir_path = Path(doc_dir)
            if dir_path.exists():
                for file_path in dir_path.rglob("*"):
                    if file_path.suffix.lower() in ['.pdf', '.txt', '.md']:
                        all_documents.append(str(file_path))

        processing_state = {
            "all_documents": all_documents,
            "total_documents": len(all_documents),
            "current_index": 0,
            "processed_count": 0,
            "failed_count": 0,
            "start_time": asyncio.get_event_loop().time()
        }

        context.set_variable("corpus_state", processing_state)
        print(f"   📁 Found {len(all_documents)} documents to process")

    # Continue processing from current position
    embeddings = context.get_variable("embeddings")
    vectorstore = context.get_variable("vectorstore")

    all_documents = processing_state["all_documents"]
    current_index = processing_state["current_index"]
    total_docs = processing_state["total_documents"]

    for i in range(current_index, total_docs):
        doc_path = all_documents[i]

        try:
            print(f"   📄 Processing {i+1}/{total_docs}: {Path(doc_path).name}")

            # Load and process document
            if doc_path.endswith('.pdf'):
                loader = PDFLoader(doc_path)
            else:
                loader = TextFileLoader(doc_path)

            doc_data = loader.load()

            # Process each page
            for page in doc_data:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                chunks = text_splitter.split_documents([page])

                # Add to vector store
                vectorstore.add_documents(chunks)

            processing_state["processed_count"] += 1

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            processing_state["failed_count"] += 1

        # Update checkpoint state
        processing_state["current_index"] = i + 1
        context.set_variable("corpus_state", processing_state)

        # Manual checkpoint every 10 documents
        if (i + 1) % 10 == 0:
            completion_pct = ((i + 1) / total_docs) * 100
            print(f"   💾 Checkpoint: {completion_pct:.1f}% complete")

    # Final statistics
    total_time = asyncio.get_event_loop().time() - processing_state["start_time"]

    print(f"✅ Corpus processing complete:")
    print(f"   📊 Processed: {processing_state['processed_count']}")
    print(f"   ❌ Failed: {processing_state['failed_count']}")
    print(f"   ⏱️ Time: {total_time:.1f}s")

@state
async def save_progress_report(context):
    """Save processing progress to disk"""
    print("💾 Saving progress report...")

    corpus_state = context.get_variable("corpus_state")
    outputs = context.get_output_keys()

    report = {
        "corpus_processing": corpus_state,
        "metrics": {key: context.get_output(key) for key in outputs},
        "timestamp": asyncio.get_event_loop().time()
    }

    with open("rag_progress_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Progress report saved")
\`\`\`

**Checkpointing benefits:**
- Resume processing after interruptions
- Progress tracking for long operations
- Persistent state management
- Resource usage optimization

## 5. Multi-Agent Coordination

Finally, let's scale with multiple coordinated agents for parallel processing:

\`\`\`python
from puffinflow.core.coordination.primitives import Semaphore, Barrier, Mutex

# Coordination primitives
embedding_semaphore = Semaphore("embedding_api", max_count=3)
vector_store_mutex = Mutex("vector_store_writes")
processing_barrier = Barrier("sync_point", parties=3)

# Specialized agents
coordinator_agent = Agent("document-coordinator")
embedding_agent_1 = Agent("embedding-worker-1")
embedding_agent_2 = Agent("embedding-worker-2")

# Shared workspace
class DocumentWorkspace:
    def __init__(self):
        self.document_queue = []
        self.processed_docs = []
        self.failed_docs = []

    async def add_documents(self, docs):
        self.document_queue.extend(docs)

    async def get_next_batch(self, batch_size=5):
        if len(self.document_queue) >= batch_size:
            batch = self.document_queue[:batch_size]
            self.document_queue = self.document_queue[batch_size:]
            return batch
        return []

workspace = DocumentWorkspace()

@state
async def coordinate_processing(context):
    """Coordinate document distribution"""
    print("📋 Coordinator: Distributing documents...")

    # Discover documents
    docs = []  # Your document discovery logic here
    await workspace.add_documents(docs)

    context.set_output("total_documents", len(docs))
    print(f"📤 Distributed {len(docs)} documents")

@state(rate_limit=8.0, max_retries=3)
async def embedding_worker(context):
    """Worker for processing embeddings"""
    agent_name = context.get_variable("agent_name", "worker")
    print(f"🔢 {agent_name}: Starting...")

    embeddings = context.get_variable("embeddings")
    processed_count = 0

    while True:
        # Get work batch
        doc_batch = await workspace.get_next_batch(batch_size=3)
        if not doc_batch:
            break

        print(f"   📦 {agent_name}: Processing {len(doc_batch)} documents")

        for doc_info in doc_batch:
            try:
                # Rate-limited embedding creation
                print(f"   🔒 {agent_name}: Requesting API access...")
                await embedding_semaphore.acquire(agent_name)

                try:
                    # Process document (your logic here)
                    print(f"   ✅ {agent_name}: Processed document")
                    processed_count += 1

                finally:
                    await embedding_semaphore.release(agent_name)

                # Exclusive vector store access
                await vector_store_mutex.acquire(agent_name)
                try:
                    # Store embeddings (your logic here)
                    pass
                finally:
                    await vector_store_mutex.release(agent_name)

            except Exception as e:
                print(f"   ❌ {agent_name}: Failed: {e}")

    context.set_output(f"{agent_name}_processed", processed_count)
    print(f"🏁 {agent_name}: Completed {processed_count} documents")

    # Wait for other workers
    await processing_barrier.wait(agent_name)
    print(f"🚀 {agent_name}: All workers complete!")

# Build multi-agent workflow
coordinator_agent.add_state("coordinate_processing", coordinate_processing)

embedding_agent_1.add_state("initialize_system", initialize_system)
embedding_agent_1.add_state("embedding_worker",
    lambda ctx: embedding_worker({**ctx.shared_state, "agent_name": "worker-1"}),
    dependencies=["initialize_system"])

embedding_agent_2.add_state("initialize_system", initialize_system)
embedding_agent_2.add_state("embedding_worker",
    lambda ctx: embedding_worker({**ctx.shared_state, "agent_name": "worker-2"}),
    dependencies=["initialize_system"])

async def run_multi_agent_system():
    """Run coordinated multi-agent RAG system"""
    print("🤝 Starting multi-agent RAG system...")

    tasks = [
        asyncio.create_task(coordinator_agent.run()),
        asyncio.create_task(embedding_agent_1.run()),
        asyncio.create_task(embedding_agent_2.run())
    ]

    await asyncio.gather(*tasks)
    print("🎉 Multi-agent processing complete!")
\`\`\`

**Multi-agent benefits:**
- Parallel processing for better throughput
- Coordinated resource access with semaphores/mutexes
- Load balancing across workers
- Synchronized completion with barriers

## Key Takeaways

This tutorial showed how Puffinflow transforms a basic RAG system into a production-ready solution:

| Feature | Basic | Enhanced | Production | Multi-Agent |
|---------|--------|----------|------------|-------------|
| **Security** | ❌ Hardcoded keys | ✅ \`context.set_secret()\` | ✅ Secure storage | ✅ Shared secrets |
| **Error Handling** | ❌ Basic | ✅ Try/catch | ✅ Retry policies | ✅ Isolated failures |
| **Rate Limiting** | ❌ None | ❌ None | ✅ \`@state(rate_limit)\` | ✅ Coordinated limits |
| **Checkpointing** | ❌ None | ❌ None | ✅ Auto-resume | ✅ Distributed state |
| **Scalability** | ❌ Single-threaded | ✅ Re-ranking | ✅ Robust processing | ✅ Parallel workers |

**Production-ready features achieved:**
- 🔐 Secure API key management with \`context.set_secret()\`
- 🚦 Rate limiting to prevent quota exhaustion
- 🛡️ Comprehensive error handling and retries
- 💾 Checkpointing for long-running processes
- 🤝 Multi-agent coordination for scalability
- 📊 Rich metrics and monitoring
- 🔄 Re-ranking for improved retrieval quality

Start with the basic implementation and incrementally add these features as your RAG system grows in complexity and scale!
`.trim();
