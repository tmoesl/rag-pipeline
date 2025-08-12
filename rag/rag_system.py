"""Main RAG system that orchestrates the entire pipeline."""

import warnings
from typing import Any

from rag.config.settings import SIMILARITY_THRESHOLD, TOP_K_RESULTS
from rag.core.document_processor import DocumentProcessor
from rag.core.embedding_service import EmbeddingService
from rag.core.llm_service import LLMService
from rag.core.prompt_builder import format_rag_prompt
from rag.core.vector_store import VectorStore

warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")


class RAGSystem:
    """Main RAG system that orchestrates document ingestion, retrieval, and generation."""

    def __init__(self):
        """Initialize the RAG system with all pipeline components."""
        # Create single VectorStore instance as source of truth
        self.vector_store = VectorStore()

        # Core services
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()

        # In-memory cache for query embeddings
        self._query_embedding_cache: dict[str, list[float]] = {}

    def ingest_document(
        self, source: str, title: str | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Process and store a document in the vector database.

        Args:
            source: Document source (URL or file path)
            title: Optional document title
            metadata: Optional document-level metadata
        """
        # Step 1: Document extraction and chunking
        chunks = self.document_processor.process_document(source, title)

        # Step 2: Generate embeddings (batch with progress)
        chunks_with_embeddings = self.embedding_service.add_embeddings_to_chunks(chunks)

        # Step 3: Store in vector database
        document_title = title or chunks[0]["metadata"]["title"]
        self.vector_store.add_document_with_chunks(
            title=document_title,
            source=source,
            chunks=chunks_with_embeddings,
            metadata=metadata,
        )

        # Print pipeline statistics
        stats = self.document_processor.get_chunk_stats(chunks)
        print(f"\nTotal chunks: {stats['total_chunks']}")
        print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}\n")

    def ingest_multiple_documents(self, sources: list[dict[str, Any]]) -> None:
        """
        Ingest multiple documents in batch. Skip existing sources.

        Args:
            sources: List of dicts with 'source', optional 'title', and optional 'metadata' keys
        """
        # Filter existing sources
        existing = self.vector_store.get_existing_sources([s["source"] for s in sources])
        new_sources = [s for s in sources if s["source"] not in existing]

        print(f"📊 {len(existing)} skipped | {len(new_sources)} to process")
        if not new_sources:
            return

        for i, source_info in enumerate(new_sources, 1):
            source = source_info["source"]
            doc_title = source_info.get("title")
            doc_metadata = source_info.get("metadata")

            print(f"\n[{i}/{len(new_sources)}] Processing: {source}")

            try:
                self.ingest_document(source=source, title=doc_title, metadata=doc_metadata)
            except Exception as e:
                print(f"❌ Error processing {source}: {e}")
                continue

    def query(
        self,
        question: str,
        k: int = TOP_K_RESULTS,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> dict[str, Any]:
        """
        Main query method that combines retrieval and generation.

        Args:
            question: The user's question
            k: Number of documents to retrieve
            threshold: Similarity threshold

        Returns:
            Dictionary containing response, context, and metadata
        """
        # Retrieve relevant context
        context = self.retrieve_context(question, k, threshold)

        if not context:
            print("No relevant context found")  # RAG-level decision
            return {
                "response": "I couldn't find any relevant information to answer your question.",
                "context": [],
                "context_count": 0,
            }

        print(f"Found {len(context)} relevant chunks")  # RAG-level result

        # Generate response
        print("Generating response...")
        messages = format_rag_prompt(question, context)
        response = self.llm_service.generate_response(messages)

        return {
            "response": response,
            "context": context,
            "context_count": len(context),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the RAG system."""
        # Vector database and generation stats
        vector_stats = self.vector_store.get_stats()
        generation_stats = self.llm_service.get_stats()

        return {
            **vector_stats,
            "embedding_model": self.embedding_service.model,
            "embedding_dimensions": self.embedding_service.dimensions,
            **generation_stats,
        }

    def retrieve_context(self, query: str, k: int, threshold: float) -> list[dict[str, Any]]:
        """Embed the query and perform vector similarity search.

        Args:
            query: The user's question
            k: Number of documents to retrieve
            threshold: Similarity threshold

        Returns:
            List of dictionaries containing context
        """

        # Step 1: Embed the query (with in-memory cache)
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            query_embedding = cached
        else:
            query_embedding = self.embedding_service.embed_query(query)
            self._query_embedding_cache[query] = query_embedding

        # Step 2: Similarity search (using vector store)
        print("Searching for relevant context...")
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding, k=k, threshold=threshold
        )
        return results

    def clear_database(self):
        """Clear all documents and chunks from the database."""
        self.vector_store.clear_database()

    def close(self):
        """Close database connections and cleanup resources."""
        self.vector_store.close()
