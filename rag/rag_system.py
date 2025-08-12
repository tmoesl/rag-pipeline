"""Main RAG system that orchestrates the entire pipeline."""

import warnings
from typing import Any

from rag.config.settings import HYBRID_SEARCH_ALPHA, SIMILARITY_THRESHOLD, TOP_K_RESULTS
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
        search_mode: str = "similarity",
        alpha: float = HYBRID_SEARCH_ALPHA,
    ) -> dict[str, Any]:
        """
        Main query method that combines retrieval and generation.

        Args:
            question: The user's question
            k: Number of documents to retrieve
            threshold: Similarity threshold for semantic search
            search_mode: Search strategy ("similarity", "keyword", "hybrid")
            alpha: Weight for hybrid search (0.0 = pure keyword, 1.0 = pure semantic)

        Returns:
            Dictionary containing response, context, metadata, and search info

        Raises:
            ValueError: If search_mode is not one of the supported modes
        """
        print(f"\n💬 Processing query with {search_mode} search: {question}")

        # Route to appropriate search method based on search_mode
        if search_mode == "similarity":
            context = self.similarity_search(question, k, threshold)
        elif search_mode == "keyword":
            context = self.keyword_search(question, k)
        elif search_mode == "hybrid":
            context = self.hybrid_search(question, k, threshold, alpha)
        else:
            raise ValueError(
                f"Invalid search_mode: '{search_mode}'. "
                f"Supported modes: 'similarity', 'keyword', 'hybrid'"
            )

        # Base result template
        result = {
            "response": "",
            "context": context or [],
            "context_count": len(context) if context else 0,
            "search_mode": search_mode,
            "search_params": {
                "k": k,
                "threshold": threshold if search_mode in ("similarity", "hybrid") else None,
                "alpha": alpha if search_mode == "hybrid" else None,
            },
        }

        if not context:
            print("No relevant context found")  # RAG-level decision
            result["response"] = "I couldn't find any relevant information to answer your question."
            return result

        print(f"Found {len(context)} relevant chunks")  # RAG-level result

        # Generate response
        print("Generating response...")
        messages = format_rag_prompt(question, context)
        result["response"] = self.llm_service.generate_response(messages)

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the RAG system."""
        # Vector database and generation stats
        vector_stats = self.vector_store.get_stats()
        generation_stats = self.llm_service.get_stats()

        return {
            **vector_stats,
            "embedding_model": self.embedding_service.model,
            "embedding_dimensions": self.embedding_service.dimensions,
            "cached_queries": len(self._query_embedding_cache),
            **generation_stats,
        }

    def get_query_embedding(self, query: str) -> list[float]:
        """
        Get cached query embedding or create new one.

        Args:
            query: The query text to embed

        Returns:
            Query embedding vector
        """
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached

        embedding = self.embedding_service.embed_query(query)
        self._query_embedding_cache[query] = embedding
        return embedding

    def similarity_search(self, query: str, k: int, threshold: float) -> list[dict[str, Any]]:
        """
        Semantic similarity search using embeddings.

        Args:
            query: The user's question
            k: Number of documents to retrieve
            threshold: Similarity threshold

        Returns:
            List of similar chunks with metadata
        """
        query_embedding = self.get_query_embedding(query)
        print("Performing semantic similarity search...")
        return self.vector_store.similarity_search(
            query_embedding=query_embedding, k=k, threshold=threshold
        )

    def keyword_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """
        Full-text search using PostgreSQL.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of matching chunks with metadata and text search rank
        """
        print("Performing keyword search...")
        return self.vector_store.keyword_search(query=query, k=k)

    def hybrid_search(
        self, query: str, k: int, threshold: float, alpha: float
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword search with RRF fusion.

        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold for semantic search
            alpha: Weight for semantic vs keyword (0.0 = pure keyword, 1.0 = pure semantic)

        Returns:
            List of chunks ranked by fused score
        """
        query_embedding = self.get_query_embedding(query)
        print(f"Performing hybrid search (α={alpha:.1f})...")

        # Get more results from each method for better fusion
        search_k = k * 3

        # Semantic search
        semantic_results = self.vector_store.similarity_search(
            query_embedding=query_embedding, k=search_k, threshold=threshold
        )

        # Keyword search
        keyword_results = self.vector_store.keyword_search(query=query, k=search_k)

        # Apply RRF fusion
        fused_results = self._apply_rrf_fusion(semantic_results, keyword_results, search_k, alpha)

        # Return top k results
        return fused_results[:k]

    def _apply_rrf_fusion(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        search_k: int,
        alpha: float,
    ) -> list[dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion algorithm to combine semantic and keyword search results.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            search_k: Number of results fetched from each search (for fallback ranking)
            alpha: Weight for semantic vs keyword (0.0 = pure keyword, 1.0 = pure semantic)

        Returns:
            List of ALL chunks ranked by fused score (caller handles truncation)
        """

        # Create rank maps for fusion (1-indexed ranking)
        semantic_ranks = {result["id"]: i + 1 for i, result in enumerate(semantic_results)}
        keyword_ranks = {result["id"]: i + 1 for i, result in enumerate(keyword_results)}

        # Collect all unique chunks
        all_chunks = {}
        for result in semantic_results:
            all_chunks[result["id"]] = result

        for result in keyword_results:
            if result["id"] not in all_chunks:
                all_chunks[result["id"]] = result

        # Apply Reciprocal Rank Fusion (RRF)
        fused_results = []
        for chunk_id, chunk in all_chunks.items():
            # Get ranks (use search_k + 1 for items not found in that search type)
            semantic_rank = semantic_ranks.get(chunk_id, search_k + 1)
            keyword_rank = keyword_ranks.get(chunk_id, search_k + 1)

            # RRF formula with k=60 (standard value from literature)
            rrf_score = alpha * (1 / (semantic_rank + 60)) + (1 - alpha) * (1 / (keyword_rank + 60))

            # Create result with fusion metadata
            chunk_copy = dict(chunk)
            chunk_copy["fusion_score"] = rrf_score
            chunk_copy["semantic_rank"] = semantic_rank if semantic_rank <= search_k else None
            chunk_copy["keyword_rank"] = keyword_rank if keyword_rank <= search_k else None

            fused_results.append(chunk_copy)

        # Sort by fusion score (highest first) and return ALL results
        fused_results.sort(key=lambda x: x["fusion_score"], reverse=True)
        return fused_results

    def clear_database(self):
        """Clear all documents and chunks from the database."""
        self.vector_store.clear_database()

    def close(self):
        """Close database connections and cleanup resources."""
        self.vector_store.close()
