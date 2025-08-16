"""Main RAG system that orchestrates the entire pipeline."""

import warnings
from typing import Any

from rag.config.settings import get_settings
from rag.core.document_processor import DocumentProcessor
from rag.core.embedding_service import EmbeddingService
from rag.core.llm_service import LLMService
from rag.core.prompt_builder import format_rag_prompt
from rag.core.reranker_service import RerankerService
from rag.core.vector_store import VectorStore
from rag.utils.logger import logger

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
        self.reranker_service = RerankerService()

        # In-memory cache for query embeddings
        self._query_embedding_cache: dict[str, list[float]] = {}

        logger.info("RAG system initialized")
        logger.info("Vector database connection established")
        logger.info("In-memory query cache ready")

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
        print(
            f"Document processed: {stats['total_chunks']} chunks,",
            f"{stats['avg_tokens_per_chunk']:.1f} avg tokens per chunk\n",
        )

    def ingest_multiple_documents(self, sources: list[dict[str, Any]]) -> None:
        """
        Ingest multiple documents in batch. Skip existing sources.

        Args:
            sources: List of dicts with 'source', optional 'title', and optional 'metadata' keys
        """
        # Filter existing sources
        existing = self.vector_store.get_existing_sources([s["source"] for s in sources])
        new_sources = [s for s in sources if s["source"] not in existing]

        logger.info(f"Batch processing: {len(existing)} skipped, {len(new_sources)} to process")
        if not new_sources:
            return

        processed = 0
        for i, source_info in enumerate(new_sources, 1):
            source = source_info["source"]
            doc_title = source_info.get("title")
            doc_metadata = source_info.get("metadata")

            print(f"\n[{i}/{len(new_sources)}] Processing: {source}")

            try:
                self.ingest_document(source=source, title=doc_title, metadata=doc_metadata)
                processed += 1
            except Exception as e:
                logger.error(f"Error processing {source}: {e}")
                continue

        logger.info(
            f"{processed}/{len(new_sources)} succeeded, {len(new_sources) - processed} failed,"
            f" {len(existing)}/{len(sources)} skipped"
        )

    def query(
        self,
        question: str,
        k: int | None = None,
        threshold: float | None = None,
        search_mode: str = "semantic",
        alpha: float | None = None,
        use_reranking: bool = False,
    ) -> dict[str, Any]:
        """
        Main query method that combines retrieval and generation.

        Args:
            question: The user's question
            k: Number of documents to retrieve (final count for generation)
            threshold: Similarity threshold for semantic search
            search_mode: Search strategy ("semantic", "keyword", "hybrid")
            alpha: Weight for hybrid search (0.0 = pure keyword, 1.0 = pure semantic)
            use_reranking: Whether to apply reranking to improve relevance

        Returns:
            Dictionary containing response, context, metadata, and search info

        Raises:
            ValueError: If search_mode is not one of the supported modes
        """
        # Set defaults from settings if not provided
        settings = get_settings()
        top_k = k or settings.top_k_results
        threshold = threshold or settings.semantic_threshold
        alpha = alpha or settings.hybrid_search_alpha

        # Reranking expansion - fetch more documents if enabled
        initial_k = settings.pre_rerank_target if use_reranking else top_k

        logger.debug(f"Processing query with {search_mode} search: {question}")
        if use_reranking:
            logger.debug(f"Reranking enabled: fetching {initial_k} docs, will rerank to {top_k}")

        # Route to appropriate search method (receives initial_k)
        if search_mode == "semantic":
            context = self.semantic_search(question, initial_k, threshold)
        elif search_mode == "keyword":
            context = self.keyword_search(question, initial_k)
        elif search_mode == "hybrid":
            context = self.hybrid_search(question, initial_k, threshold, alpha)
        else:
            raise ValueError(
                f"Invalid search_mode: '{search_mode}'. "
                f"Supported modes: 'semantic', 'keyword', 'hybrid'"
            )

        # Apply reranking if enabled to get top_k results
        if use_reranking and context:
            logger.debug(f"Applying reranking: {len(context)} → {top_k}")
            context = self.reranker_service.rerank_documents(question, context, top_k)
        else:
            logger.debug("Reranking disabled or no context available")

        # Base result template
        result = {
            "response": "",
            "context": context or [],
            "context_count": len(context) if context else 0,
            "search_params": {
                "top_k": top_k,
                "search_mode": search_mode,
                "use_reranking": use_reranking,
            },
        }

        if not context:
            logger.warning("No relevant context found")  # RAG-level decision
            result["response"] = "I couldn't find any relevant information to answer your question."
            return result

        logger.debug(f"Found {len(context)} relevant chunks")  # RAG-level result

        # Generate response
        logger.debug("Generating response...")
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

    def semantic_search(self, query: str, k: int, threshold: float) -> list[dict[str, Any]]:
        """
        Semantic search using embeddings similarity (inner product).

        Args:
            query: The user's question
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of semantically similar chunks with metadata
        """
        query_embedding = self.get_query_embedding(query)
        logger.debug("Performing semantic search...")
        return self.vector_store.semantic_search(
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
        logger.debug("Performing keyword search...")
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
        logger.debug(f"Performing hybrid search (α={alpha:.1f})...")

        # Get more results from each method for better RRF fusion quality
        fusion_k = max(k * 2, 30)  # Fixed expansion for fusion quality

        # Semantic search
        semantic_results = self.vector_store.semantic_search(
            query_embedding=query_embedding, k=fusion_k, threshold=threshold
        )

        # Keyword search
        keyword_results = self.vector_store.keyword_search(query=query, k=fusion_k)

        # Apply RRF fusion
        fused_results = self._apply_rrf_fusion(semantic_results, keyword_results, alpha)

        logger.debug(f"🔀 Semantic: {len(semantic_results)} chunks")
        logger.debug(f"🔀 Keyword: {len(keyword_results)} chunks")
        logger.debug(f"🎯 Fused {len(fused_results)} chunks")

        # Return top k results
        return fused_results[:k]

    def _apply_rrf_fusion(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        alpha: float,
    ) -> list[dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion algorithm to combine semantic and keyword search results.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            alpha: Weight for semantic vs keyword (0.0 = pure keyword, 1.0 = pure semantic)

        Returns:
            List of ALL chunks ranked by fused score (caller handles truncation)
        """

        # Create rank maps for fusion (1-indexed ranking)
        semantic_ranks = {result["id"]: i + 1 for i, result in enumerate(semantic_results)}
        keyword_ranks = {result["id"]: i + 1 for i, result in enumerate(keyword_results)}

        # Cache result lengths for fallback ranks
        semantic_len = len(semantic_results)
        keyword_len = len(keyword_results)

        # Collect all unique chunks [deduplication by id]
        all_chunks = {}
        for result in semantic_results:
            all_chunks[result["id"]] = result

        for result in keyword_results:
            if result["id"] not in all_chunks:
                all_chunks[result["id"]] = result

        # Apply Reciprocal Rank Fusion (RRF)
        fused_results = []

        for chunk_id, chunk in all_chunks.items():
            # Get ranks (use actual result count + 1 for items not found in that search type)
            semantic_rank = semantic_ranks.get(chunk_id, semantic_len + 1)
            keyword_rank = keyword_ranks.get(chunk_id, keyword_len + 1)

            # RRF formula with k=60 (standard value from literature)
            rrf_score = alpha * (1 / (semantic_rank + 60)) + (1 - alpha) * (1 / (keyword_rank + 60))

            # Create result with fusion metadata
            chunk_copy = dict(chunk)
            chunk_copy["fusion_score"] = rrf_score
            chunk_copy["semantic_rank"] = semantic_rank if semantic_rank <= semantic_len else None
            chunk_copy["keyword_rank"] = keyword_rank if keyword_rank <= keyword_len else None

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
