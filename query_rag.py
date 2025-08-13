"""Chat interface for the RAG system."""

import sys
from contextlib import suppress

from rag.rag_system import RAGSystem
from rag.utils.logger import logger


def display_chunks(context: list, search_mode: str, search_params: dict):
    """Display retrieved chunks with relevant metrics."""
    alpha = search_params.get("alpha")

    print("\nRetrieved chunks:")
    for i, doc in enumerate(context, 1):
        title = doc.get("title", "Unknown")
        idx = doc.get("chunk_index", i)
        meta = doc.get("metadata") or {}
        author = meta.get("author", "Unknown")

        sim = doc.get("similarity")
        ts_rank = doc.get("rank")
        fusion = doc.get("fusion_score")
        s_rank = doc.get("semantic_rank")
        k_rank = doc.get("keyword_rank")

        parts: list[str] = []
        if search_mode == "hybrid":
            if fusion is not None:
                parts.append(f"Fusion: {fusion:.4f}")
            parts.append(f"SemRank: {s_rank if s_rank is not None else '-'}")
            parts.append(f"KeyRank: {k_rank if k_rank is not None else '-'}")
            if sim is not None:
                parts.append(f"Sim: {sim:.3f}")
            if ts_rank is not None:
                parts.append(f"TS-Rank: {ts_rank:.4f}")
            if alpha is not None:
                parts.append(f"alpha={alpha}")
        elif search_mode == "semantic":
            if sim is not None:
                parts.append(f"Similarity: {sim:.3f}")
        elif ts_rank is not None:
            parts.append(f"TS-Rank: {ts_rank:.4f}")

        extra = f", {', '.join(parts)}" if parts else ""
        print(f"{i}. {title} (Chunk {idx}{extra}, Author: {author})")
        print(f"   Preview: {doc.get('content', '')[:200]}...\n")


def main():
    """Main function for the interactive chat interface."""
    print("====================== RAG System ======================")

    # Initialize and check RAG system
    try:
        rag = RAGSystem()
        stats = rag.get_stats()
        doc_count = stats.get("total_documents", 0)
        chunk_count = stats.get("total_chunks", 0)
        if doc_count == 0:
            print("No documents found. Please run build_db.py first.")
            sys.exit(1)
        logger.info(f"Vector database: ({doc_count} documents, {chunk_count} chunks)")
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        sys.exit(1)

    print("\nCommands: 'stats', 'quit', 'exit', 'q'")
    print("\nExample questions:")
    print("What is the main benefit of contextual retrieval?")
    print("What is the biggest advantage of using Docling for document processing?")

    # Interactive query loop
    while True:
        print("\n" + "-" * 50)
        query = input("\nYour question: ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye! 👋")
            logger.info("RAG system shutting down")
            rag.close()
            break

        if query.lower() == "stats":
            stats = rag.get_stats()
            print("\nStatistics:")
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            continue

        if not query:
            print("Please enter a question.")
            continue

        try:
            # Query the system
            result = rag.query(query)

            print("\nResponse:")
            print(result["response"])

            context = result.get("context", [])
            if not context:
                continue

            if input("\nShow retrieved chunks? (y/n): ").strip().lower() == "y":
                search_mode = result.get("search_mode", "")
                search_params = result.get("search_params", {})
                display_chunks(context, search_mode, search_params)
        except Exception as e:
            logger.error(f"Error processing query: {e}")

    # Clean up
    with suppress(Exception):
        rag.close()


if __name__ == "__main__":
    main()
