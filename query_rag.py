"""Chat interface for the RAG system."""

import sys
from contextlib import suppress

from rag.rag_system import RAGSystem


def main():
    """Main function for the interactive chat interface."""
    print("=== RAG Query System ===")

    # Initialize and check RAG system
    try:
        print("Initializing...")
        rag = RAGSystem()
        stats = rag.get_stats()
        doc_count = stats.get("total_documents", 0)
        chunk_count = stats.get("total_chunks", 0)
        if doc_count == 0:
            print("No documents found. Please run build_db.py first.")
            sys.exit(1)
        print(f"Vector database initialized! ({doc_count} documents, {chunk_count} chunks)")
        print("✅ RAG system ready!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    print("\nCommands: 'stats', 'quit', 'exit', q")
    print("\nExample questions:")
    print("What is the main benefit of contextual retrieval?")
    print("What is the biggest advantage of using Docling for document processing?")

    # Interactive query loop
    while True:
        print("\n" + "-" * 50)
        query = input("\nYour question: ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye! 👋")
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
            print("\nProcessing request...")
            result = rag.query(query)

            print("\nResponse:")
            print(result["response"])

            # Ask if user wants to see retrieved chunks
            if result["context"]:
                show_context = input("\nShow retrieved chunks? (y/n): ").strip().lower()
                if show_context == "y":
                    print("\nRetrieved chunks:")
                    for i, doc in enumerate(result["context"], 1):
                        title = doc.get("title", "Unknown")
                        idx = doc.get("chunk_index", i)
                        sim = doc.get("similarity", 0.0)
                        meta = doc.get("metadata", {})
                        author = meta.get("author", "Unknown")

                        print(
                            f"{i}. {title} (Chunk {idx}, Similarity: {sim:.2f}, Author: {author})"
                        )
                        print(f"   Preview: {doc['content'][:200]}...")
                        print()

        except Exception as e:
            print(f"\nError processing query: {str(e)}")

    # Clean up
    with suppress(Exception):
        rag.close()


if __name__ == "__main__":
    main()
