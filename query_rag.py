"""Chat interface for the RAG system."""

import sys
from contextlib import suppress

from app.src.rag_system import RAGSystem


def main():
    """Main function for the interactive chat interface."""
    print("=== RAG Pipeline Query System ===\n")

    # Initialize RAG system
    print("Initializing RAG system...")
    try:
        rag = RAGSystem()
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        sys.exit(1)

    # Check if database has documents
    doc_count = rag.vector_store.get_document_count()
    if doc_count == 0:
        print("No documents found in database. Please run build_db.py first.")
        sys.exit(1)

    chunk_count = rag.vector_store.get_chunk_count()
    print(f"✅ RAG system ready! ({doc_count} documents, {chunk_count} chunks)")

    print("\nCommands: 'stats', 'quit' or 'exit'")
    print("\nExample questions:")
    print("- What is considered ethical AI at CBA?")
    print("- What is the main benefit of contextual retrieval?")
    print("- What is the main benefit of generative AI?")
    print("- What is the main benefit of RAG?")

    # Interactive query loop
    while True:
        print("\n" + "-" * 50)
        query = input("\nYour question: ").strip()

        if query.lower() in ["quit", "exit"]:
            print("\nGoodbye! 👋")
            break

        if query.lower() == "stats":
            stats = rag.get_stats()
            print("\nRAG System Statistics:")
            for key, value in stats.items():
                print(f"- {key.replace('_', ' ').title()}: {value}")
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

            # Ask if user wants to see the retrieved context
            if result["context"]:
                show_context = input("\nShow retrieved context? (y/n): ").strip().lower()
                if show_context == "y":
                    print("\n📚 Retrieved Context:")
                    for i, doc in enumerate(result["context"], 1):
                        doc_title = doc.get("title", "Unknown Document")
                        chunk_idx = doc.get("chunk_index", i)
                        similarity = doc.get("similarity", 0.0)
                        metadata = doc.get("metadata") or {}
                        author = metadata.get("author", "Unknown Author")
                        print(
                            f"\n{i}. {doc_title} (Chunk {chunk_idx}, Similarity: {similarity:.3f}, Author: {author})"
                        )
                        print(f"   Preview: {doc['content'][:400]}...")

        except Exception as e:
            print(f"\nError processing query: {str(e)}")

    # Clean up
    with suppress(Exception):
        rag.close()


if __name__ == "__main__":
    main()
