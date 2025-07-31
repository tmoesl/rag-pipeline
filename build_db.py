"""Data ingestion script for the vector database of the RAG pipeline."""

import json
import sys
from typing import Any

from app.src.rag_system import RAGSystem


def load_documents(sources: list[str]) -> list[dict[str, Any]]:
    """Load documents from JSON files or direct paths."""
    documents = []
    for source in sources:
        if source.endswith(".json"):
            try:
                with open(source) as f:
                    data = json.load(f)
                    documents.extend(data.get("documents", []))
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error reading JSON file {source}: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            documents.append({"source": source})
    return documents


def show_usage():
    """Display usage information."""
    print("Usage:")
    print("  python build_db.py [--clear] source1 source2...  # Ingest documents")
    print("  python build_db.py [--clear] config.json         # Ingest from JSON")
    print("  python build_db.py --clear                        # Clear database only")
    print("  python build_db.py --stats                        # View statistics")
    print("\nJSON Format:")
    print('  {"documents": [{"source": "doc.pdf", "title": "...", "metadata": {...}}]}')
    print("\nExamples:")
    print("  python build_db.py doc1.pdf doc2.pdf")
    print("  python build_db.py documents.json")
    print("  python build_db.py config.json extra.pdf")
    print("  python build_db.py --clear documents.json")
    print("  python build_db.py --stats")
    print("\nOptions:")
    print("  --clear    Clear database (can be used alone or with sources)")
    print("  --stats    Show statistics only (cannot be used with sources)")


def show_statistics(rag: RAGSystem):
    """Display pipeline statistics."""
    print("=== Pipeline Statistics ===")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


def main():
    """Main function for data ingestion operations."""
    args = sys.argv[1:]

    # Show usage if no arguments or help requested
    if not args or "--help" in args or "-h" in args:
        show_usage()
        sys.exit(0)

    # Parse simple flags
    clear_db = "--clear" in args
    stats_only = "--stats" in args
    sources = [arg for arg in args if not arg.startswith("--")]

    # Validate arguments
    if stats_only and sources:
        print("Error: --stats cannot be used with sources", file=sys.stderr)
        sys.exit(1)

    if not stats_only and not sources and not clear_db:
        print("Error: No sources provided", file=sys.stderr)
        sys.exit(1)

    # Initialize RAG system
    try:
        print("=== Building Vector Database ===\n")
        print("Initializing RAG system...")
        rag = RAGSystem()
    except Exception as e:
        print(f"Error initializing RAG system: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Handle stats-only mode
        if stats_only:
            show_statistics(rag)
            return

        # Handle database clearing
        if clear_db:
            print("Clearing all documents from the database...")
            rag.clear_database()
            print("Database cleared successfully!")

        # Handle clear-only case (no sources to ingest)
        if clear_db and not sources:
            print("Database operation completed.")
            return

        # Ingest documents (only if sources provided)
        if sources:
            documents = load_documents(sources)
            print(f"Ingesting {len(documents)} document(s)...")
            rag.ingest_multiple_documents(documents)
            print("✅ Vector database built successfully")

        # Show final statistics
        print()
        show_statistics(rag)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error during data ingestion: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up resources
        rag.close()


if __name__ == "__main__":
    main()
