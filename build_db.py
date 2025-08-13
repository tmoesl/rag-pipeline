"""Data ingestion script for the vector database of the RAG pipeline."""

import json
import sys
from typing import Any

from rag.rag_system import RAGSystem
from rag.utils.logger import logger


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
                logger.error(f"Error reading JSON file {source}: {e}")
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
    print("\nOptions:")
    print("  --clear    Clear database before ingesting")
    print("  --stats    Show statistics only")


def show_statistics(rag: RAGSystem):
    """Display RAG system statistics."""
    print("\n====================== RAG System Statistics ======================")
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

    # Parse flags and sources
    clear_db = "--clear" in args
    stats_only = "--stats" in args
    sources = [arg for arg in args if not arg.startswith("--")]

    # Validate arguments
    if stats_only and sources:
        print("Error: --stats cannot be used with sources", file=sys.stderr)
        sys.exit(1)
    if not (stats_only or sources or clear_db):
        print("Error: No action specified", file=sys.stderr)
        sys.exit(1)

    # Initialize RAG system
    try:
        print("\n====================== Building Vector Database ======================")
        rag = RAGSystem()
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        sys.exit(1)

    try:
        # Handle stats-only mode
        if stats_only:
            show_statistics(rag)
            return

        # Handle database clearing
        if clear_db:
            logger.info("Clearing database...")
            rag.clear_database()
            logger.info("Database cleared")

        # Ingest documents
        if sources:
            documents = load_documents(sources)
            logger.info(f"Ingesting {len(documents)} document(s)...")
            rag.ingest_multiple_documents(documents)

        # Show statistics
        if sources or clear_db:
            print()  # Visual separation for CLI
        show_statistics(rag)

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        sys.exit(1)
    finally:
        # Clean up resources
        rag.close()


if __name__ == "__main__":
    main()
