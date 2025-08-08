"""Prompt formatting utilities for RAG generation."""

from typing import Any


def format_rag_prompt(query: str, context: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build system+user messages from query and retrieved context.

    Args:
        query: User question
        context: Retrieved chunks with fields like content, title, chunk_index, similarity

    Returns:
        Messages suitable for LLMService.generate_response
    """
    # Prepare context blocks
    context_texts: list[str] = []
    for i, doc in enumerate(context, 1):
        content = doc.get("content")
        if not content or not content.strip():
            continue

        idx = doc.get("chunk_index", i)
        title = doc.get("title", "Unknown")
        sim = doc.get("similarity")
        sim_str = f" (Similarity: {sim:.3f})" if isinstance(sim, int | float) else ""

        header = f"[Chunk {idx} from '{title}'{sim_str}]"
        context_texts.append(f"{header}\n{content}")

    combined_context = "\n\n---\n\n".join(context_texts)

    system_prompt = (
        "You are a helpful AI assistant that answers questions based on the provided context. "
        "Use the context to answer the user's question accurately and comprehensively. "
        "If the context doesn't contain enough information to fully answer the question, say so. "
        "Always cite which document(s) and chunks you're using for your answer."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {query}"},
    ]
    return messages
