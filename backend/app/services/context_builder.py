from typing import Any


MAX_CONTEXT_LENGTH = 6000
MAX_CHUNK_LENGTH = 2000


def build_context(
    search_results: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build structured context from hybrid search results.

    Returns:
        context: Formatted document context for the LLM.
        sources: Source metadata for API responses.
    """

    if not search_results:
        return "", []

    context_parts = []
    sources = []
    current_length = 0

    for result in search_results:

        document = result.get("document", "").strip()

        if not document:
            continue

        metadata = result.get("metadata") or {}

        document = document[:MAX_CHUNK_LENGTH]

        source_number = len(sources) + 1

        filename = metadata.get(
            "original_filename",
            metadata.get("source", "Unknown"),
        )

        source_content = (
            f"[Source {source_number}]\n"
            f"Filename: {filename}\n"
            f"Content:\n{document}"
        )

        if current_length + len(source_content) > MAX_CONTEXT_LENGTH:
            break

        context_parts.append(source_content)

        source = {
            "source_number": source_number,
            "filename": filename,
            "document_id": metadata.get("document_id"),
            "chunk_index": metadata.get("chunk_index"),
            "rrf_score": round(
                float(result.get("rrf_score", 0.0)),
                6,
            ),
        }

        if result.get("distance") is not None:
            source["distance"] = round(
                float(result["distance"]),
                4,
            )

        sources.append(source)

        current_length += len(source_content)

    context = "\n\n".join(context_parts)

    return context, sources