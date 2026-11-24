import re
from typing import Any


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 100


def _find_best_boundary(
    text: str,
    start: int,
    target_end: int,
) -> int:
    """
    Find a natural text boundary near the target position.

    Priority:
    1. Paragraph boundary
    2. Sentence boundary
    3. Word boundary
    4. Exact target position
    """

    if target_end >= len(text):
        return len(text)

    minimum_end = start + int(
        (target_end - start) * 0.70
    )

    search_region = text[minimum_end:target_end]

    # Prefer paragraph boundaries.
    paragraph_matches = list(
        re.finditer(r"\n+", search_region)
    )

    if paragraph_matches:
        return (
            minimum_end
            + paragraph_matches[-1].end()
        )

    # Prefer sentence boundaries.
    sentence_matches = list(
        re.finditer(r'[.!?](?:["\']?)\s+', search_region)
    )

    if sentence_matches:
        return (
            minimum_end
            + sentence_matches[-1].end()
        )

    # Fall back to word boundaries.
    whitespace_position = text.rfind(
        " ",
        minimum_end,
        target_end,
    )

    if whitespace_position > start:
        return whitespace_position + 1

    return target_end


def _find_overlap_start(
    text: str,
    chunk_end: int,
    chunk_overlap: int,
) -> int:
    """
    Calculate the next chunk's starting position.

    Attempts to begin the overlap at a word boundary.
    """

    if chunk_overlap == 0:
        return chunk_end

    next_start = max(
        0,
        chunk_end - chunk_overlap,
    )

    whitespace_position = text.find(
        " ",
        next_start,
        chunk_end,
    )

    if whitespace_position != -1:
        return whitespace_position + 1

    return next_start


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split document text into structure-aware overlapping chunks.

    Attempts to preserve paragraphs, sentences and headings.

    Returns:
        A list of dictionaries containing:
        - chunk_index
        - text
        - start_char
        - end_char
        - character_count
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative "
            "and smaller than chunk size."
        )

    # Normalize line endings while preserving paragraph structure.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unnecessary spaces without removing paragraph boundaries.
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in text.split("\n")
    ]

    text = "\n".join(
        paragraph
        for paragraph in paragraphs
        if paragraph
    )

    if not text.strip():
        return []

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(text):

        target_end = min(
            start + chunk_size,
            len(text),
        )

        end = _find_best_boundary(
            text=text,
            start=start,
            target_end=target_end,
        )

        # Prevent invalid or excessively small chunks.
        if end <= start:
            end = min(
                start + chunk_size,
                len(text),
            )

        chunk = text[start:end].strip()

        if chunk and (
            len(chunk) >= MIN_CHUNK_SIZE
            or not chunks
            or end == len(text)
        ):
            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk,
                "start_char": start,
                "end_char": end,
                "character_count": len(chunk),
            })

            chunk_index += 1

        if end >= len(text):
            break

        next_start = _find_overlap_start(
            text=text,
            chunk_end=end,
            chunk_overlap=chunk_overlap,
        )

        # Guarantee forward progress.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks
