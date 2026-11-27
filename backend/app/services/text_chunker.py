import re
from typing import Any


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def _is_numbered_heading(line: str) -> bool:
    """
    Identify numbered headings.

    Examples:
        1. ANNUAL LEAVE
        2) SICK LEAVE
        3.1 IT SUPPORT
    """

    return bool(
        re.match(
            r"^\s*\d+(?:\.\d+)*(?:[.)])?\s+\S",
            line,
        )
    )


def _find_best_boundary(
    text: str,
    start: int,
    target_end: int,
) -> int:
    """
    Find a natural text boundary near the target position.

    Priority:
    1. Line boundary, unless it separates a heading
       from its associated content.
    2. Sentence boundary.
    3. Word boundary.
    4. Exact target position.
    """

    if target_end >= len(text):
        return len(text)

    minimum_end = start + int(
        (target_end - start) * 0.70
    )

    search_region = text[minimum_end:target_end]

    # Prefer line boundaries, but never end a chunk
    # immediately after a numbered heading.
    line_matches = list(
        re.finditer(r"\n+", search_region)
    )

    valid_line_matches = []

    for match in line_matches:
        boundary = minimum_end + match.start()

        previous_line_start = (
            text.rfind("\n", start, boundary) + 1
        )

        previous_line = text[
            previous_line_start:boundary
        ].strip()

        if not _is_numbered_heading(previous_line):
            valid_line_matches.append(match)

    if valid_line_matches:
        return (
            minimum_end
            + valid_line_matches[-1].end()
        )

    # Prefer sentence boundaries.
    # Avoid interpreting numbered headings such as
    # "2. SICK LEAVE" as sentence endings.
    sentence_matches = list(
        re.finditer(
            r"""(?<!\d)[.!?](?:["']?)\s+""",
            search_region,
        )
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

    Attempts to start at a word boundary.

    If the overlap begins inside a numbered heading,
    move the starting position to the beginning
    of that heading.
    """

    if chunk_overlap == 0:
        return chunk_end

    next_start = max(
        0,
        chunk_end - chunk_overlap,
    )

    line_start = (
        text.rfind("\n", 0, next_start) + 1
    )

    line_end = text.find(
        "\n",
        next_start,
    )

    if line_end == -1:
        line_end = len(text)

    current_line = text[line_start:line_end]

    if (
        line_start < next_start
        and _is_numbered_heading(current_line)
    ):
        return line_start

    # Otherwise, start at the next word boundary.
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

    Preserves:
    - Numbered headings where possible.
    - Line and sentence boundaries.
    - Word boundaries.
    - Overlapping context between chunks.

    Does not discard short intermediate chunks.

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

    # Normalize line endings.
    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # Normalize individual lines while preserving
    # meaningful line boundaries.
    lines = [
        " ".join(line.split())
        for line in text.split("\n")
    ]

    text = "\n".join(
        line
        for line in lines
        if line
    )

    if not text:
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

        # Guarantee a valid boundary.
        if end <= start:
            end = min(
                start + chunk_size,
                len(text),
            )

        chunk = text[start:end].strip()

        # Never discard non-empty chunks.
        # Small chunks can still contain important
        # headings, definitions or policy statements.
        if chunk:
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