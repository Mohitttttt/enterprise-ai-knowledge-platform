import re
from typing import Any


CONDITIONAL_MARKERS = {
    "if",
    "unless",
    "otherwise",
    "else",
    "when",
    "whenever",
    "except",
    "alternatively",
    "instead",
}

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "do", "does", "did", "how", "what", "when",
    "where", "why", "who", "which", "can", "could",
    "would", "should", "will", "may", "might",
    "to", "of", "in", "on", "at", "for", "from",
    "with", "by", "and", "or", "but", "their",
    "they", "them", "this", "that", "these",
    "those", "it", "its", "if", "then",
    "employees", "employee", "company",
}


def tokenize(text: str) -> set[str]:
    """
    Convert text into lowercase searchable tokens.

    Normalize possessive words:
    company's -> company
    employee's -> employee
    """

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b",
        text.lower(),
    )

    normalized_tokens = {
        token[:-2] if token.endswith("'s") else token
        for token in tokens
    }

    return normalized_tokens


def calculate_relevance(
    question_tokens: set[str],
    sentence_tokens: set[str],
) -> float:
    """
    Calculate relevance using meaningful question keywords.
    """

    if not question_tokens or not sentence_tokens:
        return 0.0

    overlap = question_tokens.intersection(sentence_tokens)

    return len(overlap) / len(question_tokens)


def filter_context(
    question: str,
    search_results: list[dict[str, Any]],
    minimum_relevance: float = 0.15,
) -> list[dict[str, Any]]:

    if not question or not question.strip():
        return []

    if not 0 <= minimum_relevance <= 1:
        raise ValueError(
            "minimum_relevance must be between 0 and 1."
        )

    question_tokens = tokenize(question) - STOP_WORDS

    if not question_tokens:
        return []

    filtered_results = []

    for result in search_results:

        document = result.get("document", "").strip()

        if not document:
            continue

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                document,
            )
            if sentence.strip()
        ]

        relevant_indexes = set()

        # Identify directly relevant sentences.
        for index, sentence in enumerate(sentences):

            sentence_tokens = tokenize(sentence) - STOP_WORDS

            relevance = calculate_relevance(
                question_tokens,
                sentence_tokens,
            )

            if relevance >= minimum_relevance:
                relevant_indexes.add(index)

        if not relevant_indexes:
            continue

        selected_indexes = set(relevant_indexes)

        # Preserve adjacent conditional instructions.
        for index in relevant_indexes:

            for neighbor_index in (
                index - 1,
                index + 1,
            ):

                if not 0 <= neighbor_index < len(sentences):
                    continue

                neighbor_tokens = tokenize(
                    sentences[neighbor_index]
                )

                if neighbor_tokens.intersection(
                    CONDITIONAL_MARKERS
                ):
                    selected_indexes.add(neighbor_index)

        filtered_document = " ".join(
            sentences[index]
            for index in sorted(selected_indexes)
        )

        filtered_results.append({
            **result,
            "document": filtered_document,
        })

    return filtered_results