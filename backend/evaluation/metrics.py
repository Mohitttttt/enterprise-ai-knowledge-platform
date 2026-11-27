import re
from typing import Any


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate the proportion of relevant results in the top K."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        return 0.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1 for item in retrieved
        if item in relevant_ids
    )

    return relevant_count / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate the proportion of relevant results retrieved in the top K."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    relevant_retrieved = retrieved.intersection(
        relevant_ids
    )

    return len(relevant_retrieved) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Calculate the reciprocal rank of the first relevant result."""

    if not relevant_ids:
        return 0.0

    for rank, item in enumerate(retrieved_ids, start=1):

        if item in relevant_ids:
            return 1.0 / rank

    return 0.0


def normalize_content(text: str) -> str:
    """Normalize text for exact duplicate detection."""

    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def duplicate_rate(
    results: list[dict[str, Any]],
) -> float:
    """Calculate the proportion of results that duplicate earlier content."""

    if not results:
        return 0.0

    seen_content = set()
    duplicate_count = 0

    for result in results:

        content = normalize_content(
            result.get("document", "")
        )

        if not content:
            continue

        if content in seen_content:
            duplicate_count += 1
        else:
            seen_content.add(content)

    return duplicate_count / len(results)


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Relevance grades:
        0 = irrelevant
        1 = partially relevant
        2 = highly relevant
        3 = directly relevant
    """

    import math

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if any(
        not isinstance(grade, int) or isinstance(grade, bool)
        or grade < 0
        for grade in relevance_grades.values()
    ):
        raise ValueError("Relevance grades must be non-negative integers.")

    def calculate_dcg(grades: list[int]) -> float:
        score = 0.0

        for rank, grade in enumerate(grades, start=1):
            gain = (2 ** grade) - 1
            discount = math.log2(rank + 1)
            score += gain / discount

        return score

    retrieved_grades = [
        relevance_grades.get(chunk_id, 0)
        for chunk_id in retrieved_ids[:k]
    ]

    ideal_grades = sorted(
        relevance_grades.values(),
        reverse=True,
    )[:k]

    actual_dcg = calculate_dcg(retrieved_grades)
    ideal_dcg = calculate_dcg(ideal_grades)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg
