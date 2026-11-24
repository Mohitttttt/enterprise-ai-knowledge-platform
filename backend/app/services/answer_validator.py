import re
from typing import Any


UNSUPPORTED_ANSWER = (
    "I couldn't find sufficient information "
    "in the uploaded documents."
)


PROMPT_LEAKAGE_PATTERNS = [
    r"CONVERSATION HISTORY:",
    r"DOCUMENT CONTEXT:",
    r"LATEST USER QUESTION:",
    r"^ANSWER:",
    r"you are an enterprise knowledge assistant",
    r"based on the supplied document context",
    r"i will provide the answer directly",
]


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "do", "does", "did", "how", "what", "when",
    "where", "who", "why", "which", "can", "could",
    "would", "should", "i", "you", "we", "they",
    "he", "she", "it", "them", "their", "this",
    "that", "to", "of", "in", "on", "for", "with",
    "and", "or", "but", "as", "by", "from",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def is_empty_answer(answer: str | None) -> bool:
    return not answer or not answer.strip()


def contains_prompt_leakage(answer: str) -> bool:
    for pattern in PROMPT_LEAKAGE_PATTERNS:
        if re.search(
            pattern,
            answer,
            flags=re.IGNORECASE | re.MULTILINE,
        ):
            return True

    return False


def calculate_grounding_score(
    answer: str,
    context: str,
) -> float:
    """
    Calculate lexical overlap between answer and document context.

    This is a lightweight heuristic, not semantic fact verification.
    """

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)

    if not answer_tokens or not context_tokens:
        return 0.0

    matched_tokens = answer_tokens & context_tokens

    return len(matched_tokens) / len(answer_tokens)


def validate_answer(
    answer: str | None,
    context: str,
    minimum_grounding_score: float = 0.35,
) -> dict[str, Any]:
    """
    Validate an answer before returning it to the user.
    """

    if not 0 <= minimum_grounding_score <= 1:
        raise ValueError(
            "minimum_grounding_score must be between 0 and 1."
        )

    if is_empty_answer(answer):
        return {
            "is_valid": False,
            "reason": "empty_answer",
            "grounding_score": 0.0,
            "answer": UNSUPPORTED_ANSWER,
        }

    answer = answer.strip()

    if contains_prompt_leakage(answer):
        return {
            "is_valid": False,
            "reason": "prompt_leakage",
            "grounding_score": 0.0,
            "answer": UNSUPPORTED_ANSWER,
        }

    grounding_score = calculate_grounding_score(
        answer=answer,
        context=context,
    )

    if grounding_score < minimum_grounding_score:
        return {
            "is_valid": False,
            "reason": "insufficient_grounding",
            "grounding_score": round(grounding_score, 4),
            "answer": UNSUPPORTED_ANSWER,
        }

    return {
        "is_valid": True,
        "reason": None,
        "grounding_score": round(grounding_score, 4),
        "answer": answer,
    }