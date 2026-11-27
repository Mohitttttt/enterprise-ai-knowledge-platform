"""
Controlled synonym expansion for BM25 keyword retrieval.
"""

SYNONYMS = {
    "laptop": ["laptops", "device", "devices"],
    "monitor": ["monitors"],
    "stops": ["problem", "problems", "malfunction"],
    "breaks": [
        "problem",
        "problems",
        "malfunction",
        "repair",
        "replacement",
    ],
    "issue": ["problem", "problems"],
    "reporting": ["report", "submit", "submission"],
}


def expand_tokens(original_tokens: list[str]) -> list[str]:
    """
    Expand tokenized queries using controlled synonyms.

    Preserves all original tokens, including repetitions.
    Appends synonyms only once.
    """

    expanded_tokens = list(original_tokens)
    existing_tokens = set(original_tokens)

    for token in original_tokens:
        for synonym in SYNONYMS.get(token, []):

            if synonym not in existing_tokens:
                expanded_tokens.append(synonym)
                existing_tokens.add(synonym)

    return expanded_tokens
