import re
from decimal import Decimal
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


VALIDATION_SYNONYMS = {
    "staff": "employees",
    "entitled": "receive",
    "yearly": "annual",
    "vacation": "leave",
}


NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|neither|nor|without|cannot|can't|"
    r"isn't|aren't|wasn't|weren't|don't|doesn't|didn't|"
    r"hasn't|haven't|hadn't|won't|wouldn't|shouldn't|"
    r"couldn't|mustn't)\b",
    flags=re.IGNORECASE,
)


# Recognised unit names and their canonical representations.
UNIT_ALIASES = {
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "hour": "hour",
    "hours": "hour",
    "minute": "minute",
    "minutes": "minute",
    "second": "second",
    "seconds": "second",

    "euro": "EUR",
    "euros": "EUR",
    "eur": "EUR",
    "dollar": "USD",
    "dollars": "USD",
    "usd": "USD",
    "pound": "GBP",
    "pounds": "GBP",
    "gbp": "GBP",

    "percent": "%",
    "percentage": "%",

    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
}


def tokenize(text: str) -> set[str]:
    """Convert text into normalised tokens."""

    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        VALIDATION_SYNONYMS.get(word, word)
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    }



def extract_numeric_tokens(text: str) -> set[str]:
    """Extract and normalise numeric values from text."""

    # Remove ISO-formatted dates before extracting numbers.
    text = re.sub(
        r"\b\d{4}-\d{2}-\d{2}\b",
        " ",
        text,
    )

    matches = re.findall(
        r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?(?:\s*(?:%|percent(?:age)?\b))?(?!\w)",
        text,
        flags=re.IGNORECASE,
    )

    numbers = set()

    for match in matches:
        is_percentage = bool(
            re.search(
                r"(?:%|percent(?:age)?)$",
                match,
                flags=re.IGNORECASE,
            )
        )

        value = re.sub(
            r"\s*(?:%|percent(?:age)?)$",
            "",
            match,
            flags=re.IGNORECASE,
        ).replace(",", "")

        normalised = format(
            Decimal(value).normalize(),
            "f",
        )

        if is_percentage:
            normalised += "%"

        numbers.add(normalised)

    return numbers


def extract_numeric_unit_pairs(
    text: str,
) -> set[tuple[str, str]]:
    """
    Extract numeric values together with recognised units.

    Examples:
        20 days -> ("20", "day")
        500 euros -> ("500", "EUR")
        $500 -> ("500", "USD")
        25% -> ("25", "%")
    """

    # Remove ISO-formatted dates before extracting values.
    text = re.sub(
        r"\b\d{4}-\d{2}-\d{2}\b",
        " ",
        text,
    )

    unit_alternatives = "|".join(
        re.escape(unit)
        for unit in sorted(
            UNIT_ALIASES,
            key=len,
            reverse=True,
        )
    )

    pattern = re.compile(
        rf"(?P<prefix>€|£|\$)?\s*"
        rf"(?P<number>[+-]?(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)"
        rf"(?:\s*(?P<suffix>"
        rf"%|(?<![A-Za-z])(?:{unit_alternatives})(?![A-Za-z])"
        rf"))?",
        flags=re.IGNORECASE,
    )

    pairs = set()

    for match in pattern.finditer(text):
        prefix = match.group("prefix")
        suffix = match.group("suffix")

        unit = None

        if suffix:
            if suffix == "%":
                unit = "%"
            else:
                unit = UNIT_ALIASES.get(suffix.lower())

        elif prefix:
            unit = {
                "€": "EUR",
                "$": "USD",
                "£": "GBP",
            }[prefix]

        if unit is None:
            continue

        number = match.group("number").replace(",", "")

        normalised_number = format(
            Decimal(number).normalize(),
            "f",
        )

        pairs.add((normalised_number, unit))

    return pairs


def has_unsupported_numbers(
    answer: str,
    context: str,
) -> bool:
    """Check whether the answer introduces unsupported numeric values."""

    answer_numbers = extract_numeric_tokens(answer)
    context_numbers = extract_numeric_tokens(context)

    return not answer_numbers.issubset(context_numbers)


def split_into_claims(text: str) -> list[str]:
    """Split text into sentences and coordinated clauses."""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    claims = []

    for sentence in sentences:
        clauses = re.split(
            r"(?:,\s*|\s+)(?:and|but)\s+",
            sentence,
            flags=re.IGNORECASE,
        )

        claims.extend(
            clause.strip()
            for clause in clauses
            if clause.strip()
        )

    return claims


def contains_negation(text: str) -> bool:
    """Detect common explicit negation forms."""

    return bool(NEGATION_PATTERN.search(text))


def has_unsupported_claims(
    answer: str,
    context: str,
    minimum_coverage: float = 0.65,
) -> bool:
    """
    Check whether individual answer claims have supporting evidence.

    Evidence with matching explicit negation is prioritised.
    Numeric values and recognised units are checked against
    their corresponding evidence sentence.
    """

    claims = split_into_claims(answer)
    evidence_sentences = split_into_claims(context)

    for claim in claims:
        claim_tokens = tokenize(claim)

        if not claim_tokens:
            continue

        claim_is_negated = contains_negation(claim)

        best_coverage = 0.0
        best_evidence = ""

        best_polarity_coverage = -1.0
        best_polarity_evidence = ""

        for evidence_sentence in evidence_sentences:
            evidence_tokens = tokenize(evidence_sentence)

            coverage = (
                len(claim_tokens & evidence_tokens)
                / len(claim_tokens)
            )

            if coverage > best_coverage:
                best_coverage = coverage
                best_evidence = evidence_sentence

            evidence_is_negated = contains_negation(
                evidence_sentence
            )

            if (
                evidence_is_negated == claim_is_negated
                and coverage > best_polarity_coverage
            ):
                best_polarity_coverage = coverage
                best_polarity_evidence = evidence_sentence

        # Prefer evidence with matching negation.
        if best_polarity_evidence:
            best_coverage = best_polarity_coverage
            best_evidence = best_polarity_evidence

        if best_coverage < minimum_coverage:
            return True

        # Numeric values must be supported by this specific evidence.
        if not extract_numeric_tokens(claim).issubset(
            extract_numeric_tokens(best_evidence)
        ):
            return True

        # Recognised numeric units must also match the evidence.
        if not extract_numeric_unit_pairs(claim).issubset(
            extract_numeric_unit_pairs(best_evidence)
        ):
            return True

        # Reject claims whose explicit negation contradicts the evidence.
        if contains_negation(claim) != contains_negation(best_evidence):
            return True

    return False


def is_empty_answer(answer: str | None) -> bool:
    return not answer or not answer.strip()


def contains_prompt_leakage(answer: str) -> bool:
    """Detect accidental exposure of internal prompt instructions."""

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
    """Validate an answer before returning it to the user."""

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

    if has_unsupported_numbers(answer, context):
        return {
            "is_valid": False,
            "reason": "unsupported_numbers",
            "grounding_score": round(grounding_score, 4),
            "answer": UNSUPPORTED_ANSWER,
        }

    if has_unsupported_claims(answer, context):
        return {
            "is_valid": False,
            "reason": "unsupported_claim",
            "grounding_score": round(grounding_score, 4),
            "answer": UNSUPPORTED_ANSWER,
        }

    return {
        "is_valid": True,
        "reason": None,
        "grounding_score": round(grounding_score, 4),
        "answer": answer,
    }
