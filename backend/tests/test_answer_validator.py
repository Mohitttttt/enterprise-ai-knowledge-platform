import pytest

from app.services.answer_validator import (
    UNSUPPORTED_ANSWER,
    calculate_grounding_score,
    contains_prompt_leakage,
    is_empty_answer,
    validate_answer,
)


# --------------------------------------------------
# TEST 1: Valid Grounded Answer
# --------------------------------------------------

def test_valid_grounded_answer():

    context = (
        "Employees receive 30 days of paid annual leave "
        "per calendar year."
    )

    answer = (
        "Employees receive 30 days of paid annual leave."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is True
    assert result["reason"] is None
    assert result["answer"] == answer
    assert result["grounding_score"] >= 0.35


# --------------------------------------------------
# TEST 2: Empty Answer
# --------------------------------------------------

def test_empty_answer():

    result = validate_answer("", "Some document context.")

    assert result["is_valid"] is False
    assert result["reason"] == "empty_answer"
    assert result["answer"] == UNSUPPORTED_ANSWER


# --------------------------------------------------
# TEST 3: Whitespace Answer
# --------------------------------------------------

def test_whitespace_answer():

    result = validate_answer("   ", "Some document context.")

    assert result["is_valid"] is False
    assert result["reason"] == "empty_answer"


# --------------------------------------------------
# TEST 4: Prompt Leakage Detection
# --------------------------------------------------

def test_prompt_leakage_detection():

    answer = (
        "CONVERSATION HISTORY:\n"
        "The previous user asked about annual leave."
    )

    assert contains_prompt_leakage(answer) is True


# --------------------------------------------------
# TEST 5: Reject Prompt Leakage
# --------------------------------------------------

def test_validator_rejects_prompt_leakage():

    context = (
        "Employees receive 30 days of paid annual leave."
    )

    answer = (
        "DOCUMENT CONTEXT:\n"
        "Employees receive 30 days of paid annual leave."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is False
    assert result["reason"] == "prompt_leakage"
    assert result["answer"] == UNSUPPORTED_ANSWER


# --------------------------------------------------
# TEST 6: Unsupported Answer
# --------------------------------------------------

def test_unsupported_answer():

    context = (
        "Employees receive 30 days of paid annual leave."
    )

    answer = (
        "The company provides free international flights "
        "and unlimited bonuses."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is False
    assert result["reason"] == "insufficient_grounding"
    assert result["answer"] == UNSUPPORTED_ANSWER


# --------------------------------------------------
# TEST 7: Grounding Score
# --------------------------------------------------

def test_calculate_grounding_score():

    answer = "Employees receive 30 days annual leave."

    context = (
        "Employees receive 30 days of paid annual leave "
        "per calendar year."
    )

    score = calculate_grounding_score(answer, context)

    assert 0.0 <= score <= 1.0
    assert score > 0.5


# --------------------------------------------------
# TEST 8: Empty Context
# --------------------------------------------------

def test_empty_context():

    result = validate_answer(
        "Employees receive 30 days of annual leave.",
        "",
    )

    assert result["is_valid"] is False
    assert result["reason"] == "insufficient_grounding"


# --------------------------------------------------
# TEST 9: Invalid Grounding Threshold
# --------------------------------------------------

@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1],
)
def test_invalid_grounding_threshold(threshold):

    with pytest.raises(ValueError):
        validate_answer(
            "Employees receive 30 days of annual leave.",
            "Employees receive 30 days of annual leave.",
            minimum_grounding_score=threshold,
        )


# --------------------------------------------------
# TEST 10: Empty Answer Helper
# --------------------------------------------------

def test_is_empty_answer():

    assert is_empty_answer("") is True
    assert is_empty_answer("   ") is True
    assert is_empty_answer(None) is True
    assert is_empty_answer("Valid answer") is False