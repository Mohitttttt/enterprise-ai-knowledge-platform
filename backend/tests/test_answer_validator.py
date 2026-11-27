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

# --------------------------------------------------
# TEST 11: Reject Unsupported Numeric Claims
# --------------------------------------------------

def test_reject_unsupported_numeric_claim():

    context = "Employees receive 20 days of paid annual leave."

    answer = "Employees receive 30 days of paid annual leave."

    result = validate_answer(answer, context)

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_numbers"
    assert result["answer"] == UNSUPPORTED_ANSWER


# --------------------------------------------------
# TEST 12: Accept Supported Numeric Claims
# --------------------------------------------------

def test_accept_supported_numeric_claim():

    context = "Employees receive 20 days of paid annual leave."

    answer = "Employees receive 20 days of paid annual leave."

    result = validate_answer(answer, context)

    assert result["is_valid"] is True


# --------------------------------------------------
# TEST 13: Reject Additional Unsupported Numbers
# --------------------------------------------------

def test_reject_additional_unsupported_number():

    context = "Employees receive 20 days of paid annual leave."

    answer = "Employees receive 20 days of paid annual leave and 500 euros."

    result = validate_answer(answer, context)

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_numbers"


# --------------------------------------------------
# TEST 14: Reject Partially Unsupported Claims
# --------------------------------------------------

def test_reject_partially_unsupported_claim():

    context = (
        "Employees receive annual leave. "
        "Parental leave policies are described separately."
    )

    answer = (
        "Employees receive annual leave, "
        "and parental leave is fully paid."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_claim"
    assert result["answer"] == UNSUPPORTED_ANSWER


# --------------------------------------------------
# TEST 15: Accept Multiple Supported Claims
# --------------------------------------------------

def test_accept_multiple_supported_claims():

    context = (
        "Employees receive annual leave. "
        "Employees may also take parental leave."
    )

    answer = (
        "Employees receive annual leave. "
        "They may also take parental leave."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is True


# --------------------------------------------------
# TEST 16: Reject Contradictory Negation
# --------------------------------------------------

def test_reject_contradictory_negation():

    context = (
        "Employees are eligible for parental leave."
    )

    answer = (
        "Employees are not eligible for parental leave."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_claim"


# --------------------------------------------------
# TEST 17: Reject Numbers Associated With Wrong Claims
# --------------------------------------------------

def test_reject_number_from_unrelated_context():

    context = (
        "Annual leave entitlement is 20 days. "
        "Expense reimbursement is limited to 500 euros."
    )

    answer = (
        "Annual leave entitlement is 500 days."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is False


# --------------------------------------------------
# TEST 18: Accept Legitimate Paraphrase
# --------------------------------------------------

def test_accept_legitimate_paraphrase():

    context = (
        "Employees receive annual leave."
    )

    answer = (
        "Staff members are entitled to yearly vacation."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is True


# --------------------------------------------------
# TEST 19: Reject Unsupported Synonym-Based Claim
# --------------------------------------------------

def test_reject_unsupported_synonym_based_claim():

    context = (
        "Employees receive annual leave."
    )

    answer = (
        "Staff members receive paid parental leave."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is False


# --------------------------------------------------
# TEST 20: Accept Claims Supported Across Sentences
# --------------------------------------------------

def test_accept_claims_supported_across_sentences():

    context = (
        "Employees receive 20 days of annual leave. "
        "Employees may carry over 5 unused days."
    )

    answer = (
        "Employees receive 20 days of annual leave "
        "and may carry over 5 unused days."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is True


# --------------------------------------------------
# TEST 21: Accept Coordinated Phrases
# --------------------------------------------------

def test_accept_coordinated_phrases():

    context = (
        "Employees receive health and dental insurance."
    )

    answer = (
        "Employees receive health and dental insurance."
    )

    result = validate_answer(answer, context)

    assert result["is_valid"] is True

# --------------------------------------------------
# TEST 22: Reject Incorrect Time Unit
# --------------------------------------------------

def test_reject_unsupported_time_unit():

    context = "Employees receive 20 days of annual leave."
    answer = "Employees receive 20 weeks of annual leave."

    result = validate_answer(answer, context)

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_claim"


# --------------------------------------------------
# TEST 23: Reject Incorrect Currency Unit
# --------------------------------------------------

def test_reject_unsupported_currency_unit():

    context = "Employees can claim 500 euros."
    answer = "Employees can claim 500 dollars."

    result = validate_answer(answer, context)

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_claim"


# --------------------------------------------------
# TEST 24: Accept Correct Numeric Unit
# --------------------------------------------------

def test_accept_supported_numeric_unit():

    context = "Employees receive 20 days of annual leave."
    answer = "Employees receive 20 days of annual leave."

    result = validate_answer(answer, context)

    assert result["is_valid"] is True
# TEST 25: Accept Equivalent Percentage Formats
def test_accept_equivalent_percentage_formats():
    context = "Employees receive a 25 percent bonus."
    answer = "Employees receive a 25% bonus."

    result = validate_answer(answer, context)

    assert result["is_valid"] is True


# TEST 26: Accept Reverse Percentage Formats
def test_accept_reverse_percentage_formats():
    context = "Employees receive a 25% bonus."
    answer = "Employees receive a 25 percent bonus."

    result = validate_answer(answer, context)

    assert result["is_valid"] is True


# TEST 27: Accept Equivalent Euro Formats
def test_accept_equivalent_euro_formats():
    context = "Employees can claim 500 euros."
    answer = "Employees can claim €500."

    result = validate_answer(answer, context)

    assert result["is_valid"] is True


# TEST 28: Accept Equivalent Dollar Formats
def test_accept_equivalent_dollar_formats():
    context = "Employees can claim 500 dollars."
    answer = "Employees can claim $500."

    result = validate_answer(answer, context)

    assert result["is_valid"] is True
