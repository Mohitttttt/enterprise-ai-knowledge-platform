from app.services.answer_service import answer_from_context


def test_extracts_maternity_leave_policy_instead_of_annual_leave():
    context = """
    [Source 1]
    Filename: test_company_policy.txt
    Content:
    1. Annual Leave
    Employees receive 30 days of paid annual leave per calendar year.

    7. Maternity and Parental Leave
    This policy does not specify maternity or parental leave entitlements.
    Employees should contact Human Resources for further information.
    """

    answer = answer_from_context(
        question="How many weeks of maternity leave do employees receive?",
        context=context,
    )

    assert answer is not None
    assert "maternity" in answer.lower()
    assert "30 days" not in answer.lower()


def test_extracts_multiple_facts_for_multi_part_question():
    context = """
    [Source 1]
    Filename: test_company_policy.txt
    Content:
    1. Annual Leave
    Employees receive 30 days of paid annual leave per calendar year.
    Annual leave requests must be submitted to the employee's manager
    for approval.
    """

    answer = answer_from_context(
        question=(
            "Who approves annual leave requests, "
            "and how many paid annual leave days "
            "do employees receive each year?"
        ),
        context=context,
    )

    assert answer is not None
    assert "30 days" in answer.lower()
    assert "manager" in answer.lower()


def test_returns_none_when_question_is_unrelated():
    context = """
    [Source 1]
    Filename: company_policy.txt
    Content:
    Employees receive 30 days of paid annual leave.
    """

    answer = answer_from_context(
        question="What is the company's maternity leave policy?",
        context=context,
    )

    assert answer is None