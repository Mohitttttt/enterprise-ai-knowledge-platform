from app.services.context_filter import (
    filter_context,
    tokenize,
    calculate_relevance,
)


def test_tokenize():
    result = tokenize("Company Password Recovery 123")

    assert result == {
        "company",
        "password",
        "recovery",
        "123",
    }


def test_calculate_relevance():
    question_tokens = {
        "company",
        "password",
        "recovery",
    }

    sentence_tokens = {
        "company",
        "password",
        "portal",
    }

    score = calculate_relevance(
        question_tokens,
        sentence_tokens,
    )

    assert score == 2 / 3


def test_filter_relevant_sentences():
    search_results = [
        {
            "id": "chunk-1",
            "document": (
                "Employees can reset their passwords through "
                "the employee portal. "
                "If they cannot access the portal, "
                "they should contact IT Support. "
                "Standard working hours are 9 AM to 5 PM."
            ),
            "metadata": {
                "original_filename": "company_policy.txt",
                "chunk_index": 0,
            },
            "rrf_score": 0.016,
        }
    ]

    results = filter_context(
        question=(
            "How can employees reset passwords "
            "if the portal is inaccessible?"
        ),
        search_results=search_results,
    )

    assert len(results) == 1

    filtered_document = results[0]["document"]

    assert "reset their passwords" in filtered_document
    assert "contact IT Support" in filtered_document
    assert "Standard working hours" not in filtered_document


def test_filter_preserves_metadata():
    search_results = [
        {
            "id": "chunk-1",
            "document": (
                "Employees can reset their passwords "
                "through the employee portal."
            ),
            "metadata": {
                "original_filename": "company_policy.txt",
                "chunk_index": 0,
            },
            "rrf_score": 0.016,
        }
    ]

    results = filter_context(
        question="How can employees reset passwords?",
        search_results=search_results,
    )

    assert results[0]["id"] == "chunk-1"
    assert results[0]["metadata"]["chunk_index"] == 0
    assert results[0]["rrf_score"] == 0.016


def test_empty_question():
    results = filter_context(
        question="",
        search_results=[],
    )

    assert results == []


def test_empty_search_results():
    results = filter_context(
        question="How can employees reset passwords?",
        search_results=[],
    )

    assert results == []


def test_invalid_relevance_threshold():
    try:
        filter_context(
            question="How can employees reset passwords?",
            search_results=[],
            minimum_relevance=1.5,
        )
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_password_recovery_preserves_all_instructions():
    search_results = [
        {
            "id": "password-chunk",
            "document": (
                "Employees can reset their company passwords "
                "through the employee portal. "
                "If they cannot access the portal, they should "
                "contact the IT Support team. "
                "Employees must never share their passwords "
                "with another person. "
                "Standard working hours are Monday to Friday."
            ),
            "metadata": {
                "original_filename": "company_policy.txt",
                "chunk_index": 0,
            },
            "rrf_score": 0.016,
        }
    ]

    results = filter_context(
        question="How can employees recover their company passwords?",
        search_results=search_results,
    )

    assert len(results) == 1

    filtered_document = results[0]["document"]

    assert "employee portal" in filtered_document
    assert "contact the IT Support team" in filtered_document
    assert "never share their passwords" in filtered_document
    assert "Standard working hours" not in filtered_document
    # --------------------------------------------------
# TEST 9: Remove Unrelated Sentences With Shared Words
# --------------------------------------------------
# --------------------------------------------------
# TEST 9: Remove Unrelated Sentences With Shared Words
# --------------------------------------------------

def test_filter_removes_unrelated_sentences_with_shared_words():

    search_results = [
        {
            "id": "chunk-1",
            "document": (
                "Employees can reset their company passwords "
                "through the employee portal. "
                "If they cannot access the portal, they should "
                "contact the IT Support team. "
                "Employees must notify their manager when "
                "they are unable to attend work due to illness."
            ),
            "metadata": {
                "original_filename": "company_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            },
            "rrf_score": 0.03,
        }
    ]

    filtered = filter_context(
        question="How can employees recover their company passwords?",
        search_results=search_results,
    )

    assert len(filtered) == 1

    filtered_document = filtered[0]["document"]

    assert "employee portal" in filtered_document
    assert "IT Support team" in filtered_document

    assert "unable to attend work" not in filtered_document
    assert "due to illness" not in filtered_document


# --------------------------------------------------
# TEST 10: Reject Password Recovery for Expiration Questions
# --------------------------------------------------

def test_password_expiration_rejects_password_recovery():

    search_results = [
        {
            "id": "chunk_1",
            "document": (
                "Employees can reset their company passwords "
                "through the employee portal."
            ),
            "metadata": {
                "source": "company_policy.txt"
            },
        }
    ]

    result = filter_context(
        question="What is the company's password expiration period?",
        search_results=search_results,
    )

    assert result == []


# --------------------------------------------------
# TEST 11: Accept Password Reset for Recovery Questions
# --------------------------------------------------

def test_password_recovery_accepts_password_reset():

    search_results = [
        {
            "id": "chunk_1",
            "document": (
                "Employees can reset their company passwords "
                "through the employee portal."
            ),
            "metadata": {
                "source": "company_policy.txt"
            },
        }
    ]

    result = filter_context(
        question="How can employees recover their company passwords?",
        search_results=search_results,
    )

    assert len(result) == 1

    assert (
        "reset their company passwords"
        in result[0]["document"]
    )