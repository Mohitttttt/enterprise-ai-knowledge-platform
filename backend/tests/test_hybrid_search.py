import pytest

from app.services.hybrid_search import reciprocal_rank_fusion


# --------------------------------------------------
# TEST 1: Combine Two Search Rankings
# --------------------------------------------------

def test_rrf_combines_rankings():

    semantic_results = [
        {"id": "doc_1", "document": "Document one"},
        {"id": "doc_2", "document": "Document two"},
    ]

    keyword_results = [
        {"id": "doc_2", "document": "Document two"},
        {"id": "doc_3", "document": "Document three"},
    ]

    results = reciprocal_rank_fusion(
        rankings=[
            semantic_results,
            keyword_results,
        ],
        top_k=5,
    )

    assert len(results) == 3

    result_ids = [
        result["id"] for result in results
    ]

    assert set(result_ids) == {
        "doc_1",
        "doc_2",
        "doc_3",
    }


# --------------------------------------------------
# TEST 2: Combine Duplicate Documents
# --------------------------------------------------

def test_rrf_deduplicates_documents():

    ranking_one = [
        {"id": "doc_1", "document": "Document one"},
        {"id": "doc_2", "document": "Document two"},
    ]

    ranking_two = [
        {"id": "doc_1", "document": "Document one"},
        {"id": "doc_3", "document": "Document three"},
    ]

    results = reciprocal_rank_fusion(
        rankings=[
            ranking_one,
            ranking_two,
        ],
    )

    result_ids = [
        result["id"] for result in results
    ]

    assert len(result_ids) == len(set(result_ids))


# --------------------------------------------------
# TEST 3: Documents Ranked Highly in Both Searches
# --------------------------------------------------

def test_rrf_rewards_consistent_rankings():

    ranking_one = [
        {"id": "doc_1", "document": "Document one"},
        {"id": "doc_2", "document": "Document two"},
    ]

    ranking_two = [
        {"id": "doc_1", "document": "Document one"},
        {"id": "doc_3", "document": "Document three"},
    ]

    results = reciprocal_rank_fusion(
        rankings=[
            ranking_one,
            ranking_two,
        ],
    )

    assert results[0]["id"] == "doc_1"


# --------------------------------------------------
# TEST 4: Respect Top K
# --------------------------------------------------

def test_rrf_respects_top_k():

    ranking = [
        {"id": f"doc_{i}", "document": f"Document {i}"}
        for i in range(10)
    ]

    results = reciprocal_rank_fusion(
        rankings=[ranking],
        top_k=3,
    )

    assert len(results) == 3


# --------------------------------------------------
# TEST 5: Empty Rankings
# --------------------------------------------------

def test_rrf_empty_rankings():

    results = reciprocal_rank_fusion(
        rankings=[
            [],
            [],
        ],
    )

    assert results == []


# --------------------------------------------------
# TEST 6: Invalid Top K
# --------------------------------------------------

def test_rrf_invalid_top_k():

    with pytest.raises(ValueError):

        reciprocal_rank_fusion(
            rankings=[],
            top_k=0,
        )


# --------------------------------------------------
# TEST 7: Invalid RRF Constant
# --------------------------------------------------

def test_rrf_invalid_k():

    with pytest.raises(ValueError):

        reciprocal_rank_fusion(
            rankings=[],
            k=0,
        )


# --------------------------------------------------
# TEST 8: RRF Score Calculation
# --------------------------------------------------

def test_rrf_score_calculation():

    ranking = [
        {"id": "doc_1", "document": "Document one"},
    ]

    results = reciprocal_rank_fusion(
        rankings=[ranking],
        k=60,
    )

    expected_score = 1 / 61

    assert results[0]["rrf_score"] == pytest.approx(
        expected_score
    )
