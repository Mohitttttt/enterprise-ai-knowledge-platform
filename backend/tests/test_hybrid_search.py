import pytest

import app.services.hybrid_search as hybrid_module

from app.services.hybrid_search import (
    HybridSearch,
    reciprocal_rank_fusion,
)

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
# --------------------------------------------------
# TEST 9: Hybrid Search Expands Candidate Pool
# --------------------------------------------------

def test_hybrid_search_expands_candidate_pool(monkeypatch):

    class FakeVectorStore:

        def __init__(self):
            self.requested_n_results = None

        def search(self, query_embedding, n_results):

            self.requested_n_results = n_results

            return {
                "ids": [[f"semantic_{i}" for i in range(n_results)]],
                "documents": [
                    [f"Semantic document {i}" for i in range(n_results)]
                ],
                "metadatas": [
                    [{"source": "semantic"} for _ in range(n_results)]
                ],
                "distances": [[0.1 for _ in range(n_results)]],
            }

    class FakeKeywordSearch:

        def __init__(self):
            self.requested_top_k = None

        def search(self, query, top_k):

            self.requested_top_k = top_k

            return [
                {
                    "id": f"keyword_{i}",
                    "document": f"Keyword document {i}",
                    "metadata": {"source": "keyword"},
                    "score": 1.0,
                }
                for i in range(top_k)
            ]

    fake_vector_store = FakeVectorStore()
    fake_keyword_search = FakeKeywordSearch()

    service = HybridSearch()

    service.vector_store = fake_vector_store
    service.keyword_search = fake_keyword_search

    monkeypatch.setattr(
        hybrid_module,
        "generate_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )

    service.search("annual leave", top_k=5)

    expected_candidate_k = 15

    assert fake_vector_store.requested_n_results == expected_candidate_k

    assert fake_keyword_search.requested_top_k == expected_candidate_k


# --------------------------------------------------
# TEST 10: Hybrid Search Respects Final Top K
# --------------------------------------------------

def test_hybrid_search_respects_final_top_k(monkeypatch):

    class FakeVectorStore:

        def search(self, query_embedding, n_results):

            return {
                "ids": [[f"doc_{i}" for i in range(n_results)]],
                "documents": [
                    [f"Document {i}" for i in range(n_results)]
                ],
                "metadatas": [
                    [{} for _ in range(n_results)]
                ],
                "distances": [[0.1 for _ in range(n_results)]],
            }

    class FakeKeywordSearch:

        def search(self, query, top_k):

            return []

    service = HybridSearch()

    service.vector_store = FakeVectorStore()
    service.keyword_search = FakeKeywordSearch()

    monkeypatch.setattr(
        hybrid_module,
        "generate_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )

    results = service.search(
        "annual leave",
        top_k=5,
    )

    assert len(results) == 5
