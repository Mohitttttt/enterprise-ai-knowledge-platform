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
# --------------------------------------------------
# TEST 11: Diagnostic Search Exposes Both Rankings
# --------------------------------------------------

def test_search_with_diagnostics_exposes_rankings(monkeypatch):

    class FakeVectorStore:

        def search(self, query_embedding, n_results):
            return {
                "ids": [["doc_a", "doc_b"]],
                "documents": [["Document A", "Document B"]],
                "metadatas": [[{}, {}]],
                "distances": [[0.1, 0.2]],
            }

    class FakeKeywordSearch:

        def search(self, query, top_k):
            return [
                {
                    "id": "doc_b",
                    "document": "Document B",
                    "metadata": {},
                    "score": 2.0,
                },
                {
                    "id": "doc_c",
                    "document": "Document C",
                    "metadata": {},
                    "score": 1.0,
                },
            ]

    service = HybridSearch(rrf_k=40)
    service.vector_store = FakeVectorStore()
    service.keyword_search = FakeKeywordSearch()

    monkeypatch.setattr(
        hybrid_module,
        "generate_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )

    results = service.search_with_diagnostics(
        "annual leave",
        top_k=2,
    )

    assert len(results["semantic_results"]) == 2
    assert len(results["keyword_results"]) == 2
    assert len(results["fused_results"]) == 3

    assert results["semantic_results"][0]["id"] == "doc_a"
    assert results["keyword_results"][0]["id"] == "doc_b"


# --------------------------------------------------
# TEST 12: Diagnostic Search Calculates RRF Contributions
# --------------------------------------------------

def test_search_with_diagnostics_calculates_contributions(monkeypatch):

    class FakeVectorStore:

        def search(self, query_embedding, n_results):
            return {
                "ids": [["doc_a", "doc_b"]],
                "documents": [["Document A", "Document B"]],
                "metadatas": [[{}, {}]],
                "distances": [[0.1, 0.2]],
            }

    class FakeKeywordSearch:

        def search(self, query, top_k):
            return [
                {
                    "id": "doc_b",
                    "document": "Document B",
                    "metadata": {},
                    "score": 2.0,
                },
                {
                    "id": "doc_c",
                    "document": "Document C",
                    "metadata": {},
                    "score": 1.0,
                },
            ]

    service = HybridSearch(rrf_k=40)
    service.vector_store = FakeVectorStore()
    service.keyword_search = FakeKeywordSearch()

    monkeypatch.setattr(
        hybrid_module,
        "generate_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )

    results = service.search_with_diagnostics(
        "annual leave",
        top_k=2,
    )

    best_result = results["fused_results"][0]

    assert best_result["id"] == "doc_b"
    assert best_result["semantic_rank"] == 2
    assert best_result["keyword_rank"] == 1
    assert best_result["final_rank"] == 1

    expected_semantic = 1 / 42
    expected_keyword = 1 / 41

    assert best_result["semantic_rrf_contribution"] == pytest.approx(
        expected_semantic
    )

    assert best_result["keyword_rrf_contribution"] == pytest.approx(
        expected_keyword
    )

    assert best_result["rrf_score"] == pytest.approx(
        expected_semantic + expected_keyword
    )


# --------------------------------------------------
# TEST 13: Diagnostic Search Handles Empty Queries
# --------------------------------------------------

def test_search_with_diagnostics_handles_empty_query():

    service = HybridSearch()

    results = service.search_with_diagnostics("")

    assert results == {
        "semantic_results": [],
        "keyword_results": [],
        "fused_results": [],
    }
# --------------------------------------------------
# TEST 14: Constructor Dependency Injection
# --------------------------------------------------

def test_hybrid_search_accepts_injected_dependencies():

    class FakeVectorStore:
        pass

    class FakeKeywordSearch:
        pass

    store = FakeVectorStore()
    keyword_service = FakeKeywordSearch()

    service = HybridSearch(
        rrf_k=40,
        store=store,
        keyword_service=keyword_service,
    )

    assert service.vector_store is store
    assert service.keyword_search is keyword_service


# --------------------------------------------------
# TEST 15: Diagnostic Search Handles Empty Semantic Results
# --------------------------------------------------

def test_search_with_diagnostics_handles_empty_semantic_results(
    monkeypatch,
):

    class FakeVectorStore:

        def search(self, query_embedding, n_results):
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

    class FakeKeywordSearch:

        def search(self, query, top_k):
            return [
                {
                    "id": "doc_a",
                    "document": "Document A",
                    "metadata": {},
                    "score": 2.0,
                }
            ]

    service = HybridSearch(
        store=FakeVectorStore(),
        keyword_service=FakeKeywordSearch(),
    )

    monkeypatch.setattr(
        hybrid_module,
        "generate_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )

    results = service.search_with_diagnostics(
        "annual leave",
        top_k=2,
    )

    assert results["semantic_results"] == []
    assert len(results["keyword_results"]) == 1
    assert results["fused_results"][0]["id"] == "doc_a"


# --------------------------------------------------
# TEST 16: Diagnostic Search Rejects Invalid Top K
# --------------------------------------------------

def test_search_with_diagnostics_rejects_invalid_top_k():

    service = HybridSearch()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        service.search_with_diagnostics(
            "annual leave",
            top_k=0,
        )


# --------------------------------------------------
# TEST 17: Normal and Diagnostic Search Are Consistent
# --------------------------------------------------

def test_normal_and_diagnostic_search_return_same_top_results(
    monkeypatch,
):

    class FakeVectorStore:

        def search(self, query_embedding, n_results):
            return {
                "ids": [["doc_a", "doc_b"]],
                "documents": [["Document A", "Document B"]],
                "metadatas": [[{}, {}]],
                "distances": [[0.1, 0.2]],
            }

    class FakeKeywordSearch:

        def search(self, query, top_k):
            return [
                {
                    "id": "doc_b",
                    "document": "Document B",
                    "metadata": {},
                    "score": 2.0,
                },
                {
                    "id": "doc_c",
                    "document": "Document C",
                    "metadata": {},
                    "score": 1.0,
                },
            ]

    service = HybridSearch(
        rrf_k=40,
        store=FakeVectorStore(),
        keyword_service=FakeKeywordSearch(),
    )

    monkeypatch.setattr(
        hybrid_module,
        "generate_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )

    normal_results = service.search(
        "annual leave",
        top_k=2,
    )

    diagnostic_results = service.search_with_diagnostics(
        "annual leave",
        top_k=2,
    )

    normal_ids = [
        result["id"]
        for result in normal_results
    ]

    diagnostic_ids = [
        result["id"]
        for result in diagnostic_results["fused_results"][:2]
    ]

    assert normal_ids == diagnostic_ids


# --------------------------------------------------
# TEST 18: Production Singleton Enables Query Expansion
# --------------------------------------------------

def test_production_hybrid_search_enables_query_expansion():

    assert (
        hybrid_module.hybrid_search
        .keyword_search.enable_query_expansion
        is True
    )


# --------------------------------------------------
# TEST 19: Default Keyword Search Keeps Expansion Disabled
# --------------------------------------------------

def test_default_keyword_search_keeps_expansion_disabled():

    service = hybrid_module.KeywordSearch()

    assert service.enable_query_expansion is False


# --------------------------------------------------
# TEST 20: Default Hybrid Search Preserves Baseline
# --------------------------------------------------

def test_default_hybrid_search_preserves_baseline():

    service = HybridSearch()

    assert service.keyword_search.enable_query_expansion is False
