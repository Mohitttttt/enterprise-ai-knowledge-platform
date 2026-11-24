
import pytest

from app.services.keyword_search import (
    KeywordSearch,
    tokenize,
)


# --------------------------------------------------
# MOCK VECTOR STORE
# --------------------------------------------------

class FakeCollection:

    def get(self, include=None):

        return {
            "ids": [
                "doc_1",
                "doc_2",
                "doc_3",
            ],
            "documents": [
                "Employees receive 30 days of annual leave.",
                "Password recovery is handled through IT Support.",
                "Annual leave requests require manager approval.",
            ],
            "metadatas": [
                {"filename": "leave_policy.txt"},
                {"filename": "security_policy.txt"},
                {"filename": "leave_policy.txt"},
            ],
        }


class FakeStore:

    def __init__(self):
        self.collection = FakeCollection()


@pytest.fixture
def search_service():

    service = KeywordSearch()
    service.store = FakeStore()

    return service


# --------------------------------------------------
# TEST 1: Tokenization
# --------------------------------------------------

def test_tokenize():

    tokens = tokenize("Employee ID 10452 Error 503")

    assert tokens == [
        "employee",
        "id",
        "10452",
        "error",
        "503",
    ]


# --------------------------------------------------
# TEST 2: Retrieve Matching Documents
# --------------------------------------------------

def test_keyword_search_returns_matching_documents(
    search_service,
):

    results = search_service.search(
        query="annual leave",
        top_k=5,
    )

    assert len(results) > 0

    assert all(
        "annual leave" in result["document"].lower()
        for result in results
    )


# --------------------------------------------------
# TEST 3: Relevant Documents Rank Higher
# --------------------------------------------------

def test_keyword_search_ranks_relevant_documents(
    search_service,
):

    results = search_service.search(
        query="annual leave",
        top_k=5,
    )

    assert results[0]["id"] in {
        "doc_1",
        "doc_3",
    }

    assert results[0]["score"] > 0


# --------------------------------------------------
# TEST 4: No Matching Documents
# --------------------------------------------------

def test_keyword_search_no_matches(
    search_service,
):

    results = search_service.search(
        query="cryptocurrency blockchain",
    )

    assert results == []


# --------------------------------------------------
# TEST 5: Empty Query
# --------------------------------------------------

def test_keyword_search_empty_query(
    search_service,
):

    results = search_service.search(
        query="   ",
    )

    assert results == []


# --------------------------------------------------
# TEST 6: Invalid Top K
# --------------------------------------------------

def test_keyword_search_invalid_top_k(
    search_service,
):

    with pytest.raises(ValueError):

        search_service.search(
            query="annual leave",
            top_k=0,
        )


# --------------------------------------------------
# TEST 7: Result Metadata
# --------------------------------------------------

def test_keyword_search_returns_metadata(
    search_service,
):

    results = search_service.search(
        query="password recovery",
    )

    assert len(results) == 1

    result = results[0]

    assert result["id"] == "doc_2"
    assert result["metadata"]["filename"] == (
        "security_policy.txt"
    )
    assert result["score"] > 0
