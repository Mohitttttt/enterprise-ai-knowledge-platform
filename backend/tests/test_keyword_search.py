import pytest

from app.services.keyword_search import (
    KeywordSearch,
    tokenize,
)


# --------------------------------------------------
# MOCK VECTOR STORE
# --------------------------------------------------

class FakeCollection:

    def __init__(self):

        self.ids = [
            "doc_1",
            "doc_2",
            "doc_3",
        ]

        self.documents = [
            "Employees receive 30 days of annual leave.",
            "Password recovery is handled through IT Support.",
            "Annual leave requests require manager approval.",
        ]

        self.metadatas = [
            {"filename": "leave_policy.txt"},
            {"filename": "security_policy.txt"},
            {"filename": "leave_policy.txt"},
        ]

        self.get_calls = 0

    def get(self, include=None):

        self.get_calls += 1

        return {
            "ids": self.ids.copy(),
            "documents": self.documents.copy(),
            "metadatas": self.metadatas.copy(),
        }

    def add_record(self, document_id, document, metadata):

        self.ids.append(document_id)
        self.documents.append(document)
        self.metadatas.append(metadata)

    def delete_record(self, document_id):

        if document_id not in self.ids:
            return

        index = self.ids.index(document_id)

        self.ids.pop(index)
        self.documents.pop(index)
        self.metadatas.pop(index)


class FakeStore:

    def __init__(self):

        self.collection = FakeCollection()
        self.version = 0

    def add_document(self, document_id, document, metadata):

        self.collection.add_record(
            document_id,
            document,
            metadata,
        )

        self.version += 1

    def delete_document(self, document_id):

        if document_id not in self.collection.ids:
            return

        self.collection.delete_record(document_id)

        self.version += 1


# --------------------------------------------------
# FIXTURE
# --------------------------------------------------

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
# TEST 2: Unicode Tokenization
# --------------------------------------------------

def test_tokenize_german_unicode():

    tokens = tokenize(
        "Rückerstattung jährlich Straße"
    )

    assert tokens == [
        "rückerstattung",
        "jährlich",
        "strasse",
    ]


# --------------------------------------------------
# TEST 3: Technical Identifiers
# --------------------------------------------------

def test_tokenize_technical_identifiers():

    tokens = tokenize(
        "EMPLOYEE_ID USER_ACCOUNT 2026"
    )

    assert tokens == [
        "employee_id",
        "user_account",
        "2026",
    ]


# --------------------------------------------------
# TEST 4: Retrieve Matching Documents
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
# TEST 5: Relevant Documents Rank Higher
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
# TEST 6: No Matching Documents
# --------------------------------------------------

def test_keyword_search_no_matches(
    search_service,
):

    results = search_service.search(
        query="cryptocurrency blockchain",
    )

    assert results == []


# --------------------------------------------------
# TEST 7: Empty Query
# --------------------------------------------------

def test_keyword_search_empty_query(
    search_service,
):

    results = search_service.search(
        query="   ",
    )

    assert results == []


# --------------------------------------------------
# TEST 8: Invalid Top K
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
# TEST 9: Result Metadata
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


# --------------------------------------------------
# TEST 10: BM25 Cache Reuse
# --------------------------------------------------

def test_keyword_search_reuses_cached_index(
    search_service,
):

    search_service.search("annual leave")

    search_service.search("password recovery")

    search_service.search("manager approval")

    # The corpus should only be retrieved once.
    assert search_service.store.collection.get_calls == 1


# --------------------------------------------------
# TEST 11: Cache Invalidation After Insertion
# --------------------------------------------------

def test_keyword_search_rebuilds_cache_after_insertion(
    search_service,
):

    # First search builds the original index.
    search_service.search("annual leave")

    assert search_service.store.collection.get_calls == 1

    # Add a new document and increment the version.
    search_service.store.add_document(
        document_id="doc_4",
        document="Cryptocurrency security guidelines.",
        metadata={"filename": "crypto_policy.txt"},
    )

    # Search should detect the version change.
    results = search_service.search(
        "cryptocurrency security"
    )

    assert search_service.store.collection.get_calls == 2

    assert any(
        result["id"] == "doc_4"
        for result in results
    )


# --------------------------------------------------
# TEST 12: Cache Invalidation After Deletion
# --------------------------------------------------

def test_keyword_search_rebuilds_cache_after_deletion(
    search_service,
):

    # First search builds the original index.
    results = search_service.search("password recovery")

    assert any(
        result["id"] == "doc_2"
        for result in results
    )

    assert search_service.store.collection.get_calls == 1

    # Delete the document and increment the version.
    search_service.store.delete_document("doc_2")

    # Search must not return the deleted document.
    results = search_service.search("password recovery")

    assert search_service.store.collection.get_calls == 2

    assert all(
        result["id"] != "doc_2"
        for result in results
    )


# --------------------------------------------------
# TEST 13: Empty Corpus Cache
# --------------------------------------------------

def test_keyword_search_caches_empty_corpus():

    service = KeywordSearch()

    empty_store = FakeStore()

    empty_store.collection.ids = []
    empty_store.collection.documents = []
    empty_store.collection.metadatas = []

    service.store = empty_store

    first_results = service.search("annual leave")

    second_results = service.search("password recovery")

    assert first_results == []
    assert second_results == []

    # Empty corpus should also be cached.
    assert empty_store.collection.get_calls == 1