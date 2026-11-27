import pytest

from app.services.vector_store import VectorStore


# --------------------------------------------------
# MOCK CHROMADB COLLECTION
# --------------------------------------------------

class FakeCollection:

    def __init__(self):
        self.records = {}

    def upsert(
        self,
        ids,
        embeddings,
        documents,
        metadatas,
    ):
        for document_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
        ):
            self.records[document_id] = {
                "document": document,
                "metadata": metadata,
            }

    def get(self, where=None, include=None):

        if where:
            key, value = next(iter(where.items()))

            matching_ids = [
                document_id
                for document_id, record in self.records.items()
                if record["metadata"].get(key) == value
            ]
        else:
            matching_ids = list(self.records.keys())

        return {
            "ids": matching_ids,
            "documents": [
                self.records[document_id]["document"]
                for document_id in matching_ids
            ],
            "metadatas": [
                self.records[document_id]["metadata"]
                for document_id in matching_ids
            ],
        }

    def delete(self, ids):

        for document_id in ids:
            self.records.pop(document_id, None)

    def count(self):
        return len(self.records)


# --------------------------------------------------
# MOCK CHROMADB CLIENT
# --------------------------------------------------

class FakeClient:

    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, name, metadata):
        return self.collection


# --------------------------------------------------
# FIXTURE
# --------------------------------------------------

@pytest.fixture
def vector_store_service(monkeypatch):

    collection = FakeCollection()
    client = FakeClient(collection)

    monkeypatch.setattr(
        "app.services.vector_store.chromadb.PersistentClient",
        lambda path: client,
    )

    store = VectorStore()

    return store


# --------------------------------------------------
# TEST 1: Initial Version
# --------------------------------------------------

def test_vector_store_initial_version(vector_store_service):

    assert vector_store_service.version == 0


# --------------------------------------------------
# TEST 2: Successful Insertion
# --------------------------------------------------

def test_add_documents_increments_version(vector_store_service):

    vector_store_service.add_documents(
        ids=["doc_1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Employee annual leave policy."],
        metadatas=[{"document_id": "document_1"}],
    )

    assert vector_store_service.version == 1
    assert vector_store_service.count() == 1


# --------------------------------------------------
# TEST 3: Empty Insertion
# --------------------------------------------------

def test_empty_insertion_does_not_increment_version(
    vector_store_service,
):

    vector_store_service.add_documents(
        ids=[],
        embeddings=[],
        documents=[],
        metadatas=[],
    )

    assert vector_store_service.version == 0


# --------------------------------------------------
# TEST 4: Updating Existing Documents
# --------------------------------------------------

def test_upsert_increments_version(vector_store_service):

    vector_store_service.add_documents(
        ids=["doc_1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Original document."],
        metadatas=[{"document_id": "document_1"}],
    )

    vector_store_service.add_documents(
        ids=["doc_1"],
        embeddings=[[0.4, 0.5, 0.6]],
        documents=["Updated document."],
        metadatas=[{"document_id": "document_1"}],
    )

    assert vector_store_service.version == 2
    assert vector_store_service.count() == 1


# --------------------------------------------------
# TEST 5: Successful Deletion
# --------------------------------------------------

def test_delete_document_increments_version(
    vector_store_service,
):

    vector_store_service.add_documents(
        ids=["doc_1", "doc_2"],
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        documents=[
            "First document.",
            "Second document.",
        ],
        metadatas=[
            {"document_id": "document_1"},
            {"document_id": "document_1"},
        ],
    )

    deleted_count = vector_store_service.delete_document(
        "document_1"
    )

    assert deleted_count == 2
    assert vector_store_service.version == 2
    assert vector_store_service.count() == 0


# --------------------------------------------------
# TEST 6: Deleting Nonexistent Documents
# --------------------------------------------------

def test_delete_nonexistent_document_does_not_increment_version(
    vector_store_service,
):

    deleted_count = vector_store_service.delete_document(
        "nonexistent_document"
    )

    assert deleted_count == 0
    assert vector_store_service.version == 0


# --------------------------------------------------
# TEST 7: Failed Insertion
# --------------------------------------------------

def test_failed_insertion_does_not_increment_version(
    vector_store_service,
):

    def failing_upsert(**kwargs):
        raise RuntimeError("Simulated database failure.")

    vector_store_service.collection.upsert = failing_upsert

    with pytest.raises(RuntimeError):

        vector_store_service.add_documents(
            ids=["doc_1"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["Test document."],
            metadatas=[{"document_id": "document_1"}],
        )

    assert vector_store_service.version == 0


# --------------------------------------------------
# TEST 8: Failed Deletion
# --------------------------------------------------

def test_failed_deletion_does_not_increment_version(
    vector_store_service,
):

    vector_store_service.add_documents(
        ids=["doc_1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Test document."],
        metadatas=[{"document_id": "document_1"}],
    )

    def failing_delete(ids):
        raise RuntimeError("Simulated database failure.")

    vector_store_service.collection.delete = failing_delete

    with pytest.raises(RuntimeError):

        vector_store_service.delete_document("document_1")

    assert vector_store_service.version == 1