from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings


client = TestClient(app)


# --------------------------------------------------
# FIXTURE: Upload a Test Document and Clean It Up
# --------------------------------------------------

@pytest.fixture
def uploaded_document():

    # Generate unique content for every test.
    # This prevents conflicts with previously indexed documents.
    file_content = (
        f"1. Annual Leave\n"
        f"Employees receive 30 days of paid annual leave "
        f"per calendar year. Requests require manager approval.\n"
        f"Test identifier: {uuid4().hex}"
    ).encode("utf-8")

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "test_api_policy.txt",
                file_content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 200, response.text

    document = response.json()

    yield document

    # Automatically delete the document after the test.
    cleanup_response = client.delete(
        f"/api/v1/documents/{document['document_id']}"
    )

    # A 404 is acceptable if the test already deleted it.
    assert cleanup_response.status_code in (200, 404)


# --------------------------------------------------
# TEST 1: Root API Endpoint
# --------------------------------------------------

def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == settings.app_name
    assert data["status"] == "running"
    assert data["version"] == settings.app_version


# --------------------------------------------------
# TEST 2: OpenAPI Documentation
# --------------------------------------------------

def test_openapi_documentation():

    response = client.get("/openapi.json")

    assert response.status_code == 200

    data = response.json()

    assert "paths" in data
    assert "/" in data["paths"]


# --------------------------------------------------
# TEST 3: Required API Routes
# --------------------------------------------------

def test_required_api_routes_exist():

    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/health" in paths
    assert "/api/v1/documents/upload" in paths
    assert "/api/v1/documents/" in paths
    assert "/api/v1/search/" in paths
    assert "/api/v1/ask" in paths


# --------------------------------------------------
# TEST 4: Upload a Document
# --------------------------------------------------

def test_upload_document(uploaded_document):

    assert uploaded_document["status"] == "indexed"

    assert uploaded_document["original_filename"] == (
        "test_api_policy.txt"
    )

    assert uploaded_document["total_chunks"] >= 1
    assert uploaded_document["stored_vectors"] >= 1
    assert uploaded_document["embedding_dimension"] == 384

    assert "document_id" in uploaded_document
    assert "sha256" in uploaded_document


# --------------------------------------------------
# TEST 5: Reject Unsupported File Format
# --------------------------------------------------

def test_reject_unsupported_file():

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "test.exe",
                b"invalid file",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400

    assert "Unsupported file format" in (
        response.json()["detail"]
    )


# --------------------------------------------------
# TEST 6: Reject Empty File
# --------------------------------------------------

def test_reject_empty_file():

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert "empty" in response.json()["detail"].lower()


# --------------------------------------------------
# TEST 7: List Documents
# --------------------------------------------------

def test_list_documents(uploaded_document):

    response = client.get("/api/v1/documents/")

    assert response.status_code == 200

    data = response.json()

    assert "total_documents" in data
    assert "documents" in data

    document_ids = [
        document["document_id"]
        for document in data["documents"]
    ]

    assert uploaded_document["document_id"] in document_ids


# --------------------------------------------------
# TEST 8: Retrieve a Document by ID
# --------------------------------------------------

def test_get_document(uploaded_document):

    document_id = uploaded_document["document_id"]

    response = client.get(
        f"/api/v1/documents/{document_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_id"] == document_id
    assert data["filename"] == "test_api_policy.txt"
    assert data["total_chunks"] >= 1
    assert len(data["chunks"]) >= 1


# --------------------------------------------------
# TEST 9: Delete a Document
# --------------------------------------------------

def test_delete_document(uploaded_document):

    document_id = uploaded_document["document_id"]

    response = client.delete(
        f"/api/v1/documents/{document_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "deleted"
    assert data["document_id"] == document_id
    assert data["deleted_chunks"] >= 1

    # Verify that the document no longer exists.
    get_response = client.get(
        f"/api/v1/documents/{document_id}"
    )

    assert get_response.status_code == 404


# --------------------------------------------------
# TEST 10: Reject Retrieval of Non-Existent Document
# --------------------------------------------------

def test_get_nonexistent_document():

    response = client.get(
        "/api/v1/documents/nonexistent_document_id"
    )

    assert response.status_code == 404


# --------------------------------------------------
# TEST 11: Reject Duplicate Document Upload
# --------------------------------------------------

def test_reject_duplicate_document():

    # Generate unique content for this test.
    file_content = (
        f"Duplicate detection test: {uuid4().hex}"
    ).encode("utf-8")

    # First upload should succeed.
    first_response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "duplicate_test.txt",
                file_content,
                "text/plain",
            )
        },
    )

    assert first_response.status_code == 200, first_response.text

    document_id = first_response.json()["document_id"]

    try:
        # Second upload with identical content should fail.
        second_response = client.post(
            "/api/v1/documents/upload",
            files={
                "file": (
                    "duplicate_test.txt",
                    file_content,
                    "text/plain",
                )
            },
        )

        assert second_response.status_code == 409

        assert second_response.json()["detail"] == (
            "This document has already been uploaded."
        )

    finally:
        # Clean up the successfully uploaded document.
        client.delete(
            f"/api/v1/documents/{document_id}"
        )
    
    # --------------------------------------------------
# TEST 12: Reject Files Larger Than 10 MB
# --------------------------------------------------

def test_reject_oversized_file():

    oversized_content = b"A" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "oversized_test.txt",
                oversized_content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 413