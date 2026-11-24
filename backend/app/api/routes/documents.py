import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.document_parser import (
    extract_text,
    DocumentParsingError,
)
from app.services.text_chunker import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import vector_store


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# UPLOAD DOCUMENT
# --------------------------------------------------

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload PDF, DOCX or TXT.",
        )

    document_id = uuid4().hex
    unique_filename = f"{document_id}{extension}"
    file_path = UPLOAD_DIR / unique_filename

    file_size = 0
    sha256_hash = hashlib.sha256()

    try:

        # ------------------------------------------
        # 1. SAVE UPLOADED FILE AND CALCULATE HASH
        # ------------------------------------------

        with file_path.open("wb") as buffer:

            while chunk := await file.read(1024 * 1024):

                file_size += len(chunk)

                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File size exceeds the 10 MB limit.",
                    )

                sha256_hash.update(chunk)
                buffer.write(chunk)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        file_hash = sha256_hash.hexdigest()

        # ------------------------------------------
        # 2. CHECK FOR DUPLICATE DOCUMENT
        # ------------------------------------------

        if vector_store.document_exists_by_hash(file_hash):

            raise HTTPException(
                status_code=409,
                detail="This document has already been uploaded.",
            )

        # ------------------------------------------
        # 3. EXTRACT DOCUMENT TEXT
        # ------------------------------------------

        try:
            extracted_text = extract_text(file_path)

        except DocumentParsingError as error:

            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

        # ------------------------------------------
        # 4. SPLIT TEXT INTO CHUNKS
        # ------------------------------------------

        chunks = chunk_text(extracted_text)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text chunks could be generated.",
            )

        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        # ------------------------------------------
        # 5. GENERATE EMBEDDINGS
        # ------------------------------------------

        embeddings = generate_embeddings(chunk_texts)

        chunk_ids = [
            f"{document_id}_{chunk['chunk_index']}"
            for chunk in chunks
        ]

        # ------------------------------------------
        # 6. PREPARE DOCUMENT METADATA
        # ------------------------------------------

        metadatas = [
            {
                "document_id": document_id,
                "original_filename": file.filename,
                "file_extension": extension,
                "chunk_index": chunk["chunk_index"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "sha256": file_hash,
            }
            for chunk in chunks
        ]

        # ------------------------------------------
        # 7. STORE DOCUMENT VECTORS
        # ------------------------------------------

        vector_store.add_documents(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
        )

        # ------------------------------------------
        # 8. RETURN UPLOAD INFORMATION
        # ------------------------------------------

        return {
            "message": "Document uploaded and indexed successfully.",
            "document_id": document_id,
            "original_filename": file.filename,
            "stored_filename": unique_filename,
            "file_extension": extension,
            "file_size_bytes": file_size,
            "sha256": file_hash,
            "extracted_characters": len(extracted_text),
            "total_chunks": len(chunks),
            "embedding_dimension": len(embeddings[0]),
            "stored_vectors": len(chunk_ids),
            "status": "indexed",
        }

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except Exception as error:

        file_path.unlink(missing_ok=True)

        logger.exception(
            "Document processing failed for %s",
            file.filename,
        )

        raise HTTPException(
            status_code=500,
            detail="Document processing failed.",
        ) from error

    finally:
        await file.close()


# --------------------------------------------------
# LIST DOCUMENTS
# --------------------------------------------------

@router.get("/")
async def list_documents():

    documents = vector_store.get_documents()

    return {
        "total_documents": len(documents),
        "documents": documents,
    }


# --------------------------------------------------
# GET DOCUMENT
# --------------------------------------------------

@router.get("/{document_id}")
async def get_document(document_id: str):

    document = vector_store.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document


# --------------------------------------------------
# DELETE DOCUMENT
# --------------------------------------------------

@router.delete("/{document_id}")
async def delete_document(document_id: str):

    document = vector_store.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    stored_filename = (
        f"{document_id}{document['file_extension']}"
    )

    file_path = UPLOAD_DIR / stored_filename

    tombstone_path = file_path.with_name(
        f"{file_path.name}.{uuid4().hex}.deleting"
    )

    # ------------------------------------------
    # 1. TEMPORARILY RENAME PHYSICAL FILE
    # ------------------------------------------

    if file_path.exists():

        try:
            file_path.replace(tombstone_path)

        except OSError as error:

            logger.exception(
                "Could not stage document file for deletion: %s",
                document_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Could not prepare the document file for deletion.",
            ) from error

    # ------------------------------------------
    # 2. DELETE DOCUMENT VECTORS
    # ------------------------------------------

    try:
        deleted_chunks = vector_store.delete_document(document_id)

    except Exception as error:

        if tombstone_path.exists():

            try:
                tombstone_path.replace(file_path)

            except OSError:

                logger.exception(
                    "CRITICAL: Could not restore document file %s",
                    document_id,
                )

        logger.exception(
            "Could not delete document vectors: %s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Could not delete document vectors.",
        ) from error

    # ------------------------------------------
    # 3. REMOVE TEMPORARY FILE
    # ------------------------------------------

    cleanup_warning = None

    if tombstone_path.exists():

        try:
            tombstone_path.unlink()

        except OSError:

            cleanup_warning = (
                "Document vectors were deleted, but temporary file "
                "cleanup failed. Manual cleanup may be required."
            )

            logger.exception(
                "Temporary document cleanup failed: %s",
                tombstone_path,
            )

    response = {
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "filename": document["filename"],
        "deleted_chunks": deleted_chunks,
        "status": "deleted",
    }

    if cleanup_warning:
        response["warning"] = cleanup_warning

    return response