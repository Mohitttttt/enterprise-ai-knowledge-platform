from pathlib import Path

import chromadb


BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_DB_DIR = BASE_DIR / "data" / "chroma"

VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


class VectorStore:

    def __init__(self):

        # Tracks changes to the vector database.
        # Used by KeywordSearch to invalidate its cached BM25 index.
        self._version = 0

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name="enterprise_knowledge",
            metadata={
                "description": "Enterprise document knowledge embeddings",
                "hnsw:space": "cosine",
            },
        )

    @property
    def version(self) -> int:
        """
        Return the current vector-store version.

        The version increments whenever documents are successfully
        inserted, updated, or deleted through this instance.
        """
        return self._version

    # --------------------------------------------------
    # ADD DOCUMENTS
    # --------------------------------------------------

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):

        if not (
            len(ids)
            == len(embeddings)
            == len(documents)
            == len(metadatas)
        ):
            raise ValueError("All input lists must have equal lengths.")

        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # Increment only after a successful upsert.
        self._version += 1

    # --------------------------------------------------
    # SEARCH DOCUMENTS
    # --------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ):

        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        total_vectors = self.collection.count()

        if total_vectors == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, total_vectors),
            include=["documents", "metadatas", "distances"],
        )

    # --------------------------------------------------
    # LIST DOCUMENTS
    # --------------------------------------------------

    def get_documents(self) -> list[dict]:
        """
        Return a list of unique uploaded documents.
        """

        records = self.collection.get(
            include=["metadatas"]
        )

        documents = {}

        for metadata in records["metadatas"] or []:

            if not metadata:
                continue

            document_id = metadata.get("document_id")

            # Ignore synthetic test records without document IDs.
            if not document_id:
                continue

            if document_id not in documents:

                documents[document_id] = {
                    "document_id": document_id,
                    "filename": metadata.get("original_filename"),
                    "file_extension": metadata.get("file_extension"),
                    "sha256": metadata.get("sha256"),
                    "total_chunks": 0,
                }

            documents[document_id]["total_chunks"] += 1

        return list(documents.values())

    # --------------------------------------------------
    # GET DOCUMENT
    # --------------------------------------------------

    def get_document(self, document_id: str) -> dict | None:
        """
        Retrieve information about one document.
        """

        records = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas", "documents"],
        )

        if not records["ids"]:
            return None

        metadatas = records["metadatas"] or []

        first_metadata = metadatas[0] if metadatas else {}

        return {
            "document_id": document_id,
            "filename": first_metadata.get("original_filename"),
            "file_extension": first_metadata.get("file_extension"),
            "sha256": first_metadata.get("sha256"),
            "total_chunks": len(records["ids"]),
            "chunks": [
                {
                    "chunk_index": metadata.get("chunk_index"),
                    "content": content,
                }
                for metadata, content in zip(
                    metadatas,
                    records["documents"] or [],
                )
            ],
        }

    # --------------------------------------------------
    # DELETE DOCUMENT
    # --------------------------------------------------

    def delete_document(self, document_id: str) -> int:
        """
        Delete all vector records belonging to a document.
        Returns the number of deleted chunks.
        """

        records = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        ids = records["ids"]

        if not ids:
            return 0

        self.collection.delete(ids=ids)

        # Increment only after a successful deletion.
        self._version += 1

        return len(ids)

    # --------------------------------------------------
    # CHECK DUPLICATE DOCUMENT
    # --------------------------------------------------

    def document_exists_by_hash(self, sha256: str) -> bool:
        """
        Check whether a document with the given SHA-256 hash
        already exists in the vector database.
        """

        if not sha256:
            raise ValueError("SHA-256 hash cannot be empty.")

        records = self.collection.get(
            where={"sha256": sha256},
            include=["metadatas"],
        )

        return len(records["ids"]) > 0

    # --------------------------------------------------
    # COUNT VECTORS
    # --------------------------------------------------

    def count(self) -> int:
        return self.collection.count()


# --------------------------------------------------
# VECTOR STORE INSTANCE
# --------------------------------------------------

vector_store = VectorStore()