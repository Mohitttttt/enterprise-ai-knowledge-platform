import re
import threading
from typing import Any

from rank_bm25 import BM25Okapi

from app.services.vector_store import vector_store
from app.services.query_expansion import expand_tokens


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase Unicode-aware searchable tokens.

    Preserves:
    - English and German words
    - Unicode characters
    - Numbers
    - Technical identifiers containing underscores
    """

    return re.findall(
        r"\b\w+\b",
        text.casefold(),
        flags=re.UNICODE,
    )


class KeywordSearch:

    def __init__(
        self,
        store: Any | None = None,
        enable_query_expansion: bool = False,
    ):

        self.store = store if store is not None else vector_store
        self.enable_query_expansion = enable_query_expansion

        # Cached BM25 index.
        self._bm25: BM25Okapi | None = None

        # Records corresponding to the cached BM25 corpus.
        self._records: list[tuple] = []

        # Version associated with the cached index.
        self._cached_version: int | None = None

        # Tracks whether the cache has been initialized,
        # including when the corpus is empty.
        self._cache_initialized = False

        # Protect cache creation and updates.
        self._lock = threading.RLock()

    # --------------------------------------------------
    # BUILD OR RETRIEVE CACHED BM25 INDEX
    # --------------------------------------------------

    def _get_index(
        self,
    ) -> tuple[BM25Okapi | None, list[tuple]]:
        """
        Return the cached BM25 index and its corresponding records.

        Rebuild the index only when the vector-store version changes.

        If the store does not expose a version property, caching is
        disabled for that store to avoid serving stale results.
        """

        with self._lock:

            current_version = getattr(
                self.store,
                "version",
                None,
            )

            # Reuse the existing index when the version is unchanged.
            if (
                current_version is not None
                and self._cache_initialized
                and current_version == self._cached_version
            ):
                return self._bm25, self._records

            # Retrieve all indexed document chunks.
            records = self.store.collection.get(
                include=["documents", "metadatas"]
            )

            documents = records.get("documents") or []
            metadatas = records.get("metadatas") or []
            ids = records.get("ids") or []

            # Tokenize the document corpus.
            tokenized_corpus = [
                tokenize(document or "")
                for document in documents
            ]

            # Ignore documents without searchable tokens.
            valid_records = [
                (document_id, document, metadata, tokens)
                for document_id, document, metadata, tokens
                in zip(
                    ids,
                    documents,
                    metadatas,
                    tokenized_corpus,
                )
                if tokens
            ]

            if valid_records:

                tokenized_documents = [
                    record[3]
                    for record in valid_records
                ]

                bm25 = BM25Okapi(tokenized_documents)

            else:

                bm25 = None

            # Update the cache only after successfully retrieving
            # and processing the corpus.
            self._bm25 = bm25

            self._records = [
                (document_id, document, metadata)
                for document_id, document, metadata, _
                in valid_records
            ]

            self._cached_version = current_version
            self._cache_initialized = True

            return self._bm25, self._records

    # --------------------------------------------------
    # SEARCH DOCUMENTS USING BM25
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant document chunks using BM25.

        Returns:
            A ranked list containing document content,
            metadata, document ID and BM25 score.
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # Tokenize the search query.
        tokenized_query = tokenize(query)

        if self.enable_query_expansion:
            tokenized_query = expand_tokens(tokenized_query)

        if not tokenized_query:
            return []

        # Retrieve the cached index or rebuild it if necessary.
        bm25, valid_records = self._get_index()

        if bm25 is None or not valid_records:
            return []

        # Calculate relevance scores.
        scores = bm25.get_scores(tokenized_query)

        ranked_results = []

        for index, score in enumerate(scores):

            # Ignore documents without keyword matches.
            if score <= 0:
                continue

            document_id, document, metadata = (
                valid_records[index]
            )

            ranked_results.append({
                "id": document_id,
                "document": document,
                "metadata": metadata or {},
                "score": float(score),
            })

        # Highest relevance score first.
        ranked_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return ranked_results[:top_k]


# --------------------------------------------------
# KEYWORD SEARCH INSTANCE
# --------------------------------------------------

keyword_search = KeywordSearch()
