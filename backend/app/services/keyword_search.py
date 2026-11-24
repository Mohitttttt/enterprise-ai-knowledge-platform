
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.services.vector_store import vector_store


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase searchable tokens.
    Preserves numbers and technical identifiers.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )


class KeywordSearch:

    def __init__(self):
        self.store = vector_store

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

        # Retrieve all indexed document chunks.
        records = self.store.collection.get(
            include=["documents", "metadatas"]
        )

        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        ids = records.get("ids") or []

        if not documents:
            return []

        # Tokenize the document corpus.
        tokenized_corpus = [
            tokenize(document or "")
            for document in documents
        ]

        # Ignore empty documents.
        valid_records = [
            (document_id, document, metadata, tokens)
            for document_id, document, metadata, tokens
            in zip(ids, documents, metadatas, tokenized_corpus)
            if tokens
        ]

        if not valid_records:
            return []

        tokenized_corpus = [
            record[3]
            for record in valid_records
        ]

        tokenized_query = tokenize(query)

        if not tokenized_query:
            return []

        # Build BM25 ranking model.
        bm25 = BM25Okapi(tokenized_corpus)

        # Calculate relevance scores.
        scores = bm25.get_scores(tokenized_query)

        ranked_results = []

        for index, score in enumerate(scores):

            # Ignore documents without keyword matches.
            if score <= 0:
                continue

            document_id, document, metadata, _ = (
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