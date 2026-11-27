from typing import Any

from app.services.embedding_service import generate_embedding
from app.services.keyword_search import KeywordSearch, keyword_search
from app.services.vector_store import vector_store


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RRF_K = 40
CANDIDATE_MULTIPLIER = 3


# --------------------------------------------------
# RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    top_k: int = 5,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Combine multiple ranked result lists using
    Reciprocal Rank Fusion (RRF).

    Args:
        rankings: Multiple ranked result lists.
        top_k: Number of final results to return.
        k: RRF smoothing constant.

    Returns:
        Deduplicated and fused ranked results.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    fused_results: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            document_id = result["id"]

            if document_id not in fused_results:
                fused_results[document_id] = {
                    **result,
                    "rrf_score": 0.0,
                }

            fused_results[document_id]["rrf_score"] += (
                1.0 / (k + rank)
            )

    results = list(fused_results.values())

    results.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return results[:top_k]


# --------------------------------------------------
# HYBRID SEARCH
# --------------------------------------------------

class HybridSearch:
    """
    Combines semantic and keyword retrieval using
    Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        rrf_k: int = RRF_K,
        store: Any | None = None,
        keyword_service: Any | None = None,
    ):
        """
        Initialize the hybrid search service.

        Args:
            rrf_k: RRF smoothing constant.
            store: Optional vector-store instance.
            keyword_service: Optional keyword-search instance.
        """

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        self.vector_store = store if store is not None else vector_store

        self.keyword_search = (
            keyword_service
            if keyword_service is not None
            else (
                KeywordSearch(store=self.vector_store)
                if store is not None
                else keyword_search
            )
        )

        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve an expanded candidate pool from semantic
        and keyword search, then combine rankings using RRF.

        Args:
            query: User's search query.
            top_k: Number of final results to return.

        Returns:
            Fused and ranked search results.
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # Expand candidate pool before reranking.
        candidate_k = top_k * CANDIDATE_MULTIPLIER

        # Generate query embedding.
        query_embedding = generate_embedding(query)

        # Retrieve expanded semantic search results.
        semantic_results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=candidate_k,
        )

        semantic_ranking = []

        documents = semantic_results["documents"][0]
        metadatas = semantic_results["metadatas"][0]
        ids = semantic_results["ids"][0]
        distances = semantic_results["distances"][0]

        for document_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            semantic_ranking.append({
                "id": document_id,
                "document": document,
                "metadata": metadata or {},
                "distance": float(distance),
            })

        # Retrieve expanded keyword search results.
        keyword_ranking = self.keyword_search.search(
            query=query,
            top_k=candidate_k,
        )

        # Combine both rankings using configurable RRF.
        return reciprocal_rank_fusion(
            rankings=[
                semantic_ranking,
                keyword_ranking,
            ],
            top_k=top_k,
            k=self.rrf_k,
        )

    # --------------------------------------------------
    # RETRIEVAL DIAGNOSTICS
    # --------------------------------------------------

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Retrieve results while exposing individual retrieval
        rankings and their RRF contributions.

        Does not change normal production search behaviour.
        """

        if not query or not query.strip():
            return {
                "semantic_results": [],
                "keyword_results": [],
                "fused_results": [],
            }

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        candidate_k = top_k * CANDIDATE_MULTIPLIER

        # Semantic retrieval
        query_embedding = generate_embedding(query)

        semantic_data = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=candidate_k,
        )

        semantic_ranking = []

        for rank, (document_id, document, metadata, distance) in enumerate(
            zip(
                semantic_data["ids"][0],
                semantic_data["documents"][0],
                semantic_data["metadatas"][0],
                semantic_data["distances"][0],
            ),
            start=1,
        ):
            semantic_ranking.append({
                "id": document_id,
                "document": document,
                "metadata": metadata or {},
                "distance": float(distance),
                "semantic_rank": rank,
            })

        # Keyword retrieval
        keyword_data = self.keyword_search.search(
            query=query,
            top_k=candidate_k,
        )

        keyword_ranking = [
            {
                **result,
                "keyword_rank": rank,
            }
            for rank, result in enumerate(
                keyword_data,
                start=1,
            )
        ]

        # Build rank lookup tables
        semantic_by_id = {
            result["id"]: result
            for result in semantic_ranking
        }

        keyword_by_id = {
            result["id"]: result
            for result in keyword_ranking
        }

        # Fuse rankings using the existing RRF implementation
        fused_results = reciprocal_rank_fusion(
            rankings=[
                semantic_ranking,
                keyword_ranking,
            ],
            top_k=candidate_k,
            k=self.rrf_k,
        )

        # Add individual contributions and original ranks
        diagnostic_fused_results = []

        for final_rank, result in enumerate(
            fused_results,
            start=1,
        ):
            document_id = result["id"]

            semantic_match = semantic_by_id.get(document_id)
            keyword_match = keyword_by_id.get(document_id)

            semantic_rank = (
                semantic_match["semantic_rank"]
                if semantic_match else None
            )

            keyword_rank = (
                keyword_match["keyword_rank"]
                if keyword_match else None
            )

            semantic_contribution = (
                1.0 / (self.rrf_k + semantic_rank)
                if semantic_rank is not None
                else 0.0
            )

            keyword_contribution = (
                1.0 / (self.rrf_k + keyword_rank)
                if keyword_rank is not None
                else 0.0
            )

            diagnostic_fused_results.append({
                **result,
                "semantic_rank": semantic_rank,
                "semantic_distance": (
                    semantic_match["distance"]
                    if semantic_match else None
                ),
                "keyword_rank": keyword_rank,
                "keyword_score": (
                    keyword_match["score"]
                    if keyword_match else None
                ),
                "semantic_rrf_contribution": semantic_contribution,
                "keyword_rrf_contribution": keyword_contribution,
                "final_rank": final_rank,
            })

        return {
            "semantic_results": semantic_ranking,
            "keyword_results": keyword_ranking,
            "fused_results": diagnostic_fused_results,
        }
# --------------------------------------------------
# DEFAULT HYBRID SEARCH INSTANCE
# --------------------------------------------------

hybrid_search = HybridSearch(
    keyword_service=KeywordSearch(
        enable_query_expansion=True
    )
)
