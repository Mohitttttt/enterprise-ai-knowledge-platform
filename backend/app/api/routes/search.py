from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.embedding_service import generate_embedding
from app.services.vector_store import vector_store


router = APIRouter(
    prefix="/api/v1/search",
    tags=["Semantic Search"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


@router.post("/")
async def semantic_search(request: SearchRequest):

    if vector_store.count() == 0:
        return {
            "query": request.query,
            "total_results": 0,
            "results": [],
        }

    try:
        query_embedding = generate_embedding(request.query)

        search_results = vector_store.search(
            query_embedding=query_embedding,
            n_results=request.top_k,
        )

        documents = search_results["documents"][0]
        metadatas = search_results["metadatas"][0]
        distances = search_results["distances"][0]

        results = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            results.append({
                "content": document,
                "metadata": metadata,
                "distance": round(float(distance), 4),
                "similarity": round(
                    max(0.0, 1.0 - float(distance)),
                    4,
                ),
            })

        return {
            "query": request.query,
            "total_results": len(results),
            "results": results,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(error)}",
        ) from error