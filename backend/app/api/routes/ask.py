from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import generate_rag_answer


router = APIRouter(
    prefix="/api/v1",
    tags=["AI Assistant"],
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Question about uploaded documents",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


@router.post("/ask")
async def ask_question(request: AskRequest):

    try:
        result = generate_rag_answer(
            question=request.question,
            top_k=request.top_k,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"AI answer generation failed: {str(error)}",
        ) from error