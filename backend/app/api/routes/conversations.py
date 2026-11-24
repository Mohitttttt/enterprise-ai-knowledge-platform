from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.conversation_service import conversation_service
from app.services.rag_service import generate_rag_answer


router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["Conversations"],
)


class MessageRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="User message or question",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


@router.post("")
async def create_conversation():

    return conversation_service.create_conversation()


@router.post("/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    request: MessageRequest,
):

    if not conversation_service.conversation_exists(
        conversation_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    # Retrieve history before adding the current question.
    history = conversation_service.get_messages(
        conversation_id
    )

    # Save the user's message.
    conversation_service.add_message(
        conversation_id=conversation_id,
        role="user",
        content=request.message,
    )

    try:

        # Generate an answer using conversation-aware RAG.
        result = generate_rag_answer(
            question=request.message,
            top_k=request.top_k,
            conversation_history=history,
        )

        # Save the assistant's response.
        conversation_service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result["answer"],
        )

        return {
            "conversation_id": conversation_id,
            "question": request.message,
            "answer": result["answer"],
            "sources": result["sources"],
            "model": result["model"],
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Message processing failed: {str(error)}",
        ) from error


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):

    if not conversation_service.conversation_exists(
        conversation_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    messages = conversation_service.get_messages(
        conversation_id
    )

    return {
        "conversation_id": conversation_id,
        "messages": messages,
    }


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):

    deleted = conversation_service.delete_conversation(
        conversation_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return {
        "message": "Conversation deleted successfully.",
        "conversation_id": conversation_id,
    }