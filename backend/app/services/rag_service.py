import logging

from app.services.hybrid_search import hybrid_search
from app.services.vector_store import vector_store
from app.services.context_filter import filter_context
from app.services.context_builder import build_context
from app.services.prompt_builder import build_rag_prompt
from app.services.llm_service import ollama_service
from app.services.answer_service import answer_from_context
from app.services.answer_validator import validate_answer
from app.services.query_rewriter import rewrite_query


logger = logging.getLogger(__name__)


NO_DOCUMENTS_MESSAGE = "No documents have been uploaded yet."

NO_RELEVANT_INFORMATION = (
    "I couldn't find sufficiently relevant information "
    "in the uploaded documents to answer this question."
)

UNSUPPORTED_ANSWER = (
    "I couldn't find sufficient information "
    "in the uploaded documents."
)


def build_retrieval_query(
    question: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Build a standalone retrieval query using
    conversational query rewriting.
    """

    return rewrite_query(
        question=question,
        conversation_history=conversation_history,
    )


def generate_rag_answer(
    question: str,
    top_k: int = 5,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Generate answers using a hybrid RAG pipeline.

    Pipeline:
    1. Validate the question.
    2. Check document availability.
    3. Rewrite conversational questions.
    4. Retrieve relevant document chunks.
    5. Filter irrelevant context.
    6. Build document context and source metadata.
    7. Generate an answer using Ollama.
    8. Validate the generated answer.
    9. Fall back to extractive answering if necessary.
    10. Return the answer with source metadata.
    """

    # STEP 1: Validate the question.
    if not question or not question.strip():
        return {
            "question": question,
            "answer": "Please provide a valid question.",
            "sources": [],
            "model": "none",
        }

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # STEP 2: Check whether documents exist.
    if vector_store.count() == 0:
        return {
            "question": question,
            "answer": NO_DOCUMENTS_MESSAGE,
            "sources": [],
            "model": "none",
        }

    # STEP 3: Rewrite the question for retrieval.
    retrieval_query = build_retrieval_query(
        question=question,
        conversation_history=conversation_history,
    )

    # STEP 4: Perform hybrid search.
    search_results = hybrid_search.search(
        query=retrieval_query,
        top_k=top_k,
    )

    if not search_results:
        return {
            "question": question,
            "answer": NO_RELEVANT_INFORMATION,
            "sources": [],
            "model": "none",
        }

    # STEP 5: Filter retrieved context.
    filtered_results = filter_context(
        question=retrieval_query,
        search_results=search_results,
    )

    if not filtered_results:
        return {
            "question": question,
            "answer": NO_RELEVANT_INFORMATION,
            "sources": [],
            "model": "none",
        }

    # STEP 6: Build context and source metadata.
    context, sources = build_context(filtered_results)

    if not context:
        return {
            "question": question,
            "answer": NO_RELEVANT_INFORMATION,
            "sources": [],
            "model": "none",
        }

    # STEP 7: Generate an answer using Ollama.
    prompt = build_rag_prompt(
        question=question,
        context=context,
        conversation_history=conversation_history,
    )

    try:
        generated_answer = ollama_service.generate(
            prompt,
            temperature=0.0,
        )

        # STEP 8: Validate the generated answer.
        validation_result = validate_answer(
            answer=generated_answer,
            context=context,
            minimum_grounding_score=0.5,
        )

        if validation_result["is_valid"]:
            return {
                "question": question,
                "answer": validation_result["answer"],
                "sources": sources,
                "model": "ollama",
            }

        logger.warning(
            "Ollama answer rejected. Reason: %s. "
            "Grounding score: %s. "
            "Attempting extractive fallback.",
            validation_result["reason"],
            validation_result["grounding_score"],
        )

    except Exception:
        logger.exception(
            "Ollama generation failed. "
            "Attempting extractive fallback."
        )

    # STEP 9: Fall back to extractive answering.
    extracted_answer = answer_from_context(
        question=question,
        context=context,
        conversation_history=conversation_history,
    )

    if extracted_answer:
        return {
            "question": question,
            "answer": extracted_answer,
            "sources": sources,
            "model": "extractive",
        }

    # STEP 10: Neither answering method produced an acceptable answer.
    return {
        "question": question,
        "answer": UNSUPPORTED_ANSWER,
        "sources": sources,
        "model": "none",
    }