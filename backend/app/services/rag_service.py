from app.services.hybrid_search import hybrid_search
from app.services.vector_store import vector_store
from app.services.context_filter import filter_context
from app.services.context_builder import build_context
from app.services.prompt_builder import build_rag_prompt
from app.services.llm_service import ollama_service
from app.services.answer_service import answer_from_context
from app.services.answer_validator import validate_answer
from app.services.query_rewriter import rewrite_query


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

    The original question remains unchanged.
    Only the retrieval query is rewritten.
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
    7. Try extractive answering.
    8. Use Ollama when extraction is insufficient.
    9. Validate the generated answer.
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

    # STEP 7: Try extractive answering first.
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

    # STEP 8: Fall back to Ollama.
    prompt = build_rag_prompt(
        question=question,
        context=context,
        conversation_history=conversation_history,
    )

    try:
        answer = ollama_service.generate(
            prompt,
            temperature=0.0,
        )

    except Exception:
        answer = None

    # STEP 9: Validate the generated answer.
    validation_result = validate_answer(
        answer=answer,
        context=context,
        minimum_grounding_score=0.5,
    )

    answer = validation_result["answer"]

    # STEP 10: Return the final response.
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "model": "ollama",
    }