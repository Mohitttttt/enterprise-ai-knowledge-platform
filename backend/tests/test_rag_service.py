from unittest.mock import patch

import pytest

from app.services.rag_service import generate_rag_answer


# --------------------------------------------------
# TEST 1: Annual Leave - Extractive RAG
# --------------------------------------------------

def test_generate_rag_answer_annual_leave():

    search_results = [
        {
            "id": "chunk-1",
            "document": "Employees receive 30 days of paid annual leave.",
            "metadata": {
                "original_filename": "company_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            },
            "rrf_score": 0.0325,
        }
    ]

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid, patch(
        "app.services.rag_service.ollama_service"
    ) as mock_llm:

        mock_store.count.return_value = 1
        mock_hybrid.search.return_value = search_results

        result = generate_rag_answer(
            "How many days of annual leave do employees receive?"
        )

        assert "30 days" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["sources"][0]["filename"] == "company_policy.txt"
        assert result["model"] == "extractive"

        mock_hybrid.search.assert_called_once()
        mock_llm.generate.assert_not_called()


# --------------------------------------------------
# TEST 2: Unsupported Question - No Retrieved Results
# --------------------------------------------------

def test_generate_rag_answer_no_relevant_results():

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid, patch(
        "app.services.rag_service.ollama_service"
    ) as mock_llm:

        mock_store.count.return_value = 1
        mock_hybrid.search.return_value = []

        result = generate_rag_answer(
            "What is the company's annual employee bonus?"
        )

        assert result["sources"] == []
        assert "couldn't find sufficiently relevant information" in (
            result["answer"]
        )

        mock_llm.generate.assert_not_called()


# --------------------------------------------------
# TEST 3: No Documents Available
# --------------------------------------------------

def test_generate_rag_answer_no_documents():

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid:

        mock_store.count.return_value = 0

        result = generate_rag_answer(
            "What are the working hours?"
        )

        assert result["sources"] == []
        assert result["answer"] == "No documents have been uploaded yet."

        mock_hybrid.search.assert_not_called()


# --------------------------------------------------
# TEST 4: Empty Question
# --------------------------------------------------

def test_generate_rag_answer_empty_question():

    result = generate_rag_answer("")

    assert result["answer"] == "Please provide a valid question."
    assert result["sources"] == []


# --------------------------------------------------
# TEST 5: Invalid Top K
# --------------------------------------------------

def test_generate_rag_answer_invalid_top_k():

    with pytest.raises(ValueError):
        generate_rag_answer(
            question="What are the working hours?",
            top_k=0,
        )


# --------------------------------------------------
# TEST 6: LLM Fallback
# --------------------------------------------------

def test_generate_rag_answer_uses_llm():

    search_results = [
        {
            "id": "chunk-1",
            "document": "Working hours are Monday to Friday, 9 AM to 5 PM.",
            "metadata": {
                "original_filename": "employee_handbook.txt",
                "document_id": "doc-2",
                "chunk_index": 1,
            },
            "rrf_score": 0.025,
        }
    ]

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid, patch(
        "app.services.rag_service.answer_from_context",
        return_value=None,
    ), patch(
        "app.services.rag_service.ollama_service"
    ) as mock_llm:

        mock_store.count.return_value = 1
        mock_hybrid.search.return_value = search_results

        mock_llm.generate.return_value = (
            "Working hours are Monday to Friday, "
            "from 9 AM to 5 PM."
        )

        result = generate_rag_answer(
            "What are the working hours?"
        )

        assert "Monday to Friday" in result["answer"]
        assert "9 AM to 5 PM" in result["answer"]
        assert result["model"] == "ollama"

        mock_llm.generate.assert_called_once()


# --------------------------------------------------
# TEST 7: Empty LLM Response
# --------------------------------------------------

def test_generate_rag_answer_empty_llm_response():

    search_results = [
        {
            "id": "chunk-1",
            "document": "Employees receive 30 days of annual leave.",
            "metadata": {
                "original_filename": "company_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            },
            "rrf_score": 0.03,
        }
    ]

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid, patch(
        "app.services.rag_service.answer_from_context",
        return_value=None,
    ), patch(
        "app.services.rag_service.ollama_service"
    ) as mock_llm:

        mock_store.count.return_value = 1
        mock_hybrid.search.return_value = search_results
        mock_llm.generate.return_value = ""

        result = generate_rag_answer(
            "How many days of annual leave do employees receive?"
        )

        assert result["answer"] == (
            "I couldn't find sufficient information "
            "in the uploaded documents."
        )

        assert result["model"] == "ollama"


# --------------------------------------------------
# TEST 8: RAG Uses Filtered Context
# --------------------------------------------------

def test_generate_rag_answer_uses_filtered_context():

    search_results = [
        {
            "id": "chunk-1",
            "document": (
                "Employees can reset their company passwords "
                "through the employee portal. "
                "If they cannot access the portal, they should "
                "contact the IT Support team. "
                "Employees must never share their passwords "
                "with another person. "
                "Standard working hours are Monday to Friday."
            ),
            "metadata": {
                "original_filename": "company_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            },
            "rrf_score": 0.0325,
        }
    ]

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid, patch(
        "app.services.rag_service.answer_from_context",
        return_value=None,
    ), patch(
        "app.services.rag_service.ollama_service"
    ) as mock_llm:

        mock_store.count.return_value = 1
        mock_hybrid.search.return_value = search_results

        mock_llm.generate.return_value = (
            "Employees can reset passwords through "
            "the employee portal or contact IT Support."
        )

        result = generate_rag_answer(
            "How can employees recover their company passwords?"
        )

        prompt = mock_llm.generate.call_args.args[0]

        assert "employee portal" in prompt
        assert "IT Support team" in prompt
        assert "never share their passwords" in prompt

        assert "Standard working hours" not in prompt

        assert len(result["sources"]) == 1
        assert result["sources"][0]["filename"] == (
            "company_policy.txt"
        )

        assert result["model"] == "ollama"
        
        # --------------------------------------------------
# TEST 9: Reject Prompt Leakage from LLM
# --------------------------------------------------

def test_generate_rag_answer_rejects_prompt_leakage():

    search_results = [
        {
            "id": "chunk-1",
            "document": "Employees receive 30 days of paid annual leave.",
            "metadata": {
                "original_filename": "company_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            },
            "rrf_score": 0.0325,
        }
    ]

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid, patch(
        "app.services.rag_service.answer_from_context",
        return_value=None,
    ), patch(
        "app.services.rag_service.ollama_service"
    ) as mock_llm:

        mock_store.count.return_value = 1
        mock_hybrid.search.return_value = search_results

        mock_llm.generate.return_value = (
            "DOCUMENT CONTEXT:\n"
            "Employees receive 30 days of paid annual leave."
        )

        result = generate_rag_answer(
            "Explain the annual leave policy."
        )

        assert result["answer"] == (
            "I couldn't find sufficient information "
            "in the uploaded documents."
        )


# --------------------------------------------------
# TEST 10: Reject Unsupported LLM Answer
# --------------------------------------------------

def test_generate_rag_answer_rejects_unsupported_answer():

    search_results = [
        {
            "id": "chunk-1",
            "document": "Employees receive 30 days of paid annual leave.",
            "metadata": {
                "original_filename": "company_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            },
            "rrf_score": 0.0325,
        }
    ]

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid, patch(
        "app.services.rag_service.answer_from_context",
        return_value=None,
    ), patch(
        "app.services.rag_service.ollama_service"
    ) as mock_llm:

        mock_store.count.return_value = 1
        mock_hybrid.search.return_value = search_results

        mock_llm.generate.return_value = (
            "Employees receive unlimited paid holidays "
            "and free international flights."
        )

        result = generate_rag_answer(
            "Explain the annual leave policy."
        )

        assert result["answer"] == (
            "I couldn't find sufficient information "
            "in the uploaded documents."
        )


# --------------------------------------------------
# TEST 11: Accept Valid Grounded LLM Answer
# --------------------------------------------------

def test_generate_rag_answer_accepts_grounded_answer():

    search_results = [
        {
            "id": "chunk-1",
            "document": (
                "Employees receive 30 days of paid annual leave "
                "per calendar year."
            ),
            "metadata": {
                "original_filename": "company_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            },
            "rrf_score": 0.0325,
        }
    ]

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid, patch(
        "app.services.rag_service.answer_from_context",
        return_value=None,
    ), patch(
        "app.services.rag_service.ollama_service"
    ) as mock_llm:

        mock_store.count.return_value = 1
        mock_hybrid.search.return_value = search_results

        mock_llm.generate.return_value = (
            "Employees receive 30 days of paid annual leave "
            "per calendar year."
        )

        result = generate_rag_answer(
            "Explain the annual leave entitlement."
        )

        assert "30 days" in result["answer"]
        assert result["model"] == "ollama"
        assert len(result["sources"]) == 1

def test_generate_rag_answer_uses_rewritten_query_for_retrieval():

    search_results = [
        {
            "id": "chunk-1",
            "document": (
                "Annual leave requests must be approved "
                "by the employee's direct manager."
            ),
            "metadata": {
                "original_filename": "company_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            },
            "rrf_score": 0.0325,
        }
    ]

    conversation_history = [
        {
            "role": "user",
            "content": (
                "How many annual leave days "
                "do employees receive?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Employees receive 30 days "
                "of paid annual leave."
            ),
        },
    ]

    rewritten_query = (
        "Who approves annual leave requests?"
    )

    with patch(
        "app.services.rag_service.vector_store"
    ) as mock_store, patch(
        "app.services.rag_service.hybrid_search"
    ) as mock_hybrid, patch(
        "app.services.rag_service.rewrite_query",
        return_value=rewritten_query,
    ) as mock_rewriter, patch(
        "app.services.rag_service.filter_context",
        return_value=search_results,
    ) as mock_filter, patch(
        "app.services.rag_service.answer_from_context",
        return_value=(
            "Annual leave requests must be approved "
            "by the employee's direct manager."
        ),
    ) as mock_answer:

        mock_store.count.return_value = 1
        mock_hybrid.search.return_value = search_results

        result = generate_rag_answer(
            question="Who approves it?",
            conversation_history=conversation_history,
        )

        mock_rewriter.assert_called_once_with(
            question="Who approves it?",
            conversation_history=conversation_history,
        )

        mock_hybrid.search.assert_called_once_with(
            query=rewritten_query,
            top_k=5,
        )

        mock_filter.assert_called_once_with(
            question=rewritten_query,
            search_results=search_results,
        )

        mock_answer.assert_called_once_with(
            question="Who approves it?",
            context=mock_answer.call_args.kwargs["context"],
            conversation_history=conversation_history,
        )

        assert "direct manager" in result["answer"]
        assert result["model"] == "extractive"
        assert len(result["sources"]) == 1
