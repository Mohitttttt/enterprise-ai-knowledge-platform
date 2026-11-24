from unittest.mock import patch

from app.services.query_rewriter import (
    rewrite_query,
    build_rewriting_prompt,
)


# TEST 1: Rewrite a follow-up question

def test_rewrite_follow_up_question():

    history = [
        {
            "role": "user",
            "content": "How many annual leave days do employees receive?",
        },
        {
            "role": "assistant",
            "content": "Employees receive 30 days of paid annual leave.",
        },
    ]

    with patch(
        "app.services.query_rewriter.ollama_service"
    ) as mock_llm:

        mock_llm.generate.return_value = (
            "Who approves annual leave requests?"
        )

        result = rewrite_query(
            question="Who approves it?",
            conversation_history=history,
        )

        assert result == "Who approves annual leave requests?"
        mock_llm.generate.assert_called_once()


# TEST 2: Preserve standalone questions

def test_preserve_question_without_history():

    result = rewrite_query(
        question="What are the company working hours?",
        conversation_history=None,
    )

    assert result == "What are the company working hours?"


# TEST 3: Build rewriting prompt correctly

def test_build_rewriting_prompt():

    history = [
        {
            "role": "user",
            "content": "What is the leave policy?",
        },
        {
            "role": "assistant",
            "content": "Employees receive 30 days of annual leave.",
        },
    ]

    prompt = build_rewriting_prompt(
        question="Who approves it?",
        conversation_history=history,
    )

    assert "CONVERSATION HISTORY:" in prompt
    assert "What is the leave policy?" in prompt
    assert "Who approves it?" in prompt
    assert "STANDALONE SEARCH QUERY:" in prompt


# TEST 4: Fall back when Ollama fails

def test_fallback_when_ollama_fails():

    history = [
        {
            "role": "user",
            "content": "What is the leave policy?",
        }
    ]

    with patch(
        "app.services.query_rewriter.ollama_service"
    ) as mock_llm:

        mock_llm.generate.side_effect = RuntimeError(
            "Ollama unavailable"
        )

        result = rewrite_query(
            question="Who approves it?",
            conversation_history=history,
        )

        assert result == "What is the leave policy? Who approves it?"


# TEST 5: Fall back on empty LLM response

def test_fallback_on_empty_response():

    history = [
        {
            "role": "user",
            "content": "What is the leave policy?",
        }
    ]

    with patch(
        "app.services.query_rewriter.ollama_service"
    ) as mock_llm:

        mock_llm.generate.return_value = "   "

        result = rewrite_query(
            question="Who approves it?",
            conversation_history=history,
        )

        assert result == "What is the leave policy? Who approves it?"


# TEST 6: Reject multiline response

def test_reject_multiline_response():

    history = [
        {
            "role": "user",
            "content": "What is the leave policy?",
        }
    ]

    with patch(
        "app.services.query_rewriter.ollama_service"
    ) as mock_llm:

        mock_llm.generate.return_value = (
            "Who approves annual leave?\n"
            "The manager approves it."
        )

        result = rewrite_query(
            question="Who approves it?",
            conversation_history=history,
        )

        assert result == "What is the leave policy? Who approves it?"


# TEST 7: Reject excessively long response

def test_reject_excessively_long_response():

    history = [
        {
            "role": "user",
            "content": "What is the leave policy?",
        }
    ]

    with patch(
        "app.services.query_rewriter.ollama_service"
    ) as mock_llm:

        mock_llm.generate.return_value = "A" * 501

        result = rewrite_query(
            question="Who approves it?",
            conversation_history=history,
        )

        assert result == "What is the leave policy? Who approves it?"