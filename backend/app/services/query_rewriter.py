from app.services.llm_service import ollama_service


MAX_HISTORY_MESSAGES = 8
MAX_QUERY_LENGTH = 500

INVALID_RESPONSE_PATTERNS = [
    "i can't provide",
    "i cannot provide",
    "i can't help",
    "i cannot help",
    "is there anything else i can help",
    "as an ai",
    "i am unable to",
]


def build_rule_based_query(
    question: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Build a contextual retrieval query without using an LLM.

    Combines the latest previous user question with the
    current question to preserve conversational context.
    """

    original_question = question.strip()

    if not conversation_history:
        return original_question

    previous_user_question = None

    for message in reversed(conversation_history):
        if message.get("role") == "user":
            content = message.get("content", "").strip()

            if content:
                previous_user_question = content
                break

    if not previous_user_question:
        return original_question

    if previous_user_question.lower() == original_question.lower():
        return original_question

    contextual_query = (
        f"{previous_user_question} {original_question}"
    )

    return contextual_query[:MAX_QUERY_LENGTH]


def build_rewriting_prompt(
    question: str,
    conversation_history: list[dict],
) -> str:
    """
    Build a prompt to convert a conversational question
    into a standalone search query.
    """

    history = conversation_history[-MAX_HISTORY_MESSAGES:]

    formatted_history = []

    for message in history:
        role = message.get("role", "").upper()
        content = message.get("content", "").strip()

        if role in {"USER", "ASSISTANT"} and content:
            formatted_history.append(
                f"{role}: {content}"
            )

    history_text = "\n".join(formatted_history)

    return f"""
You are a search query rewriting engine.

Your ONLY task is to rewrite the latest user question
into a standalone question suitable for searching documents.

IMPORTANT:
- Never answer the user's question.
- Never refuse the user's question.
- Never provide advice or explanations.
- Resolve pronouns using conversation history.
- Preserve the original meaning.
- Do not invent facts.
- Return exactly ONE standalone search query.
- Do not include quotation marks or explanations.

EXAMPLES:

Conversation:
USER: How many annual leave days do employees receive?
ASSISTANT: Employees receive 30 days of paid annual leave.

Latest question:
Who approves it?

Correct output:
Who approves annual leave requests?

Another example:

Conversation:
USER: What is the company's sick leave policy?
ASSISTANT: Employees can take paid sick leave.

Latest question:
What documents do I need?

Correct output:
What documents are required for taking sick leave?

Another example:

Conversation:
USER: What are the working hours?
ASSISTANT: Employees work Monday to Friday.

Latest question:
What about weekends?

Correct output:
What are the company's weekend working hours?

NOW REWRITE THIS QUESTION:

CONVERSATION HISTORY:
{history_text}

LATEST USER QUESTION:
{question}

STANDALONE SEARCH QUERY:
""".strip()


def rewrite_query(
    question: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Rewrite a conversational question into a standalone
    search query using Ollama.

    Uses a rule-based contextual fallback if rewriting fails.
    """

    original_question = question.strip()

    if not original_question:
        return original_question

    if not conversation_history:
        return original_question

    fallback_query = build_rule_based_query(
        question=original_question,
        conversation_history=conversation_history,
    )

    prompt = build_rewriting_prompt(
        question=original_question,
        conversation_history=conversation_history,
    )

    try:
        rewritten_query = ollama_service.generate(
            prompt,
            temperature=0.0,
        )

    except Exception:
        return fallback_query

    if not rewritten_query:
        return fallback_query

    rewritten_query = rewritten_query.strip()
    rewritten_query = rewritten_query.strip('"').strip("'")

    if not rewritten_query:
        return fallback_query

    if len(rewritten_query) > MAX_QUERY_LENGTH:
        return fallback_query

    if "\n" in rewritten_query:
        return fallback_query

    if "CONVERSATION:" in rewritten_query.upper():
        return fallback_query

    if "OUTPUT ONLY" in rewritten_query.upper():
        return fallback_query

    lowered_response = rewritten_query.lower()

    if any(
        pattern in lowered_response
        for pattern in INVALID_RESPONSE_PATTERNS
    ):
        return fallback_query

    return rewritten_query
