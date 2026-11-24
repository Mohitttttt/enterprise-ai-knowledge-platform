import re


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does",
    "did", "how", "what", "when", "where", "who", "why", "which",
    "can", "could", "would", "should", "i", "you", "we", "they",
    "he", "she", "it", "them", "their", "there", "this", "that",
    "to", "of", "in", "on", "for", "with", "and", "or", "me",
    "my", "your", "please", "tell", "about",
    "employee", "employees", "receive", "day", "days", "week",
    "weeks", "year", "years", "paid", "calendar", "per",
    "policy", "request", "requests", "leave"
}


def normalize_word(word: str) -> str:
    if word.startswith("approv"):
        return "approv"

    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]

    return word


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())

    return {
        normalize_word(word)
        for word in words
        if normalize_word(word) not in STOP_WORDS
    }


def extract_sentence_records(context: str) -> list[dict]:
    records = []
    current_section = ""
    current_paragraph = []

    def save_paragraph():
        if not current_paragraph:
            return

        text = " ".join(current_paragraph)

        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                records.append({
                    "section": current_section,
                    "text": sentence
                })

        current_paragraph.clear()

    for line in context.splitlines():
        line = line.strip()

        if not line:
            save_paragraph()
            continue

        if line.startswith("[Source ") or line.startswith("Filename:"):
            save_paragraph()
            current_section = ""
            continue

        if line == "Content:":
            continue

        heading = re.match(r"^\d+\.\s+(.+)$", line)

        if heading:
            save_paragraph()
            current_section = heading.group(1)
            continue

        if (
            len(line) < 80
            and not re.search(r"[.!?]$", line)
            and len(line.split()) <= 8
        ):
            save_paragraph()
            current_section = line
            continue

        current_paragraph.append(line)

    save_paragraph()

    return records


def get_history_tokens(
    conversation_history: list[dict] | None,
) -> set[str]:
    history_tokens = set()

    if conversation_history:
        for message in conversation_history[-6:]:
            if message.get("role") == "user":
                history_tokens.update(
                    tokenize(message.get("content", ""))
                )

    return history_tokens


def answer_from_context(
    question: str,
    context: str,
    conversation_history: list[dict] | None = None,
) -> str | None:

    if not question or not question.strip():
        return None

    if not context or not context.strip():
        return None

    question_tokens = tokenize(question)
    history_tokens = get_history_tokens(conversation_history)

    records = extract_sentence_records(context)

    if not records:
        return None

    scored_records = []

    for record in records:
        sentence = record["text"]
        section = record["section"]

        sentence_tokens = tokenize(sentence)
        section_tokens = tokenize(section)

        available_tokens = sentence_tokens | section_tokens

        # Specific question terms must match the sentence or its section.
        if question_tokens and not (question_tokens & available_tokens):
            continue

        current_matches = question_tokens & sentence_tokens
        section_matches = question_tokens & section_tokens
        history_matches = history_tokens & sentence_tokens

        score = (
            len(current_matches) * 4
            + len(section_matches) * 3
            + len(history_matches)
        )

        if current_matches or section_matches:
            score += 3

        if score >= 4:
            scored_records.append((score, sentence))

    if not scored_records:
        return None

    scored_records.sort(key=lambda item: item[0], reverse=True)

    selected_sentences = {
        sentence for _, sentence in scored_records[:4]
    }

    ordered_sentences = [
        record["text"]
        for record in records
        if record["text"] in selected_sentences
    ]

    return " ".join(ordered_sentences) if ordered_sentences else None