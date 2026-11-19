
import re


# ============================================
# CONVERSATION CONFIGURATION
# ============================================

MAX_HISTORY_EXCHANGES = 5


# ============================================
# INITIALIZE CHAT HISTORY
# ============================================

def initialize_chat_history():
    """
    Initialize an empty conversation history.
    """

    return []


# ============================================
# ADD MESSAGE TO CHAT HISTORY
# ============================================

def add_message(chat_history, question, answer, sources=None):
    """
    Store a question, answer, and sources in chat history.
    Keep only the latest five exchanges.
    """

    chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources or []
    })

    if len(chat_history) > MAX_HISTORY_EXCHANGES:
        del chat_history[:-MAX_HISTORY_EXCHANGES]

    return chat_history


# ============================================
# FORMAT CHAT HISTORY
# ============================================

def format_chat_history(chat_history):
    """
    Convert chat history into readable text.
    """

    if not chat_history:
        return ""

    formatted_history = []

    for message in chat_history:

        formatted_history.append(
            f"User: {message.get('question', '')}"
        )

        formatted_history.append(
            f"Assistant: {message.get('answer', '')}"
        )

    return "\n".join(formatted_history)


# ============================================
# EXTRACT CONVERSATION TOPIC
# ============================================

def extract_conversation_topic(chat_history):
    """
    Find the most recent explicitly introduced topic.

    Recognizes questions such as:
    - What is deep learning?
    - What is the difference between INNER JOIN and LEFT JOIN?
    - Explain machine learning.
    - Define neural networks.
    """

    if not chat_history:
        return None

    topic_patterns = [

        r"what\s+is\s+the\s+difference\s+between\s+(.+?)[?.!]*$",

        r"what\s+is\s+(.+?)[?.!]*$",

        r"what\s+are\s+(.+?)[?.!]*$",

        r"explain\s+(.+?)[?.!]*$",

        r"define\s+(.+?)[?.!]*$",

        r"tell\s+me\s+about\s+(.+?)[?.!]*$"

    ]

    ignored_topics = {
        "it",
        "its",
        "they",
        "their",
        "this",
        "that",
        "these",
        "those"
    }

    for message in reversed(chat_history):

        question = message.get("question", "").strip()

        if not question:
            continue

        for pattern in topic_patterns:

            match = re.fullmatch(
                pattern,
                question,
                flags=re.IGNORECASE
            )

            if match:

                topic = match.group(1).strip()

                # Remove unrelated trailing clauses
                topic = re.split(
                    r",?\s+and\s+how\b",
                    topic,
                    maxsplit=1,
                    flags=re.IGNORECASE
                )[0]

                topic = topic.strip(" .?!")

                if topic.lower() not in ignored_topics:
                    return topic

    return None


# ============================================
# REWRITE FOLLOW-UP USER QUERY
# ============================================

def rewrite_user_query(question, chat_history):
    """
    Convert context-dependent follow-up questions
    into complete questions using the conversation topic.
    """

    if not chat_history:
        return question

    topic = extract_conversation_topic(chat_history)

    if not topic:
        return question

    question = question.strip()

    is_comparison = " and " in topic.lower()

    follow_up_patterns = [

        # ARCHITECTURE
        (
            r"(?:explain\s+)?(?:its|their)\s+architecture"
            r"(?:\s+in\s+detail)?[?.!]*",
            f"Explain the architecture of {topic} in detail."
        ),
        (
            r"what\s+is\s+(?:its|their)\s+architecture[?.!]*",
            f"What is the architecture of {topic}?"
        ),
        (
            r"explain\s+the\s+architecture"
            r"(?:\s+in\s+detail)?[?.!]*",
            f"Explain the architecture of {topic} in detail."
        ),

        # APPLICATIONS
        (
            r"(?:what\s+are\s+)?(?:its|their)\s+applications[?.!]*"
            r"|(?:what\s+are\s+)?applications[?.!]*",
            f"What are the applications of {topic}?"
        ),

        # ADVANTAGES
        (
            r"(?:what\s+are\s+)?(?:its|their)\s+advantages[?.!]*"
            r"|(?:what\s+are\s+)?advantages[?.!]*",
            f"What are the advantages of {topic}?"
        ),

        # DISADVANTAGES
        (
            r"(?:what\s+are\s+)?(?:its|their)\s+disadvantages[?.!]*"
            r"|(?:what\s+are\s+)?disadvantages[?.!]*",
            f"What are the disadvantages of {topic}?"
        ),

        # TYPES
        (
            r"(?:what\s+are\s+)?(?:its|their)\s+types[?.!]*"
            r"|(?:what\s+are\s+)?types[?.!]*",
            f"What are the types of {topic}?"
        ),

        # WORKING
        (
            r"how\s+does\s+it\s+work[?.!]*",
            (
                f"How do {topic} work?"
                if is_comparison
                else f"How does {topic} work?"
            )
        ),
        (
            r"how\s+do\s+they\s+work[?.!]*",
            f"How do {topic} work?"
        ),
        (
            r"explain\s+how\s+it\s+works[?.!]*",
            (
                f"Explain how {topic} work."
                if is_comparison
                else f"Explain how {topic} works."
            )
        ),
        (
            r"explain\s+how\s+they\s+work[?.!]*",
            f"Explain how {topic} work."
        ),

        # EXAMPLES
        (
            r"(?:can\s+you\s+)?give\s+me?\s+an?\s+example[?.!]*",
            f"Give an example of {topic}."
        ),
        (
            r"explain\s+(?:it|them)\s+with\s+an?\s+example[?.!]*",
            f"Explain {topic} with an example."
        ),

        # LIMITATIONS
        (
            r"(?:what\s+are\s+)?(?:its|their)\s+limitations[?.!]*"
            r"|(?:what\s+are\s+)?limitations[?.!]*",
            f"What are the limitations of {topic}?"
        ),

        # PERFORMANCE
        (
            r"what\s+are\s+(?:the|their)\s+differences"
            r"\s+in\s+performance[?.!]*",
            f"What are the performance differences between {topic}?"
        ),
        (
            r"how\s+do\s+they\s+compare\s+in\s+performance[?.!]*",
            f"How do {topic} compare in performance?"
        ),

        # GENERAL DIFFERENCES
        (
            r"what\s+are\s+their\s+differences[?.!]*",
            f"What are the differences between {topic}?"
        ),

        # COMPARISON
        (
            r"compare\s+it\s+with\s+machine\s+learning[?.!]*",
            f"Compare {topic} with machine learning."
        ),
        (
            r"how\s+is\s+it\s+different\s+from\s+"
            r"traditional\s+(?:ml|machine\s+learning)[?.!]*",
            f"How is {topic} different from traditional machine learning?"
        ),

        # AUTOMATIC FEATURE LEARNING
        (
            r"how\s+does\s+it\s+learn\s+features\s+automatically[?.!]*",
            f"How does {topic} learn features automatically?"
        )
    ]

    # MATCH FOLLOW-UP QUESTION
    for pattern, rewritten_query in follow_up_patterns:

        if re.fullmatch(
            pattern,
            question,
            flags=re.IGNORECASE
        ):
            return rewritten_query

    # RETURN ORIGINAL QUESTION IF NO PATTERN MATCHES
    return question


# ============================================
# GENERATE CONVERSATIONAL RAG RESPONSE
# ============================================

def generate_conversational_response(
    question,
    embedding_model,
    original_index,
    original_documents,
    uploaded_index,
    uploaded_documents,
    client,
    model_name,
    chat_history=None,
    top_k=5
):
    """
    Rewrite follow-up questions, generate a RAG response,
    and maintain conversation history.
    """

    from src.rag_pipeline import generate_rag_response

    if chat_history is None:
        chat_history = initialize_chat_history()

    # Rewrite follow-up question
    rewritten_question = rewrite_user_query(
        question,
        chat_history
    )

    # Generate RAG response
    response = generate_rag_response(
        question=rewritten_question,
        embedding_model=embedding_model,
        original_index=original_index,
        original_documents=original_documents,
        uploaded_index=uploaded_index,
        uploaded_documents=uploaded_documents,
        client=client,
        model_name=model_name,
        chat_history=chat_history,
        top_k=top_k,
        retrieval_query=rewritten_question
    )

    # Preserve original question
    response["question"] = question

    response["rewritten_question"] = rewritten_question

    response["retrieval_query"] = rewritten_question

    # Save successful conversation exchange
    if response.get("success"):

        add_message(
            chat_history,
            question,
            response.get("answer", ""),
            response.get("sources", [])
        )

    return response
