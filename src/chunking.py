
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================
# 1. CHUNKING CONFIGURATION
# ============================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ============================================
# 2. INITIALIZE TEXT SPLITTER
# ============================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# ============================================
# 3. QUESTION-ANSWER DETECTION
# ============================================

QA_PATTERN = re.compile(
    r"^[ \t]*Q:[ \t]*(.*?)\n"
    r"[ \t]*A:[ \t]*(.*?)"
    r"(?=^[ \t]*Q:[ \t]*|\Z)",
    re.MULTILINE | re.DOTALL
)


# ============================================
# 4. STRUCTURED SQL SECTION DETECTION
# ============================================

SQL_SECTION_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?:"
    r"INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL OUTER JOIN|"
    r"CROSS JOIN|SELF JOIN|SQL JOINS|JOIN SUMMARY|"
    r"SQL FUNDAMENTALS|WHERE AND HAVING|"
    r"PRIMARY KEY|FOREIGN KEY"
    r")"
    r"[ \t]*$",
    re.MULTILINE | re.IGNORECASE
)


# ============================================
# 5. DETECT SPECIAL SQL JOIN PAGE
# ============================================

def is_sql_join_page(text):

    return (
        "SQL Fundamentals and Practical Reference" in text
        and re.search(
            r"(?im)^[ \t]*Join summary[ \t]*$",
            text
        ) is not None
    )


# ============================================
# 6. PARSE STRUCTURED SQL JOIN PAGE
# ============================================

def parse_sql_join_page(text):

    summary_match = re.search(
        r"(?im)^[ \t]*Join summary[ \t]*$",
        text
    )

    if not summary_match:
        raise ValueError("JOIN summary not found.")

    main_text = text[:summary_match.start()]
    summary_text = text[summary_match.start():].strip()

    heading_pattern = re.compile(
        r"(?im)^[ \t]*(?:"
        r"INNER JOIN|"
        r"LEFT JOIN|"
        r"RIGHT JOIN and FULL OUTER JOIN|"
        r"CROSS JOIN and SELF JOIN"
        r")[ \t]*$"
    )

    headings = list(heading_pattern.finditer(main_text))

    if len(headings) != 4:
        raise ValueError(
            f"Expected 4 main heading blocks, found {len(headings)}."
        )

    # ----------------------------------------
    # Extract complete SQL queries
    # ----------------------------------------

    query_pattern = re.compile(
        r"(?ims)"
        r"(?:^[ \t]*--[^\n]*\n)?"
        r"^[ \t]*SELECT\b.*?;"
    )

    queries = [
        match.group().strip()
        for match in query_pattern.finditer(main_text)
    ]

    if len(queries) != 6:
        raise ValueError(
            f"Expected 6 SQL queries, found {len(queries)}."
        )

    # ----------------------------------------
    # Identify the JOIN type of each query
    # ----------------------------------------

    query_by_type = {}

    for query in queries:

        upper_query = query.upper()

        if "FULL OUTER JOIN" in upper_query:
            join_type = "FULL OUTER JOIN"

        elif "CROSS JOIN" in upper_query:
            join_type = "CROSS JOIN"

        elif "LEFT JOIN EMPLOYEES AS M" in upper_query:
            join_type = "SELF JOIN"

        elif "RIGHT JOIN" in upper_query:
            join_type = "RIGHT JOIN"

        elif "LEFT JOIN" in upper_query:
            join_type = "LEFT JOIN"

        elif "INNER JOIN" in upper_query:
            join_type = "INNER JOIN"

        else:
            raise ValueError(
                "Could not identify JOIN type:\n" + query
            )

        if join_type in query_by_type:
            raise ValueError(
                f"Duplicate SQL query for {join_type}."
            )

        query_by_type[join_type] = query

    # ----------------------------------------
    # Extract explanations
    # ----------------------------------------

    descriptions = {}

    for index, heading in enumerate(headings):

        start = heading.end()

        if index + 1 < len(headings):
            end = headings[index + 1].start()
        else:
            end = len(main_text)

        block = main_text[start:end].strip()

        # Remove complete SQL queries
        block = re.sub(
            query_pattern,
            "",
            block
        )

        # Remove remaining SQL comments
        block = re.sub(
            r"(?m)^[ \t]*--.*$",
            "",
            block
        ).strip()

        heading_name = heading.group().strip()

        if heading_name == "RIGHT JOIN and FULL OUTER JOIN":

            parts = re.split(
                r"(?i)\bFULL OUTER JOIN returns\b",
                block,
                maxsplit=1
            )

            if len(parts) != 2:
                raise ValueError(
                    "Could not separate RIGHT and FULL descriptions."
                )

            descriptions["RIGHT JOIN"] = re.sub(
                r"(?i)^RIGHT JOIN[ \t]*",
                "",
                parts[0]
            ).strip()

            descriptions["FULL OUTER JOIN"] = (
                "Returns " + parts[1].strip()
            )

        elif heading_name == "CROSS JOIN and SELF JOIN":

            parts = re.split(
                r"(?i)\bA self join joins\b",
                block,
                maxsplit=1
            )

            if len(parts) != 2:
                raise ValueError(
                    "Could not separate CROSS and SELF descriptions."
                )

            descriptions["CROSS JOIN"] = re.sub(
                r"(?i)^CROSS JOIN[ \t]*",
                "",
                parts[0]
            ).strip()

            descriptions["SELF JOIN"] = (
                "A self join joins " + parts[1].strip()
            )

        else:

            descriptions[heading_name] = block

    # ----------------------------------------
    # Validate all six JOIN types
    # ----------------------------------------

    join_order = [
        "INNER JOIN",
        "LEFT JOIN",
        "RIGHT JOIN",
        "FULL OUTER JOIN",
        "CROSS JOIN",
        "SELF JOIN"
    ]

    for join_type in join_order:

        if not descriptions.get(join_type):
            raise ValueError(
                f"Missing explanation: {join_type}"
            )

        if join_type not in query_by_type:
            raise ValueError(
                f"Missing SQL query: {join_type}"
            )

    # ----------------------------------------
    # Build final structured chunks
    # ----------------------------------------

    chunks = []

    introduction = main_text[:headings[0].start()].strip()

    if introduction:
        chunks.append({
            "section": "Introduction",
            "text": introduction
        })

    for join_type in join_order:

        combined_text = (
            f"{join_type}\n"
            f"{descriptions[join_type]}\n\n"
            f"{query_by_type[join_type]}"
        )

        chunks.append({
            "section": join_type,
            "text": combined_text
        })

    chunks.append({
        "section": "JOIN Summary",
        "text": summary_text
    })

    return chunks


# ============================================
# 7. SPLIT TEXT INTO STRUCTURED SECTIONS
# ============================================

def split_into_sections(text):

    matches = list(SQL_SECTION_PATTERN.finditer(text))

    if not matches:
        return [text] if text.strip() else []

    sections = []

    if text[:matches[0].start()].strip():
        sections.append(text[:matches[0].start()].strip())

    for index, match in enumerate(matches):

        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        section = text[start:end].strip()

        if section:
            sections.append(section)

    return sections


# ============================================
# 8. SPLIT LONG SECTIONS
# ============================================

def split_long_section(section):

    if len(section) <= CHUNK_SIZE:
        return [section]

    return text_splitter.split_text(section)


# ============================================
# 9. QUESTION-ANSWER AWARE CHUNKING
# ============================================

def split_text_preserving_qa(text):

    # Use the specialized parser for the SQL JOIN page
    if is_sql_join_page(text):

        structured_chunks = parse_sql_join_page(text)

        return [
            chunk["text"]
            for chunk in structured_chunks
        ]

    # Preserve normal Q&A pairs
    matches = list(QA_PATTERN.finditer(text))

    if not matches:

        sections = split_into_sections(text)

        chunks = []

        for section in sections:
            chunks.extend(
                split_long_section(section)
            )

        return chunks

    chunks = []
    previous_end = 0

    for match in matches:

        preceding_text = text[previous_end:match.start()]

        if preceding_text.strip():

            sections = split_into_sections(preceding_text)

            for section in sections:
                chunks.extend(
                    split_long_section(section)
                )

        question = match.group(1).strip()
        answer = match.group(2).strip()

        qa_text = (
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

        chunks.extend(
            split_long_section(qa_text)
        )

        previous_end = match.end()

    remaining_text = text[previous_end:]

    if remaining_text.strip():

        sections = split_into_sections(remaining_text)

        for section in sections:
            chunks.extend(
                split_long_section(section)
            )

    return chunks


# ============================================
# 10. CHUNK DOCUMENTS
# ============================================

def chunk_documents(documents):

    chunked_documents = []

    for document in documents:

        text = document["text"]

        chunks = split_text_preserving_qa(text)

        for chunk_index, chunk in enumerate(chunks):

            chunk_document = {
                "id": (
                    f"{document['id']}_chunk_"
                    f"{chunk_index + 1}"
                ),
                "text": chunk,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": chunk_index + 1,
                    "total_chunks": len(chunks)
                }
            }

            chunked_documents.append(chunk_document)

    return chunked_documents
