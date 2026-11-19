
# ============================================
# PERSONAL AI KNOWLEDGE ASSISTANT
# Streamlit Application
# ============================================
import os
from dotenv import load_dotenv
from google import genai
import hashlib
import sys
from pathlib import Path

import json
import streamlit as st
import numpy as np


# ============================================
# 1. PROJECT ROOT CONFIGURATION
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================
# GEMINI API CONFIGURATION
# ============================================

MODEL_NAME = "gemini-3.5-flash-lite"


@st.cache_resource
def get_gemini_client():

    # Load environment variables from project root
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY")

    # Validate API key
    if not api_key or not api_key.strip():

        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Check your project .env file."
        )

    # Initialize Gemini client
    client = genai.Client(api_key=api_key.strip())

    return client
# ============================================
# 2. IMPORT RAG MODULES
# ============================================

from src.document_loader import load_document

from src.chunking import chunk_documents

from src.embeddings import (
    load_embedding_model,
    generate_embeddings
)

from src.retrieval import (
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    search_faiss_index,
    search_all_knowledge
)

from src.rag_pipeline import (
    generate_rag_response,
    SIMILARITY_THRESHOLD
)

from src.conversation import (
    initialize_chat_history,
    add_message,
    format_chat_history,
    extract_conversation_topic,
    rewrite_user_query,
    generate_conversational_response
)

# ============================================
# LOAD ORIGINAL KNOWLEDGE BASE
# ============================================

@st.cache_resource
def get_original_faiss_index():

    index_path = (
        PROJECT_ROOT
        / "vectorstore"
        / "knowledge_faiss.index"
    )

    return load_faiss_index(index_path)


@st.cache_data
def get_original_documents():

    documents_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ml_knowledge_full.jsonl"
    )

    documents = []

    with open(
        documents_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if line.strip():
                documents.append(json.loads(line))

    return documents

# ============================================
# UPLOADED KNOWLEDGE BASE LOADER
# ============================================

@st.cache_resource
def get_uploaded_faiss_index():

    index_path = (
        PROJECT_ROOT
        / "vectorstore"
        / "uploaded_documents.index"
    )

    return load_faiss_index(index_path)


@st.cache_data
def get_uploaded_documents():

    documents_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "uploaded_documents.jsonl"
    )

    documents = []

    with open(documents_path, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                documents.append(json.loads(line))

    return documents
# ============================================
# EMBEDDING MODEL
# ============================================

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()
# ============================================
# 3. PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Personal AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# 4. SESSION STATE INITIALIZATION
# ============================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = initialize_chat_history()

if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []

if "uploaded_document_names" not in st.session_state:
    st.session_state.uploaded_document_names = []



# ============================================
# RESTORE SAVED UPLOADED KNOWLEDGE BASE
# ============================================

if (
    "uploaded_index" not in st.session_state
    and "uploaded_embeddings" not in st.session_state
):

    index_path = (
        PROJECT_ROOT
        / "vectorstore"
        / "uploaded_documents.index"
    )

    documents_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "uploaded_documents.jsonl"
    )

    embeddings_path = (
        PROJECT_ROOT
        / "vectorstore"
        / "uploaded_embeddings.npy"
    )

    if (
        index_path.exists()
        and documents_path.exists()
        and embeddings_path.exists()
    ):

        try:

            with st.spinner("Restoring uploaded knowledge base..."):

                uploaded_index = get_uploaded_faiss_index()

                uploaded_documents = get_uploaded_documents()

                uploaded_embeddings = np.load(embeddings_path)

            if (
                uploaded_index.ntotal == len(uploaded_documents)
                and uploaded_index.ntotal == uploaded_embeddings.shape[0]
                and uploaded_index.d == uploaded_embeddings.shape[1]
            ):

                st.session_state.uploaded_index = uploaded_index

                st.session_state.processed_documents = uploaded_documents

                st.session_state.uploaded_embeddings = uploaded_embeddings

                st.session_state.uploaded_document_names = sorted(
                    {
                        document["metadata"].get("source")
                        for document in uploaded_documents
                        if document.get("metadata", {}).get("source")
                    }
                )

                st.success(
                    "Uploaded knowledge base restored successfully!"
                )

                st.write(
                    "Uploaded FAISS vectors:",
                    uploaded_index.ntotal
                )

                st.write(
                    "Uploaded documents:",
                    len(uploaded_documents)
                )

                st.write(
                    "Uploaded embeddings:",
                    uploaded_embeddings.shape
                )

                st.write(
                    "Uploaded embedding dimensions:",
                    uploaded_index.d
                )

                st.write(
                    "Uploaded index type:",
                    type(uploaded_index).__name__
                )

            else:

                st.error(
                    "Uploaded knowledge base validation failed!"
                )

        except Exception as error:

            st.error(
                f"Knowledge base restoration error: {error}"
            )


# ============================================
# ORIGINAL KNOWLEDGE BASE VALIDATION
# ============================================

st.session_state.original_index = None
st.session_state.original_documents = []

try:

    with st.spinner("Loading original knowledge base..."):

        original_index = get_original_faiss_index()

        original_documents = get_original_documents()

    st.session_state.original_index = original_index

    st.session_state.original_documents = original_documents

    if (
        original_index.ntotal == len(original_documents)
        and original_index.d == 384
    ):

        st.success("Original knowledge base loaded successfully!")

        st.write("FAISS vectors:", original_index.ntotal)

        st.write("Original documents:", len(original_documents))

        st.write("Embedding dimensions:", original_index.d)

    else:

        st.error("Original knowledge base validation failed!")

except Exception as error:

    st.error(f"Knowledge base loading error: {error}")




# ============================================
# 5. SIDEBAR
# ============================================

with st.sidebar:

    st.title("⚙️ Settings")

    st.subheader("Knowledge Assistant")

    st.info(
        "Upload documents to build your personal "
        "knowledge base."
    )

    st.divider()

    # ============================================
    # DOCUMENT UPLOAD
    # ============================================

    st.subheader("📂 Upload Documents")

    uploaded_files = st.file_uploader(

        "Choose your documents",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        help="Upload multiple documents to your knowledge base."
    )

    st.divider()

    # ============================================
    # SAVED DOCUMENTS
    # ============================================

    st.subheader("📚 Saved Documents")

    if st.session_state.uploaded_document_names:

        for document_name in st.session_state.uploaded_document_names:

            st.write(f"📄 {document_name}")

            if st.button(
                f"🗑️ Delete {document_name}",
                key=f"delete_{document_name}"
            ):

                st.session_state.document_to_delete = document_name

    else:

        st.info("No previously uploaded documents found.")

    st.divider()
    # ============================================
# CLEAR CHAT
# ============================================

st.subheader("💬 Conversation")

if st.button("🗑️ Clear Chat", use_container_width=True):

    st.session_state.chat_history = initialize_chat_history()

    st.success("Chat history cleared successfully!")

    st.rerun()
    # ============================================
    # SELECTED DOCUMENTS
    # ============================================

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} document(s) selected."
        )

        st.markdown("**Selected files:**")

        for uploaded_file in uploaded_files:

            st.write(f"📄 {uploaded_file.name}")

    else:

        st.caption("No documents uploaded yet.")

    st.divider()

    st.caption(
        "Supported formats: PDF, DOCX, TXT, CSV, XLSX"
    )

    st.caption(
        "Powered by RAG, FAISS and Gemini"
    )




# ============================================
# DISPLAY SELECTED FILES
# ============================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} new document(s) selected."
    )

    st.markdown("**Selected files:**")

    for uploaded_file in uploaded_files:
        st.write(f"📄 {uploaded_file.name}")

else:
    st.caption(
        "No new documents selected. "
        "Previously saved documents remain available."
    )

st.divider()

st.caption("Supported formats: PDF, DOCX, TXT, CSV, XLSX")

st.caption("Powered by RAG, FAISS and Gemini")

# ============================================
# DELETE UPLOADED DOCUMENT
# ============================================

if "document_to_delete" in st.session_state:

    document_name = st.session_state.document_to_delete

    st.warning(f"Deleting document: {document_name}")

    # Identify chunks that belong to other documents
    remaining_documents = [
        document
        for document in st.session_state.processed_documents
        if document.get("metadata", {}).get("source") != document_name
    ]

    # Find the corresponding embedding rows
    keep_indices = [
        index
        for index, document in enumerate(
            st.session_state.processed_documents
        )
        if document.get("metadata", {}).get("source") != document_name
    ]

    remaining_embeddings = (
        st.session_state.uploaded_embeddings[keep_indices]
    )

    # Save updated documents
    documents_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "uploaded_documents.jsonl"
    )

    with open(documents_path, "w", encoding="utf-8") as file:
        for document in remaining_documents:
            file.write(
                json.dumps(document, ensure_ascii=False) + "\n"
            )

    # Save updated embeddings and rebuild FAISS index
    embeddings_path = (
        PROJECT_ROOT
        / "vectorstore"
        / "uploaded_embeddings.npy"
    )

    index_path = (
        PROJECT_ROOT
        / "vectorstore"
        / "uploaded_documents.index"
    )

    if remaining_documents:

        updated_index = create_faiss_index(
            remaining_embeddings
        )

        np.save(embeddings_path, remaining_embeddings)

        save_faiss_index(updated_index, index_path)

        st.session_state.uploaded_index = updated_index
        st.session_state.uploaded_embeddings = remaining_embeddings

    else:

        # Remove uploaded knowledge-base files if no documents remain
        for path in [embeddings_path, index_path]:
            if path.exists():
                path.unlink()

        st.session_state.pop("uploaded_index", None)
        st.session_state.pop("uploaded_embeddings", None)

    # Update session state
    st.session_state.processed_documents = remaining_documents

    st.session_state.uploaded_document_names = [
        name
        for name in st.session_state.uploaded_document_names
        if name != document_name
    ]

    # Remove the raw uploaded file if it exists
    raw_file_path = PROJECT_ROOT / "data" / "raw" / document_name

    if raw_file_path.exists():
        raw_file_path.unlink()

    # Clear cached uploaded knowledge-base data
    get_uploaded_faiss_index.clear()
    get_uploaded_documents.clear()

    # Clear deletion request
    del st.session_state.document_to_delete

    st.success(f"{document_name} deleted successfully!")

    st.rerun()

# ============================================
# DOCUMENT PROCESSING
# ============================================

st.subheader("📑 Document Processing")

if uploaded_files:

    if st.button(
        "Process Uploaded Documents",
        type="primary"
    ):

        all_documents = []
        new_document_names = []

        # Identify hashes of previously processed documents
        existing_hashes = set()

        for document in st.session_state.processed_documents:

            file_hash = document.get(
                "metadata", {}
            ).get("file_hash")

            if file_hash:
                existing_hashes.add(file_hash)

        # Check older documents processed before duplicate detection
        raw_folder = PROJECT_ROOT / "data" / "raw"

        for document_name in st.session_state.uploaded_document_names:

            raw_file_path = raw_folder / document_name

            if raw_file_path.exists():

                with open(raw_file_path, "rb") as file:

                    old_hash = hashlib.sha256(
                        file.read()
                    ).hexdigest()

                existing_hashes.add(old_hash)

        # Extract new documents
        with st.spinner("Checking and extracting documents..."):

            for uploaded_file in uploaded_files:

                # Calculate SHA-256 hash
                file_content = uploaded_file.getvalue()

                file_hash = hashlib.sha256(
                    file_content
                ).hexdigest()

                # Skip duplicate documents
                if file_hash in existing_hashes:

                    st.warning(
                        f"Duplicate skipped: {uploaded_file.name}"
                    )

                    continue

                # Save uploaded file
                temp_file_path = (
                    PROJECT_ROOT
                    / "data"
                    / "raw"
                    / uploaded_file.name
                )

                temp_file_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                with open(temp_file_path, "wb") as file:
                    file.write(file_content)

                # Extract document contents
                documents = load_document(temp_file_path)

                # Store file hash in metadata
                for document in documents:

                    document.setdefault(
                        "metadata", {}
                    )["file_hash"] = file_hash

                all_documents.extend(documents)

                existing_hashes.add(file_hash)

                new_document_names.append(
                    uploaded_file.name
                )

        # Process extracted documents
        if all_documents:

            with st.spinner("Splitting documents into chunks..."):

                chunked_documents = chunk_documents(
                    all_documents
                )
                
            st.write(
               "Total knowledge chunks:",
                len(st.session_state.processed_documents)
           )

            # Preserve previously processed documents
            st.session_state.processed_documents.extend(
                chunked_documents
            )

            # Save all processed documents to JSONL
            documents_path = (
                PROJECT_ROOT
                / "data"
                / "processed"
                / "uploaded_documents.jsonl"
            )

            documents_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                documents_path,
                "w",
                encoding="utf-8"
            ) as file:

                for document in st.session_state.processed_documents:

                    file.write(
                        json.dumps(document, ensure_ascii=False)
                        + "\n"
                    )

            # Update saved filenames without duplicates
            st.session_state.uploaded_document_names = sorted(
                set(
                    st.session_state.uploaded_document_names
                    + new_document_names
                )
            )

            st.success("Document processing completed!")

            st.write(
                "New extracted documents:",
                len(all_documents)
            )

            st.write(
                "New chunks generated:",
                len(chunked_documents)
            )

            st.write(
                "Total knowledge chunks:",
                len(st.session_state.processed_documents)
            )

        else:

            if not new_document_names:

                st.warning(
                    "No new documents to process. "
                    "All selected files are duplicates."
                )

            else:

                st.warning(
                    "No readable text was extracted."
                )


      
# ============================================
# EMBEDDING GENERATION
# ============================================

if st.session_state.processed_documents:

    st.subheader("🧮 Generate Document Embeddings")

    if st.button("Generate Embeddings"):

        # Check the number of processed documents
        st.write(
            "Current processed document count:",
            len(st.session_state.processed_documents)
        )

        # Load embedding model
        with st.spinner("Loading embedding model..."):

            embedding_model = get_embedding_model()

        # Extract text from all processed documents
        document_texts = [
            document["text"]
            for document in st.session_state.processed_documents
        ]

        # Generate embeddings
        with st.spinner("Generating embeddings..."):

            embeddings = generate_embeddings(
                document_texts,
                embedding_model
            )

        # Store embeddings in session state
        st.session_state.uploaded_embeddings = embeddings

        # Save embeddings to disk
        embeddings_path = (
            PROJECT_ROOT
            / "vectorstore"
            / "uploaded_embeddings.npy"
        )

        embeddings_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            embeddings_path,
            embeddings
        )

        # Display results
        st.success(
            "Embeddings generated and saved successfully!"
        )

        st.write(
            "Embedding shape:",
            embeddings.shape
        )

        st.write(
            "Data type:",
            embeddings.dtype
        )

else:

    st.info(
        "Please process documents before generating embeddings."
    )

# ============================================
# FAISS INDEX CREATION
# ============================================

if "uploaded_embeddings" in st.session_state:

    st.subheader("🔍 Build FAISS Vector Index")

    if st.button("Create FAISS Index"):

        embeddings = st.session_state.uploaded_embeddings
        documents = st.session_state.processed_documents

        # Validate document and embedding counts
        if len(embeddings) != len(documents):

            st.error(
                "Document count and embedding count do not match!"
            )

        else:

            with st.spinner("Creating FAISS index..."):

                # Create FAISS index
                uploaded_index = create_faiss_index(
                    embeddings
                )

                # Validate index
                if uploaded_index.ntotal != len(documents):

                    st.error(
                        "FAISS index validation failed!"
                    )

                else:

                    # Save FAISS index to disk
                    index_path = (
                        PROJECT_ROOT
                        / "vectorstore"
                        / "uploaded_documents.index"
                    )

                    index_path.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    save_faiss_index(
                        uploaded_index,
                        index_path
                    )

                    # Store index in session state
                    st.session_state.uploaded_index = uploaded_index

                    # Clear cached loaders so future reloads
                    # use the updated files
                    get_uploaded_faiss_index.clear()
                    get_uploaded_documents.clear()

            if (
                uploaded_index.ntotal == len(documents)
            ):

                st.success(
                    "FAISS index created and saved successfully!"
                )

                st.write(
                    "Total vectors:",
                    uploaded_index.ntotal
                )

                st.write(
                    "Embedding dimensions:",
                    uploaded_index.d
                )

                st.write(
                    "Index type:",
                    type(uploaded_index).__name__
                )

                st.write(
                    "Saved location:",
                    str(index_path)
                )


# ============================================
# SAVE UPLOADED KNOWLEDGE BASE
# ============================================

if "uploaded_index" in st.session_state:

    if st.button("Save Knowledge Base"):

        vectorstore_path = PROJECT_ROOT / "vectorstore"

        vectorstore_path.mkdir(
            parents=True,
            exist_ok=True
        )

        index_path = (
            vectorstore_path / "uploaded_documents.index"
        )

        save_faiss_index(
            st.session_state.uploaded_index,
            index_path
        )

        st.success("FAISS index saved successfully!")

        st.write("Saved location:", str(index_path))
        

# ============================================
# SAVE EMBEDDINGS AND DOCUMENT METADATA
# ============================================

if "uploaded_index" in st.session_state:

    if st.button("Save Embeddings and Documents"):

        import json
        import numpy as np

        vectorstore_path = PROJECT_ROOT / "vectorstore"
        processed_path = PROJECT_ROOT / "data" / "processed"

        vectorstore_path.mkdir(
            parents=True,
            exist_ok=True
        )

        processed_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save embeddings
        embeddings_path = (
            vectorstore_path / "uploaded_embeddings.npy"
        )

        np.save(
            embeddings_path,
            st.session_state.uploaded_embeddings
        )

        # Save document chunks
        documents_path = (
            processed_path / "uploaded_documents.jsonl"
        )

        with open(
            documents_path,
            "w",
            encoding="utf-8"
        ) as file:

            for document in st.session_state.processed_documents:
                file.write(
                    json.dumps(document, ensure_ascii=False) + "\n"
                )

        st.success("Embeddings and document metadata saved!")

        st.write("Embeddings:", str(embeddings_path))
        st.write("Documents:", str(documents_path))



# ============================================
# VALIDATE SAVED KNOWLEDGE BASE
# ============================================

if st.button("Validate Knowledge Base"):

    import json
    import numpy as np

    import faiss

    index_path = (
        PROJECT_ROOT / "vectorstore"
        / "uploaded_documents.index"
    )

    embeddings_path = (
        PROJECT_ROOT / "vectorstore"
        / "uploaded_embeddings.npy"
    )

    documents_path = (
        PROJECT_ROOT / "data" / "processed"
        / "uploaded_documents.jsonl"
    )

    # Load saved FAISS index
    saved_index = faiss.read_index(str(index_path))

    # Load saved embeddings
    saved_embeddings = np.load(embeddings_path)

    # Load saved documents
    with open(
        documents_path,
        "r",
        encoding="utf-8"
    ) as file:

        saved_documents = [
            json.loads(line)
            for line in file
        ]

    # Validate alignment
    st.write("FAISS vectors:", saved_index.ntotal)
    st.write("Embedding shape:", saved_embeddings.shape)
    st.write("Saved documents:", len(saved_documents))

    if (
        saved_index.ntotal == len(saved_documents)
        == saved_embeddings.shape[0]
        and saved_index.d == saved_embeddings.shape[1]
    ):

        st.success("Knowledge base validation successful!")

    else:

        st.error("Knowledge base validation failed!")


# ============================================
# SEARCH UPLOADED DOCUMENTS
# ============================================

if (
    "uploaded_index" in st.session_state
    and "uploaded_embeddings" in st.session_state
):

    st.subheader("🔎 Search Your Uploaded Documents")

    user_query = st.text_input(
        "Enter your question",
        placeholder="Example: What is deep learning?"
    )

    if st.button("Search Documents"):

        if user_query.strip():

            with st.spinner("Searching documents..."):

                embedding_model = get_embedding_model()

                search_results = search_faiss_index(
                    query=user_query,
                    embedding_model=embedding_model,
                    index=st.session_state.uploaded_index,
                    documents=st.session_state.processed_documents,
                    top_k=5
                )

            st.session_state.search_results = search_results

            if search_results:

                st.success(
                    f"Found {len(search_results)} relevant results."
                )

                for i, result in enumerate(search_results, start=1):

                    st.markdown(f"### Result {i}")

                    st.write(
                        "Similarity score:",
                        round(result["score"], 4)
                    )

                    st.write(
                        "Source:",
                        result["metadata"]["source"]
                    )

                    st.write(result["text"])

                    st.divider()

            else:

                st.warning("No matching documents found.")

        else:

            st.warning("Please enter a question.")


# ============================================
# UNIFIED KNOWLEDGE BASE SEARCH
# ============================================

if (
    "original_index" in st.session_state
    and "original_documents" in st.session_state
    and "uploaded_index" in st.session_state
):

    st.subheader("🌐 Search All Knowledge Sources")

    unified_query = st.text_input(
        "Ask a question across all knowledge sources",
        key="unified_query",
        placeholder="Example: Explain deep learning."
    )

    if st.button("Search All Knowledge"):

        if unified_query.strip():

            with st.spinner("Searching all knowledge sources..."):

                embedding_model = get_embedding_model()

                unified_results = search_all_knowledge(
                    query=unified_query,
                    embedding_model=embedding_model,
                    original_index=st.session_state.original_index,
                    original_documents=st.session_state.original_documents,
                    uploaded_index=st.session_state.uploaded_index,
                    uploaded_documents=st.session_state.processed_documents,
                    top_k=5
                )

            st.session_state.unified_results = unified_results

            if unified_results:

                st.success(
                    f"Found {len(unified_results)} relevant results."
                )

                for i, result in enumerate(
                    unified_results, start=1
                ):

                    st.markdown(f"### Result {i}")

                    st.write(
                        "Similarity score:",
                        round(result["score"], 4)
                    )

                    st.write(
                        "Knowledge source:",
                        result["knowledge_source"]
                    )

                    st.write(
                        "Source:",
                        result["metadata"].get("source", "Original knowledge base")
                    )

                    st.write(result["text"])

                    st.divider()

            else:

                st.warning("No relevant results found.")

        else:

            st.warning("Please enter a question.")
  
# ============================================
# 6. MAIN DASHBOARD
# ============================================

st.title("🧠 Personal AI Knowledge Assistant")

st.markdown(
    """
    ### Your intelligent document-based AI assistant

    Upload your documents, ask questions, and get
    context-aware answers using Retrieval-Augmented
    Generation (RAG).
    """
)

st.divider()


# ============================================
# 7. DOCUMENT UPLOAD INFORMATION
# ============================================

st.subheader("📂 Upload Documents")

st.write(
    "Upload your PDF, Word, Excel, CSV or text files "
    "using the sidebar."
)

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} document(s) selected successfully."
    )

else:

    st.info(
        "Select documents from the sidebar to get started."
    )


# ============================================
# 8. CHAT ASSISTANT SECTION
# ============================================

st.subheader("💬 Ask Your Knowledge Assistant")

st.write(
    "Ask questions about your uploaded documents "
    "and the original machine learning knowledge base."
)

# Display previous conversation
for message in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(message["question"])

    with st.chat_message("assistant"):

        st.write(message["answer"])

        # Display sources belonging to this answer
        if message.get("sources"):

            with st.expander("📚 View Reference Sources"):

                for index, source in enumerate(
                    message["sources"], start=1
                ):

                    st.markdown(
                        f"**Source {index}: {source['source']}**"
                    )

                    st.write(
                        "Knowledge source:",
                        source["knowledge_source"]
                    )

                    st.write(
                        "Document ID:",
                        source["document_id"]
                    )

                    st.write(
                        "Similarity:",
                        source["similarity_score"]
                    )

                    if source.get("page_number"):
                        st.write(
                            "Page:",
                            source["page_number"]
                        )

                    if source.get("sheet_name"):
                        st.write(
                            "Sheet:",
                            source["sheet_name"]
                        )

                    st.divider()


# ============================================
# CHAT INPUT
# ============================================

user_question = st.chat_input(
    "Ask a question about your knowledge base..."
)

if user_question:

    # Display user question
    with st.chat_message("user"):
        st.write(user_question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching knowledge and generating answer..."
        ):

            try:

                embedding_model = get_embedding_model()

                client = get_gemini_client()

                # ------------------------------------
                # GET AVAILABLE KNOWLEDGE BASES
                # ------------------------------------

                original_index = st.session_state.original_index
                original_documents = st.session_state.original_documents

                uploaded_index = st.session_state.get(
                    "uploaded_index"
                )

                uploaded_documents = st.session_state.get(
                    "processed_documents",
                    []
                )

                # ------------------------------------
                # GENERATE CONVERSATIONAL RESPONSE
                # ------------------------------------

                response = generate_conversational_response(
                    question=user_question,
                    embedding_model=embedding_model,
                    original_index=original_index,
                    original_documents=original_documents,
                    uploaded_index=uploaded_index,
                    uploaded_documents=uploaded_documents,
                    client=client,
                    model_name=MODEL_NAME,
                    chat_history=st.session_state.chat_history,
                    top_k=3
                )

                # ------------------------------------
                # 1. DISPLAY QUESTIONS
                # ------------------------------------

                rewritten_question = response.get(
                    "rewritten_question",
                    user_question
                )

                retrieval_query = response.get(
                    "retrieval_query",
                    rewritten_question
                )

                st.info(
                    f"Original question: {user_question}"
                )

                st.info(
                    f"Interpreted question: {rewritten_question}"
                )

                st.info(
                    f"Retrieval query: {retrieval_query}"
                )

                # ------------------------------------
                # 2. GET RETRIEVAL RESULTS
                # ------------------------------------

                retrieved_results = response.get(
                    "retrieved_results",
                    []
                )

                sources = response.get("sources", [])

                # ------------------------------------
                # 3. DISPLAY RETRIEVAL SCORE
                # ------------------------------------

                if retrieved_results:

                    highest_similarity = max(
                        result["score"]
                        for result in retrieved_results
                    )

                    st.caption(
                        f"Highest retrieval similarity: "
                        f"{highest_similarity:.4f}"
                    )

                    st.caption(
                        f"Similarity threshold: "
                        f"{SIMILARITY_THRESHOLD:.2f}"
                    )

                # ------------------------------------
                # 4. DISPLAY AI ANSWER
                # ------------------------------------

                if response.get("success", False):

                    st.markdown("### AI Answer")

                    st.write(response["answer"])

                else:

                    st.error(
                        response.get(
                            "answer",
                            "Unable to generate a response."
                        )
                    )

                # ------------------------------------
                # 5. DISPLAY REFERENCE SOURCES
                # ------------------------------------

                if sources or retrieved_results:

                    with st.expander(
                        "📚 View Reference Sources"
                    ):

                        if sources:

                            for index, source in enumerate(
                                sources,
                                start=1
                            ):

                                st.markdown(
                                    f"**Source {index}: "
                                    f"{source.get('source', 'Unknown')}**"
                                )

                                st.write(
                                    "Knowledge source:",
                                    source.get(
                                        "knowledge_source",
                                        "Unknown"
                                    )
                                )

                                st.write(
                                    "Document ID:",
                                    source.get(
                                        "document_id",
                                        "Unknown"
                                    )
                                )

                                st.write(
                                    "Similarity:",
                                    round(
                                        source.get(
                                            "similarity_score",
                                            0
                                        ),
                                        4
                                    )
                                )

                                if source.get("page_number"):
                                    st.write(
                                        "Page:",
                                        source["page_number"]
                                    )

                                if source.get("sheet_name"):
                                    st.write(
                                        "Sheet:",
                                        source["sheet_name"]
                                    )

                                st.divider()

                        else:

                            for index, result in enumerate(
                                retrieved_results,
                                start=1
                            ):

                                metadata = result.get(
                                    "metadata",
                                    {}
                                )

                                st.markdown(
                                    f"**Retrieved Result {index}: "
                                    f"{metadata.get('source', 'Unknown')}**"
                                )

                                st.write(
                                    "Document ID:",
                                    result.get("id", "Unknown")
                                )

                                st.write(
                                    "Similarity:",
                                    round(result["score"], 4)
                                )

                                st.write(
                                    "Knowledge source:",
                                    result.get(
                                        "knowledge_source",
                                        "Unknown"
                                    )
                                )

                                st.write("Retrieved content:")

                                st.write(
                                    result.get("text", "")
                                )

                                st.divider()

            except Exception as error:

                st.error(
                    f"Chat processing error: {error}"
                )


# ============================================
# 9. FOOTER
# ============================================

st.divider()

st.caption(
    "Personal AI Knowledge Assistant | Version 1.0"
)
