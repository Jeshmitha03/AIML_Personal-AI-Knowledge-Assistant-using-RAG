
from pathlib import Path

import faiss
import numpy as np


# ============================================
# 1. CREATE FAISS INDEX
# ============================================

def create_faiss_index(embeddings):
    """
    Create a FAISS inner-product index.
    Embeddings must be normalized.
    """

    if len(embeddings) == 0:
        raise ValueError("Cannot create an index without embeddings.")

    embedding_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings.astype("float32"))

    return index


# ============================================
# 2. SAVE FAISS INDEX
# ============================================

def save_faiss_index(index, file_path):
    """
    Save FAISS index to disk.
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(file_path))


# ============================================
# 3. LOAD FAISS INDEX
# ============================================

def load_faiss_index(file_path):
    """
    Load a saved FAISS index from disk.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {file_path}"
        )

    return faiss.read_index(str(file_path))


# ============================================
# 4. SEARCH FAISS INDEX
# ============================================

def search_faiss_index(
    query,
    embedding_model,
    index,
    documents,
    top_k=5
):
    """
    Search a FAISS index using semantic similarity.
    """

    if index is None or index.ntotal == 0:
        return []

    if index.ntotal != len(documents):
        raise ValueError(
            "FAISS index and document count do not match."
        )

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        document = documents[int(idx)]

        results.append({
            "score": float(score),
            "id": document["id"],
            "text": document["text"],
            "metadata": document["metadata"]
        })

    return results


# ============================================
# 5. SEARCH ALL KNOWLEDGE BASES
# ============================================

def search_all_knowledge(
    query,
    embedding_model,
    original_index,
    original_documents,
    uploaded_index=None,
    uploaded_documents=None,
    top_k=5
):
    """
    Search original and uploaded knowledge bases.

    Either knowledge base can be unavailable.
    Combine and rank the available results.
    """

    all_results = []

    # ----------------------------------------
    # Search Original Knowledge Base
    # ----------------------------------------

    if (
        original_index is not None
        and original_documents is not None
        and len(original_documents) > 0
    ):

        original_results = search_faiss_index(
            query,
            embedding_model,
            original_index,
            original_documents,
            top_k
        )

        for result in original_results:

            result["knowledge_source"] = (
                "original_knowledge_base"
            )

        all_results.extend(original_results)

    # ----------------------------------------
    # Search Uploaded Documents
    # ----------------------------------------

    if (
        uploaded_index is not None
        and uploaded_documents is not None
        and len(uploaded_documents) > 0
    ):

        uploaded_results = search_faiss_index(
            query,
            embedding_model,
            uploaded_index,
            uploaded_documents,
            top_k
        )

        for result in uploaded_results:

            result["knowledge_source"] = (
                "uploaded_documents"
            )

        all_results.extend(uploaded_results)

    # ----------------------------------------
    # Combine and Rank Results
    # ----------------------------------------

    all_results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return all_results[:top_k]
