"""
rag.py — Atlas RAG Query Module (runs on Pi 5)

Connects to ChromaDB, searches for relevant note chunks,
and builds an enhanced prompt for Ollama.

Imported by app.py — not run directly.
"""

import os

CHROMA_DIR = os.environ.get("ATLAS_CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "atlas_notes"
TOP_K = 3  # number of chunks to retrieve


def _get_collection():
    """Connect to ChromaDB and return the collection, or None if unavailable."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


def search_notes(query, top_k=TOP_K):
    """
    Search the notes database for chunks relevant to the query.
    Returns a list of matching text chunks, or an empty list.
    """
    collection = _get_collection()
    if collection is None:
        return []

    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        return [doc for doc in documents if doc.strip()]
    except Exception:
        return []


def build_rag_prompt(question, context_chunks):
    """
    Build an enhanced prompt that includes retrieved context.
    If no context is available, returns the question as-is.
    """
    if not context_chunks:
        return question

    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"Use the following context from the user's notes to answer their question. "
        f"If the context is not relevant, answer from your own knowledge.\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {question}"
    )