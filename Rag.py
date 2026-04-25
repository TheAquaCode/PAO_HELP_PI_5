import chromadb
from chromadb.utils import embedding_functions
import os

DB_PATH = "/home/admin/atlas_notes"
N_RESULTS = 3
RELEVANCE_THRESHOLD = 0.6  # only use RAG if chunks are this similar or closer (lower = more similar in chromadb)

def get_collection():
    """Always reads fresh from disk — no caching."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        ef = embedding_functions.DefaultEmbeddingFunction()
        return client.get_collection("notes", embedding_function=ef)
    except Exception as e:
        print(f"RAG not ready: {e}")
        return None

def query_notes(question, n=N_RESULTS):
    """Returns chunks only if they are actually relevant to the question."""
    collection = get_collection()
    if collection is None:
        return []
    try:
        results = collection.query(
            query_texts=[question],
            n_results=n,
            include=["documents", "distances"]
        )
        docs = results["documents"][0]
        distances = results["distances"][0]

        # Filter out chunks that are not relevant enough
        relevant = [
            doc for doc, dist in zip(docs, distances)
            if dist <= RELEVANCE_THRESHOLD
        ]

        if relevant:
            print(f"RAG: found {len(relevant)} relevant chunks (distances: {[round(d,3) for d in distances]})")
        else:
            print(f"RAG: no relevant chunks found (distances: {[round(d,3) for d in distances]}) — using base model")

        return relevant
    except Exception as e:
        print(f"RAG query error: {e}")
        return []

def build_rag_prompt(question, chunks):
    if not chunks:
        return question
    context = "\n\n".join(chunks)
    return f"""You MUST answer using ONLY the information in the notes below.
Do NOT use any outside knowledge or make anything up.
If the answer is in the notes, repeat it accurately and specifically.
If the answer is not in the notes at all, say exactly: "I don't have that in my notes."

--- NOTES START ---
{context}
--- NOTES END ---

Question: {question}
Answer:"""

def rag_enabled():
    if not os.path.exists(DB_PATH):
        return False
    collection = get_collection()
    if collection is None:
        return False
    try:
        return collection.count() > 0
    except:
        return False

def reload():
    """No-op — kept for compatibility."""
    pass
