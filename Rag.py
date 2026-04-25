import chromadb
from chromadb.utils import embedding_functions
import os

DB_PATH  = "/home/admin/atlas_notes"
N_CHUNKS = 4

def get_collection():
    if not os.path.exists(DB_PATH):
        return None
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        ef     = embedding_functions.DefaultEmbeddingFunction()
        return client.get_collection("notes", embedding_function=ef)
    except Exception as e:
        print(f"RAG collection error: {e}")
        return None

def query_notes(question):
    """Dump ALL chunks into the prompt — no filtering, no semantic search."""
    col = get_collection()
    if col is None:
        return []
    try:
        count = col.count()
        if count == 0:
            return []
        results = col.get(include=["documents"])
        docs = results["documents"]
        print(f"RAG: dumping all {len(docs)} chunks into prompt")
        return docs
    except Exception as e:
        print(f"RAG query error: {e}")
        return []

def build_rag_prompt(question, chunks):
    context = "\n\n---\n\n".join(chunks)
    return f"""INSTRUCTIONS: Read the NOTES section below. Then answer the QUESTION using ONLY what is written in the NOTES. Copy exact wording from the notes where possible. Do not add any information from outside the notes. If the answer cannot be found in the notes, respond only with: "I don't have that in my notes."

NOTES:
{context}

QUESTION: {question}

ANSWER (use only information from the NOTES above):"""

def rag_enabled():
    if not os.path.exists(DB_PATH):
        return False
    col = get_collection()
    if col is None:
        return False
    try:
        return col.count() > 0
    except:
        return False

def reload():
    pass
