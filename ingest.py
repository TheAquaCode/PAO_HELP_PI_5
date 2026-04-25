"""
ingest.py — Atlas RAG Ingestion (runs on Pi 5)

Reads PDF and text files from a local notes folder,
splits them into chunks, and stores them in ChromaDB.

Usage:
    python3 ingest.py

Put your .pdf and .txt files in the NOTES_FOLDER before running.
"""

import os
import sys

# --- Config ---
NOTES_FOLDER = os.environ.get("ATLAS_NOTES_FOLDER", "./notes")
CHROMA_DIR = os.environ.get("ATLAS_CHROMA_DIR", "./chroma_db")
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks
COLLECTION_NAME = "atlas_notes"


def read_text_file(filepath):
    """Read a plain text file and return its content."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(filepath):
    """Read a PDF file and return extracted text."""
    try:
        import PyPDF2
    except ImportError:
        print("  [!] PyPDF2 not installed. Run: pip3 install PyPDF2")
        return ""

    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    # Check notes folder exists
    if not os.path.isdir(NOTES_FOLDER):
        print(f"[!] Notes folder not found: {NOTES_FOLDER}")
        print(f"    Creating it now. Add your .pdf and .txt files, then re-run.")
        os.makedirs(NOTES_FOLDER, exist_ok=True)
        sys.exit(0)

    # Gather files
    files = [f for f in os.listdir(NOTES_FOLDER)
             if f.lower().endswith((".pdf", ".txt"))]

    if not files:
        print(f"[!] No .pdf or .txt files found in {NOTES_FOLDER}")
        print(f"    Add your class notes there and re-run.")
        sys.exit(0)

    print(f"[*] Found {len(files)} file(s) in {NOTES_FOLDER}")

    # Import ChromaDB
    try:
        import chromadb
    except ImportError:
        print("[!] ChromaDB not installed. Run: pip3 install chromadb")
        sys.exit(1)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Delete old collection if it exists so we do a clean re-ingest
    try:
        client.delete_collection(COLLECTION_NAME)
        print("[*] Cleared previous collection.")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    total_chunks = 0

    for filename in files:
        filepath = os.path.join(NOTES_FOLDER, filename)
        print(f"\n[*] Processing: {filename}")

        # Read file
        if filename.lower().endswith(".pdf"):
            text = read_pdf_file(filepath)
        else:
            text = read_text_file(filepath)

        if not text.strip():
            print(f"  [!] No text extracted from {filename}, skipping.")
            continue

        # Chunk it
        chunks = chunk_text(text)
        print(f"  [+] Split into {len(chunks)} chunks")

        # Add to ChromaDB
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        total_chunks += len(chunks)

    print(f"\n[✓] Ingestion complete. {total_chunks} chunks stored in {CHROMA_DIR}")
    print(f"    Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()