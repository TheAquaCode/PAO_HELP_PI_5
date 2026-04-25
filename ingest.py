import os
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2

# ── Config ──
USB_BASE = "/media/admin"   # base path where USB drives mount
CHUNK_SIZE = 500            # characters per chunk
CHUNK_OVERLAP = 50          # overlap between chunks
DB_PATH = "/home/admin/atlas_notes"

def find_usb():
    """Automatically find whatever USB drive is plugged in."""
    if not os.path.exists(USB_BASE):
        print(f"No media folder found at {USB_BASE}")
        return None
    drives = os.listdir(USB_BASE)
    if not drives:
        print("No USB drive detected. Plug one in and try again.")
        return None
    drive = os.path.join(USB_BASE, drives[0])
    print(f"Found USB drive: {drive}")
    return drive

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Could not read PDF {path}: {e}")
    return text

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def ingest():
    drive = find_usb()
    if not drive:
        return

    print(f"Scanning {drive} for notes...")

    all_chunks = []
    all_ids = []
    all_metas = []
    file_count = 0

    for root, dirs, files in os.walk(drive):
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = fname.lower().split(".")[-1]
            text = ""

            if ext in ("txt", "md"):
                print(f"  Reading: {fname}")
                text = read_txt(fpath)
            elif ext == "pdf":
                print(f"  Reading: {fname}")
                text = read_pdf(fpath)
            else:
                continue

            if not text.strip():
                print(f"  Skipping {fname} — empty or unreadable")
                continue

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{fname}-chunk-{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metas.append({"source": fname, "chunk": i})

            file_count += 1

    if not all_chunks:
        print("No readable files found on USB drive.")
        print("Supported types: .txt  .pdf  .md")
        return

    print(f"\nIngesting {len(all_chunks)} chunks from {file_count} file(s)...")

    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()

    # Clear old collection and rebuild fresh
    try:
        client.delete_collection("notes")
        print("Cleared old notes database.")
    except:
        pass

    collection = client.create_collection("notes", embedding_function=ef)
    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metas)

    print(f"\nDone! {len(all_chunks)} chunks stored.")
    print("Atlas can now answer questions from your notes.")

if __name__ == "__main__":
    ingest()
