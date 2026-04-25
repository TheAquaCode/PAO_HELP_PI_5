import sqlite3
import requests
import json

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH = "atlas.db"          # path to your SQLite database
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"           # change if you're using a different model
SUMMARIZE_AFTER = 20          # summarize once the conversation hits this many messages
# ────────────────────────────────────────────────────────────────────────────


def get_db():
    """Open a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    """
    Create the tables if they don't already exist.
    
    messages  — stores every chat turn (role = 'user' or 'assistant')
    summaries — stores one rolling summary per conversation
    """
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT    NOT NULL,
            role            TEXT    NOT NULL,
            content         TEXT    NOT NULL,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT    NOT NULL UNIQUE,
            summary         TEXT    NOT NULL,
            updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def get_message_count(conversation_id: str) -> int:
    """Return how many messages are stored for this conversation."""
    conn = get_db()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
        (conversation_id,)
    ).fetchone()
    conn.close()
    return row["cnt"]


def get_messages(conversation_id: str) -> list[dict]:
    """Fetch all messages for a conversation, oldest first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,)
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def save_message(conversation_id: str, role: str, content: str):
    """Append a single message to the database."""
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content)
    )
    conn.commit()
    conn.close()


def get_existing_summary(conversation_id: str) -> str | None:
    """Return the stored summary for this conversation, or None."""
    conn = get_db()
    row = conn.execute(
        "SELECT summary FROM summaries WHERE conversation_id = ?",
        (conversation_id,)
    ).fetchone()
    conn.close()
    return row["summary"] if row else None


def save_summary(conversation_id: str, summary: str):
    """Write (or overwrite) the summary for this conversation."""
    conn = get_db()
    conn.execute("""
        INSERT INTO summaries (conversation_id, summary)
        VALUES (?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            summary    = excluded.summary,
            updated_at = CURRENT_TIMESTAMP
    """, (conversation_id, summary))
    conn.commit()
    conn.close()


def _call_ollama(prompt: str) -> str:
    """Send a plain prompt to Ollama and return the response text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def summarize_conversation(conversation_id: str) -> str:
    """
    Pull the full message history, ask Ollama to summarize it,
    store the result, and return it.
    """
    messages = get_messages(conversation_id)
    if not messages:
        return ""

    # Build a readable transcript for the model
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    prompt = (
        "You are summarizing a conversation between a user and an AI assistant called Atlas. "
        "Write a concise summary (3-5 sentences) that captures the key topics discussed, "
        "any decisions made, and important context that should be remembered. "
        "Do not include greetings or filler. Be factual and specific.\n\n"
        f"CONVERSATION:\n{transcript}\n\n"
        "SUMMARY:"
    )

    summary = _call_ollama(prompt)
    save_summary(conversation_id, summary)
    print(f"[summarize] Stored summary for conversation '{conversation_id}'")
    return summary


def maybe_summarize(conversation_id: str):
    """
    Call this after every assistant reply.
    Triggers a summarization when the message count hits the threshold.
    Does nothing if the threshold hasn't been reached yet.
    """
    count = get_message_count(conversation_id)
    if count >= SUMMARIZE_AFTER and count % SUMMARIZE_AFTER == 0:
        print(f"[summarize] {count} messages reached — summarizing...")
        summarize_conversation(conversation_id)


def build_context(conversation_id: str, recent_n: int = 10) -> list[dict]:
    """
    Build the message list to send to Ollama for the next reply.
    
    If a summary exists, it's prepended as a system message so the model
    has long-term context without the full history in the prompt.
    The last `recent_n` messages are always included verbatim.
    """
    summary = get_existing_summary(conversation_id)
    all_messages = get_messages(conversation_id)
    recent = all_messages[-recent_n:]  # always keep the freshest exchanges

    context = []

    if summary:
        context.append({
            "role": "system",
            "content": f"Summary of the conversation so far: {summary}"
        })

    context.extend(recent)
    return context


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_db()
    cid = "test-conversation-1"

    # Simulate some messages
    pairs = [
        ("user", "What is the capital of France?"),
        ("assistant", "The capital of France is Paris."),
        ("user", "What language do they speak there?"),
        ("assistant", "They speak French in France."),
        ("user", "What is a popular dish from there?"),
        ("assistant", "Croissants and coq au vin are both well-known French dishes."),
    ]
    for role, content in pairs:
        save_message(cid, role, content)

    print("Messages saved. Running summarization...\n")
    summary = summarize_conversation(cid)
    print(f"Summary:\n{summary}\n")

    print("Context that would be sent to Ollama:")
    for msg in build_context(cid):
        print(f"  [{msg['role']}] {msg['content'][:80]}")
