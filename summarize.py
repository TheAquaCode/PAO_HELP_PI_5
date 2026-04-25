import sqlite3
import requests

DB = "/home/admin/chat.db"
SUMMARY_THRESHOLD = 20  # summarize after this many messages
KEEP_RECENT = 6         # always keep this many recent messages

def get_message_count():
    conn = sqlite3.connect(DB)
    count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    conn.close()
    return count

def get_all_messages():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT id, role, content FROM messages ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    return rows

def get_recent_messages_as_text(limit=50):
    """Get recent messages formatted as a conversation string."""
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT role, content FROM messages ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    rows = list(reversed(rows))
    lines = []
    for role, content in rows:
        prefix = "User" if role == "user" else "Atlas"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)

def delete_messages_by_ids(ids):
    conn = sqlite3.connect(DB)
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", ids)
    conn.commit()
    conn.close()

def insert_summary(summary_text):
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        ("assistant", f"[CONVERSATION SUMMARY]: {summary_text}")
    )
    conn.commit()
    conn.close()

def summarize_with_ollama(conversation_text):
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "phi3:mini",
            "system": "You are a summarizer. Summarize the following conversation into 2-3 concise sentences. Only include what was actually discussed.",
            "prompt": conversation_text,
            "stream": False
        }, timeout=120)
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Summarize error: {e}")
        return ""

def get_conversation_summary():
    """Returns a summary of the current chat history — called on demand."""
    text = get_recent_messages_as_text(limit=30)
    if not text:
        return "No conversation history yet."
    print("Generating conversation summary...")
    summary = summarize_with_ollama(text)
    return summary if summary else "Could not generate summary."

def maybe_summarize():
    """Auto-summarize old messages if threshold hit."""
    count = get_message_count()
    if count < SUMMARY_THRESHOLD:
        return False

    messages = get_all_messages()
    to_summarize = messages[:-KEEP_RECENT]

    if not to_summarize:
        return False

    convo = "\n".join([
        f"{'User' if r[1] == 'user' else 'Atlas'}: {r[2]}"
        for r in to_summarize
    ])

    print(f"Auto-summarizing {len(to_summarize)} old messages...")
    summary = summarize_with_ollama(convo)

    if not summary:
        return False

    ids = [r[0] for r in to_summarize]
    delete_messages_by_ids(ids)
    insert_summary(summary)
    print(f"Done. Compressed {len(to_summarize)} messages into 1 summary.")
    return True
