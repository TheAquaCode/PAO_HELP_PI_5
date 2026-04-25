from flask import Flask, request, jsonify, render_template, Response
import requests
import psutil
import sqlite3
import json
import hashlib
from rag import search_notes, build_rag_prompt
 
app = Flask(__name__)
DB = "chat.db"
 
PERSONAS = {
    "assistant": "You are Atlas, a helpful AI assistant.",
    "tutor":     "You are Atlas, a patient study tutor. Help the user understand concepts clearly.",
    "coder":     "You are Atlas, a coding assistant. Give concise working code with short explanations.",
}
 
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"
 
# Database 
 
def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            query_hash TEXT PRIMARY KEY,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
 
def save_message(role, content):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()
 
def get_history():
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT role, content, timestamp FROM messages ORDER BY timestamp DESC LIMIT 20").fetchall()
    conn.close()
    return list(reversed(rows))
 
# Response Cache
 
def _hash_query(text):
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()
 
def get_cached_response(question):
    try:
        conn = sqlite3.connect(DB)
        row = conn.execute(
            "SELECT answer FROM cache WHERE query_hash = ?",
            (_hash_query(question),)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None
 
def save_cached_response(question, answer):
    try:
        conn = sqlite3.connect(DB)
        conn.execute(
            "INSERT OR REPLACE INTO cache (query_hash, question, answer) VALUES (?, ?, ?)",
            (_hash_query(question), question.strip(), answer)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
 
#
#  Routes 
 
@app.route("/")
def index():
    return render_template("index.html")
 
@app.route("/history")
def history():
    rows = get_history()
    return jsonify([{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows])
 
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    persona = data.get("persona", "assistant")
    system_prompt = PERSONAS.get(persona, PERSONAS["assistant"])
 
    save_message("user", message)
 
    # Check cache first
    cached = get_cached_response(message)
    if cached:
        save_message("assistant", cached)
        def cached_stream():
            yield f"data: {cached}\n\n"
            yield "data: [DONE]\n\n"
        return Response(cached_stream(), mimetype="text/event-stream")
 
    # RAG: search notes and build enhanced prompt
    try:
        chunks = search_notes(message)
        prompt = build_rag_prompt(message, chunks)
    except Exception:
        prompt = message
 
    def stream():
        full_response = ""
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "system": system_prompt,
                "prompt": prompt,
                "stream": True
            }, stream=True, timeout=120)
 
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    full_response += token
                    yield f"data: {token}\n\n"
                    if chunk.get("done"):
                        save_message("assistant", full_response)
                        save_cached_response(message, full_response)
                        yield "data: [DONE]\n\n"
                        break
        except requests.exceptions.ConnectionError:
            yield "data: [Error: Cannot reach Ollama. Is it running?]\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [Error: {str(e)}]\n\n"
            yield "data: [DONE]\n\n"
 
    return Response(stream(), mimetype="text/event-stream")
 
@app.route("/stats")
def stats():
    temp = 0.0
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = round(int(f.read()) / 1000, 1)
    except:
        pass
    return jsonify({
        "cpu": psutil.cpu_percent(interval=0.5),
        "ram": psutil.virtual_memory().percent,
        "temp": temp
    })
 
@app.route("/pi3-status")
def pi3_status():
    try:
        r = requests.get("http://pi3.local:5001/ping", timeout=2)
        return jsonify({"online": r.status_code == 200})
    except:
        return jsonify({"online": False})
 
@app.route("/rag-status")
def rag_status():
    """Reports whether RAG notes are loaded and how many chunks exist."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("atlas_notes")
        count = collection.count()
        return jsonify({"active": count > 0, "docs": count})
    except Exception:
        return jsonify({"active": False, "docs": 0})
 
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000)