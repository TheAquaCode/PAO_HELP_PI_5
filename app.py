from flask import Flask, request, jsonify, render_template_string, Response
import requests
import psutil
import sqlite3
import json

app = Flask(__name__)
DB = "chat.db"

PERSONAS = {
    "assistant": "You are Atlas, a helpful AI assistant.",
    "tutor":     "You are Atlas, a patient study tutor. Help the user understand concepts clearly.",
    "coder":     "You are Atlas, a coding assistant. Give concise working code with short explanations.",
}

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

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Atlas</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0d0d0d; color: #00ff99; font-family: monospace; display: flex; height: 100vh; overflow: hidden; }

        #sidebar {
            width: 200px; background: #111; border-right: 1px solid #1a1a1a;
            padding: 16px; display: flex; flex-direction: column; gap: 20px;
        }
        #sidebar h3 { font-size: 11px; color: #444; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
        .stat-label { font-size: 11px; color: #555; margin-bottom: 2px; }
        .stat-value { font-size: 20px; color: #00ff99; }
        .stat-block { margin-bottom: 10px; }
        select {
            background: #1a1a1a; color: #00ff99; border: 1px solid #333;
            padding: 6px; font-family: monospace; width: 100%; cursor: pointer;
        }
        select:focus { outline: none; border-color: #00ff99; }
        #pi3-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: #ff4444; display: inline-block; margin-right: 6px;
        }
        #pi3-dot.online { background: #00ff99; }

        #main { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 10px; overflow: hidden; }
        #header { font-size: 18px; font-weight: bold; border-bottom: 1px solid #1a1a1a; padding-bottom: 10px; }
        #chat { flex: 1; overflow-y: auto; padding: 10px; background: #0a0a0a; border: 1px solid #1a1a1a; }
        .user-msg { color: #ffffff; margin: 8px 0; }
        .ai-msg { color: #00ff99; margin: 8px 0; white-space: pre-wrap; }
        .timestamp { font-size: 10px; color: #333; display: block; margin-bottom: 2px; }
        #status { font-size: 11px; color: #555; min-height: 16px; }
        #input-row { display: flex; gap: 8px; }
        #msg {
            flex: 1; background: #1a1a1a; color: #00ff99;
            border: 1px solid #333; padding: 10px;
            font-family: monospace; font-size: 14px;
        }
        #msg:focus { outline: none; border-color: #00ff99; }
        button {
            background: #00ff99; color: #0d0d0d; border: none;
            padding: 10px 20px; font-family: monospace;
            font-weight: bold; cursor: pointer;
        }
        button:hover { background: #00cc77; }
    </style>
</head>
<body>
    <div id="sidebar">
        <div>
            <h3>Atlas</h3>
            <div style="font-size: 12px; color: #555;">AI Terminal v2</div>
        </div>
        <div>
            <h3>System — Pi 5</h3>
            <div class="stat-block">
                <div class="stat-label">CPU</div>
                <div class="stat-value" id="cpu">--%</div>
            </div>
            <div class="stat-block">
                <div class="stat-label">RAM</div>
                <div class="stat-value" id="ram">--%</div>
            </div>
            <div class="stat-block">
                <div class="stat-label">Temp</div>
                <div class="stat-value" id="temp">--°C</div>
            </div>
        </div>
        <div>
            <h3>Persona</h3>
            <select id="persona">
                <option value="assistant">Assistant</option>
                <option value="tutor">Tutor</option>
                <option value="coder">Coder</option>
            </select>
        </div>
        <div>
            <h3>Pi 3 Voice Node</h3>
            <div style="font-size: 12px; display: flex; align-items: center;">
                <span id="pi3-dot"></span>
                <span id="pi3-label">checking...</span>
            </div>
        </div>
    </div>

    <div id="main">
        <div id="header">Atlas AI Terminal</div>
        <div id="chat"></div>
        <div id="status"></div>
        <div id="input-row">
            <input id="msg" type="text" placeholder="Type a message..." onkeydown="if(event.key==='Enter') send()"/>
            <button onclick="send()">Send</button>
        </div>
    </div>

    <script>
        window.onload = async () => {
            const res = await fetch("/history");
            const data = await res.json();
            data.forEach(m => appendMessage(m.role, m.content, m.timestamp));
            updateStats();
            setInterval(updateStats, 3000);
            checkPi3();
            setInterval(checkPi3, 8000);
        };

        function appendMessage(role, content, timestamp) {
            const chat = document.getElementById("chat");
            const div = document.createElement("div");
            div.className = role === "user" ? "user-msg" : "ai-msg";
            const ts = timestamp ? `<span class="timestamp">${timestamp}</span>` : "";
            const prefix = role === "user" ? "> " : "Atlas: ";
            div.innerHTML = ts + prefix + content;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function send(override) {
            const input = document.getElementById("msg");
            const msg = override || input.value.trim();
            if (!msg) return;
            input.value = "";
            const persona = document.getElementById("persona").value;

            appendMessage("user", msg);
            document.getElementById("status").textContent = "Atlas is thinking...";

            const chat = document.getElementById("chat");
            const aiDiv = document.createElement("div");
            aiDiv.className = "ai-msg";
            aiDiv.textContent = "Atlas: ";
            chat.appendChild(aiDiv);

            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: msg, persona: persona })
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let full = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value);
                const lines = chunk.split("\\n").filter(l => l.startsWith("data: "));
                for (const line of lines) {
                    const token = line.replace("data: ", "");
                    if (token === "[DONE]") break;
                    full += token;
                    aiDiv.textContent = "Atlas: " + full;
                    chat.scrollTop = chat.scrollHeight;
                }
            }

            document.getElementById("status").textContent = "";
        }

        async function updateStats() {
            try {
                const res = await fetch("/stats");
                const data = await res.json();
                document.getElementById("cpu").textContent = data.cpu + "%";
                document.getElementById("ram").textContent = data.ram + "%";
                document.getElementById("temp").textContent = data.temp + "°C";
            } catch {}
        }

        async function checkPi3() {
            try {
                const res = await fetch("/pi3-status");
                const data = await res.json();
                const dot = document.getElementById("pi3-dot");
                const label = document.getElementById("pi3-label");
                if (data.online) {
                    dot.classList.add("online");
                    label.textContent = "online";
                } else {
                    dot.classList.remove("online");
                    label.textContent = "offline";
                }
            } catch {
                document.getElementById("pi3-label").textContent = "offline";
            }
        }
    </script>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(HTML)

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

    def stream():
        full_response = ""
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "phi3:mini",
            "system": system_prompt,
            "prompt": message,
            "stream": True
        }, stream=True)

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                full_response += token
                yield f"data: {token}\n\n"
                if chunk.get("done"):
                    save_message("assistant", full_response)
                    yield "data: [DONE]\n\n"
                    break

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

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000)
