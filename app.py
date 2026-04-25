from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
import requests as req
import psutil
import sqlite3
import json
import threading
import subprocess
import queue
from datetime import datetime

app = Flask(__name__)
DB     = "/home/admin/chat.db"
OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL  = "llama3.2"
MAX_T  = 150

# ── SSE clients ──
clients      = []
clients_lock = threading.Lock()

def push(etype, data):
    msg = json.dumps({"type": etype, "data": data})
    with clients_lock:
        dead = []
        for q in clients:
            try:
                q.put_nowait(msg)
            except:
                dead.append(q)
        for q in dead:
            clients.remove(q)

# ── Shared state (thread safe) ──
_state_lock   = threading.Lock()
_pi5_status   = "idle"
_current_mode = "ai"

def get_status():
    with _state_lock:
        return _pi5_status

def set_status(s):
    global _pi5_status
    with _state_lock:
        _pi5_status = s

def get_mode():
    with _state_lock:
        return _current_mode

def set_mode_val(m):
    global _current_mode
    with _state_lock:
        _current_mode = m

# ── DB ──
db_lock = threading.Lock()

def initdb():
    with db_lock:
        conn = sqlite3.connect(DB)
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT, content TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()
        conn.close()

def savemsg(role, content):
    with db_lock:
        conn = sqlite3.connect(DB)
        conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
        conn.commit()
        conn.close()

def gethistory(n=20):
    with db_lock:
        conn = sqlite3.connect(DB)
        rows = conn.execute(
            "SELECT role, content, ts FROM messages ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()
        conn.close()
    return list(reversed(rows))

# ── RAG ──
try:
    from rag import query_notes, build_rag_prompt, rag_enabled, reload as rag_reload
except:
    def rag_enabled(): return False
    def query_notes(q): return []
    def build_rag_prompt(q, c): return q
    def rag_reload(): pass

# ── Web search ──
try:
    from search import search_web, build_search_prompt
except:
    def search_web(q): return []
    def build_search_prompt(q, r): return q, False

# ── Summarize ──
try:
    from summarize import maybe_summarize, get_conversation_summary
    HAS_SUM = True
except:
    HAS_SUM = False
    def maybe_summarize(): pass
    def get_conversation_summary(): return "No summary available."

SUM_KEYS = ["summarize", "summary", "recap", "what did we talk", "what have we discussed"]

def is_sum(msg):
    return any(k in msg.lower() for k in SUM_KEYS)

def build(message, mode):
    # Summary works in any mode
    if is_sum(message) and HAS_SUM:
        return None, None, "summary", get_conversation_summary()

    # Notes mode
    if mode == "rag":
        if not rag_enabled():
            return None, None, "no_notes", "No notes loaded. Plug in USB and press 💾 USB."
        chunks = query_notes(message)
        if not chunks:
            return None, None, "no_notes", "No notes found on the drive."
        context = "\n\n---\n\n".join(chunks)
        prompt = (
            f"NOTES:\n{context}\n\n"
            f"TASK: Find and quote the part of the NOTES that answers this question: {message}\n"
            f"Copy the relevant sentence(s) from the NOTES exactly as written. "
            f"Start your answer with: According to the notes,"
        )
        system = "You are a note reader. Find the answer in the provided notes and quote it directly. Never use outside knowledge."
        return prompt, system, "rag", None

    # Web mode
    if mode == "web":
        results = search_web(message)
        prompt, ok = build_search_prompt(message, results)
        if ok:
            system = "You are a web search assistant. Answer using ONLY the search results provided. Be concise."
            return prompt, system, "web", None
        return message, "You are Atlas. Answer in 2 sentences max.", "ai", None

    # AI mode
    return message, "You are Atlas. Answer in 2 sentences max unless asked to elaborate.", "ai", None

def ollama_call(system, prompt, stream=False):
    return req.post(OLLAMA, json={
        "model":       MODEL,
        "system":      system,
        "prompt":      prompt,
        "stream":      stream,
        "num_predict": MAX_T,
        "options":     {"num_ctx": 2048, "temperature": 0.7}
    }, stream=stream, timeout=180)

HTML = """<!DOCTYPE html>
<html>
<head>
<title>Atlas</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;overflow:hidden;background:#0d0d0d;color:#00ff99;font-family:monospace;font-size:12px}
#top{display:flex;align-items:center;background:#111;border-bottom:1px solid #222;padding:4px 8px;gap:8px;height:32px;flex-shrink:0}
#logo{font-weight:bold;font-size:14px;white-space:nowrap}
#stats{display:flex;gap:10px;font-size:10px}
.sv{display:flex;flex-direction:column;align-items:center}
.sk{font-size:7px;color:#444;margin-bottom:1px}
.dot{width:8px;height:8px;border-radius:50%;background:#ff4444;display:inline-block;margin-right:3px}
.dot.on{background:#00ff99}
#modes{display:flex;gap:4px;margin-left:auto}
.mb{background:#1a1a1a;color:#555;border:1px solid #333;padding:3px 8px;font-family:monospace;font-size:9px;cursor:pointer;border-radius:3px;transition:all .2s}
.mb:hover{color:#00ff99;border-color:#00ff99}
.mb.on{color:#0d0d0d;border-color:transparent}
.mb.on.ai{background:#00ff99}
.mb.on.rag{background:#0099ff}
.mb.on.web{background:#ffaa00}
#usbbtn{background:#1a1a1a;color:#00ff99;border:1px solid #333;padding:3px 7px;font-family:monospace;font-size:10px;cursor:pointer;border-radius:3px}
#wrap{display:flex;flex-direction:column;height:calc(100vh - 32px);padding:8px;gap:6px}
#box{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;background:#0a0a0a;border:1px solid #1a1a1a;border-radius:8px;padding:16px;min-height:0}
#icon{font-size:40px;line-height:1}
#lbl{font-size:13px;color:#555;text-align:center}
#sub{font-size:12px;color:#fff;text-align:center;word-wrap:break-word;max-width:95%;display:none}
#cdwrap{display:none;flex-direction:column;align-items:center;gap:2px}
#cdnum{font-size:52px;font-weight:bold;color:#ffaa00;line-height:1}
#cdunit{font-size:10px;color:#555}
#pgwrap{display:none;width:85%;flex-direction:column;gap:5px}
#pgbg{width:100%;height:8px;background:#1a1a1a;border-radius:4px;overflow:hidden}
#pgfill{height:100%;width:0%;background:#00ff99;border-radius:4px;transition:width .5s ease}
#pgpct{font-size:10px;color:#555;text-align:center}
#resp{background:#0a0a0a;border:1px solid #1a1a1a;border-radius:6px;padding:8px;flex-shrink:0;max-height:100px;overflow-y:auto;word-wrap:break-word;white-space:pre-wrap}
#resp::-webkit-scrollbar{width:2px}
#resp::-webkit-scrollbar-thumb{background:#222}
#rmeta{font-size:8px;color:#333;display:block;margin-bottom:3px}
#rtext{color:#333;font-size:11px}
#bar{display:flex;gap:5px;flex-shrink:0}
#inp{flex:1;background:#1a1a1a;color:#00ff99;border:1px solid #333;padding:6px 8px;font-family:monospace;font-size:11px;border-radius:3px;outline:none}
#inp:focus{border-color:#00ff99}
.gbtn{background:#00ff99;color:#0d0d0d;border:none;padding:6px 12px;font-family:monospace;font-weight:bold;font-size:11px;cursor:pointer;border-radius:3px}
.gbtn:active{background:#00cc77}
#logbtn{background:#1a1a1a;color:#00ff99;border:1px solid #333;padding:6px 8px;font-family:monospace;font-size:10px;cursor:pointer;border-radius:3px}
#statusbar{font-size:9px;color:#333;text-align:center;flex-shrink:0;min-height:12px}
#hist{display:none;position:fixed;inset:0;background:#0d0d0d;z-index:100;flex-direction:column;padding:8px;gap:6px}
#hist.show{display:flex}
#histhdr{display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
#histhdr b{font-size:14px}
#histclose{background:#1a1a1a;color:#00ff99;border:1px solid #333;padding:4px 12px;font-family:monospace;font-size:10px;cursor:pointer;border-radius:3px}
#histlist{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:4px;padding:4px 0}
.hu{color:#fff;background:#1a1a1a;padding:4px 8px;border-radius:4px;align-self:flex-end;max-width:90%;font-size:11px;word-wrap:break-word}
.ha{color:#00ff99;font-size:11px;word-wrap:break-word;white-space:pre-wrap;max-width:95%}
.hts{font-size:8px;color:#333;display:block;margin-bottom:2px}
#histlist::-webkit-scrollbar{width:3px}
#histlist::-webkit-scrollbar-thumb{background:#1a1a1a}
</style>
</head>
<body>
<div id="top">
  <span id="logo">⚡ Atlas</span>
  <div id="stats">
    <div class="sv"><span class="sk">CPU</span><span id="scpu">--%</span></div>
    <div class="sv"><span class="sk">RAM</span><span id="sram">--%</span></div>
    <div class="sv"><span class="sk">TMP</span><span id="stmp">--°</span></div>
    <div class="sv" style="flex-direction:row;align-items:center"><span id="dp3" class="dot"></span><span class="sk">P3</span></div>
    <div class="sv" style="flex-direction:row;align-items:center"><span id="ddb" class="dot"></span><span class="sk">DB</span></div>
  </div>
  <div id="modes">
    <button class="mb ai on" id="btn-ai"  onclick="setMode('ai')">AI</button>
    <button class="mb rag"  id="btn-rag" onclick="setMode('rag')">📚</button>
    <button class="mb web"  id="btn-web" onclick="setMode('web')">🌐</button>
  </div>
  <button id="usbbtn" onclick="doIngest()">💾 USB</button>
</div>
<div id="wrap">
  <div id="box">
    <div id="icon">🎙️</div>
    <div id="lbl">Waiting to be called</div>
    <div id="sub"></div>
    <div id="cdwrap"><div id="cdnum">10</div><div id="cdunit">seconds to speak</div></div>
    <div id="pgwrap"><div id="pgbg"><div id="pgfill"></div></div><div id="pgpct">0%</div></div>
  </div>
  <div id="resp"><span id="rmeta">Last response</span><span id="rtext">—</span></div>
  <div id="bar">
    <input id="inp" type="text" placeholder="Type to Atlas..." onkeydown="if(event.key==='Enter')send()">
    <button class="gbtn" onclick="send()">Go</button>
    <button id="logbtn" onclick="toggleLog()">📋 Log</button>
  </div>
  <div id="statusbar"></div>
</div>
<div id="hist">
  <div id="histhdr"><b>📋 Chat Log</b><button id="histclose" onclick="toggleLog()">✕ Close</button></div>
  <div id="histlist"></div>
</div>
<script>
var curMode='ai', logOpen=false, pgInt=null, pgV=0;
function g(id){return document.getElementById(id)}
function s(id,v){g(id).textContent=v}
function clearSt(){
  g('sub').style.display='none';
  g('cdwrap').style.display='none';
  g('pgwrap').style.display='none';
  if(pgInt){clearInterval(pgInt);pgInt=null;}
}
function idle()  {clearSt();s('icon','🎙️');s('lbl','Waiting to be called');}
function woke()  {clearSt();s('icon','👂');s('lbl','Wake word detected!');}
function tscr()  {clearSt();s('icon','✍️');s('lbl','Transcribing...');}
function cdown(n){clearSt();s('icon','👂');s('lbl','');g('cdwrap').style.display='flex';s('cdnum',n);}
function heard(t){clearSt();s('icon','💬');s('lbl','You said:');g('sub').style.display='block';s('sub',t);}
function proc(q){
  clearSt();s('icon','⚙️');
  s('lbl',q.length>50?q.slice(0,50)+'…':q);
  g('pgwrap').style.display='flex';
  pgV=0;g('pgfill').style.width='0%';s('pgpct','0%');
  pgInt=setInterval(function(){
    if(pgV<60)pgV+=2;
    else if(pgV<85)pgV+=0.5;
    else if(pgV<95)pgV+=0.1;
    var p=Math.round(pgV);
    g('pgfill').style.width=p+'%';s('pgpct',p+'%');
  },500);
}
function done(reply,src){
  clearSt();s('icon','✅');s('lbl','Done');
  var m=g('rmeta');
  m.style.color='#333';m.textContent='Last response';
  if(src==='rag'){m.style.color='#0099ff';m.textContent='📚 from notes';}
  if(src==='web'){m.style.color='#ffaa00';m.textContent='🌐 from web';}
  if(src==='summary'){m.style.color='#888';m.textContent='📝 summary';}
  if(src==='no_notes'){m.style.color='#ff4444';m.textContent='⚠️ no notes';}
  g('rtext').style.color='#00ff99';
  s('rtext',reply);
  setTimeout(idle,3000);
}
function connect(){
  var es=new EventSource('/events');
  es.onmessage=function(e){
    var m=JSON.parse(e.data),t=m.type,d=m.data;
    if(t==='idle')idle();
    else if(t==='wake_detected')woke();
    else if(t==='countdown')cdown(d);
    else if(t==='transcribing')tscr();
    else if(t==='heard')heard(d);
    else if(t==='processing')proc(d);
    else if(t==='ai_message'){done(d.content,d.source);if(logOpen)loadLog();}
    else if(t==='error'){idle();s('statusbar','Error: '+d);setTimeout(function(){s('statusbar','');},4000);}
    else if(t==='status')s('statusbar',d);
    else if(t==='mode_changed'){applyMode(d);}
  };
  es.onerror=function(){setTimeout(connect,3000);};
}
function applyMode(m){
  curMode=m;
  document.querySelectorAll('.mb').forEach(function(b){b.classList.remove('on');});
  var btn=document.querySelector('.mb.'+m);
  if(btn)btn.classList.add('on');
}
function setMode(m){
  applyMode(m);
  var names={ai:'AI mode',rag:'📚 Notes mode',web:'🌐 Web mode'};
  s('statusbar',names[m]);
  setTimeout(function(){s('statusbar','');},2000);
  fetch('/set-mode',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:m})
  }).catch(function(){});
}
async function send(){
  var inp=g('inp'),msg=inp.value.trim();
  if(!msg)return;inp.value='';
  heard(msg);proc(msg);
  var full='';
  try{
    var r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
    var reader=r.body.getReader(),dec=new TextDecoder();
    while(true){
      var chunk=await reader.read();
      if(chunk.done)break;
      dec.decode(chunk.value).split('\\n').forEach(function(line){
        if(!line.startsWith('data: '))return;
        var raw=line.slice(6);
        if(raw==='[DONE]'){done(full,curMode==='web'?'web':curMode==='rag'?'rag':'ai');if(logOpen)loadLog();}
        else full+=raw;
      });
    }
  }catch(e){idle();s('statusbar','Send error');}
}
function toggleLog(){
  logOpen=!logOpen;
  if(logOpen){g('hist').classList.add('show');loadLog();}
  else g('hist').classList.remove('show');
}
async function loadLog(){
  try{
    var r=await fetch('/history');
    var data=await r.json();
    var list=g('histlist');
    list.innerHTML='';
    data.forEach(function(m){
      var div=document.createElement('div');
      div.className=m.role==='user'?'hu':'ha';
      if(m.ts){var ts=document.createElement('span');ts.className='hts';ts.textContent=m.ts.split('.')[0];div.appendChild(ts);}
      div.appendChild(document.createTextNode(m.role==='user'?'> '+m.content:'Atlas: '+m.content));
      list.appendChild(div);
    });
    list.scrollTop=list.scrollHeight;
  }catch(e){console.log('history error',e);}
}
function stats(){
  fetch('/stats').then(function(r){return r.json();}).then(function(d){
    s('scpu',d.cpu+'%');s('sram',d.ram+'%');s('stmp',d.temp+'°');
  }).catch(function(){});
}
function p3(){
  fetch('/pi3-status').then(function(r){return r.json();}).then(function(d){
    if(d.online)g('dp3').classList.add('on');else g('dp3').classList.remove('on');
  }).catch(function(){});
}
function ragcheck(){
  fetch('/rag-status').then(function(r){return r.json();}).then(function(d){
    if(d.active)g('ddb').classList.add('on');else g('ddb').classList.remove('on');
  }).catch(function(){});
}
async function doIngest(){
  g('usbbtn').textContent='⏳';g('usbbtn').disabled=true;
  s('statusbar','Reading USB...');
  try{
    var r=await fetch('/ingest',{method:'POST'});
    var d=await r.json();
    s('statusbar',d.message);ragcheck();
  }catch(e){s('statusbar','Ingest failed');}
  g('usbbtn').textContent='💾 USB';g('usbbtn').disabled=false;
}
window.onload=function(){
  idle();connect();
  // Sync mode from server on load
  fetch('/current-mode').then(function(r){return r.json();}).then(function(d){applyMode(d.mode);}).catch(function(){});
  stats();setInterval(stats,5000);
  p3();setInterval(p3,10000);
  ragcheck();setInterval(ragcheck,15000);
};
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/history")
def history():
    try:
        rows = gethistory()
        return jsonify([{"role": r[0], "content": r[1], "ts": r[2]} for r in rows])
    except Exception as e:
        print(f"[history error] {e}")
        return jsonify([])

@app.route("/events")
def events():
    q = queue.Queue()
    with clients_lock:
        clients.append(q)
    def stream():
        while True:
            try:
                yield f"data: {q.get(timeout=20)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type':'heartbeat'})}\n\n"
    return Response(stream_with_context(stream()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/notify", methods=["POST"])
def notify():
    d = request.get_json()
    push(d.get("type", ""), d.get("data", ""))
    return jsonify({"ok": True})

@app.route("/pi3-status-feed")
def pi3feed():
    return jsonify({"msg": get_status()})

@app.route("/set-mode", methods=["POST"])
def set_mode():
    try:
        d = request.get_json()
        m = d.get("mode", "ai")
        if m not in ("ai", "rag", "web"):
            return jsonify({"error": "invalid mode"}), 400
        set_mode_val(m)
        push("mode_changed", m)
        print(f"[mode] switched to {m}")
        return jsonify({"ok": True, "mode": m})
    except Exception as e:
        print(f"[set-mode error] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/current-mode")
def current_mode_route():
    return jsonify({"mode": get_mode()})

@app.route("/debug")
def debug():
    return jsonify({
        "model":       MODEL,
        "mode":        get_mode(),
        "pi5_status":  get_status(),
        "rag_enabled": rag_enabled(),
        "rag_chunks":  len(query_notes("test")) if rag_enabled() else 0
    })

@app.route("/chat", methods=["POST"])
def chat():
    d       = request.get_json()
    message = d.get("message", "")
    mode    = get_mode()  # always use server-side mode

    savemsg("user", message)
    set_status(f"received: {message[:40]}")
    if HAS_SUM:
        threading.Thread(target=maybe_summarize, daemon=True).start()

    prompt, system, src, direct = build(message, mode)
    set_status(f"generating ({src})...")

    if direct is not None:
        savemsg("assistant", direct)
        set_status("idle")
        def ss():
            for ch in direct: yield f"data: {ch}\n\n"
            yield "data: [DONE]\n\n"
        return Response(ss(), mimetype="text/event-stream")

    def stream():
        full = ""
        try:
            r = ollama_call(system, prompt, stream=True)
            for line in r.iter_lines():
                if line:
                    c   = json.loads(line)
                    tok = c.get("response", "")
                    full += tok
                    yield f"data: {tok}\n\n"
                    if c.get("done"):
                        savemsg("assistant", full)
                        set_status("idle")
                        yield "data: [DONE]\n\n"
                        break
        except Exception as e:
            print(f"[chat error] {e}")
            set_status("idle")
            yield "data: [DONE]\n\n"

    return Response(stream(), mimetype="text/event-stream")

@app.route("/voice", methods=["POST"])
def voice():
    d       = request.get_json()
    message = d.get("message", "")
    mode    = get_mode()  # always use server-side mode
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    savemsg("user", message)
    set_status(f"received: {message[:40]}")
    if HAS_SUM:
        threading.Thread(target=maybe_summarize, daemon=True).start()

    prompt, system, src, direct = build(message, mode)
    set_status(f"generating ({src})...")

    if direct is not None:
        savemsg("assistant", direct)
        set_status("idle")
        push("ai_message", {"content": direct, "source": src, "timestamp": ts})
        return jsonify({"response": direct})

    try:
        r     = ollama_call(system, prompt, stream=False)
        reply = r.json().get("response", "")
        savemsg("assistant", reply)
        set_status("idle")
        push("ai_message", {"content": reply, "source": src, "timestamp": ts})
        return jsonify({"response": reply})
    except Exception as e:
        print(f"[voice error] {e}")
        set_status("idle")
        push("error", str(e)[:60])
        return jsonify({"response": "Error"}), 500

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        r = subprocess.run(
            ["python3", "/home/admin/ingest.py"],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode == 0:
            rag_reload()
            return jsonify({"message": "Notes loaded! Switch to 📚 mode.", "success": True})
        return jsonify({"message": f"Ingest error: {r.stderr[:80]}", "success": False})
    except Exception as e:
        return jsonify({"message": str(e), "success": False})

@app.route("/rag-status")
def ragstatus():
    return jsonify({"active": rag_enabled()})

@app.route("/stats")
def stats():
    temp = 0.0
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = round(int(f.read()) / 1000, 1)
    except:
        pass
    return jsonify({
        "cpu":  psutil.cpu_percent(interval=0.3),
        "ram":  psutil.virtual_memory().percent,
        "temp": temp
    })

@app.route("/pi3-status")
def pi3status():
    try:
        r = req.get("http://pi3.local:5001/ping", timeout=2)
        return jsonify({"online": r.status_code == 200})
    except:
        return jsonify({"online": False})

if __name__ == "__main__":
    initdb()
    app.run(host="0.0.0.0", port=5000, threaded=True)
