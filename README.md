# PAO_HELP_PI_5

PAO Help is the Raspberry Pi 5 side of your Atlas assistant stack.
It provides the Flask web UI, chat streaming to Ollama, local chat history/cache,
and optional RAG note lookup.

## Project Structure

- app.py: main Flask server
- templates/index.html: web UI template
- chat.db: created automatically on first run

## Requirements

- Python 3.10+
- Ollama running locally at http://localhost:11434
- Model pulled in Ollama (default in app.py is phi3:mini)

Install Python packages:

```bash
pip install flask requests psutil chromadb
```

## Run

From this folder:

```bash
python app.py
```

Server starts on:

- http://0.0.0.0:5000
- Open in browser: http://localhost:5000

## Notes

- The UI HTML is stored in templates/index.html and rendered by Flask.
- app.py imports search_notes and build_rag_prompt from rag.py.
	Ensure rag.py exists in this project and is configured for your ChromaDB notes.
- Pi 3 status check expects a voice node endpoint at:
	http://pi3.local:5001/ping
