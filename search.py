from duckduckgo_search import DDGS

MAX_RESULTS = 3
MAX_CHARS = 2000

def search_web(query):
    """Search DuckDuckGo and return text snippets."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=MAX_RESULTS):
                results.append({
                    "title": r.get("title", ""),
                    "body":  r.get("body", ""),
                    "url":   r.get("href", "")
                })
        print(f"Web search for '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def build_search_prompt(question, results):
    """Inject search results into the prompt."""
    if not results:
        return question, False

    context = ""
    total = 0
    for r in results:
        snippet = f"Title: {r['title']}\n{r['body']}\n\n"
        if total + len(snippet) > MAX_CHARS:
            break
        context += snippet
        total += len(snippet)

    prompt = f"""Answer the following question using ONLY the web search results below.
Be concise and direct. If the results contain the answer state it clearly.

--- WEB SEARCH RESULTS ---
{context}
--- END RESULTS ---

Question: {question}
Answer:"""

    return prompt, True
