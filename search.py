import requests
from bs4 import BeautifulSoup
import urllib.parse

# ── Config ───────────────────────────────────────────────────────────────────
MAX_RESULTS = 5       # how many DuckDuckGo results to pull
SNIPPET_LENGTH = 400  # max characters to keep from each result's description
REQUEST_TIMEOUT = 8   # seconds before giving up on a slow page
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {
    # Pretend to be a normal browser so DDG doesn't block the request
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux aarch64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _ddg_search(query: str) -> list[dict]:
    """
    Scrape DuckDuckGo HTML results for `query`.
    Returns a list of dicts with 'title', 'url', and 'snippet'.
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[search] DuckDuckGo request failed: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for result in soup.select(".result__body"):
        title_tag = result.select_one(".result__title")
        url_tag = result.select_one(".result__url")
        snippet_tag = result.select_one(".result__snippet")

        title = title_tag.get_text(strip=True) if title_tag else "No title"
        url = url_tag.get_text(strip=True) if url_tag else ""
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

        if not snippet:
            continue  # skip results with no useful text

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet[:SNIPPET_LENGTH]
        })

        if len(results) >= MAX_RESULTS:
            break

    return results


def search(query: str) -> dict:
    """
    Search DuckDuckGo for `query` and return:
      - results : raw list of {title, url, snippet}
      - injected: formatted string ready to paste into an Ollama prompt
    
    Use `injected` the same way you inject RAG note chunks.
    """
    print(f"[search] Querying DuckDuckGo: '{query}'")
    results = _ddg_search(query)

    if not results:
        return {
            "results": [],
            "injected": "[Web search returned no results. Answer from your own knowledge.]"
        }

    # Build the formatted block to inject into the prompt
    lines = ["The following are live web search results. Use them to answer the question.\n"]
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    {r['snippet']}")
        lines.append(f"    Source: {r['url']}\n")

    injected = "\n".join(lines)

    return {
        "results": results,
        "injected": injected
    }


def build_search_prompt(user_question: str, persona_system_prompt: str = "") -> str:
    """
    Convenience function: runs a search and returns a complete prompt
    you can send straight to Ollama.
    
    Usage in app.py:
        from search import build_search_prompt
        prompt = build_search_prompt(user_message, current_persona)
        # send prompt to Ollama as normal
    """
    result = search(user_question)
    injected = result["injected"]

    parts = []
    if persona_system_prompt:
        parts.append(persona_system_prompt.strip())
    parts.append(injected)
    parts.append(f"USER QUESTION: {user_question}")
    parts.append("Answer based on the search results above. Be concise and accurate.")

    return "\n\n".join(parts)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = "What is Retrieval Augmented Generation?"
    result = search(query)

    print(f"Found {len(result['results'])} results\n")
    print("── Injected prompt block ──────────────────────────────")
    print(result["injected"])
