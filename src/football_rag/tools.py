"""Agent tools: search the local football index and wikipedia."""

import requests 
from langchain_core.tools import tool

from football_rag import config
from football_rag.ingest import load_store

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "rag-practice/1.0 (github.com/JohnKarounis)"
HTTP_TIMEOUT = 10
WIKI_RESULTS = 3
WIKI_EXTRACT_CHARS = 1500
RECENT_POOL = 20 

store = load_store()
retriever = store.as_retriever(search_kwargs={'k': config.TOP_K})


def format_docs(docs):
    """Join retrieved chunks into one context string."""

    return "\n\n".join(d.page_content for d in docs)


@tool(parse_docstring=True)
def search_football_docs(query: str, prefer_recent: bool = False) -> str:
    """Search a collection of football articles (2022 FIFA World Cup, UEFA Euro 2024, FIFA World Cup history, Premier League, offside rule). Returns relevant passages.

    Args:
        query: A specific search query, e.g. "2022 World Cup final result".
        prefer_recent: Set true for questions about the latest, most recent or current event. Returns the newest relevant passages first.
    """

    if not prefer_recent:
        docs = retriever.invoke(query)
        return format_docs(docs)

    pool = store.similarity_search(query, k=RECENT_POOL)
    pool = sorted(pool, key=lambda d: d.metadata.get("max_year", 0), reverse=True)
    return format_docs(pool[:config.TOP_K])


@tool(parse_docstring=True)
def search_wikipedia(query: str) -> str:
    """Search Wikipedia. Always try search_football_docs first; use this only if it does not contain the answer or the topic is outside the local documents. Returns the intros of the top 3 matching Wikipedia articles.

    Args:
        query: A specific search query, e.g. "2024 Copa America final".
    """ 

    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": query, "gsrlimit": WIKI_RESULTS, "prop": "extracts",
        "exintro": 1, "explaintext": 1, "redirects": 1,
    }

    try:
        response = requests.get(
            WIKI_API_URL, 
            params=params,
            headers={'User-Agent': USER_AGENT},
            timeout=HTTP_TIMEOUT
        )

        response.raise_for_status()

    except requests.RequestException as e:
        return f'Wikipedia search failed: {e}'

    data = response.json()
    if "query" not in data:
        return "No wikipedia results found"

    pages = data['query']['pages']
    pages = sorted(pages.values(), key=lambda p: p['index'])

    results = []
    for page in pages:
        extract = page.get('extract', '')[:WIKI_EXTRACT_CHARS]
        entry = f"Title: {page['title']}\n{extract}"
        results.append(entry)

    return "\n\n".join(results)


agent_tools = [search_football_docs, search_wikipedia]
