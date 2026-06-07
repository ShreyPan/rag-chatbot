from ddgs import DDGS  # type: ignore
from retrieval import retrieve


def search_documents(query, collection):
    chunks = retrieve(query, collection)
    return "\n\n".join(chunks)


def search_web(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)

    if not results:
        return "No results found."

    formatted = ""
    for r in results:
        formatted += f"Title: {r['title']}\n"
        formatted += f"Summary: {r['body']}\n"
        formatted += f"URL: {r['href']}\n\n"
    return formatted
