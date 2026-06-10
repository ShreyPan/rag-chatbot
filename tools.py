from ddgs import DDGS  # type: ignore
from retrieval import retrieve


def search_documents(query, collection):
    try:
        chunks = retrieve(query, collection)
        if not chunks:
            return "No relevant information found in documents."
        return "\n\n".join(chunks)
    except Exception as e:
        return f"Error searching documents: {e}"


def search_web(query):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)

        if not results:
            return "No results found."

        if not results:
            return "No results found."

        formatted = ""
        for r in results:
            formatted += f"Title: {r['title']}\n"
            formatted += f"Summary: {r['body']}\n"
            formatted += f"URL: {r['href']}\n\n"
        return formatted

    except Exception as e:
        return f"Web search failed: {e}. Please try again or rephrase your question."
