from ddgs import DDGS  # type: ignore


def search_documents(query, collection):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "metadatas"]
        )

        if not results["documents"][0]:
            return "No relevant information found in documents."

        formatted = ""
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            source = meta.get("source", "unknown")
            formatted += f"[Source: {source}]\n{doc}\n\n"

        return formatted

    except Exception as e:
        return f"Error searching documents: {e}"


def search_web(query):
    try:
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

    except Exception as e:
        return f"Web search failed: {e}. Please try again or rephrase your question."
