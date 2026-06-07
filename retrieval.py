def retrieve(question, collection, n_results=3):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results["documents"][0]
