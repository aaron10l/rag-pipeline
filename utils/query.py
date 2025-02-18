def query_chromadb(collection, query: str, model, n_results: int = 5) -> dict:
    query_embedding = model.encode([query])
    return collection.query(query_embeddings=query_embedding, n_results=n_results)

def format_context(results: dict) -> str:
    formatted = "\n\n".join(results["documents"][0])
    return formatted

def format_prompt(query: str, context: str) -> str:
    prompt = f"""
    Here is some context that may be relevant to the query:
    {context}

    Answer the following query in detail, using both the provided context and your own knowledge:
    {query}
    """
    return prompt
