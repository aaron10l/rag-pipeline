def query_chromadb(collection, query: str, model, n_results: int = 5) -> dict:
    query_embedding = model.encode([query])
    return collection.query(query_embeddings=query_embedding, n_results=n_results)

def format_context(results: dict) -> str:
    return "".join(results["documents"][0])

def format_prompt(query: str, context: str) -> str:
    return f"QUERY: {query}\nCONTEXT: {context}\n"
