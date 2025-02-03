def generate_embeddings(chunks: list, model) -> list:
    return model.encode(chunks)

def store_embeddings_in_chromadb(collection, file_chunks, model):
    for file_path, chunks in file_chunks.items():
        embeddings = generate_embeddings(chunks, model)
        for idx, embedding in enumerate(embeddings):
            collection.add(
                documents=[chunks[idx]],
                metadatas=[{"file": file_path, "chunk": idx}],
                ids=[f"{file_path}-{idx}"]
            )
