import chromadb
import glob
import requests
from chromadb.config import Settings
from typing import List, Dict

def initialize_chromadb(collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient("./chromadb_store")  # PersistentClient for local DB
    return client.get_or_create_collection(name=collection_name)

def load_and_chunk_text(file_path: str, chunk_size: int = 512, stride: int = 256) -> List[str]:
    # updated chunking to use sliding window with overlap(stride)
    print(f"chunking file: {file_path}")
    with open(file_path, 'r') as file:
        text = file.read()
    return [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, stride)]

def process_text_files(folder_path: str, chunk_size: int = 512) -> Dict[str, List[str]]:
    file_paths = glob.glob(f"{folder_path}/*.txt")
    file_chunks = {}
    for file_path in file_paths:
        file_chunks[file_path] = load_and_chunk_text(file_path, chunk_size)
    return file_chunks

def generate_embeddings(chunks: List[str], model) -> List[List[float]]:
    return model.encode(chunks)

def store_embeddings_in_chromadb(
    collection: chromadb.Collection,
    file_chunks: Dict[str, List[str]],
    model
):
    for file_path, chunks in file_chunks.items():
        print(f"storing embeddings for file: {file_path}")
        embeddings = generate_embeddings(chunks, model)
        for idx, embedding in enumerate(embeddings):
            collection.add(
                documents=[chunks[idx]],
                metadatas=[{"file": file_path, "chunk": idx}],
                ids=[f"{file_path}-{idx}"]
            )

def query_chromadb(
    collection: chromadb.Collection,
    query: str,
    model,
    n_results: int = 5
) -> Dict:
    query_embedding = model.encode([query])
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results

def format_context(results: Dict) -> str:
    context = "".join(results["documents"][0])
    return context

def format_prompt(query, context):
    formatted = f"""
        Answer the QUERY below using the CONTEXT below. Use the CONTEXT below only if it is relevant to the QUERY, otherwise use general knowledge.\n
        QUERY: {query} \n
        CONTEXT: {context}\n
    """
    return formatted

def query_ollama(prompt: str, host: str = "http://localhost", port: int = 11434) -> str:
    url = f"{host}:{port}/api/generate"
    payload = {"prompt": prompt,
               "model": "llama3.2",
               "stream": False
               }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("response", "No response from Ollama.")
    else:
        return f"Error: {response.status_code}, {response.text}"

