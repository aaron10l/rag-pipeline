import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import glob
import requests


def initialize_chromadb(collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient("./chromadb_store")  # PersistentClient for local DB
    return client.get_or_create_collection(name=collection_name)


# updated chunking to use sliding window with overlap(stride)
def load_and_chunk_text(file_path: str, chunk_size: int = 512, stride: int = 256) -> List[str]:
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
    formatted = f"Here is some context you may or may not choose to use: \n {context}.\n Here is the original query: \n {query}"
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


# Streamlit App
st.title("RAG Pipeline with Ollama")

# Sidebar for initialization
with st.sidebar:
    st.header("Pipeline Initialization")
    folder_path = st.text_input("Path to Text Files", "./text-data/text/")
    initialize_button = st.button("Initialize Pipeline")

# Initialize components
if "collection" not in st.session_state:
    st.session_state.collection = None

if initialize_button:
    # initializing the vector db for the first time
    st.write("Initializing pipeline...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = initialize_chromadb("pdf_texts")

    if not collection.count() > 0:
        st.write("Indexing text data...")
        file_chunks = process_text_files(folder_path)
        store_embeddings_in_chromadb(collection, file_chunks, embedding_model)
        st.success("Pipeline initialized and embeddings stored!")
    else:
        st.success("Pipeline already initialized!")
    
    # Save to session state
    st.session_state.collection = collection
else:
    collection = st.session_state.collection

# Main Query Area
st.header("Query the RAG Pipeline")

if collection is None:
    st.warning("initialize the pipeline first!")
else:
    query = st.text_area("Enter your query:")
    submit_query = st.button("Submit Query")

    if submit_query and query:
        st.write("Retrieving context...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Reinitialize if needed
        results = query_chromadb(collection, query, embedding_model)
        context = format_context(results)
        st.write(f"**Retrieved Context:**\n{context}")
        prompt = format_prompt(query, context)

        # Send to Ollama
        st.write("Querying Ollama...")
        response = query_ollama(prompt)
        st.write(f"**Ollama Response:**\n{response}")

