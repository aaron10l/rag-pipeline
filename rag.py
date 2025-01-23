import streamlit as st
from sentence_transformers import SentenceTransformer
import time
import chromadb
from chromadb.config import Settings
from xml.etree import ElementTree
import requests
import json

import utils

def initialize_chromadb(collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient("./chromadb_store")  # PersistentClient for local DB
    return client.get_or_create_collection(name=collection_name)

def fetch_pmc_full_text(query: str, papers_metadata, max_results: int = 5) -> None:
    """
    Query PubMed Central and fetch the full text body of the top results. Saves the full-text body of articles as .txt files.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Use esearch to get article IDs
    search_url = f"{base_url}/esearch.fcgi"
    search_params = {
        "db": "pmc",
        "term": query,
        "retmax": max_results,
        "retmode": "xml"
    }
    search_response = requests.get(search_url, params=search_params)
    search_response.raise_for_status()

    # Parse the response to extract IDs
    search_tree = ElementTree.fromstring(search_response.content)
    pmc_ids = [id_elem.text for id_elem in search_tree.findall(".//Id")]
    print(f"PMC IDs: {pmc_ids}")

    if not pmc_ids:
        print("No results found.")
        return

    # Fetch the text body of each individual paper
    fetch_url = f"{base_url}/efetch.fcgi"

    for pmc_id in pmc_ids:
        if pmc_id not in papers_metadata:
            fetch_params = {
                "db": "pmc",
                "id": pmc_id,
                "rettype": "full",
                "retmode": "xml"
            }
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()
            time.sleep(1)

            # Parse the XML to extract the full text body
            article_tree = ElementTree.fromstring(fetch_response.content)
            body_elements = article_tree.findall(".//body")

            if not body_elements:
                print(f"No full text body found for article {pmc_id}.")
                continue

            full_text = "\n".join(ElementTree.tostring(body, encoding="unicode", method="text") for body in body_elements)

            # save the full text body to a file
            file_name = f"corpus/PMC_{pmc_id}.txt"
            with open(file_name, "w", encoding="utf-8") as file:
                file.write(full_text)
            
            # Extract metadata fields
            title_elements = article_tree.findall(".//article-title")
            title = title_elements[0].text if title_elements else "Unknown Title"

            journal_elements = article_tree.findall(".//journal-title")
            journal = journal_elements[0].text if journal_elements else "Unknown Journal"

            publication_date_elements = article_tree.findall(".//pub-date")
            publication_date = (
                publication_date_elements[0].text if publication_date_elements else "Unknown Date"
            )

            papers_metadata[pmc_id] = {
                "title": title,
                "file_name": file_name,
                "journal": journal,
                "publication_date": publication_date,
                "file_name": file_name
            }
            print(f"Saved full text body of article {pmc_id} as {file_name}")
        else:
            print(f"skipping article {pmc_id}, already been saved")

    return papers_metadata

def main():
    st.title("RAG Pipeline with Ollama")

    # sidebar for initialization
    with st.sidebar:
        st.header("Pipeline Initialization")
        gene_symbol = st.text_input("Gene Symbol", "TP53")
        folder_path = st.text_input("Folder Path", "./corpus/")
        initialize_button = st.button("Initialize Pipeline")

    # Initialize components
    if "collection" not in st.session_state:
        st.session_state.collection = None

    if initialize_button:
        st.write("Initializing pipeline...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        collection = initialize_chromadb("gene_data")

        queries = [
            f"interaction partners of {gene_symbol}",
            f"protein interaction regions of {gene_symbol}",
            f"functional sites of {gene_symbol}"
        ]

        papers_metadata = {}

        # Fetch and process papers using fetch_pmc_full_text
        for query in queries:
            print(f"processing query {query}...")
            papers_metadata = fetch_pmc_full_text(query, papers_metadata)
            print(f"there are {len(papers_metadata)} papers downloaded.")
        #TODO: complete the process papers and store function

        # CHUNKING, VECTORIZING, AND STORING PAPERS IN CHROMADB
        if not collection.count() > 0:
            st.write("storing data in chromadb")
            file_chunks = utils.process_text_files(folder_path)
            utils.store_embeddings_in_chromadb(collection, file_chunks, embedding_model)
            st.success("Pipeline initialized and relevant data retrieved!")
        else:
            st.write("there is already vectorized data in the chromadb.")
        st.session_state.collection = collection

# Main Query Area
    st.header("Query the RAG Pipeline")

    if st.session_state.collection is None:
        st.warning("Initialize the pipeline first!")
    else:
        query = st.text_area("Enter your query:")
        submit_query = st.button("Submit Query")

        if submit_query and query:
            st.write("Retrieving context...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Reinitialize if needed
            results = {}  # Placeholder for query results from ChromaDB

            # Retrieve and format context
            context = "\n".join(results.get("documents", ["No results found."]))
            st.write(f"**Retrieved Context:**\n{context}")

            # Format and send prompt to Ollama
            prompt = f"Here is some context: \n{context}.\nOriginal query: \n{query}"
            st.write("Querying Ollama...")
            response = "Ollama response placeholder."  # Placeholder for Ollama query
            st.write(f"**Ollama Response:**\n{response}")

if __name__ == "__main__":
    main()
