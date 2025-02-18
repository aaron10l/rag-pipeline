import streamlit as st
from sentence_transformers import SentenceTransformer

from data.chroma import initialize_chromadb
from data.pubmed import fetch_pmc_full_text
from utils.processing import process_text_files, convert_pdf_to_xml, convert_xml_to_txt
from utils.embedding import store_embeddings_in_chromadb
from utils.query import query_chromadb, format_context, format_prompt
from utils.ollama import query_ollama

def main():
    st.title("RAG Pipeline with Ollama")

    # Sidebar initialization
    with st.sidebar:
        st.header("Pipeline Initialization")
        gene_symbol = st.text_input("Gene Symbol", "NKX2-1")
        folder_path = st.text_input("Folder Path", "./corpus/")
        initialize_button = st.button("Initialize Pipeline")

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

        # for query in queries:
            # papers_metadata = fetch_pmc_full_text(query, papers_metadata, folder_path)

        # convert papers to xml using grobid
        xml_files_path = convert_pdf_to_xml(folder_path)
        
        # convert xml to txt using treeparser
        text_files_path = convert_xml_to_txt(xml_files_path)

        # Store processed data in ChromaDB
        if not collection.count() > 0:
            file_chunks = process_text_files(text_files_path)
            store_embeddings_in_chromadb(collection, file_chunks, embedding_model)
            st.success("Pipeline initialized and relevant data retrieved!")
        else:
            st.write("Existing vectorized data found in ChromaDB.")

        st.session_state.collection = collection

    # Querying interface
    st.header("Query the RAG Pipeline")
    
    if st.session_state.collection:
        query = st.text_area("Enter your query:")
        submit_query = st.button("Submit Query")

        if submit_query and query:
            st.write("Retrieving context...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            results = query_chromadb(st.session_state.collection, query, embedding_model)
            context = format_context(results)
            st.write(f"**Retrieved Context:**\n{context}")

            prompt = format_prompt(query, context)
            st.write("Querying Ollama...")
            response = query_ollama(prompt)
            st.write(f"**Ollama Response:**\n{response}")
    else:
        st.warning("Initialize the pipeline first!")

if __name__ == "__main__":
    main()
