import chromadb

def initialize_chromadb(collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient("./chromadb_store")
    return client.get_or_create_collection(name=collection_name)
