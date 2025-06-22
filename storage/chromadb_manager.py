import chromadb
from config import CHROMA_DB_PATH

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def store_final_version(collection_name: str, doc_id: str, document: str):
    # Use get_or_create to avoid errors on first write
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(documents=[document], ids=[doc_id])
    print(f"Stored document '{doc_id}' in collection '{collection_name}'.")

def query_collection(collection_name: str, query_text: str, n_results: int = 3):
    """
    Queries a collection for n_results similar documents.
    Accepts n_results as a parameter.
    """
    try:
        collection = client.get_collection(name=collection_name)
        # Use the n_results parameter in the actual ChromaDB query
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        # Ensure 'documents' key exists and is not empty before accessing
        return results['documents'][0] if results.get('documents') else []
    except Exception as e:
        print(f"Error querying collection '{collection_name}': {e}")
        return []

def get_chroma_stats():
    """Returns statistics about the ChromaDB instance."""
    collections = client.list_collections()
    total_docs = sum(c.count() for c in collections)
    return {"collection_count": len(collections), "document_count": total_docs}