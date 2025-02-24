"""
This module handles the storage of text data in a ChromaDB vector database.
"""
import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Example sentence transformer model

def initialize_chromadb(vector_db_path: str):
    """Initializes ChromaDB instance for a notebook."""
    return chromadb.PersistentClient(path=vector_db_path)

def store_text_in_chromadb(notebook_name: str, texts: list):
    """Stores extracted text from PDFs into a ChromaDB vector database."""
    notebook_path = os.path.join("notebooks", notebook_name)
    vector_db_path = os.path.join(notebook_path, "chromadb")

    # Initialize ChromaDB
    client = initialize_chromadb(vector_db_path)
    collection = client.get_or_create_collection(
        name=notebook_name,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=CHROMA_EMBEDDING_MODEL)
    )

    # Add texts to ChromaDB
    for doc in texts:
        collection.add(
            ids=[doc["filename"]],
            documents=[doc["text"]]
        )

    print(f"Added {len(texts)} new documents to ChromaDB for {notebook_name}.")
