"""
This module handles the storage of text data in a ChromaDB vector database.
"""
import os
import json
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
CHROMA_EMBEDDING_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

def initialize_chromadb(vector_db_path: str):
    """Initializes ChromaDB instance for a notebook."""
    return chromadb.PersistentClient(path=vector_db_path)

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks for better retrieval.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of dictionaries containing chunk text and metadata
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position with overlap
        end = min(start + chunk_size, text_length)
        
        # If this is not the first chunk and we're not at the end, 
        # try to find a good break point (newline or period)
        if start > 0 and end < text_length:
            # Look for a newline or period in the last 100 chars of the chunk
            last_part = text[max(end - 100, start):end]
            
            # Try to find a newline first
            newline_pos = last_part.rfind('\n')
            if newline_pos != -1:
                end = max(end - 100, start) + newline_pos + 1
            else:
                # Try to find a period followed by space or newline
                period_pos = last_part.rfind('. ')
                if period_pos != -1:
                    end = max(end - 100, start) + period_pos + 2
        
        # Extract the chunk
        chunk_text = text[start:end].strip()
        
        if chunk_text:  # Only add non-empty chunks
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "start": start,
                "end": end
            })
        
        # Move to next chunk with overlap
        start = end - chunk_overlap
        
        # Ensure we're making progress
        if start >= end:
            start = end
    
    return chunks

def store_text_in_chromadb(notebook_name: str, texts: list, 
                           chunk_size: int = DEFAULT_CHUNK_SIZE,
                           chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    """
    Stores extracted text from PDFs into a ChromaDB vector database.
    
    Args:
        notebook_name: Name of the notebook
        texts: List of dictionaries containing filename and text
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
    """
    notebook_path = os.path.join("notebooks", notebook_name)
    vector_db_path = os.path.join(notebook_path, "chromadb")
    metadata_path = os.path.join(notebook_path, "metadata.json")

    # Initialize ChromaDB
    client = initialize_chromadb(vector_db_path)
    collection = client.get_or_create_collection(
        name=notebook_name,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=CHROMA_EMBEDDING_MODEL)
    )
    
    # Load metadata to track chunks
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Initialize chunks tracking if not present
    if "chunks" not in metadata:
        metadata["chunks"] = {}
    
    total_chunks = 0
    
    # Process each document
    for doc in texts:
        filename = doc["filename"]
        text = doc["text"]
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        # Add chunks to ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = chunk["id"]
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "filename": filename,
                "start": chunk["start"],
                "end": chunk["end"],
                "source": filename
            })
        
        # Store in ChromaDB
        if ids:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        
        # Update metadata
        metadata["chunks"][filename] = [chunk["id"] for chunk in chunks]
        total_chunks += len(chunks)
    
    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Added {total_chunks} chunks from {len(texts)} documents to ChromaDB for {notebook_name}.")

def query_vector_db(notebook_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the vector database for relevant chunks.
    
    Args:
        notebook_name: Name of the notebook
        query: Query string
        top_k: Number of results to return
        
    Returns:
        List of dictionaries containing text and metadata
    """
    notebook_path = os.path.join("notebooks", notebook_name)
    vector_db_path = os.path.join(notebook_path, "chromadb")
    
    # Initialize ChromaDB
    client = initialize_chromadb(vector_db_path)
    collection = client.get_or_create_collection(
        name=notebook_name,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=CHROMA_EMBEDDING_MODEL)
    )
    
    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted_results = []
    
    if results and "documents" in results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            
            formatted_results.append({
                "text": doc,
                "metadata": metadata,
                "relevance_score": 1.0 - (distance / 2) if distance is not None else None  # Convert distance to score
            })
    
    return formatted_results
