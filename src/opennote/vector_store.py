"""
This module handles the storage of text data in a ChromaDB vector database.
"""
import os
import json
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction  # type: ignore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
# Default to a more powerful embedding model if available
CHROMA_EMBEDDING_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL", "all-mpnet-base-v2")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CHUNK_MIN_SIZE = 200  # Minimum chunk size to avoid tiny chunks

def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name to meet ChromaDB requirements:
    - 3-63 characters
    - Starts and ends with alphanumeric
    - Contains only alphanumeric, underscores, or hyphens
    - No consecutive periods
    - Not a valid IPv4 address
    """
    # Replace spaces and other invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Ensure it starts and ends with alphanumeric
    if not sanitized[0].isalnum():
        sanitized = 'c' + sanitized
    if not sanitized[-1].isalnum():
        sanitized = sanitized + '0'
    
    # Ensure length is between 3-63 characters
    if len(sanitized) < 3:
        sanitized = sanitized + '0' * (3 - len(sanitized))
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
        # Ensure it still ends with alphanumeric
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + '0'
    
    return sanitized

def initialize_chromadb(vector_db_path: str):
    """Initializes ChromaDB instance for a notebook."""
    return chromadb.PersistentClient(path=vector_db_path)

def extract_sections(text: str) -> List[Dict[str, Any]]:
    """
    Extract logical sections from text based on structure.
    
    Args:
        text: The text to process
        
    Returns:
        List of dictionaries containing section text and metadata
    """
    if not text:
        return []
    
    # Split text by double newlines which often indicate paragraph or section breaks
    raw_sections = re.split(r'\n\s*\n', text)
    sections = []
    position = 0
    
    for section in raw_sections:
        if section.strip():  # Skip empty sections
            # Check if the section looks like a heading
            is_heading = False
            lines = section.split('\n')
            if len(lines) == 1 and len(lines[0]) < 100 and not lines[0].endswith('.'):
                is_heading = True
                
            section_length = len(section)
            sections.append({
                "text": section.strip(),
                "start": position,
                "end": position + section_length,
                "is_heading": is_heading
            })
            position += section_length + 2  # +2 for the removed double newline
    
    return sections

def calculate_chunk_importance(text: str) -> float:
    """
    Calculate importance score for a chunk based on content analysis.
    Uses simple heuristics without relying on NLTK tokenization.
    
    Args:
        text: Chunk text
        
    Returns:
        Importance score between 0 and 1
    """
    if not text:
        return 0.0
    
    score = 0.0
    
    # Length factor (normalize against typical chunk size)
    length = len(text)
    length_score = min(1.0, length / 1000)
    score += length_score * 0.3
    
    # Numbers presence
    numbers_count = len(re.findall(r'\d+', text))
    numbers_score = min(1.0, numbers_count / 10)
    score += numbers_score * 0.2
    
    # Simple information density using basic tokenization
    words = re.findall(r'\b\w+\b', text.lower())
    
    if words:
        # Common English stopwords (simplified list)
        common_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'that', 'this',
            'it', 'as', 'be', 'from', 'have', 'has', 'had', 'not', 'no'
        }
        non_stop_words = [w for w in words if w not in common_stopwords and len(w) > 1]
        density_score = len(non_stop_words) / len(words)
        score += density_score * 0.5
    
    # Bonus for structured content markers (headings, lists, etc.)
    if re.search(r'^#+\s', text, re.MULTILINE):  # Markdown-style headings
        score += 0.1
    
    if re.search(r'^(\*|-|\d+\.)\s', text, re.MULTILINE):  # Lists
        score += 0.05
    
    return min(1.0, score)  # Cap at 1.0

def semantic_chunk_text(text: str, 
                       chunk_size: int = DEFAULT_CHUNK_SIZE, 
                       chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                       min_chunk_size: int = DEFAULT_CHUNK_MIN_SIZE) -> List[Dict[str, Any]]:
    """
    Split text into semantically meaningful chunks for better retrieval.
    Uses document structure and content analysis for intelligent chunking.
    
    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        min_chunk_size: Minimum size for a chunk
        
    Returns:
        List of dictionaries containing chunk text and metadata
    """
    if not text:
        return []
    
    # First extract logical sections from the document
    sections = extract_sections(text)
    
    # Now create semantic chunks from these sections
    chunks = []
    current_chunk = []
    current_length = 0
    last_end = 0
    
    for section in sections:
        section_text = section["text"]
        section_length = len(section_text)
        
        # If this is a heading, add it to the next chunk
        if section.get("is_heading", False):
            # If we already have content, create a chunk before adding heading to next chunk
            if current_length > min_chunk_size:
                chunk_text = "\n\n".join(current_chunk)
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start": last_end - current_length,
                    "end": section["start"],
                    "importance": calculate_chunk_importance(chunk_text)
                })
                # Start new chunk with heading
                current_chunk = [section_text]
                current_length = section_length
                last_end = section["end"]
            else:
                # Add heading to current chunk
                current_chunk.append(section_text)
                current_length += section_length
                last_end = section["end"]
            continue
        
        # If the current section can fit in the current chunk, add it
        if current_length + section_length <= chunk_size:
            current_chunk.append(section_text)
            current_length += section_length
            last_end = section["end"]
            continue
        
        # If the section is too large for a single chunk, we need to split it
        if section_length > chunk_size:
            # First, finalize current chunk if it's not empty
            if current_length > min_chunk_size:
                chunk_text = "\n\n".join(current_chunk)
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start": last_end - current_length,
                    "end": section["start"],
                    "importance": calculate_chunk_importance(chunk_text)
                })
                current_chunk = []
                current_length = 0
            
            # Split the large section into sentences using regex
            sentences = re.split(r'(?<=[.!?])\s+', section_text)
            temp_chunk = []
            temp_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if temp_length + sentence_length <= chunk_size:
                    temp_chunk.append(sentence)
                    temp_length += sentence_length
                else:
                    # Finalize temp chunk if not empty
                    if temp_length > min_chunk_size:
                        chunk_text = " ".join(temp_chunk)
                        chunk_id = str(uuid.uuid4())
                        chunks.append({
                            "id": chunk_id,
                            "text": chunk_text,
                            "start": section["start"] + (section_length - temp_length),
                            "end": section["start"] + section_length,
                            "importance": calculate_chunk_importance(chunk_text)
                        })
                    
                    # Start new temp chunk with current sentence
                    temp_chunk = [sentence]
                    temp_length = sentence_length
            
            # Add remaining sentences as a chunk
            if temp_length > min_chunk_size:
                chunk_text = " ".join(temp_chunk)
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start": section["start"] + (section_length - temp_length),
                    "end": section["end"],
                    "importance": calculate_chunk_importance(chunk_text)
                })
            
            last_end = section["end"]
        else:
            # Section doesn't fit in current chunk, finalize current chunk
            if current_length > min_chunk_size:
                chunk_text = "\n\n".join(current_chunk)
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start": last_end - current_length,
                    "end": section["start"],
                    "importance": calculate_chunk_importance(chunk_text)
                })
            
            # Start new chunk with current section
            current_chunk = [section_text]
            current_length = section_length
            last_end = section["end"]
    
    # Add the final chunk if not empty
    if current_length > min_chunk_size:
        chunk_text = "\n\n".join(current_chunk)
        chunk_id = str(uuid.uuid4())
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "start": last_end - current_length,
            "end": last_end,
            "importance": calculate_chunk_importance(chunk_text)
        })
    
    return chunks

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks for better retrieval.
    This function now uses semantic chunking for improved quality.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of dictionaries containing chunk text and metadata
    """
    return semantic_chunk_text(text, chunk_size, chunk_overlap)

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

    # Sanitize collection name for ChromaDB
    sanitized_name = sanitize_collection_name(notebook_name)
    existing_collections = client.list_collections()
    if sanitized_name in existing_collections:
        raise ValueError(f"Collection name '{sanitized_name}' already exists.")
    collection = client.get_or_create_collection(
        name=sanitized_name,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=CHROMA_EMBEDDING_MODEL)
    )

    # Load metadata to track chunks
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Initialize chunks tracking if not present
    if "chunks" not in metadata:
        metadata["chunks"] = {}

    # Store the sanitized collection name in metadata
    metadata["collection_name"] = sanitized_name

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
                "source": filename,
                "importance": chunk.get("importance", 0.5)  # Include chunk importance
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

def enhance_query(query: str) -> str:
    """
    Enhance the query by removing stopwords and normalizing text.
    
    Args:
        query: The original user query
        
    Returns:
        Enhanced query string
    """
    # Basic query cleaning
    query = query.strip().lower()
    
    # Remove common question prefixes that don't add semantic meaning
    prefixes = [
        "what is", "how to", "can you tell me about", "tell me about",
        "describe", "explain", "what are", "who is", "when was"
    ]
    
    for prefix in prefixes:
        if query.startswith(prefix):
            query = query[len(prefix):].strip()
            break
    
    return query

def query_vector_db(
    notebook_name: str, 
    query: str, 
    top_k: int = 5,
    importance_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Query the vector database for relevant chunks with enhanced retrieval.
    
    Args:
        notebook_name: Name of the notebook
        query: Query string
        top_k: Number of results to return
        importance_weight: Weight to give chunk importance in final ranking (0-1)
        
    Returns:
        List of dictionaries containing text and metadata
    """
    notebook_path = os.path.join("notebooks", notebook_name)
    vector_db_path = os.path.join(notebook_path, "chromadb")
    metadata_path = os.path.join(notebook_path, "metadata.json")

    # Enhance the query
    enhanced_query = enhance_query(query)

    # Initialize ChromaDB
    client = initialize_chromadb(vector_db_path)

    # Get the sanitized collection name from metadata if available
    collection_name = notebook_name
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if "collection_name" in metadata:
                    collection_name = metadata["collection_name"]
                else:
                    # If not in metadata, sanitize it now
                    collection_name = sanitize_collection_name(notebook_name)
        except:
            # Fallback to sanitizing the name
            collection_name = sanitize_collection_name(notebook_name)
    else:
        # Fallback to sanitizing the name
        collection_name = sanitize_collection_name(notebook_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=CHROMA_EMBEDDING_MODEL)
    )

    # Retrieve more results than needed to rerank
    search_k = min(top_k * 2, 20)  # Get more candidates for reranking
    
    # Query the collection
    results = collection.query(
        query_texts=[enhanced_query],
        n_results=search_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format and rerank results
    formatted_results = []
    
    if results and "documents" in results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            
            # Calculate vector similarity score (0-1 range)
            vector_score = 1.0 - (distance / 2) if distance is not None else 0.5
            
            # Get importance from metadata or default to 0.5
            importance = float(metadata.get("importance", 0.5))
            
            # Calculate combined score with weighting
            combined_score = (
                (1 - importance_weight) * vector_score + 
                importance_weight * importance
            )
            
            formatted_results.append({
                "text": doc,
                "metadata": metadata,
                "vector_score": vector_score,
                "importance": importance,
                "relevance_score": combined_score  # Combined score as relevance
            })
    
    # Sort by combined relevance score and take top_k
    formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return formatted_results[:top_k]
