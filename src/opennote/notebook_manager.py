"""
This module handles the creation and management of user-defined notebooks.

Each notebook has its own directory containing:
- PDF documents
- ChromaDB vector database
- Metadata file

"""
import os
import json

NOTEBOOKS_DIR = "notebooks"

def create_notebook(notebook_name: str):
    """Creates a new notebook with a designated folder and metadata file."""
    notebook_path = os.path.join(NOTEBOOKS_DIR, notebook_name)
    docs_path = os.path.join(notebook_path, "docs")
    chroma_path = os.path.join(notebook_path, "chromadb")

    if os.path.exists(notebook_path):
        print(f"Notebook '{notebook_name}' already exists.")
        return notebook_path

    os.makedirs(docs_path, exist_ok=True)
    os.makedirs(chroma_path, exist_ok=True)

    # Initialize metadata file
    metadata = {
        "notebook_name": notebook_name,
        "documents": [],
        "vector_db_path": chroma_path
    }
    with open(os.path.join(notebook_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Notebook '{notebook_name}' created at {notebook_path}")
    return notebook_path
