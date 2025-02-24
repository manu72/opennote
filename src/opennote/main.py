"""
Script creates a notebook, adds PDFs, processes them, and generates the ChromaDB vector store
"""
import sys
from opennote.notebook_manager import create_notebook
from opennote.pdf_processor import process_new_pdfs
from opennote.vector_store import store_text_in_chromadb

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <notebook_name>")
        return

    notebook_name = sys.argv[1]

    # Create or check the notebook
    create_notebook(notebook_name)

    # Process new PDFs
    extracted_texts = process_new_pdfs(notebook_name)
    if extracted_texts:
        store_text_in_chromadb(notebook_name, extracted_texts)

if __name__ == "__main__":
    main()
