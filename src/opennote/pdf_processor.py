"""
This module handles the extraction of text from PDFs and the processing of newly added PDFs.
"""
import os
import json
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def process_new_pdfs(notebook_name: str):
    """Processes newly added PDFs in a notebook folder."""
    notebook_path = os.path.join("notebooks", notebook_name)
    docs_path = os.path.join(notebook_path, "docs")
    metadata_path = os.path.join(notebook_path, "metadata.json")

    if not os.path.exists(docs_path):
        print(f"No such notebook: {notebook_name}")
        return
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    processed_docs = set(metadata["documents"])
    new_texts = []

    for file in os.listdir(docs_path):
        if file.endswith(".pdf") and file not in processed_docs:
            pdf_path = os.path.join(docs_path, file)
            extracted_text = extract_text_from_pdf(pdf_path)
            if extracted_text:
                new_texts.append({"filename": file, "text": extracted_text})
                processed_docs.add(file)

    # Update metadata file
    metadata["documents"] = list(processed_docs)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return new_texts  # Returns extracted text for vectorization
