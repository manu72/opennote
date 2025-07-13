"""
This module handles the extraction of text from PDFs and the processing of newly added PDFs.
"""
import os
import json
import re
from typing import Dict, List, Optional
import fitz  # PyMuPDF

def is_header_or_footer(text: str, page_height: float, y_pos: float, threshold: float = 0.1) -> bool:
    """
    Determine if text block is likely a header or footer based on position and content.
    
    Args:
        text: The text content
        page_height: Height of the page
        y_pos: Y position of the text block
        threshold: Position threshold for headers/footers (percentage of page height)
        
    Returns:
        Boolean indicating if text is likely header/footer
    """
    # Check position (top or bottom of page)
    pos_threshold = page_height * threshold
    is_top = y_pos < pos_threshold
    is_bottom = y_pos > (page_height - pos_threshold)
    
    # Check content patterns common in headers/footers
    contains_page_number = bool(re.search(r'\b(?:page|pg)\.?\s*\d+\b', text.lower()) or 
                               re.match(r'^\s*\d+\s*$', text))
    contains_date = bool(re.search(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}', text))
    
    return (is_top or is_bottom) and (contains_page_number or contains_date or len(text) < 50)

def extract_structured_text_from_pdf(pdf_path: str) -> Dict[str, str]:
    """
    Extracts text from a PDF file preserving structure and metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with structured text content and metadata
    """
    try:
        doc = fitz.open(pdf_path)
        result = {
            "title": os.path.basename(pdf_path),
            "pages": [],
            "toc": []
        }
        
        # Extract TOC if available
        toc = doc.get_toc()
        if toc:
            result["toc"] = toc
        
        # Extract metadata
        metadata = doc.metadata
        if metadata:
            for key, value in metadata.items():
                if value and key.lower() != "format":  # Skip format key
                    result[key.lower()] = value
        
        # Process each page
        full_text = []
        for page_num, page in enumerate(doc):
            page_dict = {"page_num": page_num + 1, "content": ""}
            
            # Extract text blocks with position information
            text_blocks = []
            blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            
            for block in blocks:
                if block["type"] == 0:  # Text block
                    text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"]
                        text += "\n"
                    
                    # Skip headers and footers
                    y_pos = block["bbox"][1]  # Top y-coordinate
                    if not is_header_or_footer(text, page_height, y_pos):
                        text_blocks.append({
                            "text": text.strip(),
                            "bbox": block["bbox"],
                            "is_bold": any(span.get("font", "").lower().find("bold") >= 0 
                                          for line in block["lines"] 
                                          for span in line["spans"])
                        })
            
            # Join text blocks
            page_text = "\n\n".join([block["text"] for block in text_blocks])
            page_dict["content"] = page_text
            result["pages"].append(page_dict)
            
            # Add to full text
            full_text.append(page_text)
        
        # Combined text with page breaks marked
        result["text"] = "\n\n--- Page Break ---\n\n".join(full_text)
        
        return result
    
    except FileNotFoundError:
        error_msg = f"PDF file not found: {pdf_path}"
        print(error_msg)
        return {"title": os.path.basename(pdf_path), "text": "", "pages": [], "error": error_msg}
    except PermissionError:
        error_msg = f"Permission denied accessing PDF: {pdf_path}"
        print(error_msg)
        return {"title": os.path.basename(pdf_path), "text": "", "pages": [], "error": error_msg}
    except fitz.FileDataError as e:
        error_msg = f"Corrupted or invalid PDF file: {pdf_path} - {str(e)}"
        print(error_msg)
        return {"title": os.path.basename(pdf_path), "text": "", "pages": [], "error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error processing {pdf_path}: {str(e)}"
        print(error_msg)
        return {"title": os.path.basename(pdf_path), "text": "", "pages": [], "error": error_msg}

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    This is a compatibility wrapper around the new structured extraction function.
    """
    try:
        structured_result = extract_structured_text_from_pdf(pdf_path)
        return structured_result["text"]
    except (FileNotFoundError, PermissionError, fitz.FileDataError) as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error processing {pdf_path}: {e}")
        return ""

def process_new_pdfs(notebook_name: str, preserve_structure: bool = True) -> List[Dict[str, str]]:
    """
    Processes newly added PDFs in a notebook folder.
    
    Args:
        notebook_name: Name of the notebook
        preserve_structure: Whether to use enhanced structure-preserving extraction
        
    Returns:
        List of dictionaries with extracted text and metadata
    """
    notebook_path = os.path.join("notebooks", notebook_name)
    docs_path = os.path.join(notebook_path, "docs")
    metadata_path = os.path.join(notebook_path, "metadata.json")
    structured_path = os.path.join(notebook_path, "structured_docs")

    if not os.path.exists(docs_path):
        print(f"No such notebook: {notebook_name}")
        return []

    # Create structured docs directory if using enhanced extraction
    if preserve_structure:
        os.makedirs(structured_path, exist_ok=True)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Initialize document structure tracking if not present
    if "document_structure" not in metadata:
        metadata["document_structure"] = {}

    processed_docs = set(metadata["documents"])
    new_texts = []
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf") and f not in processed_docs]
    
    if not pdf_files:
        print("No new PDF files to process.")
        return []
    
    print(f"Processing {len(pdf_files)} new PDF file(s)...")

    for i, file in enumerate(pdf_files, 1):
        print(f"  [{i}/{len(pdf_files)}] Processing {file}...")
        pdf_path = os.path.join(docs_path, file)
        
        if preserve_structure:
            # Extract structured information
            structured_data = extract_structured_text_from_pdf(pdf_path)
            extracted_text = structured_data["text"]
            
            # Save structured data for future use
            struct_file_path = os.path.join(structured_path, f"{os.path.splitext(file)[0]}.json")
            with open(struct_file_path, "w") as f:
                json.dump(structured_data, f, indent=4)
            
            # Store structure information in metadata
            metadata["document_structure"][file] = {
                "title": structured_data.get("title", file),
                "author": structured_data.get("author", ""),
                "structured_path": struct_file_path,
                "has_toc": bool(structured_data.get("toc", [])),
                "page_count": len(structured_data.get("pages", [])),
            }
        else:
            # Use legacy extraction
            extracted_text = extract_text_from_pdf(pdf_path)
        
        if extracted_text:
            new_texts.append({"filename": file, "text": extracted_text})
            processed_docs.add(file)
            print(f"    ✓ Successfully processed {file}")
        else:
            print(f"    ✗ Failed to extract text from {file}")

    # Update metadata file
    metadata["documents"] = list(processed_docs)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return new_texts  # Returns extracted text for vectorization
