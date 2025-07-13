"""
Tests for the pdf_processor module.
"""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import json

from src.opennote.pdf_processor import (
    is_header_or_footer, 
    extract_text_from_pdf, 
    extract_structured_text_from_pdf,
    process_new_pdfs
)


class TestPDFProcessor(unittest.TestCase):
    """Test cases for pdf_processor functions."""

    def test_is_header_or_footer(self):
        """Test header/footer detection logic."""
        page_height = 800.0
        
        # Test header position (top of page)
        self.assertTrue(is_header_or_footer("Page 1", page_height, 20.0))
        
        # Test footer position (bottom of page)
        self.assertTrue(is_header_or_footer("© 2024 Company", page_height, 780.0))
        
        # Test middle of page with normal content
        self.assertFalse(is_header_or_footer("This is normal document content that should not be filtered.", page_height, 400.0))
        
        # Test page number patterns
        self.assertTrue(is_header_or_footer("Page 5", page_height, 30.0))
        self.assertTrue(is_header_or_footer("5", page_height, 30.0))
        self.assertTrue(is_header_or_footer("pg. 10", page_height, 770.0))
        
        # Test date patterns
        self.assertTrue(is_header_or_footer("01/01/2024", page_height, 25.0))
        self.assertTrue(is_header_or_footer("2024-01-01", page_height, 775.0))

    @patch('src.opennote.pdf_processor.fitz.open')
    def test_extract_text_from_pdf_success(self, mock_fitz_open):
        """Test successful PDF text extraction."""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.metadata = {"title": "Test Document"}
        mock_doc.get_toc.return_value = []
        
        # Mock page content
        mock_page.rect.height = 800
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,  # Text block
                    "bbox": [100, 300, 500, 350],  # Middle of page
                    "lines": [
                        {
                            "spans": [{"text": "This is test content"}]
                        }
                    ]
                }
            ]
        }
        
        mock_fitz_open.return_value = mock_doc
        
        # Test extraction
        result = extract_structured_text_from_pdf("test.pdf")
        
        self.assertIn("title", result)
        self.assertIn("text", result)
        self.assertIn("pages", result)
        self.assertTrue(len(result["pages"]) > 0)

    @patch('src.opennote.pdf_processor.fitz.open')
    def test_extract_text_from_pdf_file_error(self, mock_fitz_open):
        """Test PDF extraction with file error."""
        mock_fitz_open.side_effect = FileNotFoundError("File not found")
        
        result = extract_structured_text_from_pdf("nonexistent.pdf")
        
        self.assertIn("error", result)
        self.assertEqual(result["text"], "")

    def test_process_new_pdfs_no_notebook(self):
        """Test processing PDFs when notebook doesn't exist."""
        result = process_new_pdfs("nonexistent_notebook")
        self.assertEqual(result, [])

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('builtins.open')
    @patch('src.opennote.pdf_processor.extract_structured_text_from_pdf')
    def test_process_new_pdfs_success(self, mock_extract, mock_open, mock_listdir, mock_exists):
        """Test successful processing of new PDFs."""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["test.pdf", "existing.pdf", "not_pdf.txt"]
        
        # Mock metadata file
        metadata = {
            "documents": ["existing.pdf"],
            "document_structure": {}
        }
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(metadata)
        
        # Mock PDF extraction
        mock_extract.return_value = {
            "text": "Extracted text content",
            "title": "Test PDF",
            "pages": [{"page_num": 1, "content": "Page content"}]
        }
        
        # Test processing
        with patch('json.load', return_value=metadata), \
             patch('json.dump'), \
             patch('os.makedirs'):
            
            result = process_new_pdfs("test_notebook")
            
            # Should process only the new PDF (test.pdf)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["filename"], "test.pdf")
            self.assertEqual(result[0]["text"], "Extracted text content")


if __name__ == "__main__":
    unittest.main()