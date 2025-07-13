"""
Tests for the vector_store module, focusing on semantic chunking and retrieval improvements.
"""
import os
import sys
import tempfile
import unittest
import shutil
# from pathlib import Path
# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opennote.vector_store import (
    semantic_chunk_text, chunk_text, extract_sections,
    calculate_chunk_importance, enhance_query,
    initialize_chromadb, sanitize_collection_name
)

class TestVectorStore(unittest.TestCase):
    """Test case for vector_store module functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    def test_extract_sections(self):
        """Test the extract_sections function."""
        test_text = """# Heading 1

This is the first paragraph.
It has multiple lines.

This is the second paragraph.

## Heading 2

This is another paragraph under heading 2.
"""
        sections = extract_sections(test_text)

        # Print sections for debugging
        print("Detected sections:")
        for i, section in enumerate(sections):
            print(f"Section {i}: '{section['text']}' (is_heading: {section.get('is_heading', False)})")

        # Check that we have multiple sections
        self.assertGreater(len(sections), 2)

        # Verify first section is a heading
        self.assertTrue(sections[0].get("is_heading", False), "First section should be detected as a heading")

        # Check that we can find both heading and content
        heading_sections = [s for s in sections if s.get("is_heading", False)]
        content_sections = [s for s in sections if not s.get("is_heading", False)]

        # Verify we have at least one heading and one content section
        self.assertGreaterEqual(len(heading_sections), 1, "Should detect at least one heading")
        self.assertGreaterEqual(len(content_sections), 1, "Should detect at least one content section")

    def test_chunk_importance(self):
        """Test the calculate_chunk_importance function."""
        simple_text = "Hello world, this is a test."
        rich_text = """
        According to research by Smith et al. (2023), the annual revenue increased by 15.7% 
        in Q3, reaching $127.5 million compared to the same period last year. 
        Additionally, customer acquisition costs decreased by 8.3%, while the retention rate 
        improved from 76% to 81.5% across all market segments.
        """

        simple_score = calculate_chunk_importance(simple_text)
        rich_score = calculate_chunk_importance(rich_text)

        # Check that scores are in expected range
        self.assertTrue(0 <= simple_score <= 1)
        self.assertTrue(0 <= rich_score <= 1)

        # Rich text should have higher importance score due to numbers and information density
        self.assertGreater(rich_score, simple_score)

    def test_semantic_chunking(self):
        """Test the semantic_chunking function."""
        test_text = """# Introduction

This document provides an overview of the project. 
It contains multiple sections with detailed information.

# Section 1

This is the content of section 1.
It has multiple paragraphs.

This is the second paragraph of section 1.

# Section 2

This section covers additional information.
It's important to understand the context.

# Conclusion

In summary, this document demonstrates the chunking functionality.
"""
        chunks = semantic_chunk_text(test_text, chunk_size=200, chunk_overlap=50, min_chunk_size=10)

        # Verify we have chunks
        self.assertGreater(len(chunks), 0)

        # Each chunk should have expected keys
        expected_keys = {"id", "text", "start", "end", "importance"}
        for chunk in chunks:
            for key in expected_keys:
                self.assertIn(key, chunk)

        # Check that headings and corresponding content are kept together
        # At least one chunk should contain both "Section" and content
        found_section_with_content = False
        for chunk in chunks:
            if "Section" in chunk["text"] and "content" in chunk["text"].lower():
                found_section_with_content = True
                break

        self.assertTrue(found_section_with_content)

    def test_enhance_query(self):
        """Test the enhance_query function."""
        queries = [
            "What is the capital of France?",
            "Tell me about machine learning",
            "How to bake a cake",
            "capital of France"
        ]

        enhanced_queries = [enhance_query(q) for q in queries]

        # Check that common question prefixes are removed
        self.assertEqual(enhanced_queries[0], "the capital of france?")
        self.assertEqual(enhanced_queries[1], "machine learning")
        self.assertEqual(enhanced_queries[2], "bake a cake")
        self.assertEqual(enhanced_queries[3], "capital of france")

    def test_sanitize_collection_name(self):
        """Test the sanitize_collection_name function."""
        test_cases = [
            ("My Collection!", "My_Collection_"),
            ("a" * 100, "a" * 63),
            ("123", "123"),
            ("-abc-", "c_abc_0"),
        ]

        for input_name, expected_output in test_cases:
            sanitized = sanitize_collection_name(input_name)
            # Check length constraints
            self.assertTrue(3 <= len(sanitized) <= 63)
            # Check first and last chars are alphanumeric
            self.assertTrue(sanitized[0].isalnum())
            self.assertTrue(sanitized[-1].isalnum())
            # Only contains allowed characters
            self.assertRegex(sanitized, r'^[a-zA-Z0-9_-]+$')

if __name__ == "__main__":
    unittest.main()
