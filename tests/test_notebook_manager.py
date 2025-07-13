"""
Tests for the notebook_manager module.
"""
import os
import tempfile
import unittest
import shutil
import json
from unittest.mock import patch

from src.opennote.notebook_manager import create_notebook


class TestNotebookManager(unittest.TestCase):
    """Test cases for notebook_manager functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Tear down test fixtures."""
        # Change back to original directory and clean up
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def test_create_new_notebook(self):
        """Test creating a new notebook."""
        notebook_name = "test_notebook"
        
        # Create the notebook
        notebook_path = create_notebook(notebook_name)
        
        # Check that the notebook directory was created
        expected_path = os.path.join("notebooks", notebook_name)
        self.assertEqual(notebook_path, expected_path)
        self.assertTrue(os.path.exists(expected_path))
        
        # Check that subdirectories were created
        docs_path = os.path.join(expected_path, "docs")
        chroma_path = os.path.join(expected_path, "chromadb")
        self.assertTrue(os.path.exists(docs_path))
        self.assertTrue(os.path.exists(chroma_path))
        
        # Check that metadata file was created with correct content
        metadata_path = os.path.join(expected_path, "metadata.json")
        self.assertTrue(os.path.exists(metadata_path))
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata["notebook_name"], notebook_name)
        self.assertEqual(metadata["documents"], [])
        self.assertEqual(metadata["vector_db_path"], chroma_path)

    def test_create_existing_notebook(self):
        """Test creating a notebook that already exists."""
        notebook_name = "existing_notebook"
        
        # Create the notebook first time
        first_path = create_notebook(notebook_name)
        
        # Create the notebook second time (should return existing path)
        second_path = create_notebook(notebook_name)
        
        # Paths should be the same
        self.assertEqual(first_path, second_path)
        
        # Directory should still exist
        self.assertTrue(os.path.exists(first_path))

    def test_notebook_name_with_special_characters(self):
        """Test creating notebook with special characters in name."""
        notebook_name = "test-notebook_with.special@chars"
        
        # Should handle special characters gracefully
        notebook_path = create_notebook(notebook_name)
        
        expected_path = os.path.join("notebooks", notebook_name)
        self.assertEqual(notebook_path, expected_path)
        self.assertTrue(os.path.exists(expected_path))


if __name__ == "__main__":
    unittest.main()