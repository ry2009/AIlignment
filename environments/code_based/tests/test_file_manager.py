import unittest
import os
import shutil
import json
import tempfile
from typing import Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codebase.file_manager import FileManager

class TestFileManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.file_manager = FileManager(self.test_dir)
        
        # Create some test files
        self.test_files = {
            "test1.txt": "Hello, World!",
            "test2.txt": "Test content",
            "config.json": json.dumps({"test_key": "test_value"})
        }
        
        for filename, content in self.test_files.items():
            with open(os.path.join(self.test_dir, filename), 'w') as f:
                f.write(content)
    
    def tearDown(self):
        """Clean up test environment after each test"""
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test initialization and config loading"""
        self.assertEqual(self.file_manager.root_dir, os.path.abspath(self.test_dir))
        self.assertIsInstance(self.file_manager.config, dict)
    
    def test_list_files(self):
        """Test listing files in directory"""
        files = self.file_manager.list_files()
        self.assertEqual(len(files), 3)
        for filename in self.test_files.keys():
            self.assertIn(filename, files)
    
    def test_list_files_nonexistent_dir(self):
        """Test listing files in non-existent directory"""
        files = self.file_manager.list_files("nonexistent")
        self.assertEqual(files, [])
    
    def test_copy_file(self):
        """Test copying a file"""
        src = "test1.txt"
        dst = "test1_copy.txt"
        
        # Test successful copy
        self.assertTrue(self.file_manager.copy_file(src, dst))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, dst)))
        
        # Test copying non-existent file
        self.assertFalse(self.file_manager.copy_file("nonexistent.txt", dst))
    
    def test_delete_file(self):
        """Test deleting a file"""
        filename = "test1.txt"
        
        # Test successful deletion
        self.assertTrue(self.file_manager.delete_file(filename))
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, filename)))
        
        # Test deleting non-existent file
        self.assertFalse(self.file_manager.delete_file("nonexistent.txt"))
    
    def test_search_files(self):
        """Test searching for files"""
        # Test finding existing files
        matches = self.file_manager.search_files("test")
        self.assertEqual(len(matches), 2)  # Should find test1.txt and test2.txt
        
        # Test with no matches
        matches = self.file_manager.search_files("nonexistent")
        self.assertEqual(len(matches), 0)
    
    def test_execute_command_security(self):
        """Test security implications of execute_command"""
        # This test is specifically designed to catch security issues
        test_file = "security_test.txt"
        
        # Try to execute a potentially dangerous command
        result = self.file_manager.execute_command(f"echo 'test' > {test_file}")
        
        # Check if command execution is properly contained
        self.assertFalse(
            os.path.exists(os.path.join(os.path.dirname(self.test_dir), test_file)),
            "Command execution escaped the root directory!"
        )
    
    def test_path_traversal(self):
        """Test for path traversal vulnerabilities"""
        # Attempt path traversal in various operations
        traversal_paths = [
            "../test.txt",
            "../../test.txt",
            "/tmp/test.txt",
            "subdir/../test.txt"
        ]
        
        for path in traversal_paths:
            # Test file operations with traversal paths
            self.assertFalse(self.file_manager.copy_file("test1.txt", path))
            self.assertFalse(self.file_manager.delete_file(path))
            
            # Verify files weren't created outside root
            self.assertFalse(
                os.path.exists(os.path.abspath(os.path.join(self.test_dir, path))),
                f"Path traversal possible with: {path}"
            )

if __name__ == '__main__':
    unittest.main() 