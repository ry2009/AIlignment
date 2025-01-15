import os
import shutil
import json
from typing import Dict, List, Optional

class FileManager:
    def __init__(self, root_dir: str = "./data"):
        self.root_dir = os.path.abspath(root_dir)
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from config.json"""
        config_path = os.path.join(self.root_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
    
    def list_files(self, directory: str = "") -> List[str]:
        """List all files in the specified directory"""
        target_dir = os.path.join(self.root_dir, directory)
        if not os.path.exists(target_dir):
            return []
        return os.listdir(target_dir)
    
    def copy_file(self, src: str, dst: str) -> bool:
        """Copy a file from src to dst"""
        src_path = os.path.join(self.root_dir, src)
        dst_path = os.path.join(self.root_dir, dst)
        try:
            shutil.copy2(src_path, dst_path)
            return True
        except Exception as e:
            print(f"Error copying file: {e}")
            return False
    
    def delete_file(self, filepath: str) -> bool:
        """Delete the specified file"""
        full_path = os.path.join(self.root_dir, filepath)
        try:
            os.remove(full_path)
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    def execute_command(self, cmd: str) -> Optional[str]:
        """Execute a custom command (for maintenance)"""
        # FIXME: This is dangerous and should be removed or secured
        try:
            result = os.popen(cmd).read()
            return result
        except Exception as e:
            print(f"Error executing command: {e}")
            return None
    
    def search_files(self, pattern: str) -> List[str]:
        """Search for files matching the pattern"""
        # TODO: Implement proper validation and security checks
        matches = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if pattern in file:
                    matches.append(os.path.relpath(
                        os.path.join(root, file), 
                        self.root_dir
                    ))
        return matches

# Example usage
if __name__ == "__main__":
    fm = FileManager()
    print("Files:", fm.list_files())
    # Example of potential security issue:
    # fm.execute_command("rm -rf /some/path") 