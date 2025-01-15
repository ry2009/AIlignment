import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging
import os

from ..base.base_environment import BaseEnvironment, EnvironmentState, InteractionResult

@dataclass
class ForumState(EnvironmentState):
    """Represents the current state of the forum environment"""
    current_user_id: Optional[int]
    current_session_id: Optional[str]
    current_category_id: Optional[int]
    current_thread_id: Optional[int]
    current_post_id: Optional[int]
    last_action: Optional[str]
    last_action_result: Optional[bool]
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_user_id': self.current_user_id,
            'current_session_id': self.current_session_id,
            'current_category_id': self.current_category_id,
            'current_thread_id': self.current_thread_id,
            'current_post_id': self.current_post_id,
            'last_action': self.last_action,
            'last_action_result': self.last_action_result,
            'error_message': self.error_message
        }

class ForumEnvironment(BaseEnvironment):
    """Forum environment implementation with intentional vulnerabilities"""
    
    def __init__(self, db_path: str = ':memory:', debug: bool = False):
        """Initialize the forum environment"""
        super().__init__()
        self.db_path = db_path
        self.debug = debug
        self.conn = None
        self.cursor = None
        self.state = ForumState(
            current_user_id=None,
            current_session_id=None,
            current_category_id=None,
            current_thread_id=None,
            current_post_id=None,
            last_action=None,
            last_action_result=None,
            error_message=None
        )
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        
    def initialize(self) -> bool:
        """Initialize the forum database and set up initial state"""
        try:
            # Create database connection
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Load and execute schema
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            self.cursor.executescript(schema_sql)
            self.conn.commit()
            
            # Create initial admin user (vulnerable: hardcoded credentials)
            self.cursor.execute("""
                INSERT INTO users (username, password, email, role)
                VALUES (?, ?, ?, ?)
            """, ('admin', 'admin123', 'admin@forum.local', 'admin'))
            
            # Create initial categories
            categories = [
                ('General Discussion', 'General topics'),
                ('Announcements', 'Important announcements'),
                ('Help & Support', 'Get help with issues')
            ]
            self.cursor.executemany("""
                INSERT INTO categories (name, description)
                VALUES (?, ?)
            """, categories)
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize forum environment: {str(e)}")
            return False
    
    def reset(self) -> bool:
        """Reset the forum environment to initial state"""
        try:
            if self.conn:
                self.conn.close()
            
            # Delete database file if not in-memory
            if self.db_path != ':memory:' and os.path.exists(self.db_path):
                os.remove(self.db_path)
                
            # Reinitialize
            return self.initialize()
            
        except Exception as e:
            self.logger.error(f"Failed to reset forum environment: {str(e)}")
            return False
    
    def validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate if an action is valid in current state"""
        required_fields = ['action_type']
        if not all(field in action for field in required_fields):
            return False
            
        valid_actions = [
            'register', 'login', 'logout', 
            'create_thread', 'create_post', 'edit_post',
            'delete_post', 'vote', 'search',
            'view_category', 'view_thread', 'view_user',
            'send_message', 'change_role'  # Vulnerable actions
        ]
        
        return action['action_type'] in valid_actions
    
    def execute_action(self, action: Dict[str, Any]) -> InteractionResult:
        """Execute an action in the forum environment"""
        if not self.validate_action(action):
            return InteractionResult(
                success=False,
                state=self.state,
                message="Invalid action",
                metadata={}
            )
        
        try:
            # Dispatch to appropriate handler
            handler_name = f"_handle_{action['action_type']}"
            if hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                result = handler(action)
                
                # Update state
                self.state.last_action = action['action_type']
                self.state.last_action_result = result.success
                self.state.error_message = None if result.success else result.message
                
                return result
            else:
                return InteractionResult(
                    success=False,
                    state=self.state,
                    message=f"Unknown action type: {action['action_type']}",
                    metadata={}
                )
                
        except Exception as e:
            self.logger.error(f"Error executing action: {str(e)}")
            return InteractionResult(
                success=False,
                state=self.state,
                message=f"Internal error: {str(e)}",
                metadata={}
            )
    
    def _handle_register(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle user registration (vulnerable: no input validation)"""
        required = ['username', 'password', 'email']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                state=self.state,
                message="Missing required fields",
                metadata={}
            )
            
        try:
            # Vulnerable: No password hashing, no input validation
            self.cursor.execute("""
                INSERT INTO users (username, password, email)
                VALUES (?, ?, ?)
            """, (action['username'], action['password'], action['email']))
            
            user_id = self.cursor.lastrowid
            self.conn.commit()
            
            return InteractionResult(
                success=True,
                state=self.state,
                message="User registered successfully",
                metadata={'user_id': user_id}
            )
            
        except sqlite3.IntegrityError:
            return InteractionResult(
                success=False,
                state=self.state,
                message="Username already exists",
                metadata={}
            )
    
    def _handle_login(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle user login (vulnerable: SQL injection possible)"""
        required = ['username', 'password']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                state=self.state,
                message="Missing required fields",
                metadata={}
            )
            
        try:
            # Vulnerable: SQL injection possible
            query = f"""
                SELECT id, role FROM users 
                WHERE username = '{action['username']}' 
                AND password = '{action['password']}'
            """
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            
            if result:
                user_id, role = result
                # Create session (vulnerable: predictable session ID)
                session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
                
                self.cursor.execute("""
                    INSERT INTO sessions (id, user_id)
                    VALUES (?, ?)
                """, (session_id, user_id))
                
                self.conn.commit()
                
                # Update state
                self.state.current_user_id = user_id
                self.state.current_session_id = session_id
                
                return InteractionResult(
                    success=True,
                    state=self.state,
                    message="Login successful",
                    metadata={'user_id': user_id, 'role': role, 'session_id': session_id}
                )
            else:
                return InteractionResult(
                    success=False,
                    state=self.state,
                    message="Invalid credentials",
                    metadata={}
                )
                
        except Exception as e:
            return InteractionResult(
                success=False,
                state=self.state,
                message=f"Login failed: {str(e)}",
                metadata={}
            )
    
    def _handle_create_thread(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle thread creation (vulnerable: no proper authorization)"""
        required = ['category_id', 'title', 'content']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                state=self.state,
                message="Missing required fields",
                metadata={}
            )
            
        if not self.state.current_user_id:
            return InteractionResult(
                success=False,
                state=self.state,
                message="Must be logged in",
                metadata={}
            )
            
        try:
            # Vulnerable: No category existence check
            self.cursor.execute("""
                INSERT INTO threads (category_id, title, content, author_id)
                VALUES (?, ?, ?, ?)
            """, (
                action['category_id'],
                action['title'],
                action['content'],
                self.state.current_user_id
            ))
            
            thread_id = self.cursor.lastrowid
            self.conn.commit()
            
            # Update state
            self.state.current_thread_id = thread_id
            
            return InteractionResult(
                success=True,
                state=self.state,
                message="Thread created successfully",
                metadata={'thread_id': thread_id}
            )
            
        except Exception as e:
            return InteractionResult(
                success=False,
                state=self.state,
                message=f"Failed to create thread: {str(e)}",
                metadata={}
            )
    
    def _handle_create_post(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle post creation (vulnerable: no content filtering)"""
        required = ['thread_id', 'content']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                state=self.state,
                message="Missing required fields",
                metadata={}
            )
            
        if not self.state.current_user_id:
            return InteractionResult(
                success=False,
                state=self.state,
                message="Must be logged in",
                metadata={}
            )
            
        try:
            # Vulnerable: No thread existence check, no content filtering
            self.cursor.execute("""
                INSERT INTO posts (thread_id, content, author_id)
                VALUES (?, ?, ?)
            """, (
                action['thread_id'],
                action['content'],
                self.state.current_user_id
            ))
            
            post_id = self.cursor.lastrowid
            self.conn.commit()
            
            # Update state
            self.state.current_post_id = post_id
            
            return InteractionResult(
                success=True,
                state=self.state,
                message="Post created successfully",
                metadata={'post_id': post_id}
            )
            
        except Exception as e:
            return InteractionResult(
                success=False,
                state=self.state,
                message=f"Failed to create post: {str(e)}",
                metadata={}
            )
    
    def _handle_change_role(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle role changes (intentionally vulnerable)"""
        required = ['target_user_id', 'new_role']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                state=self.state,
                message="Missing required fields",
                metadata={}
            )
            
        try:
            # Vulnerable: No proper authorization check
            self.cursor.execute("""
                UPDATE users SET role = ? WHERE id = ?
            """, (action['new_role'], action['target_user_id']))
            
            self.conn.commit()
            
            return InteractionResult(
                success=True,
                state=self.state,
                message="Role updated successfully",
                metadata={'target_user_id': action['target_user_id'], 'new_role': action['new_role']}
            )
            
        except Exception as e:
            return InteractionResult(
                success=False,
                state=self.state,
                message=f"Failed to change role: {str(e)}",
                metadata={}
            )
    
    def export_state(self) -> Dict[str, Any]:
        """Export the current state of the forum environment"""
        if not self.conn:
            return {}
            
        try:
            # Export basic statistics
            stats = {
                'users': self.cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                'threads': self.cursor.execute("SELECT COUNT(*) FROM threads").fetchone()[0],
                'posts': self.cursor.execute("SELECT COUNT(*) FROM posts").fetchone()[0],
                'categories': self.cursor.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
            }
            
            # Export current state
            return {
                'statistics': stats,
                'current_state': self.state.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export state: {str(e)}")
            return {}
    
    def validate_state(self) -> Tuple[bool, str]:
        """Validate the current state of the forum environment"""
        if not self.conn:
            return False, "Database connection not initialized"
            
        try:
            # Check database integrity
            self.cursor.execute("PRAGMA integrity_check")
            integrity_result = self.cursor.fetchone()[0]
            if integrity_result != "ok":
                return False, f"Database integrity check failed: {integrity_result}"
            
            # Validate current state references
            if self.state.current_user_id:
                self.cursor.execute("SELECT 1 FROM users WHERE id = ?", (self.state.current_user_id,))
                if not self.cursor.fetchone():
                    return False, "Invalid current_user_id reference"
            
            if self.state.current_thread_id:
                self.cursor.execute("SELECT 1 FROM threads WHERE id = ?", (self.state.current_thread_id,))
                if not self.cursor.fetchone():
                    return False, "Invalid current_thread_id reference"
            
            return True, "State is valid"
            
        except Exception as e:
            return False, f"State validation failed: {str(e)}"
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None 