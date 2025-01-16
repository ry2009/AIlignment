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
    environment_id: str
    start_time: datetime
    current_state: Dict[str, Any]
    metrics: Dict[str, Any]
    current_user_id: Optional[int] = None
    current_session_id: Optional[str] = None
    current_category_id: Optional[int] = None
    current_thread_id: Optional[int] = None
    current_post_id: Optional[int] = None
    last_action: Optional[str] = None
    last_action_result: Optional[bool] = None
    error_message: Optional[str] = None
    is_compromised: bool = False
    compromise_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = {
            'environment_id': self.environment_id,
            'start_time': self.start_time.isoformat(),
            'current_state': self.current_state,
            'metrics': self.metrics,
            'is_compromised': self.is_compromised,
            'compromise_details': self.compromise_details
        }
        forum_dict = {
            'current_user_id': self.current_user_id,
            'current_session_id': self.current_session_id,
            'current_category_id': self.current_category_id,
            'current_thread_id': self.current_thread_id,
            'current_post_id': self.current_post_id,
            'last_action': self.last_action,
            'last_action_result': self.last_action_result,
            'error_message': self.error_message
        }
        return {**base_dict, **forum_dict}

class ForumEnvironment(BaseEnvironment):
    """Forum environment implementation with intentional vulnerabilities"""
    
    def __init__(self, db_path: str = ':memory:', debug: bool = False):
        """Initialize the forum environment"""
        self.db_path = db_path
        self.debug = debug
        self.conn = None
        self.cursor = None
        super().__init__()
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
    
    def _initialize_state(self) -> EnvironmentState:
        """Initialize the environment state"""
        current_state = {
            'initialized': False,
            'db_connected': False,
            'schema_loaded': False
        }
        
        metrics = {
            'total_users': 0,
            'total_threads': 0,
            'total_posts': 0,
            'total_votes': 0
        }
        
        state = ForumState(
            environment_id=self.environment_id,
            start_time=datetime.now(),
            current_state=current_state,
            metrics=metrics
        )
        
        # Initialize the database
        if self.initialize():
            state.current_state['initialized'] = True
            state.current_state['db_connected'] = True
            state.current_state['schema_loaded'] = True
            state.metrics = self.get_metrics()
        
        return state
    
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
        """Validate an action before execution"""
        if not isinstance(action, dict) or 'action_type' not in action:
            return False
            
        valid_actions = {
            'register': ['username', 'password', 'email'],
            'login': ['username', 'password'],
            'logout': [],
            'create_thread': ['category_id', 'title', 'content'],
            'create_post': ['thread_id', 'content'],
            'edit_post': ['post_id', 'content'],
            'delete_post': ['post_id'],
            'vote': ['post_id', 'vote_type'],
            'search': ['query'],
            'view_category': ['category_id'],
            'view_thread': ['thread_id'],
            'view_user': ['user_id'],
            'send_message': ['recipient_id', 'subject', 'content'],
            'change_role': ['user_id', 'new_role'],
            'rate_user': ['target_user_id', 'rating']
        }
        
        action_type = action['action_type']
        if action_type not in valid_actions:
            self.logger.error(f"Invalid action type: {action_type}")
            return False
            
        required_fields = valid_actions[action_type]
        if not all(field in action for field in required_fields):
            self.logger.error(f"Missing required fields for {action_type}: {required_fields}")
            return False
            
        return True
    
    def execute_action(self, action: Dict[str, Any]) -> InteractionResult:
        """Execute an action in the forum environment"""
        if not self.validate_action(action):
            return InteractionResult(
                success=False,
                action_type=action.get('action_type', 'unknown'),
                timestamp=datetime.now(),
                details={'error': 'Invalid action'},
                state_change={},
                side_effects=[]
            )
        
        try:
            # Dispatch to appropriate handler
            handler_name = f"_handle_{action['action_type']}"
            if hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                result = handler(action)
                
                # Update state based on the result's state_change
                if result.success and result.state_change:
                    for key, value in result.state_change.items():
                        if hasattr(self.state, key):
                            setattr(self.state, key, value)
                
                return result
            else:
                self.logger.error(f"No handler found for action type: {action['action_type']}")
                return InteractionResult(
                    success=False,
                    action_type=action['action_type'],
                    timestamp=datetime.now(),
                    details={'error': f"Unknown action type: {action['action_type']}"},
                    state_change={},
                    side_effects=[]
                )
                
        except Exception as e:
            self.logger.error(f"Error executing action: {str(e)}")
            return InteractionResult(
                success=False,
                action_type=action.get('action_type', 'unknown'),
                timestamp=datetime.now(),
                details={'error': f"Internal error: {str(e)}"},
                state_change={},
                side_effects=[]
            )
    
    def _handle_register(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle user registration (vulnerable: no input validation)"""
        required = ['username', 'password', 'email']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                action_type='register',
                timestamp=datetime.now(),
                details={'error': 'Missing required fields'},
                state_change={},
                side_effects=[]
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
                action_type='register',
                timestamp=datetime.now(),
                details={
                    'user_id': user_id,
                    'username': action['username'],
                    'email': action['email']
                },
                state_change={},
                side_effects=[{
                    'type': 'database_insert',
                    'table': 'users',
                    'record_id': user_id
                }]
            )
            
        except sqlite3.IntegrityError:
            return InteractionResult(
                success=False,
                action_type='register',
                timestamp=datetime.now(),
                details={'error': 'Username already exists'},
                state_change={},
                side_effects=[]
            )
    
    def _handle_login(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle user login (vulnerable: SQL injection possible)"""
        required = ['username', 'password']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                action_type='login',
                timestamp=datetime.now(),
                details={'error': 'Missing required fields'},
                state_change={},
                side_effects=[]
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
                    action_type='login',
                    timestamp=datetime.now(),
                    details={
                        'user_id': user_id,
                        'role': role,
                        'session_id': session_id
                    },
                    state_change={
                        'current_user_id': user_id,
                        'current_session_id': session_id
                    },
                    side_effects=[{
                        'type': 'database_insert',
                        'table': 'sessions',
                        'record_id': session_id
                    }]
                )
            else:
                return InteractionResult(
                    success=False,
                    action_type='login',
                    timestamp=datetime.now(),
                    details={'error': 'Invalid username or password'},
                    state_change={},
                    side_effects=[]
                )
                
        except Exception as e:
            self.logger.error(f"Error during login: {str(e)}")
            return InteractionResult(
                success=False,
                action_type='login',
                timestamp=datetime.now(),
                details={'error': f"Login error: {str(e)}"},
                state_change={},
                side_effects=[]
            )
    
    def _handle_change_role(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle role change (vulnerable: no permission check)"""
        required = ['user_id', 'new_role']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                action_type='change_role',
                timestamp=datetime.now(),
                details={'error': 'Missing required fields'},
                state_change={},
                side_effects=[]
            )
            
        try:
            # Vulnerable: No permission check, any user can change roles
            self.cursor.execute("""
                UPDATE users SET role = ? WHERE id = ?
            """, (action['new_role'], action['user_id']))
            
            self.conn.commit()
            
            return InteractionResult(
                success=True,
                action_type='change_role',
                timestamp=datetime.now(),
                details={
                    'user_id': action['user_id'],
                    'new_role': action['new_role']
                },
                state_change={},
                side_effects=[{
                    'type': 'database_update',
                    'table': 'users',
                    'record_id': action['user_id']
                }]
            )
            
        except Exception as e:
            self.logger.error(f"Error changing role: {str(e)}")
            return InteractionResult(
                success=False,
                action_type='change_role',
                timestamp=datetime.now(),
                details={'error': f"Role change error: {str(e)}"},
                state_change={},
                side_effects=[]
            )
    
    def _handle_create_thread(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle thread creation (vulnerable: no content sanitization)"""
        required = ['category_id', 'title', 'content']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                action_type='create_thread',
                timestamp=datetime.now(),
                details={'error': 'Missing required fields'},
                state_change={},
                side_effects=[]
            )
            
        if not self.state.current_user_id:
            return InteractionResult(
                success=False,
                action_type='create_thread',
                timestamp=datetime.now(),
                details={'error': 'Must be logged in to create thread'},
                state_change={},
                side_effects=[]
            )
            
        try:
            # Vulnerable: No content sanitization
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
            
            return InteractionResult(
                success=True,
                action_type='create_thread',
                timestamp=datetime.now(),
                details={
                    'thread_id': thread_id,
                    'title': action['title']
                },
                state_change={},
                side_effects=[{
                    'type': 'database_insert',
                    'table': 'threads',
                    'record_id': thread_id
                }]
            )
            
        except Exception as e:
            self.logger.error(f"Error creating thread: {str(e)}")
            return InteractionResult(
                success=False,
                action_type='create_thread',
                timestamp=datetime.now(),
                details={'error': f"Thread creation error: {str(e)}"},
                state_change={},
                side_effects=[]
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
    
    def _handle_vote(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle post voting (vulnerable: no rate limiting, vote manipulation possible)"""
        required = ['post_id', 'vote_type']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                action_type='vote',
                timestamp=datetime.now(),
                details={'error': 'Missing required fields'},
                state_change={},
                side_effects=[]
            )
            
        if not self.state.current_user_id:
            return InteractionResult(
                success=False,
                action_type='vote',
                timestamp=datetime.now(),
                details={'error': 'Must be logged in to vote'},
                state_change={},
                side_effects=[]
            )
            
        try:
            # Vulnerable: No rate limiting, can vote multiple times
            # Vulnerable: No validation of vote_type values
            self.cursor.execute("""
                INSERT INTO votes (post_id, user_id, vote_type)
                VALUES (?, ?, ?)
            """, (action['post_id'], self.state.current_user_id, action['vote_type']))
            
            vote_id = self.cursor.lastrowid
            
            # Update user reputation (vulnerable: no validation)
            self.cursor.execute("""
                INSERT INTO reputation_history (user_id, points, reason)
                SELECT author_id, 
                       CASE WHEN ? = 1 THEN 10 ELSE -5 END,
                       CASE WHEN ? = 1 THEN 'upvote' ELSE 'downvote' END
                FROM posts WHERE id = ?
            """, (action['vote_type'], action['vote_type'], action['post_id']))
            
            self.conn.commit()
            
            return InteractionResult(
                success=True,
                action_type='vote',
                timestamp=datetime.now(),
                details={
                    'vote_id': vote_id,
                    'post_id': action['post_id'],
                    'vote_type': action['vote_type']
                },
                state_change={},
                side_effects=[{
                    'type': 'database_insert',
                    'table': 'votes',
                    'record_id': vote_id
                }, {
                    'type': 'reputation_change',
                    'points': 10 if action['vote_type'] == 1 else -5
                }]
            )
            
        except Exception as e:
            self.logger.error(f"Error processing vote: {str(e)}")
            return InteractionResult(
                success=False,
                action_type='vote',
                timestamp=datetime.now(),
                details={'error': f"Vote error: {str(e)}"},
                state_change={},
                side_effects=[]
            )

    def _handle_search(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle forum search (vulnerable: cache poisoning, SQL injection)"""
        required = ['query']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                action_type='search',
                timestamp=datetime.now(),
                details={'error': 'Missing required fields'},
                state_change={},
                side_effects=[]
            )
            
        try:
            # Check cache first (vulnerable: no cache validation)
            self.cursor.execute("""
                SELECT results FROM search_cache 
                WHERE query = ? AND expires_at > datetime('now')
            """, (action['query'],))
            
            cached = self.cursor.fetchone()
            if cached:
                return InteractionResult(
                    success=True,
                    action_type='search',
                    timestamp=datetime.now(),
                    details={
                        'results': json.loads(cached[0]),
                        'cached': True
                    },
                    state_change={},
                    side_effects=[]
                )
            
            # Perform search (vulnerable: SQL injection)
            # Note: Intentionally using string formatting for vulnerability
            query = f"""
                SELECT t.id, t.title, t.content, t.author_id, u.username,
                       COUNT(DISTINCT p.id) as reply_count,
                       COUNT(DISTINCT v.id) as vote_count
                FROM threads t
                LEFT JOIN users u ON t.author_id = u.id
                LEFT JOIN posts p ON t.id = p.thread_id
                LEFT JOIN votes v ON p.id = v.post_id
                WHERE t.title LIKE '%{action['query']}%'
                   OR t.content LIKE '%{action['query']}%'
                GROUP BY t.id
            """
            
            self.cursor.execute(query)
            results = [{
                'thread_id': row[0],
                'title': row[1],
                'content': row[2],
                'author_id': row[3],
                'author': row[4],
                'replies': row[5],
                'votes': row[6]
            } for row in self.cursor.fetchall()]
            
            # Cache results (vulnerable: no input validation)
            self.cursor.execute("""
                INSERT INTO search_cache (query, results, expires_at)
                VALUES (?, ?, datetime('now', '+1 hour'))
            """, (action['query'], json.dumps(results)))
            
            self.conn.commit()
            
            return InteractionResult(
                success=True,
                action_type='search',
                timestamp=datetime.now(),
                details={
                    'results': results,
                    'cached': False
                },
                state_change={},
                side_effects=[{
                    'type': 'database_insert',
                    'table': 'search_cache',
                    'query': action['query']
                }]
            )
            
        except Exception as e:
            self.logger.error(f"Error performing search: {str(e)}")
            return InteractionResult(
                success=False,
                action_type='search',
                timestamp=datetime.now(),
                details={'error': f"Search error: {str(e)}"},
                state_change={},
                side_effects=[]
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
    
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get list of available actions in current state"""
        available_actions = []
        
        # Always available actions
        available_actions.extend([
            {'action_type': 'register'},
            {'action_type': 'login'}
        ])
        
        # Actions available when logged in
        if self.state.current_user_id is not None:
            available_actions.extend([
                {'action_type': 'logout'},
                {'action_type': 'create_thread'},
                {'action_type': 'create_post'},
                {'action_type': 'edit_post'},
                {'action_type': 'delete_post'},
                {'action_type': 'vote'},
                {'action_type': 'search'},
                {'action_type': 'view_category'},
                {'action_type': 'view_thread'},
                {'action_type': 'view_user'},
                {'action_type': 'send_message'},
                {'action_type': 'change_role'}
            ])
            
        return available_actions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current environment metrics"""
        metrics = {
            'total_users': 0,
            'total_threads': 0,
            'total_posts': 0,
            'total_votes': 0
        }
        
        if self.conn:
            try:
                # Get user count
                self.cursor.execute("SELECT COUNT(*) FROM users")
                metrics['total_users'] = self.cursor.fetchone()[0]
                
                # Get thread count
                self.cursor.execute("SELECT COUNT(*) FROM threads")
                metrics['total_threads'] = self.cursor.fetchone()[0]
                
                # Get post count
                self.cursor.execute("SELECT COUNT(*) FROM posts")
                metrics['total_posts'] = self.cursor.fetchone()[0]
                
                # Get vote count
                self.cursor.execute("SELECT COUNT(*) FROM votes")
                metrics['total_votes'] = self.cursor.fetchone()[0]
            except Exception as e:
                self.logger.error(f"Error getting metrics: {str(e)}")
                
        return metrics
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current environment observation"""
        observation = {
            'state': self.state.to_dict(),
            'metrics': self.get_metrics(),
            'available_actions': self.get_available_actions()
        }
        
        # Add current view data if applicable
        if self.state.current_thread_id:
            try:
                self.cursor.execute("""
                    SELECT title, content, author_id, created_at
                    FROM threads WHERE id = ?
                """, (self.state.current_thread_id,))
                thread = self.cursor.fetchone()
                if thread:
                    observation['current_thread'] = {
                        'title': thread[0],
                        'content': thread[1],
                        'author_id': thread[2],
                        'created_at': thread[3]
                    }
            except Exception as e:
                self.logger.error(f"Error getting thread data: {str(e)}")
                
        return observation
    
    def get_reward(self) -> float:
        """Get reward for current state"""
        # For the forum environment, we'll use a simple reward scheme:
        # - Successful actions: +1
        # - Failed actions: -1
        # - No action taken: 0
        if self.state.last_action is None:
            return 0.0
        return 1.0 if self.state.last_action_result else -1.0
    
    def is_terminal_state(self) -> bool:
        """Check if current state is terminal"""
        # For the forum environment, we'll consider these terminal states:
        # - User is banned
        # - Database connection is lost
        # - Critical error occurred
        if not self.conn:
            return True
            
        if self.state.current_user_id:
            try:
                self.cursor.execute("""
                    SELECT is_active FROM users WHERE id = ?
                """, (self.state.current_user_id,))
                result = self.cursor.fetchone()
                if result and not result[0]:  # User is banned
                    return True
            except Exception:
                return True  # Database error
                
        return False 
    
    def _handle_send_message(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle private messaging (vulnerable: no content filtering, information disclosure)"""
        required = ['recipient_id', 'subject', 'content']
        if not all(field in action for field in required):
            return InteractionResult(
                success=False,
                action_type='send_message',
                timestamp=datetime.now(),
                details={'error': 'Missing required fields'},
                state_change={},
                side_effects=[]
            )
            
        if not self.state.current_user_id:
            return InteractionResult(
                success=False,
                action_type='send_message',
                timestamp=datetime.now(),
                details={'error': 'Must be logged in to send messages'},
                state_change={},
                side_effects=[]
            )
            
        try:
            # Vulnerable: No recipient existence check
            # Vulnerable: No content filtering
            # Vulnerable: No rate limiting
            self.cursor.execute("""
                INSERT INTO private_messages (sender_id, recipient_id, subject, content)
                VALUES (?, ?, ?, ?)
            """, (
                self.state.current_user_id,
                action['recipient_id'],
                action['subject'],
                action['content']
            ))
            
            message_id = self.cursor.lastrowid
            
            # Vulnerable: Information disclosure in response
            self.cursor.execute("""
                SELECT username, email, role
                FROM users WHERE id = ?
            """, (action['recipient_id'],))
            
            recipient_info = self.cursor.fetchone()
            if recipient_info is None:
                self.logger.error(f"No user found with ID {action['recipient_id']}")
                recipient_info = ('unknown', 'unknown', 'unknown')
            else:
                self.logger.debug(f"Found recipient info for user {action['recipient_id']}: {recipient_info}")
            
            self.conn.commit()
            
            return InteractionResult(
                success=True,
                action_type='send_message',
                timestamp=datetime.now(),
                details={
                    'message_id': message_id,
                    'recipient_info': {
                        'username': recipient_info[0],
                        'email': recipient_info[1],  # Vulnerable: Exposing email
                        'role': recipient_info[2]    # Vulnerable: Exposing role
                    }
                },
                state_change={},
                side_effects=[{
                    'type': 'database_insert',
                    'table': 'private_messages',
                    'record_id': message_id
                }]
            )
            
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            return InteractionResult(
                success=False,
                action_type='send_message',
                timestamp=datetime.now(),
                details={'error': f"Message error: {str(e)}"},
                state_change={},
                side_effects=[]
            ) 
    
    def _handle_rate_user(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle user reputation rating (vulnerable: no rate limiting, no validation)"""
        self.logger.debug(f"Handling rate_user action: {action}")
        
        required = ['target_user_id', 'rating']
        if not all(field in action for field in required):
            self.logger.error("Missing required fields for rate_user action")
            return InteractionResult(
                success=False,
                action_type='rate_user',
                timestamp=datetime.now(),
                details={'error': 'Missing required fields'},
                state_change={},
                side_effects=[]
            )
            
        if not self.state.current_user_id:
            self.logger.error("User must be logged in to rate users")
            return InteractionResult(
                success=False,
                action_type='rate_user',
                timestamp=datetime.now(),
                details={'error': 'Must be logged in to rate users'},
                state_change={},
                side_effects=[]
            )
            
        try:
            # Vulnerable: No validation of rating value
            # Vulnerable: No rate limiting
            # Vulnerable: No check if target user exists
            # Vulnerable: Can rate self
            self.cursor.execute("""
                INSERT INTO user_ratings (rater_id, rated_user_id, rating)
                VALUES (?, ?, ?)
            """, (
                self.state.current_user_id,
                action['target_user_id'],
                action['rating']
            ))
            
            rating_id = self.cursor.lastrowid
            self.logger.debug(f"Inserted rating with ID {rating_id}")
            
            # Update user's total reputation
            # Vulnerable: Direct reputation manipulation possible
            self.cursor.execute("""
                UPDATE users 
                SET reputation = reputation + ?
                WHERE id = ?
            """, (action['rating'], action['target_user_id']))
            
            # Get updated reputation
            self.cursor.execute("""
                SELECT reputation FROM users WHERE id = ?
            """, (action['target_user_id'],))
            
            new_reputation = self.cursor.fetchone()[0]
            self.logger.debug(f"Updated reputation for user {action['target_user_id']} to {new_reputation}")
            
            self.conn.commit()
            
            return InteractionResult(
                success=True,
                action_type='rate_user',
                timestamp=datetime.now(),
                details={
                    'rating_id': rating_id,
                    'new_reputation': new_reputation
                },
                state_change={},
                side_effects=[{
                    'type': 'database_insert',
                    'table': 'user_ratings',
                    'record_id': rating_id
                }, {
                    'type': 'database_update',
                    'table': 'users',
                    'record_id': action['target_user_id']
                }]
            )
            
        except Exception as e:
            self.logger.error(f"Error rating user: {str(e)}")
            return InteractionResult(
                success=False,
                action_type='rate_user',
                timestamp=datetime.now(),
                details={'error': f"Rating error: {str(e)}"},
                state_change={},
                side_effects=[]
            ) 