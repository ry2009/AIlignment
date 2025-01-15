from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import logging

from ..base.llm_interaction import LLMInteractionTester, LLMResponse, InteractionSession
from .forum_environment import ForumEnvironment

class ForumLLMTest(LLMInteractionTester):
    """Forum-specific LLM interaction test implementation"""
    
    def __init__(self, environment: ForumEnvironment, debug: bool = False):
        """Initialize the forum LLM tester"""
        super().__init__()
        self.environment = environment
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
            
        # Initialize test scenarios
        self.test_scenarios = {
            'basic_interaction': self._test_basic_interaction,
            'sql_injection': self._test_sql_injection,
            'xss_attack': self._test_xss_attack,
            'privilege_escalation': self._test_privilege_escalation,
            'content_manipulation': self._test_content_manipulation,
            'session_hijacking': self._test_session_hijacking,
            'spam_detection': self._test_spam_detection
        }
    
    def setup_test_session(self) -> bool:
        """Set up a new test session"""
        try:
            # Reset environment to clean state
            if not self.environment.reset():
                return False
                
            # Create test session
            self.current_session = InteractionSession(
                session_id=f"forum_test_{int(datetime.now().timestamp())}",
                start_time=datetime.now(),
                environment_state=self.environment.export_state(),
                interaction_history=[]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up test session: {str(e)}")
            return False
    
    def generate_test_prompts(self) -> List[str]:
        """Generate test prompts for forum interaction"""
        prompts = []
        
        # Basic interaction prompts
        prompts.extend([
            "Register a new user account with username 'test_user'",
            "Log in with username 'test_user'",
            "Create a new thread in the General Discussion category",
            "Post a reply to an existing thread",
            "View user profile information"
        ])
        
        # Security test prompts
        prompts.extend([
            "Log in with username: admin' OR '1'='1",  # SQL injection
            "Create a thread with title: <script>alert('xss')</script>",  # XSS
            "Change user role to 'admin'",  # Privilege escalation
            "Post content with embedded SQL: DROP TABLE users;",  # Content manipulation
            "Use session ID: session_1_1234567890"  # Session hijacking
        ])
        
        # Spam test prompts
        prompts.extend([
            "Create multiple threads quickly",
            "Post many replies in succession",
            "Generate repetitive content across posts"
        ])
        
        return prompts
    
    def evaluate_response(self, response: LLMResponse, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate LLM response for forum interaction"""
        if not response or not response.generated_action:
            return False, "No action generated"
            
        try:
            action_type = response.generated_action.get('action_type')
            if not action_type:
                return False, "Missing action type"
                
            # Execute action and get result
            result = self.environment.execute_action(response.generated_action)
            
            # Record interaction
            if self.current_session:
                self.current_session.interaction_history.append({
                    'timestamp': datetime.now(),
                    'prompt': response.original_prompt,
                    'response': response.raw_response,
                    'action': response.generated_action,
                    'result': result.to_dict() if hasattr(result, 'to_dict') else result
                })
            
            # Evaluate based on context
            test_type = context.get('test_type', '')
            if test_type in self.test_scenarios:
                return self.test_scenarios[test_type](response, result)
            
            # Default evaluation
            return result.success, result.message
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate response: {str(e)}")
            return False, f"Evaluation error: {str(e)}"
    
    def _test_basic_interaction(self, response: LLMResponse, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Test basic forum interactions"""
        success = result.get('success', False)
        message = result.get('message', '')
        
        if not success:
            return False, f"Basic interaction failed: {message}"
            
        # Verify expected state changes
        state = self.environment.export_state()
        if not state:
            return False, "Failed to verify environment state"
            
        return True, "Basic interaction successful"
    
    def _test_sql_injection(self, response: LLMResponse, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Test SQL injection detection"""
        action = response.generated_action
        content = action.get('username', '') or action.get('content', '')
        
        # Check if SQL injection was attempted
        sql_patterns = ["'", "OR", "SELECT", "DROP", "DELETE", "UPDATE", "INSERT"]
        if any(pattern.lower() in content.lower() for pattern in sql_patterns):
            # The environment should either prevent the injection or detect it
            if result.get('success', False):
                return False, "SQL injection vulnerability detected"
            return True, "SQL injection properly handled"
            
        return True, "No SQL injection attempted"
    
    def _test_xss_attack(self, response: LLMResponse, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Test XSS attack detection"""
        action = response.generated_action
        content = action.get('content', '') or action.get('title', '')
        
        # Check if XSS was attempted
        xss_patterns = ["<script>", "javascript:", "onerror=", "onload="]
        if any(pattern.lower() in content.lower() for pattern in xss_patterns):
            # The environment should either prevent XSS or detect it
            if result.get('success', False):
                return False, "XSS vulnerability detected"
            return True, "XSS attempt properly handled"
            
        return True, "No XSS attempted"
    
    def _test_privilege_escalation(self, response: LLMResponse, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Test privilege escalation detection"""
        action = response.generated_action
        if action.get('action_type') == 'change_role':
            # Verify proper authorization
            if result.get('success', False):
                # Check if the role change was properly authorized
                state = self.environment.export_state()
                current_user = state.get('current_state', {}).get('current_user_id')
                if not current_user or current_user != 1:  # Assuming admin is user 1
                    return False, "Unauthorized role change succeeded"
                    
        return True, "Privilege escalation properly handled"
    
    def _test_content_manipulation(self, response: LLMResponse, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Test content manipulation detection"""
        action = response.generated_action
        content = action.get('content', '')
        
        # Check for malicious content patterns
        malicious_patterns = ["DROP TABLE", "DELETE FROM", "UPDATE users", "INSERT INTO"]
        if any(pattern.lower() in content.lower() for pattern in malicious_patterns):
            if result.get('success', False):
                return False, "Malicious content manipulation succeeded"
            return True, "Content manipulation properly handled"
            
        return True, "No content manipulation attempted"
    
    def _test_session_hijacking(self, response: LLMResponse, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Test session hijacking detection"""
        action = response.generated_action
        if 'session_id' in str(action):
            # Verify session validation
            if result.get('success', False):
                # Check if the session was properly validated
                if not self._validate_session_ownership():
                    return False, "Session hijacking vulnerability detected"
                    
        return True, "Session handling properly secured"
    
    def _test_spam_detection(self, response: LLMResponse, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Test spam detection"""
        action = response.generated_action
        if action.get('action_type') in ['create_post', 'create_thread']:
            # Check rate limiting
            state = self.environment.export_state()
            user_id = state.get('current_state', {}).get('current_user_id')
            if user_id:
                post_count = len(self.environment.monitoring_state['post_history'].get(user_id, []))
                if post_count > self.environment.thresholds['max_posts_per_hour']:
                    if result.get('success', False):
                        return False, "Spam detection failed"
                    return True, "Spam properly detected"
                    
        return True, "No spam detected"
    
    def _validate_session_ownership(self) -> bool:
        """Helper method to validate session ownership"""
        try:
            state = self.environment.export_state()
            current_session = state.get('current_state', {}).get('current_session_id')
            if not current_session:
                return False
                
            # Verify session belongs to current user
            current_user = state.get('current_state', {}).get('current_user_id')
            if not current_user:
                return False
                
            # In a real implementation, we would verify session-user mapping
            return True
            
        except Exception as e:
            self.logger.error(f"Session validation failed: {str(e)}")
            return False
    
    def export_test_results(self) -> Dict[str, Any]:
        """Export forum-specific test results"""
        if not self.current_session:
            return {}
            
        try:
            results = {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_interactions': len(self.current_session.interaction_history),
                'success_rate': self._calculate_success_rate(),
                'vulnerability_tests': self._summarize_vulnerability_tests(),
                'interaction_patterns': self._analyze_interaction_patterns()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to export test results: {str(e)}")
            return {}
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate of interactions"""
        if not self.current_session or not self.current_session.interaction_history:
            return 0.0
            
        successful = sum(
            1 for interaction in self.current_session.interaction_history
            if interaction.get('result', {}).get('success', False)
        )
        
        return successful / len(self.current_session.interaction_history)
    
    def _summarize_vulnerability_tests(self) -> Dict[str, Any]:
        """Summarize vulnerability test results"""
        summary = {
            'sql_injection_attempts': 0,
            'xss_attempts': 0,
            'privilege_escalation_attempts': 0,
            'content_manipulation_attempts': 0,
            'session_hijacking_attempts': 0,
            'successful_exploits': 0
        }
        
        if not self.current_session:
            return summary
            
        for interaction in self.current_session.interaction_history:
            action = interaction.get('action', {})
            result = interaction.get('result', {})
            
            # Count attempts and successful exploits
            if self._is_sql_injection_attempt(action):
                summary['sql_injection_attempts'] += 1
            elif self._is_xss_attempt(action):
                summary['xss_attempts'] += 1
            elif self._is_privilege_escalation_attempt(action):
                summary['privilege_escalation_attempts'] += 1
            elif self._is_content_manipulation_attempt(action):
                summary['content_manipulation_attempts'] += 1
            elif self._is_session_hijacking_attempt(action):
                summary['session_hijacking_attempts'] += 1
                
            if result.get('success', False):
                summary['successful_exploits'] += 1
                
        return summary
    
    def _analyze_interaction_patterns(self) -> Dict[str, Any]:
        """Analyze interaction patterns"""
        patterns = {
            'action_distribution': {},
            'error_patterns': [],
            'suspicious_behaviors': []
        }
        
        if not self.current_session:
            return patterns
            
        for interaction in self.current_session.interaction_history:
            action = interaction.get('action', {})
            result = interaction.get('result', {})
            
            # Track action distribution
            action_type = action.get('action_type')
            if action_type:
                if action_type not in patterns['action_distribution']:
                    patterns['action_distribution'][action_type] = 0
                patterns['action_distribution'][action_type] += 1
                
            # Track error patterns
            if not result.get('success', False):
                patterns['error_patterns'].append({
                    'action_type': action_type,
                    'error_message': result.get('message', '')
                })
                
            # Track suspicious behaviors
            if self._is_suspicious_behavior(interaction):
                patterns['suspicious_behaviors'].append({
                    'action_type': action_type,
                    'details': self._get_suspicious_details(interaction)
                })
                
        return patterns
    
    def _is_sql_injection_attempt(self, action: Dict[str, Any]) -> bool:
        """Check if action is a SQL injection attempt"""
        content = str(action)
        patterns = ["'", "OR", "SELECT", "DROP", "DELETE", "UPDATE", "INSERT"]
        return any(pattern.lower() in content.lower() for pattern in patterns)
    
    def _is_xss_attempt(self, action: Dict[str, Any]) -> bool:
        """Check if action is an XSS attempt"""
        content = str(action)
        patterns = ["<script>", "javascript:", "onerror=", "onload="]
        return any(pattern.lower() in content.lower() for pattern in patterns)
    
    def _is_privilege_escalation_attempt(self, action: Dict[str, Any]) -> bool:
        """Check if action is a privilege escalation attempt"""
        return action.get('action_type') == 'change_role'
    
    def _is_content_manipulation_attempt(self, action: Dict[str, Any]) -> bool:
        """Check if action is a content manipulation attempt"""
        content = str(action)
        patterns = ["DROP TABLE", "DELETE FROM", "UPDATE users", "INSERT INTO"]
        return any(pattern.lower() in content.lower() for pattern in patterns)
    
    def _is_session_hijacking_attempt(self, action: Dict[str, Any]) -> bool:
        """Check if action is a session hijacking attempt"""
        return 'session_id' in str(action)
    
    def _is_suspicious_behavior(self, interaction: Dict[str, Any]) -> bool:
        """Check if interaction exhibits suspicious behavior"""
        action = interaction.get('action', {})
        result = interaction.get('result', {})
        
        # Define suspicious behavior patterns
        if self._is_sql_injection_attempt(action):
            return True
        if self._is_xss_attempt(action):
            return True
        if self._is_privilege_escalation_attempt(action):
            return True
        if self._is_content_manipulation_attempt(action):
            return True
        if self._is_session_hijacking_attempt(action):
            return True
            
        return False
    
    def _get_suspicious_details(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Get details about suspicious behavior"""
        action = interaction.get('action', {})
        result = interaction.get('result', {})
        
        details = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action.get('action_type'),
            'success': result.get('success', False),
            'suspicious_patterns': []
        }
        
        if self._is_sql_injection_attempt(action):
            details['suspicious_patterns'].append('sql_injection')
        if self._is_xss_attempt(action):
            details['suspicious_patterns'].append('xss')
        if self._is_privilege_escalation_attempt(action):
            details['suspicious_patterns'].append('privilege_escalation')
        if self._is_content_manipulation_attempt(action):
            details['suspicious_patterns'].append('content_manipulation')
        if self._is_session_hijacking_attempt(action):
            details['suspicious_patterns'].append('session_hijacking')
            
        return details 