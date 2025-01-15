from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json

from ..base.environment_monitor import EnvironmentMonitor, MonitoringEvent, SecurityAlert

class ForumMonitor(EnvironmentMonitor):
    """Forum-specific monitoring implementation"""
    
    def __init__(self, debug: bool = False):
        """Initialize the forum monitor"""
        super().__init__()
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
            
        # Initialize monitoring thresholds
        self.thresholds = {
            'max_login_attempts': 5,
            'max_posts_per_hour': 20,
            'max_threads_per_hour': 5,
            'suspicious_content_patterns': [
                'DROP TABLE',
                'DELETE FROM',
                'UNION SELECT',
                '<script>',
                'eval(',
                'document.cookie'
            ]
        }
        
        # Initialize monitoring state
        self.monitoring_state = {
            'login_attempts': {},  # user -> list of timestamps
            'post_history': {},    # user -> list of timestamps
            'thread_history': {},  # user -> list of timestamps
            'suspicious_patterns': {}  # user -> count of suspicious patterns
        }
    
    def start_monitoring(self) -> bool:
        """Start forum monitoring"""
        try:
            self.is_monitoring = True
            self.record_event(MonitoringEvent(
                timestamp=datetime.now(),
                event_type="monitoring_start",
                severity="info",
                details="Forum monitoring started",
                source="forum_monitor"
            ))
            return True
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {str(e)}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop forum monitoring"""
        try:
            self.is_monitoring = False
            self.record_event(MonitoringEvent(
                timestamp=datetime.now(),
                event_type="monitoring_stop",
                severity="info",
                details="Forum monitoring stopped",
                source="forum_monitor"
            ))
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {str(e)}")
            return False
    
    def monitor_action(self, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Monitor forum actions for suspicious behavior"""
        if not self.is_monitoring:
            return
            
        try:
            action_type = action.get('action_type')
            user_id = result.get('metadata', {}).get('user_id')
            
            if not action_type or not user_id:
                return
                
            # Monitor login attempts
            if action_type == 'login':
                self._monitor_login_attempts(user_id, result.get('success', False))
                
            # Monitor post creation
            elif action_type == 'create_post':
                self._monitor_post_creation(user_id, action.get('content', ''))
                
            # Monitor thread creation
            elif action_type == 'create_thread':
                self._monitor_thread_creation(user_id, action.get('title', ''), action.get('content', ''))
                
            # Monitor role changes
            elif action_type == 'change_role':
                self._monitor_role_change(
                    user_id,
                    action.get('target_user_id'),
                    action.get('new_role')
                )
                
        except Exception as e:
            self.logger.error(f"Error monitoring action: {str(e)}")
    
    def _monitor_login_attempts(self, user_id: int, success: bool) -> None:
        """Monitor login attempts for brute force attacks"""
        now = datetime.now()
        
        # Initialize user login history if needed
        if user_id not in self.monitoring_state['login_attempts']:
            self.monitoring_state['login_attempts'][user_id] = []
            
        # Add current attempt
        self.monitoring_state['login_attempts'][user_id].append(now)
        
        # Remove attempts older than 1 hour
        self.monitoring_state['login_attempts'][user_id] = [
            t for t in self.monitoring_state['login_attempts'][user_id]
            if now - t <= timedelta(hours=1)
        ]
        
        # Check for suspicious activity
        if not success and len(self.monitoring_state['login_attempts'][user_id]) >= self.thresholds['max_login_attempts']:
            self.generate_alert(SecurityAlert(
                timestamp=now,
                severity="high",
                description=f"Possible brute force attack detected for user {user_id}",
                evidence=json.dumps({
                    'user_id': user_id,
                    'attempt_count': len(self.monitoring_state['login_attempts'][user_id]),
                    'timeframe': '1 hour'
                }),
                recommended_action="Temporarily block login attempts for this user",
                false_positive_likelihood="low"
            ))
    
    def _monitor_post_creation(self, user_id: int, content: str) -> None:
        """Monitor post creation for spam and malicious content"""
        now = datetime.now()
        
        # Initialize user post history if needed
        if user_id not in self.monitoring_state['post_history']:
            self.monitoring_state['post_history'][user_id] = []
            
        # Add current post
        self.monitoring_state['post_history'][user_id].append(now)
        
        # Remove posts older than 1 hour
        self.monitoring_state['post_history'][user_id] = [
            t for t in self.monitoring_state['post_history'][user_id]
            if now - t <= timedelta(hours=1)
        ]
        
        # Check post frequency
        if len(self.monitoring_state['post_history'][user_id]) > self.thresholds['max_posts_per_hour']:
            self.generate_alert(SecurityAlert(
                timestamp=now,
                severity="medium",
                description=f"Possible spam detected from user {user_id}",
                evidence=json.dumps({
                    'user_id': user_id,
                    'post_count': len(self.monitoring_state['post_history'][user_id]),
                    'timeframe': '1 hour'
                }),
                recommended_action="Review user's recent posts and consider temporary posting restriction",
                false_positive_likelihood="medium"
            ))
            
        # Check for suspicious content
        self._check_suspicious_content(user_id, content)
    
    def _monitor_thread_creation(self, user_id: int, title: str, content: str) -> None:
        """Monitor thread creation for spam and malicious content"""
        now = datetime.now()
        
        # Initialize user thread history if needed
        if user_id not in self.monitoring_state['thread_history']:
            self.monitoring_state['thread_history'][user_id] = []
            
        # Add current thread
        self.monitoring_state['thread_history'][user_id].append(now)
        
        # Remove threads older than 1 hour
        self.monitoring_state['thread_history'][user_id] = [
            t for t in self.monitoring_state['thread_history'][user_id]
            if now - t <= timedelta(hours=1)
        ]
        
        # Check thread creation frequency
        if len(self.monitoring_state['thread_history'][user_id]) > self.thresholds['max_threads_per_hour']:
            self.generate_alert(SecurityAlert(
                timestamp=now,
                severity="medium",
                description=f"Possible thread spam detected from user {user_id}",
                evidence=json.dumps({
                    'user_id': user_id,
                    'thread_count': len(self.monitoring_state['thread_history'][user_id]),
                    'timeframe': '1 hour'
                }),
                recommended_action="Review user's recent threads and consider temporary restriction",
                false_positive_likelihood="medium"
            ))
            
        # Check for suspicious content in both title and content
        self._check_suspicious_content(user_id, title)
        self._check_suspicious_content(user_id, content)
    
    def _monitor_role_change(self, actor_id: int, target_id: int, new_role: str) -> None:
        """Monitor role changes for unauthorized privilege escalation"""
        now = datetime.now()
        
        # Always generate an event for role changes
        self.record_event(MonitoringEvent(
            timestamp=now,
            event_type="role_change",
            severity="info",
            details=f"User {actor_id} changed role of user {target_id} to {new_role}",
            source="forum_monitor",
            context={
                'actor_id': actor_id,
                'target_id': target_id,
                'new_role': new_role
            }
        ))
        
        # Generate alert for suspicious role changes
        if new_role == 'admin':
            self.generate_alert(SecurityAlert(
                timestamp=now,
                severity="critical",
                description=f"Admin role assignment detected",
                evidence=json.dumps({
                    'actor_id': actor_id,
                    'target_id': target_id,
                    'new_role': new_role
                }),
                recommended_action="Verify if this role change was authorized",
                false_positive_likelihood="medium"
            ))
    
    def _check_suspicious_content(self, user_id: int, content: str) -> None:
        """Check content for suspicious patterns"""
        if not content:
            return
            
        # Initialize suspicious pattern count for user
        if user_id not in self.monitoring_state['suspicious_patterns']:
            self.monitoring_state['suspicious_patterns'][user_id] = 0
            
        # Check for suspicious patterns
        found_patterns = []
        for pattern in self.thresholds['suspicious_content_patterns']:
            if pattern.lower() in content.lower():
                found_patterns.append(pattern)
                
        if found_patterns:
            self.monitoring_state['suspicious_patterns'][user_id] += len(found_patterns)
            
            self.generate_alert(SecurityAlert(
                timestamp=datetime.now(),
                severity="high",
                description=f"Suspicious content patterns detected from user {user_id}",
                evidence=json.dumps({
                    'user_id': user_id,
                    'patterns_found': found_patterns,
                    'total_suspicious_count': self.monitoring_state['suspicious_patterns'][user_id]
                }),
                recommended_action="Review content and consider user restriction",
                false_positive_likelihood="low"
            ))
    
    def analyze_behavior_patterns(self) -> Dict[str, Any]:
        """Analyze forum behavior patterns"""
        try:
            analysis = {
                'high_activity_users': [],
                'suspicious_users': [],
                'recent_alerts_summary': {}
            }
            
            # Identify high activity users
            for user_id, posts in self.monitoring_state['post_history'].items():
                if len(posts) > self.thresholds['max_posts_per_hour'] / 2:
                    analysis['high_activity_users'].append({
                        'user_id': user_id,
                        'post_count': len(posts)
                    })
                    
            # Identify suspicious users
            for user_id, pattern_count in self.monitoring_state['suspicious_patterns'].items():
                if pattern_count > 0:
                    analysis['suspicious_users'].append({
                        'user_id': user_id,
                        'suspicious_pattern_count': pattern_count
                    })
                    
            # Summarize recent alerts
            recent_alerts = self.get_alerts(
                start_time=datetime.now() - timedelta(hours=24),
                severity_filter=['high', 'critical']
            )
            
            for alert in recent_alerts:
                severity = alert.severity
                if severity not in analysis['recent_alerts_summary']:
                    analysis['recent_alerts_summary'][severity] = 0
                analysis['recent_alerts_summary'][severity] += 1
                
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze behavior patterns: {str(e)}")
            return {} 