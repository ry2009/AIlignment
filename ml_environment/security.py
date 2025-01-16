import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import time
import re
import hashlib
from dataclasses import dataclass
import threading
from collections import defaultdict

@dataclass
class SecurityEvent:
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: float
    session_id: str
    details: Dict

class SecurityMonitor:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.events_path = base_path / 'logs' / 'security'
        self.events_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('SecurityMonitor')
        self._setup_logging()
        
        # Initialize security state
        self.active_sessions: Dict[str, Dict] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        self._security_lock = threading.Lock()

    def _setup_logging(self):
        handler = logging.FileHandler(self.base_path / 'logs' / 'security.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _save_event(self, event: SecurityEvent):
        """Save security event to disk."""
        event_file = self.events_path / f"events_{event.session_id}.json"
        event_data = {
            'event_type': event.event_type,
            'severity': event.severity,
            'description': event.description,
            'timestamp': event.timestamp,
            'session_id': event.session_id,
            'details': event.details
        }
        
        with open(event_file, 'a') as f:
            f.write(json.dumps(event_data) + '\n')

    def monitor_session(self, session_id: str, user_id: str, ip_address: str):
        """Start monitoring a new session."""
        with self._security_lock:
            if ip_address in self.blocked_ips:
                self.log_security_event(
                    session_id=session_id,
                    event_type='access_denied',
                    severity='high',
                    description=f"Blocked IP attempted to create session: {ip_address}",
                    details={'ip_address': ip_address, 'user_id': user_id}
                )
                raise ValueError("Access denied: IP is blocked")
            
            self.active_sessions[session_id] = {
                'user_id': user_id,
                'ip_address': ip_address,
                'start_time': time.time(),
                'last_activity': time.time(),
                'suspicious_activities': 0
            }

    def check_data_poisoning(self, data: Dict, session_id: str) -> bool:
        """Check for potential data poisoning attempts."""
        suspicious_patterns = [
            r'(?i)(drop|delete|truncate|alter)\s+table',  # SQL injection
            r'<script.*?>.*?</script>',  # XSS
            r'\b(exec|eval|system)\s*\(',  # Code injection
            r'(?i)(union\s+select|information_schema)',  # SQL injection
        ]
        
        data_str = json.dumps(data)
        for pattern in suspicious_patterns:
            if re.search(pattern, data_str):
                self.log_security_event(
                    session_id=session_id,
                    event_type='data_poisoning_attempt',
                    severity='high',
                    description=f"Suspicious pattern detected in data",
                    details={'pattern': pattern}
                )
                return True
        return False

    def check_model_extraction(self, session_id: str, request_count: int, 
                             time_window: float) -> bool:
        """Check for potential model extraction attempts."""
        with self._security_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            
            current_time = time.time()
            time_diff = current_time - session['last_activity']
            
            if time_diff <= time_window and request_count > 100:  # Threshold can be adjusted
                self.log_security_event(
                    session_id=session_id,
                    event_type='model_extraction_attempt',
                    severity='high',
                    description=f"High frequency model access detected",
                    details={
                        'request_count': request_count,
                        'time_window': time_window
                    }
                )
                return True
            
            session['last_activity'] = current_time
            return False

    def check_resource_abuse(self, session_id: str, resource_usage: Dict[str, float]) -> bool:
        """Check for resource abuse patterns."""
        thresholds = {
            'cpu_percent': 90,
            'memory_percent': 90,
            'gpu_utilization': 95
        }
        
        violations = []
        for resource, usage in resource_usage.items():
            if resource in thresholds and usage > thresholds[resource]:
                violations.append(resource)
        
        if violations:
            self.log_security_event(
                session_id=session_id,
                event_type='resource_abuse',
                severity='medium',
                description=f"Resource usage exceeded thresholds",
                details={
                    'violations': violations,
                    'usage': resource_usage
                }
            )
            return True
        return False

    def check_model_tampering(self, model_hash: str, session_id: str) -> bool:
        """Check for model tampering attempts."""
        # Implementation depends on how models are stored and versioned
        pass

    def log_security_event(self, session_id: str, event_type: str, severity: str,
                          description: str, details: Dict):
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            timestamp=time.time(),
            session_id=session_id,
            details=details
        )
        
        self._save_event(event)
        
        if severity in ['high', 'critical']:
            self.logger.warning(f"Security event: {description}")
            self._handle_severe_event(event)
        else:
            self.logger.info(f"Security event: {description}")

    def _handle_severe_event(self, event: SecurityEvent):
        """Handle severe security events."""
        with self._security_lock:
            session = self.active_sessions.get(event.session_id)
            if not session:
                return
            
            session['suspicious_activities'] += 1
            
            # Block IP if multiple suspicious activities detected
            if session['suspicious_activities'] >= 3:
                self.blocked_ips.add(session['ip_address'])
                self.log_security_event(
                    session_id=event.session_id,
                    event_type='ip_blocked',
                    severity='critical',
                    description=f"IP blocked due to multiple suspicious activities",
                    details={'ip_address': session['ip_address']}
                )

    def get_security_events(self, session_id: str, 
                          severity_filter: Optional[List[str]] = None) -> List[SecurityEvent]:
        """Retrieve security events for a session."""
        event_file = self.events_path / f"events_{session_id}.json"
        if not event_file.exists():
            return []
        
        events = []
        with open(event_file, 'r') as f:
            for line in f:
                event_data = json.loads(line)
                if severity_filter is None or event_data['severity'] in severity_filter:
                    events.append(SecurityEvent(**event_data))
        
        return events

    def is_session_blocked(self, session_id: str) -> bool:
        """Check if a session is blocked."""
        with self._security_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                return False
            return session['ip_address'] in self.blocked_ips 