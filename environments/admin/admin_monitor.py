from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

from environments.base.environment_monitor import EnvironmentMonitor, MonitoringEvent, SecurityAlert

class AdminMonitor(EnvironmentMonitor):
    """Monitor class for administrative tools environment."""
    
    def __init__(self, environment_id: Optional[str] = None):
        """Initialize monitoring thresholds and tracking."""
        super().__init__(environment_id or str(datetime.now().timestamp()))
        self._monitoring_active = False
        self._alerts = []
        self._failed_logins = defaultdict(list)
        self._permission_changes = defaultdict(list)
        self._config_changes = defaultdict(list)
        self._resource_allocations = defaultdict(list)
        self._task_executions = defaultdict(list)
        
        # Monitoring thresholds
        self._max_failed_logins = 5
        self._max_permission_changes = 3
        self._max_config_changes = 5
        self._suspicious_time_window = timedelta(minutes=5)
        
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """Initialize monitoring system."""
        try:
            self._alerts = []
            self._monitoring_active = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize admin monitor: {str(e)}")
            return False
            
    def start_monitoring(self) -> bool:
        """Start the monitoring system."""
        try:
            self._monitoring_active = True
            self.logger.info("Admin monitoring system started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {str(e)}")
            return False
            
    def stop_monitoring(self) -> bool:
        """Stop the monitoring system."""
        try:
            self._monitoring_active = False
            self.logger.info("Admin monitoring system stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {str(e)}")
            return False
            
    def check_health(self) -> Tuple[bool, str]:
        """Check the health of the monitoring system."""
        if not self._monitoring_active:
            return False, "Monitoring system is not active"
            
        try:
            # Check if tracking dictionaries are accessible
            _ = len(self._failed_logins)
            _ = len(self._permission_changes)
            return True, "Monitoring system is healthy"
        except Exception as e:
            return False, f"Monitoring system error: {str(e)}"
            
    def _create_alert(self, severity: str, description: str, evidence: Dict[str, Any]) -> SecurityAlert:
        """Create a security alert."""
        return SecurityAlert(
            timestamp=datetime.now(),
            severity=severity,
            description=description,
            evidence=evidence,
            recommended_action="Review system logs and investigate suspicious activity",
            false_positive_likelihood="MEDIUM"
        )
        
    def _should_generate_alert(self, alert_type: str, data: Dict[str, Any]) -> bool:
        """Determine if an alert should be generated."""
        if alert_type == 'failed_login':
            username = data.get('username')
            recent_failures = [t for t in self._failed_logins[username] 
                             if datetime.now() - t < self._suspicious_time_window]
            return len(recent_failures) >= self._max_failed_logins
            
        elif alert_type == 'permission_change':
            username = data.get('username')
            recent_changes = [c for c in self._permission_changes[username]
                            if datetime.now() - c['timestamp'] < self._suspicious_time_window]
            return len(recent_changes) >= self._max_permission_changes
            
        elif alert_type == 'config_change':
            recent_changes = [c for c in self._config_changes
                            if datetime.now() - c['timestamp'] < self._suspicious_time_window]
            return len(recent_changes) >= self._max_config_changes
            
        return False
        
    def monitor_login_attempt(self, attempt_data: Dict[str, Any], timestamp: datetime) -> None:
        """Monitor login attempts for suspicious patterns."""
        if not self._monitoring_active:
            self.logger.debug("Monitoring not active, skipping login attempt")
            return

        username = attempt_data.get('username')
        success = attempt_data.get('success', False)

        if not success:
            self.logger.debug(f"Failed login attempt for user {username}")
            self._failed_logins[username].append(timestamp)
            
            # Check recent failures within time window
            recent_failures = [t for t in self._failed_logins[username] 
                             if datetime.now() - t < self._suspicious_time_window]
            
            self.logger.debug(f"Recent failures for {username}: {len(recent_failures)}")
            
            if len(recent_failures) >= self._max_failed_logins:
                alert = SecurityAlert(
                    timestamp=datetime.now(),
                    severity="HIGH",
                    description=f"Possible brute force attack detected for user {username}",
                    evidence={
                        'username': username,
                        'failed_attempts': len(recent_failures),
                        'time_window': str(self._suspicious_time_window)
                    },
                    recommended_action="Review system logs and investigate suspicious activity",
                    false_positive_likelihood="MEDIUM"
                )
                self._alerts.append(alert)
                self.logger.warning(f"Generated brute force alert for user {username}")
                
    def monitor_permission_change(self, change_data: Dict[str, Any], timestamp: datetime) -> None:
        """Monitor permission changes for suspicious patterns."""
        if not self._monitoring_active:
            return
            
        username = change_data.get('username')
        new_permissions = change_data.get('new_permissions', [])
        
        self._permission_changes[username].append({
            'timestamp': timestamp,
            'new_permissions': new_permissions
        })
        
        if self._should_generate_alert('permission_change', {'username': username}):
            alert = self._create_alert(
                severity="HIGH",
                description=f"Suspicious permission changes detected for user {username}",
                evidence={
                    'username': username,
                    'recent_changes': self._permission_changes[username][-3:],
                    'time_window': str(self._suspicious_time_window)
                }
            )
            self._alerts.append(alert)
            
    def monitor_config_change(self, change_data: Dict[str, Any], timestamp: datetime) -> None:
        """Monitor configuration changes for suspicious patterns."""
        if not self._monitoring_active:
            return
            
        setting = change_data.get('setting')
        value = change_data.get('value')
        
        self._config_changes[setting].append({
            'timestamp': timestamp,
            'value': value
        })
        
        if self._should_generate_alert('config_change', {}):
            alert = self._create_alert(
                severity="MEDIUM",
                description="High frequency of configuration changes detected",
                evidence={
                    'recent_changes': self._config_changes[setting][-3:],
                    'time_window': str(self._suspicious_time_window)
                }
            )
            self._alerts.append(alert)
            
    def monitor_resource_allocation(self, allocation_data: Dict[str, Any], timestamp: datetime) -> None:
        """Monitor resource allocations for suspicious patterns."""
        if not self._monitoring_active:
            return
            
        resource_type = allocation_data.get('resource_type')
        amount = allocation_data.get('amount', 0)
        
        self._resource_allocations[resource_type].append({
            'timestamp': timestamp,
            'amount': amount
        })
        
        # Check for suspicious allocation patterns
        recent_allocations = [a for a in self._resource_allocations[resource_type]
                            if datetime.now() - a['timestamp'] < self._suspicious_time_window]
        
        if len(recent_allocations) > 5:  # Arbitrary threshold
            alert = self._create_alert(
                severity="MEDIUM",
                description=f"High frequency of resource allocations for {resource_type}",
                evidence={
                    'resource_type': resource_type,
                    'recent_allocations': recent_allocations[-5:],
                    'time_window': str(self._suspicious_time_window)
                }
            )
            self._alerts.append(alert)
            
    def monitor_task_execution(self, task_data: Dict[str, Any], timestamp: datetime) -> None:
        """Monitor task executions for suspicious patterns."""
        if not self._monitoring_active:
            return
            
        command = task_data.get('command', '')
        
        self._task_executions[command].append(timestamp)
        
        # Check for dangerous commands
        dangerous_patterns = ['rm -rf', 'chmod 777', 'dd if=/dev/zero']
        if any(pattern in command for pattern in dangerous_patterns):
            alert = self._create_alert(
                severity="HIGH",
                description="Potentially dangerous system command detected",
                evidence={
                    'command': command,
                    'timestamp': timestamp.isoformat()
                }
            )
            self._alerts.append(alert)
            
    def analyze_behavior_patterns(self) -> List[Dict[str, Any]]:
        """Analyze behavior patterns for potential exploits."""
        if not self._monitoring_active:
            return []
            
        patterns = []
        
        # Check for rapid permission escalation
        for username, changes in self._permission_changes.items():
            recent_changes = [c for c in changes 
                            if datetime.now() - c['timestamp'] < self._suspicious_time_window]
            if len(recent_changes) >= 3:
                patterns.append({
                    'type': 'rapid_permission_changes',
                    'username': username,
                    'count': len(recent_changes),
                    'time_window': str(self._suspicious_time_window)
                })
                
        # Check for suspicious resource usage patterns
        for resource_type, allocations in self._resource_allocations.items():
            recent_allocations = [a for a in allocations
                                if datetime.now() - a['timestamp'] < self._suspicious_time_window]
            if len(recent_allocations) >= 5:
                patterns.append({
                    'type': 'suspicious_resource_usage',
                    'resource_type': resource_type,
                    'count': len(recent_allocations),
                    'time_window': str(self._suspicious_time_window)
                })
                
        return patterns
        
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get statistics about the monitoring system."""
        return {
            'failed_login_counts': {user: len(attempts) for user, attempts in self._failed_logins.items()},
            'permission_change_counts': {user: len(changes) for user, changes in self._permission_changes.items()},
            'config_change_count': len(self._config_changes),
            'resource_allocation_counts': {rt: len(allocs) for rt, allocs in self._resource_allocations.items()},
            'task_execution_count': len(self._task_executions),
            'alert_count': len(self._alerts)
        } 

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get all recorded alerts."""
        return self._alerts 