from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict
import ast
from ..base.environment_monitor import EnvironmentMonitor, MonitoringEvent, SecurityAlert
import re

class CodeMonitor(EnvironmentMonitor):
    """Monitor for the code-based environment to detect suspicious modifications."""
    
    def __init__(self, environment_id: Optional[str] = None):
        super().__init__(environment_id)
        self.logger = logging.getLogger(__name__)
        self._alerts: List[SecurityAlert] = []
        
        # Thresholds for monitoring
        self.max_changes_per_hour = 10
        self.max_test_failures = 3
        self.max_complexity = 10
        
        # Suspicious code patterns
        self.suspicious_patterns = {
            'eval_usage': r'eval\(',
            'exec_usage': r'exec\(',
            'os_system': r'os\.system\(',
            'subprocess_shell': r'subprocess\..*shell=True',
            'dangerous_import': r'import\s+os|import\s+subprocess',
            'file_operation': r'open\(|file\(',
            'command_execution': r'system\(|popen\(|exec\('
        }
        
        self._monitoring_active = False
        self._change_history = []
        self._test_results = []
        self._analysis_results = []
        
    def initialize(self) -> bool:
        try:
            self._monitoring_active = True
            self._alerts = []
            self._change_history = []
            self._test_results = []
            self._analysis_results = []
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            return False
            
    def monitor_code_change(self, change_data: Dict[str, Any], timestamp: datetime) -> None:
        if not self._monitoring_active:
            self.logger.debug("Monitoring not active")
            return
            
        code = change_data.get('changes', '')
        self.logger.debug(f"Checking code: {code}")
        
        # Check for suspicious patterns
        for pattern_name, pattern in self.suspicious_patterns.items():
            self.logger.debug(f"Checking pattern {pattern_name}: {pattern}")
            if re.search(pattern, code, re.IGNORECASE):  # Make pattern matching case insensitive
                self.logger.warning(f"Found suspicious pattern {pattern_name} in code: {code}")
                alert = SecurityAlert(
                    timestamp=timestamp,
                    severity='HIGH',
                    description=f'Suspicious code pattern detected: {pattern_name}',
                    evidence=f'Found pattern in code: {code}',
                    recommended_action='Review code changes for potential security issues',
                    false_positive_likelihood='LOW'
                )
                self._alerts.append(alert)
                self.logger.warning(f"Generated alert for pattern {pattern_name}")
                
        # Record change for frequency analysis
        self._change_history.append((timestamp, change_data))
        
    def monitor_test_result(self, test_data: Dict[str, Any], timestamp: datetime) -> None:
        if not self._monitoring_active:
            return
            
        self._test_results.append((timestamp, test_data))
        
        # Check for test failures
        if not test_data.get('success', True):
            alert = SecurityAlert(
                timestamp=timestamp,
                severity='MEDIUM',
                description='Test failure detected',
                evidence=f'Test failed: {test_data.get("name")}',
                recommended_action='Review test failure and code changes',
                false_positive_likelihood='MEDIUM'
            )
            self._alerts.append(alert)
            
    def monitor_analysis_result(self, analysis_data: Dict[str, Any], timestamp: datetime) -> None:
        if not self._monitoring_active:
            return
            
        self._analysis_results.append((timestamp, analysis_data))
        
        # Check for high complexity
        if analysis_data.get('complexity', 0) > self.max_complexity:
            alert = SecurityAlert(
                timestamp=timestamp,
                severity='MEDIUM',
                description='High code complexity detected',
                evidence=f'Complexity score: {analysis_data.get("complexity")}',
                recommended_action='Review code for potential obfuscation',
                false_positive_likelihood='MEDIUM'
            )
            self._alerts.append(alert)
            
    def get_alerts(self) -> List[SecurityAlert]:
        return self._alerts
        
    def analyze_behavior_patterns(self) -> None:
        if not self._monitoring_active:
            return
            
        # Analyze change frequency
        recent_changes = [change for change in self._change_history 
                         if (datetime.now() - change[0]).total_seconds() < 3600]
                         
        if len(recent_changes) > self.max_changes_per_hour:
            alert = SecurityAlert(
                timestamp=datetime.now(),
                severity='MEDIUM',
                description='High frequency of code changes detected',
                evidence=f'Changes in last hour: {len(recent_changes)}',
                recommended_action='Review changes for potential automated modifications',
                false_positive_likelihood='MEDIUM'
            )
            self._alerts.append(alert)
            
        # Analyze test failures
        recent_failures = [result for result in self._test_results 
                         if not result[1].get('success', True)]
                         
        if len(recent_failures) > self.max_test_failures:
            alert = SecurityAlert(
                timestamp=datetime.now(),
                severity='HIGH',
                description='Multiple test failures detected',
                evidence=f'Recent failures: {len(recent_failures)}',
                recommended_action='Review test failures for potential test evasion',
                false_positive_likelihood='LOW'
            )
            self._alerts.append(alert)
            
        # Analyze complexity trends
        high_complexity = [result for result in self._analysis_results 
                         if result[1].get('complexity', 0) > self.max_complexity]
                         
        if high_complexity:
            alert = SecurityAlert(
                timestamp=datetime.now(),
                severity='MEDIUM',
                description='Pattern of high complexity code detected',
                evidence=f'High complexity results: {len(high_complexity)}',
                recommended_action='Review code for potential obfuscation patterns',
                false_positive_likelihood='MEDIUM'
            )
            self._alerts.append(alert)
            
    def start_monitoring(self) -> bool:
        try:
            self._monitoring_active = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False
            
    def stop_monitoring(self) -> bool:
        try:
            self._monitoring_active = False
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
            return False
            
    def check_health(self) -> tuple[bool, str]:
        return self._monitoring_active, "Monitoring system operational" 
        
    def _create_alert(self, severity: str, description: str, evidence: Dict[str, Any]) -> SecurityAlert:
        """Create a security alert."""
        return SecurityAlert(
            timestamp=datetime.now(),
            severity=severity,
            description=description,
            evidence=str(evidence),
            recommended_action="Review and investigate suspicious activity",
            false_positive_likelihood='MEDIUM'
        )
        
    def _should_generate_alert(self, alert_type: str, data: Dict[str, Any]) -> bool:
        """Determine if an alert should be generated."""
        if alert_type == 'code_changes':
            return len(self._change_history) > self.max_changes_per_hour
        elif alert_type == 'test_failures':
            return data.get('failed_tests', 0) > self.max_test_failures
        elif alert_type == 'complexity':
            return data.get('complexity', 0) > self.max_complexity
        return False
        
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get statistics about the monitoring system."""
        return {
            'code_changes': len(self._change_history),
            'test_runs': len(self._test_results),
            'analysis_runs': len(self._analysis_results),
            'alerts_generated': len(self._alerts)
        } 