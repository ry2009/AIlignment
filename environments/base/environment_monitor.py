from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import logging

@dataclass
class MonitoringEvent:
    """Represents a single monitoring event"""
    timestamp: datetime
    event_type: str
    severity: str
    details: Dict[str, Any]
    source: str
    context: Optional[Dict[str, Any]] = None

@dataclass
class SecurityAlert:
    """Represents a security-related alert"""
    timestamp: datetime
    severity: str
    description: str
    evidence: Dict[str, Any]
    recommended_action: str
    false_positive_likelihood: float

class EnvironmentMonitor(ABC):
    """Base class for environment-specific monitoring"""
    
    def __init__(self, environment_id: str):
        self.environment_id = environment_id
        self.events: List[MonitoringEvent] = []
        self.alerts: List[SecurityAlert] = []
        self.logger = logging.getLogger(f"monitor.{environment_id}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the monitoring system"""
        pass
    
    @abstractmethod
    def start_monitoring(self) -> bool:
        """Start monitoring activities"""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> bool:
        """Stop monitoring activities"""
        pass
    
    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """Check monitoring system health"""
        pass
    
    def record_event(self, event: MonitoringEvent):
        """Record a monitoring event"""
        self.events.append(event)
        self.logger.info(f"Event recorded: {event.event_type}")
        
        # Analyze event for potential security implications
        if self._should_generate_alert(event):
            alert = self._create_alert(event)
            self.alerts.append(alert)
            self.logger.warning(f"Security alert generated: {alert.description}")
    
    def get_events(self, 
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  event_types: Optional[List[str]] = None) -> List[MonitoringEvent]:
        """Get filtered monitoring events"""
        filtered_events = self.events
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        if event_types:
            filtered_events = [e for e in filtered_events if e.event_type in event_types]
        
        return filtered_events
    
    def get_alerts(self,
                  min_severity: str = "low",
                  start_time: Optional[datetime] = None) -> List[SecurityAlert]:
        """Get filtered security alerts"""
        severity_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_severity_level = severity_levels.get(min_severity.lower(), 0)
        
        filtered_alerts = [
            alert for alert in self.alerts
            if severity_levels.get(alert.severity.lower(), 0) >= min_severity_level
        ]
        
        if start_time:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.timestamp >= start_time
            ]
        
        return filtered_alerts
    
    def export_data(self, filepath: str):
        """Export monitoring data to file"""
        data = {
            "environment_id": self.environment_id,
            "events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "details": event.details,
                    "source": event.source,
                    "context": event.context
                }
                for event in self.events
            ],
            "alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "description": alert.description,
                    "evidence": alert.evidence,
                    "recommended_action": alert.recommended_action,
                    "false_positive_likelihood": alert.false_positive_likelihood
                }
                for alert in self.alerts
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @abstractmethod
    def _should_generate_alert(self, event: MonitoringEvent) -> bool:
        """Determine if an event should generate a security alert"""
        pass
    
    @abstractmethod
    def _create_alert(self, event: MonitoringEvent) -> SecurityAlert:
        """Create a security alert from an event"""
        pass
    
    @abstractmethod
    def analyze_behavior_patterns(self, 
                                timeframe_minutes: int = 60) -> Dict[str, Any]:
        """Analyze behavior patterns within a given timeframe"""
        pass
    
    @abstractmethod
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics"""
        pass