from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import logging

from .environment_monitor import EnvironmentMonitor

@dataclass
class EnvironmentState:
    """Represents the current state of an environment"""
    environment_id: str
    start_time: datetime
    current_state: Dict[str, Any]
    metrics: Dict[str, Any]
    is_compromised: bool = False
    compromise_details: Optional[Dict[str, Any]] = None

@dataclass
class InteractionResult:
    """Represents the result of an interaction with the environment"""
    success: bool
    action_type: str
    timestamp: datetime
    details: Dict[str, Any]
    state_change: Dict[str, Any]
    side_effects: List[Dict[str, Any]]

class BaseEnvironment(ABC):
    """Base class for all control environments"""
    
    def __init__(self, 
                 environment_id: Optional[str] = None,
                 monitor_class: Optional[type] = None):
        self.environment_id = environment_id or str(uuid.uuid4())
        self.logger = logging.getLogger(f"environment.{self.environment_id}")
        self.state = self._initialize_state()
        
        # Initialize monitoring
        if monitor_class:
            self.monitor = monitor_class(self.environment_id)
        else:
            self.monitor = None
    
    @abstractmethod
    def _initialize_state(self) -> EnvironmentState:
        """Initialize the environment state"""
        pass
    
    @abstractmethod
    def reset(self) -> EnvironmentState:
        """Reset the environment to its initial state"""
        pass
    
    @abstractmethod
    def validate_action(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate if an action is permissible"""
        pass
    
    @abstractmethod
    def execute_action(self, action: Dict[str, Any]) -> InteractionResult:
        """Execute an action in the environment"""
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get list of available actions in current state"""
        pass
    
    def record_interaction(self, action: Dict[str, Any], result: InteractionResult):
        """Record an interaction with the environment"""
        if self.monitor:
            self.monitor.record_event({
                "timestamp": datetime.now(),
                "event_type": "interaction",
                "severity": "info",
                "details": {
                    "action": action,
                    "result": {
                        "success": result.success,
                        "action_type": result.action_type,
                        "details": result.details,
                        "state_change": result.state_change
                    }
                },
                "source": "environment",
                "context": {
                    "environment_id": self.environment_id,
                    "current_state": self.state.current_state
                }
            })
    
    def check_security(self) -> Dict[str, Any]:
        """Check environment security status"""
        if self.monitor:
            alerts = self.monitor.get_alerts(min_severity="medium")
            return {
                "is_compromised": self.state.is_compromised,
                "compromise_details": self.state.compromise_details,
                "active_alerts": len(alerts),
                "alert_details": [
                    {
                        "severity": alert.severity,
                        "description": alert.description,
                        "timestamp": alert.timestamp
                    }
                    for alert in alerts
                ]
            }
        return {
            "is_compromised": self.state.is_compromised,
            "compromise_details": self.state.compromise_details,
            "active_alerts": 0,
            "alert_details": []
        }
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get environment-specific metrics"""
        pass
    
    @abstractmethod
    def validate_state(self) -> Tuple[bool, List[str]]:
        """Validate the current state of the environment"""
        pass
    
    def export_state(self, filepath: str):
        """Export environment state to file"""
        state_data = {
            "environment_id": self.state.environment_id,
            "start_time": self.state.start_time.isoformat(),
            "current_state": self.state.current_state,
            "metrics": self.state.metrics,
            "is_compromised": self.state.is_compromised,
            "compromise_details": self.state.compromise_details
        }
        
        if self.monitor:
            self.monitor.export_data(f"{filepath}_monitor.json")
        
        with open(f"{filepath}_state.json", 'w') as f:
            json.dump(state_data, f, indent=2)
    
    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation of the environment"""
        pass
    
    @abstractmethod
    def is_terminal_state(self) -> Tuple[bool, Optional[str]]:
        """Check if environment is in a terminal state"""
        pass
    
    @abstractmethod
    def get_reward(self, action: Dict[str, Any], 
                  result: InteractionResult) -> float:
        """Calculate reward for an action"""
        pass 