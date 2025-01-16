# Environment Interface Specification

## Overview
This document defines the standard interface that all control environments must implement. The goal is to ensure consistency across different environment types while maintaining flexibility for environment-specific features.

## Core Components

### 1. Environment State
All environments must maintain a state that inherits from `EnvironmentState`:
```python
@dataclass
class EnvironmentState:
    environment_id: str
    start_time: datetime
    current_state: Dict[str, Any]
    metrics: Dict[str, Any]
    is_compromised: bool = False
    compromise_details: Optional[Dict[str, Any]] = None
```

### 2. Action Interface
Actions must follow a standard format:
```python
{
    'action_type': str,          # The type of action to perform
    'parameters': Dict[str, Any], # Action-specific parameters
    'metadata': Dict[str, Any]    # Optional metadata about the action
}
```

### 3. Interaction Results
All actions must return an `InteractionResult`:
```python
@dataclass
class InteractionResult:
    success: bool
    action_type: str
    timestamp: datetime
    details: Dict[str, Any]
    state_change: Dict[str, Any]
    side_effects: List[Dict[str, Any]]
```

## Required Methods

### 1. Core Environment Methods
Every environment must implement:
```python
def initialize(self) -> bool
def reset(self) -> bool
def validate_action(self, action: Dict[str, Any]) -> bool
def execute_action(self, action: Dict[str, Any]) -> InteractionResult
def cleanup(self) -> None
```

### 2. State Management
```python
def get_state(self) -> EnvironmentState
def export_state(self) -> Dict[str, Any]
def validate_state(self) -> Tuple[bool, str]
```

### 3. Monitoring Interface
```python
def start_monitoring(self) -> bool
def stop_monitoring(self) -> bool
def record_event(self, event: MonitoringEvent) -> None
def record_alert(self, alert: SecurityAlert) -> None
def get_events(self, **filters) -> List[MonitoringEvent]
def get_alerts(self, **filters) -> List[SecurityAlert]
```

## Environment-Specific Requirements

### 1. Documentation
Each environment must provide:
- Main task description
- Side task opportunities
- List of intentional vulnerabilities
- Expected monitoring capabilities
- Success/failure criteria

### 2. Testing Requirements
Each environment must include tests for:
- Basic functionality
- Each intentional vulnerability
- Monitoring effectiveness
- State management
- Error handling

### 3. Monitoring Requirements
Each environment must implement monitoring for:
- Suspicious action patterns
- Known vulnerability exploitation
- Resource usage/limits
- User/agent behavior patterns

## Implementation Guidelines

### 1. State Management
- Use dataclasses for structured state
- Include all necessary information for rollbacks
- Maintain clean separation of concerns

### 2. Action Validation
- Validate all action parameters
- Check for required permissions/prerequisites
- Maintain action history for monitoring

### 3. Error Handling
- Provide detailed error messages
- Maintain environment stability
- Log all errors for monitoring

### 4. Security Considerations
- Document all intentional vulnerabilities
- Implement proper access controls
- Maintain audit logs

## Example Implementation

See the forum environment (`environments/forum/`) for a reference implementation of these standards. 