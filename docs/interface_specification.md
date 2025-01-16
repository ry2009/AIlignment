# Environment Interface Specification

## Overview
This document defines the standard interfaces that all environments in the AI Control Dataset must implement. These standards ensure consistency, interoperability, and proper monitoring across different environment types.

## Base Environment Interface

### Core Components
1. **State Management**
   ```python
   class EnvironmentState:
       environment_id: str
       start_time: datetime
       current_state: Dict[str, Any]
       metrics: Dict[str, Any]
       is_compromised: bool
       compromise_details: Optional[Dict[str, Any]]
   ```

2. **Interaction Results**
   ```python
   class InteractionResult:
       success: bool
       action_type: str
       timestamp: datetime
       details: Dict[str, Any]
       state_change: Dict[str, Any]
       side_effects: List[Dict[str, Any]]
   ```

### Required Methods
1. **Initialization**
   - `initialize() -> bool`
   - `reset() -> bool`

2. **Action Management**
   - `validate_action(action: Dict[str, Any]) -> bool`
   - `execute_action(action: Dict[str, Any]) -> InteractionResult`
   - `get_available_actions() -> List[Dict[str, Any]]`

3. **State Access**
   - `get_metrics() -> Dict[str, Any]`
   - `get_observation() -> Dict[str, Any]`
   - `validate_state() -> Tuple[bool, str]`
   - `is_terminal_state() -> bool`

4. **Reward System**
   - `get_reward() -> float`

## Monitoring Interface

### Event Types
1. **MonitoringEvent**
   ```python
   class MonitoringEvent:
       timestamp: datetime
       event_type: str
       severity: str
       details: Dict[str, Any]
       source: str
       context: Optional[Dict[str, Any]]
   ```

2. **SecurityAlert**
   ```python
   class SecurityAlert:
       timestamp: datetime
       severity: str
       description: str
       evidence: Dict[str, Any]
       recommended_action: str
       false_positive_likelihood: str
   ```

### Required Methods
1. **Control**
   - `start_monitoring() -> bool`
   - `stop_monitoring() -> bool`
   - `initialize() -> bool`

2. **Health & Status**
   - `check_health() -> Tuple[bool, str]`
   - `get_monitoring_stats() -> Dict[str, Any]`

3. **Event Management**
   - `record_event(event: MonitoringEvent) -> None`
   - `record_alert(alert: SecurityAlert) -> None`
   - `get_events(filters: Dict[str, Any]) -> List[MonitoringEvent]`
   - `get_alerts() -> List[SecurityAlert]`

4. **Analysis**
   - `analyze_patterns() -> List[Dict[str, Any]]`
   - `analyze_behavior_patterns() -> List[Dict[str, Any]]`

## Implementation Requirements

### 1. Action Format
All actions must be dictionaries with at least:
- `action_type: str` - The type of action being performed
- Additional parameters specific to the action type

### 2. State Management
- States must be serializable to JSON
- States must include all necessary information for monitoring
- State changes must be tracked and logged

### 3. Monitoring Integration
- All significant actions must generate monitoring events
- Security-relevant actions must be analyzed for potential alerts
- Pattern analysis must be performed regularly

### 4. Error Handling
- All methods must handle exceptions gracefully
- Error states must be properly logged and reported
- Recovery mechanisms should be provided where possible

### 5. Documentation Requirements
Each environment must provide:
- List of supported actions and their parameters
- Description of state structure and components
- Documentation of intentional vulnerabilities
- Examples of valid interactions

## Environment-Specific Extensions

### 1. Code-Based Environment
Additional required methods:
- `get_code_context() -> Dict[str, Any]`
- `validate_code_changes(changes: Dict[str, Any]) -> bool`
- `apply_code_changes(changes: Dict[str, Any]) -> InteractionResult`

### 2. Game Environment
Additional required methods:
- `get_game_state() -> Dict[str, Any]`
- `validate_game_rules(action: Dict[str, Any]) -> bool`

### 3. Forum Environment
Additional required methods:
- `get_user_context() -> Dict[str, Any]`
- `validate_permissions(action: Dict[str, Any]) -> bool` 