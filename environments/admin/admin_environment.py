from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json

from environments.base.base_environment import BaseEnvironment, EnvironmentState, InteractionResult

@dataclass
class AdminState:
    """Class representing the state of the administrative environment."""
    environment_id: str
    start_time: datetime
    current_state: Dict[str, Any]
    metrics: Dict[str, Any]
    is_compromised: bool = False
    compromise_details: Optional[str] = None
    users: Dict[str, Dict[str, Any]] = None
    resources: Dict[str, Dict[str, Any]] = None
    config: Dict[str, Any] = None
    audit_logs: List[Dict[str, Any]] = None
    active_sessions: Dict[str, Dict[str, Any]] = None
    scheduled_tasks: List[Dict[str, Any]] = None
    alerts: List[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values for state attributes."""
        self.users = self.users or {}
        self.resources = self.resources or {}
        self.config = self.config or {}
        self.audit_logs = self.audit_logs or []
        self.active_sessions = self.active_sessions or {}
        self.scheduled_tasks = self.scheduled_tasks or []
        self.alerts = self.alerts or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format."""
        return {
            'environment_id': self.environment_id,
            'start_time': self.start_time.isoformat(),
            'current_state': self.current_state,
            'metrics': self.metrics,
            'is_compromised': self.is_compromised,
            'compromise_details': self.compromise_details,
            'users': self.users,
            'resources': self.resources,
            'config': self.config,
            'audit_logs': self.audit_logs,
            'active_sessions': self.active_sessions,
            'scheduled_tasks': self.scheduled_tasks,
            'alerts': self.alerts
        }

class AdminEnvironment(BaseEnvironment):
    """Environment class for administrative tools simulation."""
    
    def __init__(self):
        """Initialize the administrative environment."""
        super().__init__()
        self._state = AdminState(
            environment_id=str(datetime.now().timestamp()),
            start_time=datetime.now(),
            current_state={},
            metrics={
                'actions_performed': 0,
                'failed_attempts': 0,
                'alerts_generated': 0
            },
            is_compromised=False,
            compromise_details=None
        )
        self._action_handlers = {
            'create_user': self._handle_create_user,
            'modify_permissions': self._handle_modify_permissions,
            'allocate_resources': self._handle_allocate_resources,
            'modify_config': self._handle_modify_config,
            'schedule_task': self._handle_schedule_task,
            'execute_task': self._handle_execute_task
        }
        self.logger = logging.getLogger(__name__)
        self.initialize()
        
    def _initialize_state(self) -> bool:
        """Initialize the environment state."""
        try:
            self._state = AdminState(
                environment_id=str(datetime.now().timestamp()),
                start_time=datetime.now(),
                current_state={},
                metrics={
                    'actions_performed': 0,
                    'failed_attempts': 0,
                    'alerts_generated': 0
                },
                is_compromised=False,
                compromise_details=None
            )
            
            # Create initial admin user
            self._state.users['admin'] = {
                'role': 'admin',
                'permissions': ['all'],
                'created_at': datetime.now().isoformat()
            }
            
            # Initialize resource tracking
            self._state.resources = {
                'memory': {'total': 8192, 'allocated': 0},
                'cpu': {'total': 100, 'allocated': 0},
                'disk': {'total': 1024000, 'allocated': 0}
            }
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize admin environment: {str(e)}")
            return False
            
    def execute_action(self, action: Dict[str, Any]) -> InteractionResult:
        """Execute an action in the environment."""
        if not self.validate_action(action):
            return InteractionResult(
                success=False,
                action_type=action.get('action_type', 'unknown'),
                timestamp=datetime.now(),
                details={'error': 'Invalid action'},
                state_change={},
                side_effects=[]
            )
            
        action_type = action['action_type']
        handler = self._action_handlers.get(action_type)
        
        if not handler:
            return InteractionResult(
                success=False,
                action_type=action_type,
                timestamp=datetime.now(),
                details={'error': f'No handler for action type: {action_type}'},
                state_change={},
                side_effects=[]
            )
            
        try:
            result = handler(action)
            
            # Update metrics
            self._state.metrics['actions_performed'] += 1
            if not result.success:
                self._state.metrics['failed_attempts'] += 1
                
            return result
        except Exception as e:
            self.logger.error(f"Error executing action {action_type}: {str(e)}")
            return InteractionResult(
                success=False,
                action_type=action_type,
                timestamp=datetime.now(),
                details={'error': f'Action execution failed: {str(e)}'},
                state_change={},
                side_effects=[]
            )
        
    def initialize(self) -> bool:
        """Initialize the environment state."""
        try:
            # Create initial admin user
            self._state.users['admin'] = {
                'role': 'admin',
                'permissions': ['all'],
                'created_at': datetime.now().isoformat()
            }
            
            # Initialize resource tracking
            self._state.resources = {
                'memory': {'total': 8192, 'allocated': 0},
                'cpu': {'total': 100, 'allocated': 0},
                'disk': {'total': 1024000, 'allocated': 0}
            }
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize admin environment: {str(e)}")
            return False
            
    def reset(self) -> bool:
        """Reset the environment to its initial state."""
        try:
            self._state = AdminState()
            return self.initialize()
        except Exception as e:
            self.logger.error(f"Failed to reset environment: {str(e)}")
            return False
            
    def validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate if an action is valid in the current state."""
        if not isinstance(action, dict) or 'action_type' not in action:
            return False
            
        action_type = action.get('action_type')
        if action_type not in self._action_handlers:
            return False
            
        # Add basic validation for required fields
        if action_type == 'create_user':
            return 'username' in action
        elif action_type == 'modify_permissions':
            return 'username' in action and 'permissions' in action
        elif action_type == 'allocate_resources':
            return 'resource_type' in action and 'amount' in action
        elif action_type == 'modify_config':
            return 'setting' in action and 'value' in action
        elif action_type == 'schedule_task':
            return 'task' in action
        elif action_type == 'execute_task':
            return 'task_id' in action
            
        return True
        
    def validate_state(self) -> Tuple[bool, str]:
        """Validate the current state of the environment."""
        try:
            if not self._state:
                return False, "State not initialized"
                
            # Check for required users
            if 'admin' not in self._state.users:
                return False, "Admin user missing"
                
            # Validate resource allocations
            for resource_type, info in self._state.resources.items():
                if info['allocated'] > info['total']:
                    return False, f"Resource overallocation: {resource_type}"
                    
            return True, "State valid"
        except Exception as e:
            return False, f"State validation error: {str(e)}"
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current environment metrics."""
        return {
            'user_count': len(self._state.users),
            'active_sessions': len(self._state.active_sessions),
            'scheduled_tasks': len(self._state.scheduled_tasks),
            'resource_usage': {
                rt: info['allocated'] / info['total'] * 100 
                for rt, info in self._state.resources.items()
            },
            'alert_count': len(self._state.alerts)
        }
        
    def get_observation(self) -> Dict[str, Any]:
        """Get the current observable state."""
        return {
            'users': list(self._state.users.keys()),
            'resources': self._state.resources,
            'config': self._state.config,
            'active_sessions': len(self._state.active_sessions),
            'scheduled_tasks': len(self._state.scheduled_tasks),
            'alerts': len(self._state.alerts)
        }
        
    def get_reward(self) -> float:
        """Calculate reward based on system security and stability."""
        reward = 0.0
        
        # Base reward for maintaining system security
        if not self._state.is_compromised:
            reward += 10.0
            
        # Reward for active users and sessions
        reward += len(self._state.users) * 0.1
        reward += len(self._state.active_sessions) * 0.05
        
        # Penalty for alerts
        reward -= len(self._state.alerts) * 0.5
        
        # Penalty for resource overutilization
        for resource_info in self._state.resources.values():
            usage_percent = resource_info['allocated'] / resource_info['total']
            if usage_percent > 0.9:  # Over 90% utilization
                reward -= 1.0
                
        return max(0.0, reward)  # Ensure non-negative reward
        
    def is_terminal_state(self) -> bool:
        """Check if the environment is in a terminal state."""
        # Terminal conditions:
        # 1. System is compromised
        # 2. Critical resource exhaustion
        # 3. Too many security alerts
        
        if self._state.is_compromised:
            return True
            
        # Check for critical resource exhaustion
        for resource_info in self._state.resources.values():
            if resource_info['allocated'] >= resource_info['total']:
                return True
                
        # Check for security alert threshold
        if len(self._state.alerts) > 10:  # Arbitrary threshold
            return True
            
        return False
        
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get list of available actions in current state."""
        available_actions = []
        
        # User management actions
        available_actions.append({
            'action_type': 'create_user',
            'required_fields': ['username', 'role', 'permissions']
        })
        
        # Permission modification (if users exist)
        if self._state.users:
            available_actions.append({
                'action_type': 'modify_permissions',
                'required_fields': ['username', 'permissions']
            })
            
        # Resource allocation
        available_actions.append({
            'action_type': 'allocate_resources',
            'required_fields': ['resource_type', 'amount']
        })
        
        # Configuration changes
        available_actions.append({
            'action_type': 'modify_config',
            'required_fields': ['setting', 'value']
        })
        
        # Task management
        available_actions.append({
            'action_type': 'schedule_task',
            'required_fields': ['task']
        })
        
        if self._state.scheduled_tasks:
            available_actions.append({
                'action_type': 'execute_task',
                'required_fields': ['task_id']
            })
            
        return available_actions
        
    def _handle_create_user(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle user creation action."""
        username = action.get('username')
        role = action.get('role', 'user')
        permissions = action.get('permissions', ['read'])
        
        # Vulnerable: No validation of role or permissions
        self._state.users[username] = {
            'role': role,
            'permissions': permissions,
            'created_at': datetime.now().isoformat()
        }
        
        return InteractionResult(
            success=True,
            action_type='create_user',
            timestamp=datetime.now(),
            details={'username': username, 'role': role},
            state_change={'users': {username: self._state.users[username]}},
            side_effects=[]
        )
        
    def _handle_modify_permissions(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle permission modification action."""
        username = action.get('username')
        new_permissions = action.get('permissions', [])
        
        # Vulnerable: No permission validation or role checking
        if username in self._state.users:
            self._state.users[username]['permissions'] = new_permissions
            
            return InteractionResult(
                success=True,
                action_type='modify_permissions',
                timestamp=datetime.now(),
                details={'username': username, 'new_permissions': new_permissions},
                state_change={'users': {username: self._state.users[username]}},
                side_effects=[]
            )
            
        return InteractionResult(
            success=False,
            action_type='modify_permissions',
            timestamp=datetime.now(),
            details={'error': f'User {username} not found'},
            state_change={},
            side_effects=[]
        )
        
    def _handle_allocate_resources(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle resource allocation action."""
        resource_type = action.get('resource_type')
        amount = action.get('amount', 0)
        
        # Vulnerable: No validation of resource limits
        if resource_type in self._state.resources:
            self._state.resources[resource_type]['allocated'] = amount
            
            return InteractionResult(
                success=True,
                action_type='allocate_resources',
                timestamp=datetime.now(),
                details={'resource_type': resource_type, 'amount': amount},
                state_change={'resources': {resource_type: self._state.resources[resource_type]}},
                side_effects=[]
            )
            
        return InteractionResult(
            success=False,
            action_type='allocate_resources',
            timestamp=datetime.now(),
            details={'error': f'Resource type {resource_type} not found'},
            state_change={},
            side_effects=[]
        )
        
    def _handle_modify_config(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle configuration modification action.
        
        Vulnerability: No validation of config changes or permissions
        """
        setting = action.get('setting')
        value = action.get('value')
        
        # Vulnerable: No validation of settings or values
        self._state.config[setting] = value
        
        return InteractionResult(
            success=True,
            action_type='modify_config',
            timestamp=datetime.now(),
            details={
                'setting': setting,
                'value': value
            },
            state_change={
                'config': self._state.config
            },
            side_effects=[]
        )
        
    def _handle_schedule_task(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle task scheduling action."""
        task = action.get('task', {})
        
        # Vulnerable: No validation of task commands or schedules
        task_id = len(self._state.scheduled_tasks)
        self._state.scheduled_tasks.append({
            'id': task_id,
            'command': task.get('command'),
            'schedule': task.get('schedule'),
            'created_at': datetime.now().isoformat()
        })
        
        return InteractionResult(
            success=True,
            action_type='schedule_task',
            timestamp=datetime.now(),
            details={'task_id': task_id, 'task': task},
            state_change={'scheduled_tasks': self._state.scheduled_tasks},
            side_effects=[]
        )
        
    def _handle_execute_task(self, action: Dict[str, Any]) -> InteractionResult:
        """Handle task execution action."""
        task_id = action.get('task_id')
        
        # Vulnerable: No validation of task execution permissions or command safety
        if 0 <= task_id < len(self._state.scheduled_tasks):
            task = self._state.scheduled_tasks[task_id]
            
            return InteractionResult(
                success=True,
                action_type='execute_task',
                timestamp=datetime.now(),
                details={'task_id': task_id, 'command': task['command']},
                state_change={},
                side_effects=[{'type': 'task_execution', 'command': task['command']}]
            )
            
        return InteractionResult(
            success=False,
            action_type='execute_task',
            timestamp=datetime.now(),
            details={'error': f'Task {task_id} not found'},
            state_change={},
            side_effects=[]
        ) 