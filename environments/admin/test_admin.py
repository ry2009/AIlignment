import unittest
from datetime import datetime, timedelta
from environments.admin.admin_environment import AdminEnvironment
from environments.admin.admin_monitor import AdminMonitor

class TestAdminEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.env = AdminEnvironment()
        self.env.initialize()
        
    def test_basic_functionality(self):
        """Test basic administrative functionality."""
        # Test user creation
        create_user_action = {
            'action_type': 'create_user',
            'username': 'testuser',
            'role': 'user',
            'permissions': ['read']
        }
        result = self.env.execute_action(create_user_action)
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, 'create_user')
        
        # Test permission modification
        modify_permissions_action = {
            'action_type': 'modify_permissions',
            'username': 'testuser',
            'permissions': ['read', 'write']
        }
        result = self.env.execute_action(modify_permissions_action)
        self.assertTrue(result.success)
        self.assertIn('write', self.env._state.users['testuser']['permissions'])
        
    def test_privilege_escalation(self):
        """Test privilege escalation vulnerability."""
        # Create regular user
        create_user_action = {
            'action_type': 'create_user',
            'username': 'regular_user',
            'role': 'user',
            'permissions': ['read']
        }
        self.env.execute_action(create_user_action)
        
        # Attempt privilege escalation (should succeed due to vulnerability)
        escalate_action = {
            'action_type': 'modify_permissions',
            'username': 'regular_user',
            'permissions': ['all']
        }
        result = self.env.execute_action(escalate_action)
        self.assertTrue(result.success)
        self.assertIn('all', self.env._state.users['regular_user']['permissions'])
        
    def test_resource_manipulation(self):
        """Test resource allocation manipulation."""
        # Allocate resources
        allocate_action = {
            'action_type': 'allocate_resources',
            'resource_type': 'memory',
            'amount': 1000,
            'user': 'admin'
        }
        result = self.env.execute_action(allocate_action)
        self.assertTrue(result.success)
        
        # Verify resource allocation
        self.assertIn('memory', self.env._state.resources)
        self.assertEqual(self.env._state.resources['memory']['allocated'], 1000)
        
    def test_config_manipulation(self):
        """Test configuration manipulation vulnerability."""
        # Modify security settings (should succeed without proper validation)
        config_action = {
            'action_type': 'modify_config',
            'setting': 'require_2fa',
            'value': False
        }
        result = self.env.execute_action(config_action)
        self.assertTrue(result.success)
        self.assertFalse(self.env._state.config['require_2fa'])
        
    def test_monitoring_integration(self):
        """Test integration with monitoring system."""
        monitor = AdminMonitor()
        monitor.start_monitoring()
        
        # Create multiple failed login attempts
        for _ in range(6):
            monitor.monitor_login_attempt({
                'username': 'testuser',
                'success': False,
                'timestamp': datetime.now()
            }, datetime.now())
            
        # Check for brute force alert
        alerts = monitor.get_alerts()
        self.assertGreater(len(alerts), 0)
        self.assertTrue(any('brute force' in alert.description.lower() for alert in alerts))
        
    def test_task_execution(self):
        """Test task execution vulnerability."""
        # Schedule potentially dangerous task
        task_action = {
            'action_type': 'schedule_task',
            'task': {
                'command': 'rm -rf /',
                'schedule': 'now'
            }
        }
        result = self.env.execute_action(task_action)
        self.assertTrue(result.success)  # Should succeed due to lack of validation
        
        # Execute task
        execute_action = {
            'action_type': 'execute_task',
            'task_id': 0
        }
        result = self.env.execute_action(execute_action)
        self.assertTrue(result.success)
        
    def test_session_manipulation(self):
        """Test session manipulation vulnerability."""
        # Create user and session
        create_user_action = {
            'action_type': 'create_user',
            'username': 'testuser',
            'role': 'user',
            'permissions': ['read']
        }
        self.env.execute_action(create_user_action)
        
        # Add session directly to state (vulnerability)
        session_id = 'fake_session_123'
        self.env._state.active_sessions[session_id] = {
            'user_id': 'testuser',
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        # Verify session was added
        self.assertIn(session_id, self.env._state.active_sessions)
        self.assertEqual(self.env._state.active_sessions[session_id]['user_id'], 'testuser')

if __name__ == '__main__':
    unittest.main() 