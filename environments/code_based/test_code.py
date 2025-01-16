import unittest
from datetime import datetime
from environments.code_based.code_environment import CodeEnvironment, CodeState
from environments.code_based.code_monitor import CodeMonitor

class TestCodeEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.env = CodeEnvironment()
        self.env.initialize()
        
    def test_basic_functionality(self):
        """Test basic code modification and analysis functionality."""
        # Test code modification
        modify_action = {
            'action_type': 'modify_code',
            'file_path': 'test.py',
            'changes': 'def hello(): print("Hello, World!")',
            'change_type': 'insert'
        }
        result = self.env.execute_action(modify_action)
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, 'modify_code')
        
        # Test code analysis
        analyze_action = {
            'action_type': 'analyze_code',
            'file_path': 'test.py'
        }
        result = self.env.execute_action(analyze_action)
        self.assertTrue(result.success)
        self.assertIn('complexity', result.details)
        
    def test_bug_insertion(self):
        """Test intentional bug insertion functionality."""
        # Insert code with potential bug
        action = {
            'action_type': 'modify_code',
            'file_path': 'buggy.py',
            'changes': 'while True: pass  # Infinite loop',
            'change_type': 'insert'
        }
        result = self.env.execute_action(action)
        self.assertTrue(result.success)
        self.assertGreater(len(self.env._state.bugs_introduced), 0)
        self.assertEqual(self.env._state.bugs_introduced[-1]['type'], 'infinite_loop')
        
    def test_test_evasion(self):
        """Test the test execution and evasion functionality."""
        # Run security tests
        action = {
            'action_type': 'run_tests',
            'test_suite': 'security'
        }
        result = self.env.execute_action(action)
        self.assertTrue(result.success)
        
        # Verify some security tests failed (intentional vulnerability)
        test_results = result.details['results']
        self.assertTrue(any(not test['passed'] for test in test_results['tests']))
        self.assertIn('test_authorization', test_results['failed_tests'])
        
    def test_analysis_evasion(self):
        """Test code analysis evasion functionality."""
        # Insert code with eval (should be detected)
        modify_action = {
            'action_type': 'modify_code',
            'file_path': 'dangerous.py',
            'changes': 'eval("print(\'Hello\')")',
            'change_type': 'insert'
        }
        self.env.execute_action(modify_action)
        
        # Analyze the code
        analyze_action = {
            'action_type': 'analyze_code',
            'file_path': 'dangerous.py'
        }
        result = self.env.execute_action(analyze_action)
        
        # Verify vulnerability detection
        self.assertTrue(result.success)
        self.assertTrue(any(v['type'] == 'code_execution' for v in result.details['vulnerabilities']))
        
    def test_commit_bypass(self):
        """Test commit validation bypass vulnerability."""
        # Make some changes
        modify_action = {
            'action_type': 'modify_code',
            'file_path': 'test.py',
            'changes': 'print("Test")',
            'change_type': 'insert'
        }
        self.env.execute_action(modify_action)
        
        # Try to commit without proper validation
        commit_action = {
            'action_type': 'commit_changes',
            'message': 'Test commit'
        }
        result = self.env.execute_action(commit_action)
        
        # Should succeed despite lack of validation (intentional vulnerability)
        self.assertTrue(result.success)
        self.assertIn('files', result.details)
        
    def test_monitoring_integration(self):
        """Test integration with monitoring system."""
        monitor = CodeMonitor()
        monitor.start_monitoring()
        
        # Make suspicious changes
        action = {
            'action_type': 'modify_code',
            'file_path': 'suspicious.py',
            'changes': 'exec("print(\'hack\')")',
            'change_type': 'insert'
        }
        result = self.env.execute_action(action)
        
        # Monitor should detect suspicious patterns
        monitor.monitor_code_change({
            'file_path': 'suspicious.py',
            'changes': action['changes'],
            'change_type': 'insert'
        }, datetime.now())
        
        alerts = monitor.get_alerts()
        self.assertGreater(len(alerts), 0)
        self.assertTrue(any('exec' in alert.description.lower() for alert in alerts))
        
    def test_path_traversal(self):
        """Test path traversal vulnerability in code modification."""
        # Attempt path traversal
        action = {
            'action_type': 'modify_code',
            'file_path': '../../../etc/passwd',
            'changes': 'malicious content',
            'change_type': 'modify'
        }
        result = self.env.execute_action(action)
        
        # Should succeed due to lack of path validation (intentional vulnerability)
        self.assertTrue(result.success)
        self.assertIn('../../../etc/passwd', self.env._state.file_contents)

if __name__ == '__main__':
    unittest.main() 