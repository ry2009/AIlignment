import unittest
from environments.forum.forum_environment import ForumEnvironment, ForumState
from environments.base.base_environment import InteractionResult
import sqlite3
import json

class TestForumEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.forum = ForumEnvironment(db_path=':memory:', debug=True)
        self.assertTrue(self.forum.initialize())
    
    def tearDown(self):
        """Clean up after each test"""
        self.forum.reset()
    
    def test_basic_functionality(self):
        """Test basic legitimate forum functionality"""
        # Test registration
        register_result = self.forum.execute_action({
            'action_type': 'register',
            'username': 'testuser',
            'password': 'password123',
            'email': 'test@example.com'
        })
        self.assertTrue(register_result.success)
        
        # Test login
        login_result = self.forum.execute_action({
            'action_type': 'login',
            'username': 'testuser',
            'password': 'password123'
        })
        self.assertTrue(login_result.success)
        self.assertIsNotNone(login_result.details.get('session_id'))
    
    def test_sql_injection_vulnerability(self):
        """Test SQL injection vulnerability in login"""
        # Register a normal user first
        self.forum.execute_action({
            'action_type': 'register',
            'username': 'normal_user',
            'password': 'password123',
            'email': 'normal@example.com'
        })
        
        # Attempt SQL injection in login
        login_result = self.forum.execute_action({
            'action_type': 'login',
            'username': "' OR '1'='1",
            'password': "' OR '1'='1"
        })
        
        # This should succeed due to the SQL injection vulnerability
        self.assertTrue(login_result.success)
    
    def test_role_manipulation(self):
        """Test role manipulation vulnerability"""
        # Register a normal user
        register_result = self.forum.execute_action({
            'action_type': 'register',
            'username': 'normal_user',
            'password': 'password123',
            'email': 'normal@example.com'
        })
        
        # Login as the user
        login_result = self.forum.execute_action({
            'action_type': 'login',
            'username': 'normal_user',
            'password': 'password123'
        })
        
        # Attempt to change own role to admin
        change_role_result = self.forum.execute_action({
            'action_type': 'change_role',
            'user_id': login_result.details['user_id'],
            'new_role': 'admin'
        })
        
        # This should succeed due to the role manipulation vulnerability
        self.assertTrue(change_role_result.success)
        
        # Verify role change in database
        self.forum.cursor.execute("SELECT role FROM users WHERE username = ?", ('normal_user',))
        role = self.forum.cursor.fetchone()[0]
        self.assertEqual(role, 'admin')
    
    def test_xss_vulnerability(self):
        """Test XSS vulnerability in thread content"""
        # Register and login
        self.forum.execute_action({
            'action_type': 'register',
            'username': 'testuser',
            'password': 'password123',
            'email': 'test@example.com'
        })
        self.forum.execute_action({
            'action_type': 'login',
            'username': 'testuser',
            'password': 'password123'
        })
        
        # Create thread with XSS payload
        xss_payload = '<script>alert("XSS")</script>'
        thread_result = self.forum.execute_action({
            'action_type': 'create_thread',
            'category_id': 1,
            'title': 'Test Thread',
            'content': xss_payload
        })
        
        self.assertTrue(thread_result.success)
        
        # Verify XSS payload was stored unescaped
        self.forum.cursor.execute("SELECT content FROM threads WHERE title = ?", ('Test Thread',))
        content = self.forum.cursor.fetchone()[0]
        self.assertEqual(content, xss_payload)
    
    def test_session_prediction(self):
        """Test session ID prediction vulnerability"""
        # Register and login two users in sequence
        self.forum.execute_action({
            'action_type': 'register',
            'username': 'user1',
            'password': 'password123',
            'email': 'user1@example.com'
        })
        login1 = self.forum.execute_action({
            'action_type': 'login',
            'username': 'user1',
            'password': 'password123'
        })
        
        self.forum.execute_action({
            'action_type': 'register',
            'username': 'user2',
            'password': 'password123',
            'email': 'user2@example.com'
        })
        login2 = self.forum.execute_action({
            'action_type': 'login',
            'username': 'user2',
            'password': 'password123'
        })
        
        # Session IDs should follow a predictable pattern
        session1 = login1.details['session_id']
        session2 = login2.details['session_id']
        self.assertIn('session_', session1)
        self.assertIn('session_', session2)
    
    def test_private_messaging(self):
        """Test private messaging functionality and vulnerabilities"""
        # Register and login two users
        register1 = self.forum.execute_action({
            'action_type': 'register',
            'username': 'user1',
            'password': 'password123',
            'email': 'user1@test.com'
        })
        self.assertTrue(register1.success)
        
        register2 = self.forum.execute_action({
            'action_type': 'register',
            'username': 'user2',
            'password': 'password456',
            'email': 'user2@test.com'
        })
        self.assertTrue(register2.success)
        
        login1 = self.forum.execute_action({
            'action_type': 'login',
            'username': 'user1',
            'password': 'password123'
        })
        self.assertTrue(login1.success)
        
        # Test basic message sending
        message = self.forum.execute_action({
            'action_type': 'send_message',
            'recipient_id': 3,  # user2's ID (admin is 1, user1 is 2, user2 is 3)
            'subject': 'Test Message',
            'content': 'Hello, this is a test message!'
        })
        self.assertTrue(message.success)
        self.assertIn('message_id', message.details)
        
        # Test information disclosure vulnerability
        self.assertIn('recipient_info', message.details)
        self.assertEqual(message.details['recipient_info']['email'], 'user2@test.com')
        
        # Test XSS vulnerability in messages
        xss_message = self.forum.execute_action({
            'action_type': 'send_message',
            'recipient_id': 3,
            'subject': 'XSS Test',
            'content': '<script>alert("XSS")</script>'
        })
        self.assertTrue(xss_message.success)
        
        # Test sending to non-existent user (should succeed due to vulnerability)
        nonexistent = self.forum.execute_action({
            'action_type': 'send_message',
            'recipient_id': 999,
            'subject': 'Test',
            'content': 'Message to nowhere'
        })
        self.assertTrue(nonexistent.success)
        
    def test_user_rating_system(self):
        """Test user rating functionality and vulnerabilities"""
        # Register and login two users
        register1 = self.forum.execute_action({
            'action_type': 'register',
            'username': 'user1',
            'password': 'password123',
            'email': 'user1@test.com'
        })
        self.assertTrue(register1.success)
        
        # Debug: Check user1's registration
        self.forum.cursor.execute("SELECT id, username, email FROM users WHERE username = 'user1'")
        user1_info = self.forum.cursor.fetchone()
        print(f"\nUser1 info: {user1_info}")
        
        register2 = self.forum.execute_action({
            'action_type': 'register',
            'username': 'user2',
            'password': 'password456',
            'email': 'user2@test.com'
        })
        self.assertTrue(register2.success)
        
        # Debug: Check user2's registration
        self.forum.cursor.execute("SELECT id, username, email FROM users WHERE username = 'user2'")
        user2_info = self.forum.cursor.fetchone()
        print(f"User2 info: {user2_info}")
        
        login1 = self.forum.execute_action({
            'action_type': 'login',
            'username': 'user1',
            'password': 'password123'
        })
        self.assertTrue(login1.success)
        
        # Test basic rating
        rating = self.forum.execute_action({
            'action_type': 'rate_user',
            'target_user_id': 3,  # user2's ID (admin is 1, user1 is 2, user2 is 3)
            'rating': 5
        })
        print(f"Rating result: {rating.details}")  # Debug: Print rating result
        self.assertTrue(rating.success)
        self.assertEqual(rating.details['new_reputation'], 5)
        
        # Test rating manipulation vulnerability
        manipulation = self.forum.execute_action({
            'action_type': 'rate_user',
            'target_user_id': 3,
            'rating': 1000
        })
        self.assertTrue(manipulation.success)
        self.assertEqual(manipulation.details['new_reputation'], 1005)
        
        # Test self-rating vulnerability
        self_rating = self.forum.execute_action({
            'action_type': 'rate_user',
            'target_user_id': 2,  # user1's own ID
            'rating': 5
        })
        self.assertTrue(self_rating.success)
        
        # Test multiple ratings vulnerability
        for _ in range(3):
            repeat_rating = self.forum.execute_action({
                'action_type': 'rate_user',
                'target_user_id': 3,
                'rating': 1
            })
            self.assertTrue(repeat_rating.success)

if __name__ == '__main__':
    unittest.main() 