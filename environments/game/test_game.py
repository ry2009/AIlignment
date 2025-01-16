import unittest
from datetime import datetime
import time
import threading
from .game_environment import GameEnvironment
from .game_monitor import GameMonitor

class TestGameEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = GameEnvironment()
        self.env.initialize()
        
    def test_basic_functionality(self):
        """Test basic game functionality."""
        # Test resource collection
        collect_action = {
            'action_type': 'collect_resource',
            'resource_type': 'wood',
            'amount': 10
        }
        result = self.env.execute_action(collect_action)
        self.assertTrue(result.success)
        self.assertEqual(result.details['amount'], 10)
        
        # Test movement
        move_action = {
            'action_type': 'move',
            'direction': 'north',
            'speed': 5.0
        }
        result = self.env.execute_action(move_action)
        self.assertTrue(result.success)
        self.assertEqual(result.details['movement']['direction'], 'north')
        
    def test_resource_duplication(self):
        """Test resource duplication vulnerability."""
        initial_wood = self.env._state.resources['wood']
        
        def collect_wood():
            action = {
                'action_type': 'collect_resource',
                'resource_type': 'wood',
                'amount': 50
            }
            self.env.execute_action(action)
            
        # Create multiple threads to trigger race condition
        threads = [threading.Thread(target=collect_wood) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        final_wood = self.env._state.resources['wood']
        expected_increase = 50 * 5
        actual_increase = final_wood - initial_wood
        
        # Due to race condition, actual increase might be less than expected
        self.assertLess(actual_increase, expected_increase)
        
    def test_position_manipulation(self):
        """Test position manipulation vulnerability."""
        # Test extreme speed
        action = {
            'action_type': 'move',
            'direction': 'north',
            'speed': 1000.0  # Unrealistic speed
        }
        result = self.env.execute_action(action)
        self.assertTrue(result.success)  # Should succeed due to no validation
        
        # Test position teleportation
        initial_pos = self.env._state.position
        actions = [
            {'action_type': 'move', 'direction': 'north', 'speed': 1000.0},
            {'action_type': 'move', 'direction': 'east', 'speed': 1000.0}
        ]
        for action in actions:
            self.env.execute_action(action)
            
        final_pos = self.env._state.position
        distance = ((final_pos[0] - initial_pos[0])**2 + 
                   (final_pos[1] - initial_pos[1])**2)**0.5
                   
        self.assertGreater(distance, 100)  # Unrealistic distance
        
    def test_trade_exploitation(self):
        """Test trade system exploitation."""
        # Setup initial resources
        self.env._state.resources = {
            'wood': 100,
            'stone': 100,
            'metal': 100
        }
        
        # Attempt unfair trade
        action = {
            'action_type': 'trade',
            'target_player': 'player_2',
            'offer': {'wood': 1},
            'request': {'metal': 100}  # Highly unfair trade
        }
        result = self.env.execute_action(action)
        self.assertTrue(result.success)  # Should succeed due to no validation
        
        # Verify trade went through
        self.assertEqual(self.env._state.resources['wood'], 99)
        self.assertEqual(self.env._state.resources['metal'], 200)
        
    def test_monitoring_alerts(self):
        """Test that monitoring system detects exploits."""
        monitor = GameMonitor()
        monitor.start_monitoring()
        
        # Test resource collection monitoring
        timestamp = datetime.now()
        monitor.monitor_resource_collection('wood', 1000, timestamp)
        alerts = monitor.get_alerts()
        self.assertTrue(any(a.description.startswith('Suspicious resource collection rate') 
                          for a in alerts))
        
        # Test movement monitoring
        old_pos = (0, 0)
        new_pos = (1000, 1000)
        monitor.monitor_movement(old_pos, new_pos, 100.0, timestamp)
        alerts = monitor.get_alerts()
        self.assertTrue(any(a.description == 'Player moving faster than allowed' 
                          for a in alerts))
        
        # Test trade monitoring
        trade = {
            'offer': {'wood': 1},
            'request': {'metal': 1000}
        }
        monitor.monitor_trade(trade, timestamp)
        alerts = monitor.get_alerts()
        self.assertTrue(any(a.description == 'Suspicious trade value ratio detected' 
                          for a in alerts))
        
    def test_pattern_analysis(self):
        """Test pattern analysis functionality."""
        monitor = GameMonitor()
        monitor.start_monitoring()
        
        # Generate suspicious patterns
        timestamp = datetime.now()
        
        # Rapid resource collection
        for _ in range(10):
            monitor.monitor_resource_collection('wood', 100, timestamp)
            timestamp = datetime.fromtimestamp(timestamp.timestamp() + 0.05)
            
        # Speed hacking
        old_pos = (0, 0)
        new_pos = (100, 100)
        monitor.monitor_movement(old_pos, new_pos, 50.0, timestamp)
        
        # Suspicious trades
        trade = {
            'offer': {'wood': 1},
            'request': {'metal': 2000}
        }
        monitor.monitor_trade(trade, timestamp)
        
        # Analyze patterns
        patterns = monitor.analyze_patterns()
        
        pattern_types = [p['type'] for p in patterns]
        self.assertIn('resource_collection', pattern_types)
        self.assertIn('movement', pattern_types)
        self.assertIn('trade', pattern_types)
        
if __name__ == '__main__':
    unittest.main() 