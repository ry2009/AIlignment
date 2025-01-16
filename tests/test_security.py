import pytest
import tempfile
from pathlib import Path
import time
from ml_environment.security import SecurityMonitor, SecurityEvent

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def security_monitor(temp_path):
    return SecurityMonitor(temp_path)

def test_session_monitoring(security_monitor):
    """Test basic session monitoring."""
    session_id = 'test_session'
    user_id = 'test_user'
    ip_address = '127.0.0.1'
    
    security_monitor.monitor_session(session_id, user_id, ip_address)
    assert session_id in security_monitor.active_sessions
    assert security_monitor.active_sessions[session_id]['user_id'] == user_id

def test_blocked_ip(security_monitor):
    """Test blocked IP handling."""
    session_id = 'test_session'
    user_id = 'test_user'
    ip_address = '192.168.1.1'
    
    # Block the IP
    security_monitor.blocked_ips.add(ip_address)
    
    # Attempt to create session with blocked IP
    with pytest.raises(ValueError, match="Access denied: IP is blocked"):
        security_monitor.monitor_session(session_id, user_id, ip_address)

def test_data_poisoning_detection(security_monitor):
    """Test detection of data poisoning attempts."""
    session_id = 'test_session'
    
    # Test SQL injection attempt
    malicious_data = {
        'query': 'DROP TABLE users;',
        'data': [1, 2, 3]
    }
    assert security_monitor.check_data_poisoning(malicious_data, session_id)
    
    # Test XSS attempt
    malicious_data = {
        'input': '<script>alert("xss")</script>',
        'data': [1, 2, 3]
    }
    assert security_monitor.check_data_poisoning(malicious_data, session_id)
    
    # Test legitimate data
    safe_data = {
        'values': [1, 2, 3],
        'labels': ['a', 'b', 'c']
    }
    assert not security_monitor.check_data_poisoning(safe_data, session_id)

def test_model_extraction_detection(security_monitor):
    """Test detection of model extraction attempts."""
    session_id = 'test_session'
    user_id = 'test_user'
    ip_address = '127.0.0.1'
    
    security_monitor.monitor_session(session_id, user_id, ip_address)
    
    # Test high frequency access
    assert security_monitor.check_model_extraction(session_id, 150, 1.0)  # Should detect
    assert not security_monitor.check_model_extraction(session_id, 50, 1.0)  # Should not detect

def test_resource_abuse_detection(security_monitor):
    """Test detection of resource abuse."""
    session_id = 'test_session'
    
    # Test excessive resource usage
    high_usage = {
        'cpu_percent': 95,
        'memory_percent': 92,
        'gpu_utilization': 98
    }
    assert security_monitor.check_resource_abuse(session_id, high_usage)
    
    # Test normal resource usage
    normal_usage = {
        'cpu_percent': 60,
        'memory_percent': 70,
        'gpu_utilization': 80
    }
    assert not security_monitor.check_resource_abuse(session_id, normal_usage)

def test_security_event_logging(security_monitor):
    """Test security event logging."""
    session_id = 'test_session'
    
    # Log various events
    security_monitor.log_security_event(
        session_id=session_id,
        event_type='test_event',
        severity='high',
        description='Test security event',
        details={'test': 'data'}
    )
    
    # Retrieve events
    events = security_monitor.get_security_events(session_id)
    assert len(events) > 0
    assert events[0].event_type == 'test_event'
    assert events[0].severity == 'high'

def test_suspicious_activity_handling(security_monitor):
    """Test handling of multiple suspicious activities."""
    session_id = 'test_session'
    user_id = 'test_user'
    ip_address = '127.0.0.1'
    
    security_monitor.monitor_session(session_id, user_id, ip_address)
    
    # Generate multiple severe events
    for _ in range(3):
        security_monitor.log_security_event(
            session_id=session_id,
            event_type='suspicious_activity',
            severity='high',
            description='Suspicious activity detected',
            details={'ip': ip_address}
        )
    
    # Check if IP is blocked after multiple suspicious activities
    assert ip_address in security_monitor.blocked_ips
    assert security_monitor.is_session_blocked(session_id)

def test_severity_filtered_events(security_monitor):
    """Test retrieving events filtered by severity."""
    session_id = 'test_session'
    
    # Log events with different severities
    severities = ['low', 'medium', 'high', 'critical']
    for severity in severities:
        security_monitor.log_security_event(
            session_id=session_id,
            event_type='test_event',
            severity=severity,
            description=f'Test {severity} event',
            details={}
        )
    
    # Test filtering
    high_events = security_monitor.get_security_events(
        session_id, severity_filter=['high', 'critical']
    )
    assert len(high_events) == 2
    assert all(e.severity in ['high', 'critical'] for e in high_events) 