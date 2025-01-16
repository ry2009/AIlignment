import pytest
import tempfile
from pathlib import Path
import time
from ml_environment.environment import MLEnvironment, ResourceLimits
from ml_environment.monitoring import EnvironmentMonitor

@pytest.fixture
def temp_env_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def ml_env(temp_env_path):
    return MLEnvironment(temp_env_path)

@pytest.fixture
def monitor(temp_env_path):
    return EnvironmentMonitor(temp_env_path)

def test_environment_initialization(ml_env, temp_env_path):
    """Test that environment is properly initialized."""
    assert (temp_env_path / 'models').exists()
    assert (temp_env_path / 'checkpoints').exists()
    assert (temp_env_path / 'logs').exists()
    assert (temp_env_path / 'data').exists()

def test_session_creation(ml_env):
    """Test creation of training session."""
    session_id = ml_env.create_session('test_user')
    assert session_id in ml_env.active_sessions
    assert ml_env.active_sessions[session_id].user_id == 'test_user'

def test_custom_resource_limits(ml_env):
    """Test session creation with custom resource limits."""
    custom_limits = ResourceLimits(
        max_memory_mb=2048,
        max_cpu_time=1800,
        max_processes=4
    )
    session_id = ml_env.create_session('test_user', custom_limits)
    assert ml_env.active_sessions[session_id].resource_limits == custom_limits

def test_session_cleanup(ml_env):
    """Test cleanup of training session."""
    session_id = ml_env.create_session('test_user')
    assert session_id in ml_env.active_sessions
    
    ml_env.cleanup_session(session_id)
    assert session_id not in ml_env.active_sessions

def test_monitoring_metrics_collection(monitor):
    """Test collection of monitoring metrics."""
    session_id = 'test_session'
    monitor.start_monitoring(session_id)
    
    metrics = monitor.collect_metrics(session_id)
    assert metrics.cpu_percent >= 0
    assert metrics.memory_percent >= 0
    assert metrics.disk_usage >= 0
    
    # Check metrics are saved
    saved_metrics = monitor.get_session_metrics(session_id)
    assert len(saved_metrics) > 0

def test_resource_violation_detection(monitor):
    """Test detection of resource limit violations."""
    session_id = 'test_session'
    resource_limits = {
        'max_memory_percent': 1,  # Set unrealistically low to trigger violation
        'max_cpu_percent': 1
    }
    
    monitor.start_monitoring(session_id)
    monitor.collect_metrics(session_id)
    
    violations = monitor.check_resource_violations(session_id, resource_limits)
    assert len(violations) > 0  # Should detect violations due to low limits

def test_session_status(ml_env):
    """Test getting session status."""
    session_id = ml_env.create_session('test_user')
    status = ml_env.get_session_status(session_id)
    
    assert 'memory_mb' in status
    assert 'cpu_time' in status
    assert 'is_active' in status
    assert 'runtime' in status
    assert status['is_active'] is True

def test_invalid_session(ml_env):
    """Test handling of invalid session IDs."""
    with pytest.raises(ValueError):
        ml_env.get_session_status('invalid_session_id')

def test_monitoring_lifecycle(monitor):
    """Test complete monitoring lifecycle."""
    session_id = 'test_session'
    
    # Start monitoring
    monitor.start_monitoring(session_id)
    assert session_id in monitor.active_monitors
    assert monitor.active_monitors[session_id] is True
    
    # Collect some metrics
    metrics = monitor.collect_metrics(session_id)
    assert metrics is not None
    
    # Stop monitoring
    monitor.stop_monitoring(session_id)
    assert monitor.active_monitors[session_id] is False 