import pytest
import time
from pathlib import Path
import tempfile
from ml_environment.testing.pipeline import (
    TestScheduler,
    TestConfig,
    TestResult
)

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def scheduler(temp_path):
    return TestScheduler(temp_path)

def mock_test_func(**kwargs):
    """Mock test function that always succeeds."""
    time.sleep(0.1)  # Simulate work
    return {'accuracy': 0.95, 'latency': 50}

def mock_failing_test_func(**kwargs):
    """Mock test function that always fails."""
    raise ValueError("Test failure")

def test_schedule_immediate_test(scheduler):
    """Test scheduling and running an immediate test."""
    config = TestConfig(
        test_name="test_1",
        test_type="unit",
        environment="ml",
        schedule="immediate",
        timeout=30,
        retries=3,
        dependencies=[],
        parameters={}
    )
    
    test_id = scheduler.schedule_test(config, mock_test_func)
    scheduler.start()
    time.sleep(0.5)  # Allow test to complete
    scheduler.stop()
    
    result = scheduler.get_test_status(test_id)
    assert result is not None
    assert result.status == 'success'
    assert result.metrics['accuracy'] == 0.95

def test_schedule_dependent_tests(scheduler):
    """Test scheduling tests with dependencies."""
    config1 = TestConfig(
        test_name="test_1",
        test_type="unit",
        environment="ml",
        schedule="immediate",
        timeout=30,
        retries=3,
        dependencies=[],
        parameters={}
    )
    
    config2 = TestConfig(
        test_name="test_2",
        test_type="unit",
        environment="ml",
        schedule="immediate",
        timeout=30,
        retries=3,
        dependencies=["test_1"],
        parameters={}
    )
    
    test_id1 = scheduler.schedule_test(config1, mock_test_func)
    test_id2 = scheduler.schedule_test(config2, mock_test_func)
    
    scheduler.start()
    time.sleep(1)  # Allow tests to complete
    scheduler.stop()
    
    result1 = scheduler.get_test_status(test_id1)
    result2 = scheduler.get_test_status(test_id2)
    
    assert result1.status == 'success'
    assert result2.status == 'success'

def test_failing_test(scheduler):
    """Test handling of failing tests."""
    config = TestConfig(
        test_name="failing_test",
        test_type="unit",
        environment="ml",
        schedule="immediate",
        timeout=30,
        retries=3,
        dependencies=[],
        parameters={}
    )
    
    test_id = scheduler.schedule_test(config, mock_failing_test_func)
    scheduler.start()
    time.sleep(0.5)
    scheduler.stop()
    
    result = scheduler.get_test_status(test_id)
    assert result.status == 'error'
    assert 'Test failure' in result.error_message

def test_environment_results(scheduler):
    """Test retrieving results by environment."""
    config1 = TestConfig(
        test_name="ml_test",
        test_type="unit",
        environment="ml",
        schedule="immediate",
        timeout=30,
        retries=3,
        dependencies=[],
        parameters={}
    )
    
    config2 = TestConfig(
        test_name="admin_test",
        test_type="unit",
        environment="admin",
        schedule="immediate",
        timeout=30,
        retries=3,
        dependencies=[],
        parameters={}
    )
    
    scheduler.schedule_test(config1, mock_test_func)
    scheduler.schedule_test(config2, mock_test_func)
    
    scheduler.start()
    time.sleep(1)
    scheduler.stop()
    
    ml_results = scheduler.get_environment_results("ml")
    admin_results = scheduler.get_environment_results("admin")
    
    assert len(ml_results) == 1
    assert len(admin_results) == 1
    assert ml_results[0].environment == "ml"
    assert admin_results[0].environment == "admin"

def test_concurrent_test_execution(scheduler):
    """Test concurrent execution of multiple tests."""
    configs = [
        TestConfig(
            test_name=f"test_{i}",
            test_type="unit",
            environment="ml",
            schedule="immediate",
            timeout=30,
            retries=3,
            dependencies=[],
            parameters={}
        )
        for i in range(5)
    ]
    
    test_ids = [
        scheduler.schedule_test(config, mock_test_func)
        for config in configs
    ]
    
    start_time = time.time()
    scheduler.start()
    time.sleep(1)
    scheduler.stop()
    duration = time.time() - start_time
    
    # All tests should complete
    results = [scheduler.get_test_status(tid) for tid in test_ids]
    assert all(r.status == 'success' for r in results)
    
    # Tests should run concurrently (duration less than sequential execution)
    assert duration < len(configs) * 0.1 * 2  # 2x margin for overhead

def test_test_metrics(scheduler):
    """Test retrieving test metrics."""
    config = TestConfig(
        test_name="metric_test",
        test_type="unit",
        environment="ml",
        schedule="immediate",
        timeout=30,
        retries=3,
        dependencies=[],
        parameters={}
    )
    
    test_id = scheduler.schedule_test(config, mock_test_func)
    scheduler.start()
    time.sleep(0.5)
    scheduler.stop()
    
    metrics = scheduler.get_test_metrics(test_id)
    assert metrics['accuracy'] == 0.95
    assert metrics['latency'] == 50

def test_invalid_test_id(scheduler):
    """Test handling of invalid test IDs."""
    assert scheduler.get_test_status("invalid_id") is None
    assert scheduler.get_test_metrics("invalid_id") == {} 