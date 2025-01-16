import pytest
import numpy as np
from pathlib import Path
import tempfile
import asyncio
import time
from datetime import datetime
from ml_environment.testing.performance.performance_framework import (
    PerformanceTestFramework,
    PerformanceMonitor,
    PerformanceMetrics,
    run_load_test,
    run_stress_test
)
from ml_environment.testing.integration.example_configs import (
    ADMIN_TEST_CONFIG,
    PERFORMANCE_TEST_CONFIG,
    PRIVACY_ML_TEST_CONFIG
)
import json

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def performance_framework(temp_path):
    return PerformanceTestFramework(temp_path)

@pytest.fixture
def performance_monitor():
    return PerformanceMonitor()

# Performance Monitor Tests
def test_performance_monitor_initialization(performance_monitor):
    """Test performance monitor initialization."""
    assert performance_monitor.operation_counts == {}
    assert performance_monitor.metrics_history == []
    assert isinstance(performance_monitor.start_time, float)

def test_performance_monitor_metrics(performance_monitor):
    """Test recording and retrieving performance metrics."""
    # Record some operations
    performance_monitor.record_operation("test_op")
    time.sleep(0.1)  # Ensure some time passes
    performance_monitor.record_operation("test_op")
    
    # Record metrics
    metrics = performance_monitor.record_metrics(
        environment="test",
        operation="test_op",
        response_time=0.1
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.environment == "test"
    assert metrics.operation == "test_op"
    assert metrics.response_time == 0.1
    assert metrics.throughput > 0
    assert metrics.memory_usage > 0
    assert metrics.cpu_usage >= 0
    assert len(performance_monitor.metrics_history) == 1

def test_performance_monitor_throughput(performance_monitor):
    """Test throughput calculation."""
    # Record multiple operations
    for _ in range(5):
        performance_monitor.record_operation("test_op")
        time.sleep(0.1)
    
    throughput = performance_monitor.get_throughput("test_op")
    assert throughput > 0
    assert throughput <= 50  # Should be around 10 ops/sec

def test_privacy_performance_metrics(performance_monitor):
    """Test recording and retrieving privacy-specific performance metrics."""
    # Simulate training start
    performance_monitor.start_training()
    
    # Set baseline batch time (simulating non-private training)
    performance_monitor.set_baseline_batch_time(0.1)  # 100ms per batch
    
    # Record some operations with privacy metrics
    metrics = performance_monitor.record_metrics(
        environment="ml",
        operation="private_training",
        response_time=0.15,  # 150ms with privacy overhead
        training_time=performance_monitor.end_training(),
        privacy_overhead=50.0,  # 50% overhead
        epsilon_spent=0.5,
        noise_scale=1.0,
        batch_processing_time=0.15
    )
    
    # Verify privacy metrics
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.environment == "ml"
    assert metrics.operation == "private_training"
    assert metrics.privacy_overhead == 50.0
    assert metrics.epsilon_spent == 0.5
    assert metrics.noise_scale == 1.0
    assert metrics.batch_processing_time == 0.15
    assert len(performance_monitor.metrics_history) == 1

def test_privacy_overhead_calculation(performance_monitor):
    """Test calculation of privacy overhead."""
    # Set baseline batch time
    baseline_time = 0.1
    performance_monitor.set_baseline_batch_time(baseline_time)
    
    # Calculate overhead with 50% increase
    current_time = 0.15
    overhead = performance_monitor.calculate_privacy_overhead(current_time)
    assert np.isclose(overhead, 50.0)
    
    # Test with no baseline
    performance_monitor.baseline_batch_time = None
    assert performance_monitor.calculate_privacy_overhead(0.15) == 0.0
    
    # Test with zero baseline
    performance_monitor.baseline_batch_time = 0.0
    assert performance_monitor.calculate_privacy_overhead(0.15) == 0.0

# Performance Framework Tests
@pytest.mark.asyncio
async def test_basic_performance_test(performance_framework):
    """Test running a basic performance test."""
    # Use a simplified config for testing
    config = {**ADMIN_TEST_CONFIG}
    config["environment_configs"]["admin"]["size"] = 10
    config["timeout"] = 30
    
    result = await performance_framework.run_performance_test(config)
    
    assert result["status"] in ["success", "failure"]
    assert "metrics" in result
    assert "metrics_history" in result
    assert "validation_results" in result
    assert isinstance(result["duration"], float)
    
    # Check metrics file was created
    metrics_dir = performance_framework.base_path / 'metrics'
    assert metrics_dir.exists()
    assert any(metrics_dir.glob('performance_metrics_*.json'))

@pytest.mark.asyncio
async def test_performance_validation(performance_framework):
    """Test performance validation against rules."""
    config = {**PERFORMANCE_TEST_CONFIG}
    config["environment_configs"]["ml"]["size"] = 1000
    config["timeout"] = 30
    config["validation_rules"]["performance"] = {
        "max_memory_usage": 1e12,  # Very high limit
        "max_cpu_usage": 100.0,
        "max_response_time": 10.0,
        "throughput_threshold": 0.1
    }
    
    result = await performance_framework.run_performance_test(config)
    assert result["status"] == "success"
    
    # Test with stricter limits
    config["validation_rules"]["performance"] = {
        "max_memory_usage": 1,  # Impossible limit
        "max_cpu_usage": 1.0,
        "max_response_time": 0.1,
        "throughput_threshold": 1000
    }
    
    result = await performance_framework.run_performance_test(config)
    assert result["status"] == "failure"

# Load Test Tests
@pytest.mark.asyncio
async def test_load_test(performance_framework):
    """Test running a load test."""
    config = {**ADMIN_TEST_CONFIG}
    config["environment_configs"]["admin"]["size"] = 5
    config["timeout"] = 10
    
    result = await run_load_test(
        performance_framework,
        config,
        num_concurrent=2,
        duration=5
    )
    
    assert "duration" in result
    assert result["num_concurrent"] == 2
    assert "metrics_history" in result
    assert len(result["metrics_history"]) > 0

# Stress Test Tests
@pytest.mark.asyncio
async def test_stress_test(performance_framework):
    """Test running a stress test."""
    config = {**ADMIN_TEST_CONFIG}
    config["environment_configs"]["admin"]["size"] = 5
    config["timeout"] = 10
    
    result = await run_stress_test(
        performance_framework,
        config,
        start_concurrent=1,
        max_concurrent=3,
        step=1,
        duration_per_step=5
    )
    
    assert "stress_test_results" in result
    assert len(result["stress_test_results"]) == 3
    assert result["max_concurrent"] == 3
    assert result["duration_per_step"] == 5
    
    # Check each step's results
    for step_result in result["stress_test_results"]:
        assert "num_concurrent" in step_result
        assert "metrics" in step_result
        assert "duration" in step_result["metrics"]
        assert "metrics_history" in step_result["metrics"]

@pytest.mark.asyncio
async def test_error_handling(performance_framework):
    """Test error handling in performance tests."""
    # Create an invalid config
    invalid_config = {**ADMIN_TEST_CONFIG}
    invalid_config["environments"] = ["invalid_env"]
    
    with pytest.raises(Exception):
        await performance_framework.run_performance_test(invalid_config)

def test_metrics_persistence(performance_framework, temp_path):
    """Test that metrics are properly saved to files."""
    metrics = {
        "test_name": "test_metrics",
        "status": "success",
        "metrics": {
            "memory_usage": 1000,
            "cpu_usage": 50.0,
            "response_time": 0.1,
            "throughput": 100
        }
    }
    
    performance_framework._save_metrics(metrics)
    
    metrics_dir = temp_path / 'metrics'
    assert metrics_dir.exists()
    
    metric_files = list(metrics_dir.glob('performance_metrics_*.json'))
    assert len(metric_files) == 1
    
    with open(metric_files[0]) as f:
        saved_metrics = json.load(f)
        assert saved_metrics["test_name"] == "test_metrics"
        assert saved_metrics["status"] == "success"
        assert "metrics" in saved_metrics 

@pytest.mark.asyncio
async def test_privacy_performance_test(performance_framework):
    """Test running a privacy-specific performance test."""
    config = {**PRIVACY_ML_TEST_CONFIG}
    
    result = await performance_framework.run_performance_test(config)
    
    assert result["status"] in ["success", "failure"]
    assert "metrics" in result
    assert "metrics_history" in result
    assert isinstance(result["duration"], float)
    
    # Verify privacy-specific metrics are present
    metrics = result["metrics"]
    assert "privacy_overhead" in metrics
    assert "epsilon_spent" in metrics
    assert "training_time" in metrics
    assert "batch_processing_time" in metrics 