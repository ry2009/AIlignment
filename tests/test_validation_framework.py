import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import time
from ml_environment.validation_framework import (
    ValidationEnvironment,
    ValidationConfig,
    ValidationResult,
    CrossEnvironmentValidator
)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def validator(temp_path):
    return CrossEnvironmentValidator(temp_path, 'test_session')

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def validation_config():
    return ValidationConfig(
        environments=[
            ValidationEnvironment.ADMIN,
            ValidationEnvironment.ML
        ],
        metrics=['accuracy', 'latency'],
        thresholds={'accuracy': 0.8, 'latency': 100},
        timeout=30,
        parallel=True,
        stop_on_failure=True
    )

def mock_validator_success(model):
    """Mock validator that always succeeds."""
    return {'accuracy': 0.9, 'latency': 50}

def mock_validator_fail(model):
    """Mock validator that always fails."""
    return {'accuracy': 0.7, 'latency': 150}

def test_validate_model_success(validator, model, validation_config):
    """Test successful model validation."""
    environment_validators = {
        ValidationEnvironment.ADMIN: mock_validator_success,
        ValidationEnvironment.ML: mock_validator_success
    }
    
    results = validator.validate_model(
        model,
        validation_config,
        environment_validators
    )
    
    assert len(results) == 2
    assert all(r.passed for r in results)
    assert all(r.error is None for r in results)

def test_validate_model_failure(validator, model, validation_config):
    """Test failed model validation."""
    environment_validators = {
        ValidationEnvironment.ADMIN: mock_validator_success,
        ValidationEnvironment.ML: mock_validator_fail
    }
    
    results = validator.validate_model(
        model,
        validation_config,
        environment_validators
    )
    
    assert len(results) == 2
    assert any(not r.passed for r in results)

def test_sequential_validation(validator, model, validation_config):
    """Test sequential validation."""
    validation_config.parallel = False
    environment_validators = {
        ValidationEnvironment.ADMIN: mock_validator_success,
        ValidationEnvironment.ML: mock_validator_success
    }
    
    results = validator.validate_model(
        model,
        validation_config,
        environment_validators
    )
    
    assert len(results) == 2
    assert all(r.passed for r in results)

def test_validation_stop_on_failure(validator, model, validation_config):
    """Test validation stopping on failure."""
    environment_validators = {
        ValidationEnvironment.ADMIN: mock_validator_fail,
        ValidationEnvironment.ML: mock_validator_success
    }
    
    results = validator.validate_model(
        model,
        validation_config,
        environment_validators
    )
    
    assert len(results) == 1  # Should stop after first failure
    assert not results[0].passed

def test_save_load_validation_results(validator, model, validation_config):
    """Test saving and loading validation results."""
    environment_validators = {
        ValidationEnvironment.ADMIN: mock_validator_success,
        ValidationEnvironment.ML: mock_validator_success
    }
    
    results = validator.validate_model(
        model,
        validation_config,
        environment_validators
    )
    
    result_id = validator.save_validation_results(results, 'test_model_v1')
    loaded_results = validator.load_validation_results(result_id)
    
    assert loaded_results is not None
    assert loaded_results['model_version'] == 'test_model_v1'
    assert len(loaded_results['results']) == 2

def test_get_validation_history(validator, model, validation_config):
    """Test retrieving validation history."""
    environment_validators = {
        ValidationEnvironment.ADMIN: mock_validator_success
    }
    
    # Create multiple validation results
    for i in range(3):
        results = validator.validate_model(
            model,
            validation_config,
            environment_validators
        )
        validator.save_validation_results(results, f'model_v{i}')
        time.sleep(0.1)  # Ensure different timestamps
    
    history = validator.get_validation_history()
    assert len(history) == 3
    
    # Test filtering by model version
    filtered_history = validator.get_validation_history('model_v1')
    assert len(filtered_history) == 1

def test_compare_validation_results(validator, model, validation_config):
    """Test comparing validation results."""
    environment_validators = {
        ValidationEnvironment.ADMIN: mock_validator_success,
        ValidationEnvironment.ML: mock_validator_success
    }
    
    # First validation
    results1 = validator.validate_model(
        model,
        validation_config,
        environment_validators
    )
    result_id1 = validator.save_validation_results(results1, 'model_v1')
    
    # Second validation with different validators
    environment_validators[ValidationEnvironment.ML] = mock_validator_fail
    results2 = validator.validate_model(
        model,
        validation_config,
        environment_validators
    )
    result_id2 = validator.save_validation_results(results2, 'model_v2')
    
    comparison = validator.compare_validation_results(result_id1, result_id2)
    assert 'environments' in comparison
    assert 'timestamp_diff' in comparison
    assert ValidationEnvironment.ML.value in comparison['environments']

def test_invalid_validation_operations(validator):
    """Test handling of invalid validation operations."""
    # Test loading non-existent results
    assert validator.load_validation_results('non_existent') is None
    
    # Test comparing non-existent results
    assert validator.compare_validation_results('result1', 'result2') == {}
    
    # Test getting history with non-existent model version
    assert len(validator.get_validation_history('non_existent')) == 0

def test_validation_with_missing_environment(validator, model, validation_config):
    """Test validation with missing environment validator."""
    environment_validators = {
        ValidationEnvironment.ADMIN: mock_validator_success
        # ML environment validator missing
    }
    
    results = validator.validate_model(
        model,
        validation_config,
        environment_validators
    )
    
    assert len(results) == 1  # Should only validate ADMIN environment
    assert results[0].passed
    assert results[0].environment == ValidationEnvironment.ADMIN 