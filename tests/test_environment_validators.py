import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import json
from ml_environment.environment_validators import (
    EnvironmentValidators,
    AdminValidationConfig,
    ForumValidationConfig,
    GameValidationConfig,
    CodeValidationConfig,
    MLValidationConfig
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
def validators(temp_path):
    return EnvironmentValidators(temp_path, 'test_session')

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def admin_config():
    return AdminValidationConfig()

@pytest.fixture
def forum_config():
    return ForumValidationConfig()

@pytest.fixture
def game_config():
    return GameValidationConfig()

@pytest.fixture
def code_config():
    return CodeValidationConfig()

@pytest.fixture
def ml_config():
    return MLValidationConfig()

def test_admin_validation(validators, model, admin_config):
    """Test administrative environment validation."""
    metrics = validators.validate_admin(model, admin_config)
    
    assert 'resource_usage' in metrics
    assert 'avg_response_time' in metrics
    assert 'success_rate' in metrics
    
    assert 0 <= metrics['resource_usage'] <= 1
    assert metrics['avg_response_time'] > 0
    assert 0 <= metrics['success_rate'] <= 1

def test_forum_validation(validators, model, forum_config):
    """Test forum environment validation."""
    metrics = validators.validate_forum(model, forum_config)
    
    assert 'avg_toxicity' in metrics
    assert 'avg_response_time' in metrics
    assert 'content_quality' in metrics
    
    assert 0 <= metrics['avg_toxicity'] <= 1
    assert metrics['avg_response_time'] > 0
    assert 0 <= metrics['content_quality'] <= 1

def test_game_validation(validators, model, game_config):
    """Test game environment validation."""
    metrics = validators.validate_game(model, game_config)
    
    assert 'win_rate' in metrics
    assert 'avg_decision_time' in metrics
    assert 'fairness_score' in metrics
    
    assert 0 <= metrics['win_rate'] <= 1
    assert metrics['avg_decision_time'] > 0
    assert 0 <= metrics['fairness_score'] <= 1

def test_code_validation(validators, model, code_config):
    """Test code environment validation."""
    metrics = validators.validate_code(model, code_config)
    
    assert 'code_quality' in metrics
    assert 'security_score' in metrics
    assert 'test_coverage' in metrics
    
    assert 0 <= metrics['code_quality'] <= 1
    assert 0 <= metrics['security_score'] <= 1
    assert 0 <= metrics['test_coverage'] <= 1

def test_ml_validation(validators, model, ml_config):
    """Test ML environment validation."""
    metrics = validators.validate_ml(model, ml_config)
    
    assert 'accuracy' in metrics
    assert 'latency' in metrics
    assert 'privacy_score' in metrics
    assert 'robustness' in metrics
    
    assert 0 <= metrics['accuracy'] <= 1
    assert metrics['latency'] > 0
    assert 0 <= metrics['privacy_score'] <= 1
    assert 0 <= metrics['robustness'] <= 1

def test_validation_with_invalid_model(validators, admin_config):
    """Test validation with invalid model."""
    invalid_model = None
    metrics = validators.validate_admin(invalid_model, admin_config)
    assert metrics == {}

def test_validation_with_custom_thresholds():
    """Test validation with custom threshold configurations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        validators = EnvironmentValidators(path, 'test_session')
        model = SimpleModel()
        
        # Custom admin config
        custom_admin_config = AdminValidationConfig(
            resource_usage_threshold=0.9,
            response_time_threshold=50,
            concurrent_users=5
        )
        metrics = validators.validate_admin(model, custom_admin_config)
        assert metrics != {}
        
        # Custom ML config
        custom_ml_config = MLValidationConfig(
            accuracy_threshold=0.9,
            latency_threshold=50,
            privacy_budget_threshold=0.8
        )
        metrics = validators.validate_ml(model, custom_ml_config)
        assert metrics != {}

def test_concurrent_validation(validators, model, admin_config):
    """Test concurrent validation requests."""
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(validators.validate_admin, model, admin_config)
            for _ in range(5)
        ]
        results = [f.result() for f in futures]
    
    assert len(results) == 5
    assert all(isinstance(r, dict) for r in results)

def test_validation_with_config_file(validators, model, temp_path):
    """Test validation using configuration from file."""
    config_path = temp_path / 'test_config.json'
    config = {
        'admin_environment': {
            'validation': {
                'resource_usage_threshold': 0.8,
                'response_time_threshold': 100,
                'concurrent_users': 10,
                'test_duration': 60
            }
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
    
    admin_config = AdminValidationConfig(
        **loaded_config['admin_environment']['validation']
    )
    metrics = validators.validate_admin(model, admin_config)
    assert metrics != {}

def test_error_handling(validators, model):
    """Test error handling in validators."""
    # Test with invalid config
    invalid_config = None
    metrics = validators.validate_admin(model, invalid_config)
    assert metrics == {}
    
    # Test with invalid model
    invalid_model = "not a model"
    metrics = validators.validate_ml(invalid_model, MLValidationConfig())
    assert metrics == {}

def test_validation_metrics_ranges(validators, model):
    """Test that validation metrics are within expected ranges."""
    # Test admin metrics
    admin_metrics = validators.validate_admin(model, AdminValidationConfig())
    assert 0 <= admin_metrics['resource_usage'] <= 1
    assert admin_metrics['avg_response_time'] >= 0
    
    # Test ML metrics
    ml_metrics = validators.validate_ml(model, MLValidationConfig())
    assert 0 <= ml_metrics['accuracy'] <= 1
    assert ml_metrics['latency'] >= 0
    assert 0 <= ml_metrics['privacy_score'] <= 1 