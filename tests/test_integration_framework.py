import pytest
import numpy as np
from pathlib import Path
import tempfile
import asyncio
from datetime import datetime, timedelta
from ml_environment.testing.integration.test_framework import (
    IntegrationTestFramework,
    IntegrationTestConfig
)
from ml_environment.testing.integration.validation_helpers import (
    check_usage_pattern,
    check_content_distribution,
    check_game_balance,
    check_feature_distribution,
    check_label_distribution,
    check_feature_correlation,
    validate_admin_forum_interaction,
    validate_forum_game_interaction,
    validate_game_ml_interaction
)

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def test_framework(temp_path):
    return IntegrationTestFramework(temp_path)

# Test Framework Tests
@pytest.mark.asyncio
async def test_basic_integration_test(test_framework):
    """Test running a basic integration test."""
    config = IntegrationTestConfig(
        name="test_integration",
        environments=["admin", "forum"],
        test_type="cross_environment",
        dependencies=[],
        timeout=60,
        environment_configs={
            "admin": {
                "size": 10,
                "resource_types": ["cpu"],
                "user_count": 5,
                "time_periods": 24
            },
            "forum": {
                "size": 20,
                "num_users": 5,
                "topics": ["general"],
                "post_length_range": (10, 100)
            }
        },
        validation_rules={
            "admin": {
                "resource_rules": {
                    "max_spike": 50,
                    "max_sustained": 80
                }
            },
            "forum": {
                "content_rules": {
                    "topic_distribution": {"general": 1.0}
                }
            }
        },
        metrics=["admin.resource_usage.cpu.mean_usage"],
        success_criteria={
            "admin.resource_usage.cpu.mean_usage": 0.0
        }
    )
    
    result = await test_framework.run_test(config)
    assert result["status"] in ["success", "failure"]
    assert "metrics" in result
    assert "validation_results" in result

@pytest.mark.asyncio
async def test_cross_environment_validation(test_framework):
    """Test cross-environment validation."""
    config = IntegrationTestConfig(
        name="cross_env_test",
        environments=["forum", "game"],
        test_type="cross_environment",
        dependencies=[],
        timeout=60,
        environment_configs={
            "forum": {
                "size": 20,
                "num_users": 5,
                "topics": ["gaming"],
                "post_length_range": (10, 100)
            },
            "game": {
                "num_episodes": 10,
                "episode_length": 5,
                "num_players": 5,
                "difficulty_levels": ["easy"],
                "win_rate_range": (0.4, 0.6)
            }
        },
        validation_rules={
            "forum_game": {
                "performance_activity_correlation": 0.5
            }
        },
        metrics=["forum_game.correlation_valid"],
        success_criteria={
            "forum_game.correlation_valid": True
        }
    )
    
    result = await test_framework.run_test(config)
    assert "forum_game" in result["validation_results"]
    assert "correlation_valid" in result["validation_results"]["forum_game"]

# Validation Helper Tests
def test_check_usage_pattern():
    """Test resource usage pattern validation."""
    usage = np.random.normal(50, 10, (10, 24))
    
    # Test spike detection
    rules = {"max_spike": 20}
    assert check_usage_pattern(usage, rules)
    
    # Add a spike
    usage[0, 12] = 200
    assert not check_usage_pattern(usage, rules)
    
    # Test sustained usage
    usage = np.ones((10, 24)) * 50
    rules = {"max_sustained": 60, "sustained_window": 3}
    assert check_usage_pattern(usage, rules)
    
    usage[:, 10:15] = 70
    assert not check_usage_pattern(usage, rules)

def test_check_content_distribution():
    """Test forum content distribution validation."""
    posts = [
        {"topic": "tech", "content_length": 100, "engagement_score": 0.5}
        for _ in range(50)
    ] + [
        {"topic": "gaming", "content_length": 150, "engagement_score": 0.7}
        for _ in range(50)
    ]
    
    # Test topic distribution
    rules = {
        "topic_distribution": {"tech": 0.5, "gaming": 0.5},
        "topic_tolerance": 0.1
    }
    assert check_content_distribution(posts, rules)
    
    # Test with invalid distribution
    rules["topic_distribution"] = {"tech": 0.8, "gaming": 0.2}
    assert not check_content_distribution(posts, rules)

def test_check_game_balance():
    """Test game balance validation."""
    episodes = [
        {
            "difficulty": "easy",
            "won": True,
            "rewards": np.array([1.0, 1.2, 0.8]),
            "completion_time": 120
        }
        for _ in range(40)
    ] + [
        {
            "difficulty": "hard",
            "won": False,
            "rewards": np.array([0.5, 0.3, 0.4]),
            "completion_time": 180
        }
        for _ in range(60)
    ]
    
    # Test win rate by difficulty
    rules = {
        "win_rate_by_difficulty": {
            "easy": 1.0,
            "hard": 0.0
        },
        "win_rate_tolerance": 0.1
    }
    assert check_game_balance(episodes, rules)
    
    # Test completion time range
    rules = {
        "completion_time_range": (60, 240)
    }
    assert check_game_balance(episodes, rules)
    
    rules["completion_time_range"] = (60, 150)
    assert not check_game_balance(episodes, rules)

def test_check_feature_distribution():
    """Test feature distribution validation."""
    # Generate normally distributed features
    features = np.random.normal(0, 1, (1000, 5))
    
    # Test normal distribution check
    rules = {
        "distribution_type": "normal",
        "p_threshold": 0.05
    }
    assert check_feature_distribution(features, rules)
    
    # Generate uniform features
    features = np.random.uniform(0, 1, (1000, 5))
    
    # Test uniform distribution check
    rules = {
        "distribution_type": "uniform",
        "p_threshold": 0.05
    }
    assert check_feature_distribution(features, rules)

def test_check_label_distribution():
    """Test label distribution validation."""
    # Test classification labels
    labels = np.concatenate([
        np.zeros(500),
        np.ones(500)
    ])
    
    rules = {
        "class_balance": [0.5, 0.5],
        "balance_tolerance": 0.1
    }
    assert check_label_distribution(labels, rules)
    
    # Test regression labels
    labels = np.random.uniform(0, 10, 1000)
    rules = {
        "value_range": (0, 10)
    }
    assert check_label_distribution(labels, rules)
    
    rules["value_range"] = (0, 5)
    assert not check_label_distribution(labels, rules)

def test_check_feature_correlation():
    """Test feature correlation validation."""
    # Generate independent features
    features = np.random.normal(0, 1, (1000, 5))
    
    rules = {
        "max_correlation": 0.5
    }
    assert check_feature_correlation(features, rules)
    
    # Generate correlated features
    features = np.random.normal(0, 1, (1000, 2))
    features = np.column_stack([
        features,
        features[:, 0] * 0.9 + np.random.normal(0, 0.1, 1000)
    ])
    
    assert not check_feature_correlation(features, rules)

def test_validate_admin_forum_interaction():
    """Test admin-forum interaction validation."""
    admin_data = {
        "resources": {
            "cpu": {"usage": np.random.normal(50, 10, 24)}
        },
        "timestamps": list(range(0, 24 * 3600, 3600))
    }
    
    forum_data = {
        "posts": [
            {
                "timestamp": i * 3600
                for i in range(24)
            }
        ]
    }
    
    rules = {
        "resource_activity_correlation": 0.5
    }
    
    results = validate_admin_forum_interaction(admin_data, forum_data, rules)
    assert "resource_activity_correlation" in results
    assert "correlation_valid" in results

def test_validate_forum_game_interaction():
    """Test forum-game interaction validation."""
    forum_data = {
        "posts": [
            {"user_id": i % 5, "timestamp": i * 3600}
            for i in range(20)
        ]
    }
    
    game_data = {
        "players": [
            {"player_id": i}
            for i in range(5)
        ],
        "episodes": [
            {
                "rewards": np.random.normal(1, 0.1, 10)
            }
            for _ in range(10)
        ]
    }
    
    rules = {
        "performance_activity_correlation": 0.5
    }
    
    results = validate_forum_game_interaction(forum_data, game_data, rules)
    assert "performance_activity_correlation" in results
    assert "correlation_valid" in results

def test_validate_game_ml_interaction():
    """Test game-ML interaction validation."""
    game_data = {
        "episodes": [
            {"won": bool(i % 2)}
            for i in range(10)
        ]
    }
    
    ml_data = {
        "predictions": np.array([i % 2 for i in range(10)])
    }
    
    rules = {
        "prediction_accuracy": 0.8
    }
    
    results = validate_game_ml_interaction(game_data, ml_data, rules)
    assert "prediction_accuracy" in results
    assert "accuracy_valid" in results 