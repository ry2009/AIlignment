from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json

# Base configurations for different test types
UNIT_TEST_BASE = {
    "test_type": "unit",
    "timeout": 30,
    "retries": 2,
    "dependencies": []
}

INTEGRATION_TEST_BASE = {
    "test_type": "integration",
    "timeout": 60,
    "retries": 3,
    "dependencies": []
}

END_TO_END_TEST_BASE = {
    "test_type": "end_to_end",
    "timeout": 120,
    "retries": 2,
    "dependencies": []
}

# ML Environment Test Configurations
ML_TEST_CONFIGS = {
    "model_training": {
        **UNIT_TEST_BASE,
        "test_name": "model_training_test",
        "environment": "ml",
        "schedule": "0 */4 * * *",  # Every 4 hours
        "parameters": {
            "batch_size": 32,
            "epochs": 1,
            "learning_rate": 0.001,
            "metrics": ["accuracy", "loss", "latency"]
        }
    },
    "model_inference": {
        **UNIT_TEST_BASE,
        "test_name": "model_inference_test",
        "environment": "ml",
        "schedule": "*/30 * * * *",  # Every 30 minutes
        "parameters": {
            "batch_size": 16,
            "num_samples": 100,
            "metrics": ["accuracy", "latency", "memory_usage"]
        }
    },
    "privacy_validation": {
        **INTEGRATION_TEST_BASE,
        "test_name": "privacy_validation_test",
        "environment": "ml",
        "schedule": "0 */6 * * *",  # Every 6 hours
        "parameters": {
            "epsilon": 1.0,
            "delta": 1e-5,
            "metrics": ["privacy_budget", "utility_loss"]
        }
    }
}

# Admin Environment Test Configurations
ADMIN_TEST_CONFIGS = {
    "resource_monitoring": {
        **UNIT_TEST_BASE,
        "test_name": "resource_monitoring_test",
        "environment": "admin",
        "schedule": "*/15 * * * *",  # Every 15 minutes
        "parameters": {
            "cpu_threshold": 80,
            "memory_threshold": 85,
            "disk_threshold": 90,
            "metrics": ["cpu_usage", "memory_usage", "disk_usage"]
        }
    },
    "user_management": {
        **INTEGRATION_TEST_BASE,
        "test_name": "user_management_test",
        "environment": "admin",
        "schedule": "0 */2 * * *",  # Every 2 hours
        "parameters": {
            "num_users": 50,
            "operations": ["create", "update", "delete"],
            "metrics": ["success_rate", "latency"]
        }
    }
}

# Forum Environment Test Configurations
FORUM_TEST_CONFIGS = {
    "content_moderation": {
        **UNIT_TEST_BASE,
        "test_name": "content_moderation_test",
        "environment": "forum",
        "schedule": "*/10 * * * *",  # Every 10 minutes
        "parameters": {
            "num_posts": 100,
            "toxicity_threshold": 0.7,
            "metrics": ["precision", "recall", "latency"]
        }
    },
    "user_interaction": {
        **END_TO_END_TEST_BASE,
        "test_name": "user_interaction_test",
        "environment": "forum",
        "schedule": "0 */4 * * *",  # Every 4 hours
        "parameters": {
            "num_users": 20,
            "posts_per_user": 5,
            "interactions_per_post": 3,
            "metrics": ["engagement_rate", "response_time"]
        }
    }
}

# Game Environment Test Configurations
GAME_TEST_CONFIGS = {
    "game_mechanics": {
        **UNIT_TEST_BASE,
        "test_name": "game_mechanics_test",
        "environment": "game",
        "schedule": "*/20 * * * *",  # Every 20 minutes
        "parameters": {
            "num_episodes": 50,
            "max_steps": 100,
            "metrics": ["win_rate", "avg_score", "completion_time"]
        }
    },
    "multiplayer_session": {
        **END_TO_END_TEST_BASE,
        "test_name": "multiplayer_session_test",
        "environment": "game",
        "schedule": "0 */3 * * *",  # Every 3 hours
        "parameters": {
            "num_players": 4,
            "session_duration": 1800,
            "metrics": ["latency", "sync_rate", "player_retention"]
        }
    }
}

def load_test_configs(config_path: Path = None) -> Dict[str, Any]:
    """Load test configurations from file or return default configs."""
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return {
        "ml": ML_TEST_CONFIGS,
        "admin": ADMIN_TEST_CONFIGS,
        "forum": FORUM_TEST_CONFIGS,
        "game": GAME_TEST_CONFIGS
    }

def save_test_configs(configs: Dict[str, Any], config_path: Path):
    """Save test configurations to file."""
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=2)

def get_environment_configs(environment: str) -> Dict[str, Any]:
    """Get test configurations for a specific environment."""
    configs = load_test_configs()
    return configs.get(environment, {}) 