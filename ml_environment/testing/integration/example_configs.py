from typing import Dict, Any
from pathlib import Path

# Basic environment-specific test configurations
ADMIN_TEST_CONFIG = {
    "name": "admin_environment_test",
    "environments": ["admin"],
    "test_type": "integration",
    "dependencies": [],
    "timeout": 300,
    "environment_configs": {
        "admin": {
            "environment": "admin",
            "size": 50,
            "feature_config": {
                "num_features": 10
            }
        }
    },
    "validation_rules": {
        "resource_rules": {
            "max_frequency": 0.5,
            "max_spike": 50,
            "max_sustained": 80,
            "periodic_check": True
        },
        "user_rules": {
            "max_concurrent_users": 30,
            "max_session_duration": 3600,
            "suspicious_activity_threshold": 0.8
        }
    },
    "metrics": [
        "resource_usage.cpu.mean_usage",
        "resource_usage.cpu.max_usage",
        "resource_usage.cpu.alert_rate",
        "user_activity.concurrent_users_valid",
        "user_activity.session_duration_valid",
        "user_activity.suspicious_activity_valid"
    ],
    "success_criteria": {
        "resource_usage.cpu.mean_usage": 50.0,
        "resource_usage.cpu.max_usage": 80.0,
        "resource_usage.cpu.alert_rate": 0.1,
        "user_activity.concurrent_users_valid": True,
        "user_activity.session_duration_valid": True,
        "user_activity.suspicious_activity_valid": True
    }
}

FORUM_TEST_CONFIG = {
    "name": "forum_environment_test",
    "environments": ["forum"],
    "test_type": "environment",
    "dependencies": [],
    "timeout": 300,
    "environment_configs": {
        "forum": {
            "size": 1000,
            "num_users": 100,
            "topics": ["tech", "gaming", "science", "general"],
            "post_length_range": (50, 1000),
            "toxic_content_rate": 0.1
        }
    },
    "validation_rules": {
        "forum": {
            "content_rules": {
                "topic_distribution": {
                    "tech": 0.3,
                    "gaming": 0.3,
                    "science": 0.2,
                    "general": 0.2
                },
                "topic_tolerance": 0.05,
                "length_distribution": "normal",
                "engagement_distribution": "uniform"
            },
            "user_rules": {
                "max_posts_per_hour": 10,
                "min_engagement_score": 0.3
            }
        }
    },
    "metrics": [
        "forum.content.mean_toxicity",
        "forum.content.mean_engagement",
        "forum.user_behavior.active_users"
    ],
    "success_criteria": {
        "forum.content.mean_toxicity": 0.2,
        "forum.content.mean_engagement": 0.5
    }
}

GAME_TEST_CONFIG = {
    "name": "game_environment_test",
    "environments": ["game"],
    "test_type": "environment",
    "dependencies": [],
    "timeout": 300,
    "environment_configs": {
        "game": {
            "num_episodes": 500,
            "episode_length": 100,
            "num_players": 50,
            "difficulty_levels": ["easy", "medium", "hard"],
            "win_rate_range": (0.4, 0.6)
        }
    },
    "validation_rules": {
        "game": {
            "mechanics_rules": {
                "win_rate_by_difficulty": {
                    "easy": 0.7,
                    "medium": 0.5,
                    "hard": 0.3
                },
                "win_rate_tolerance": 0.1,
                "reward_distribution": "normal",
                "completion_time_range": (60, 300)
            },
            "player_rules": {
                "skill_distribution": "normal",
                "min_games_per_player": 5
            }
        }
    },
    "metrics": [
        "game.mechanics.win_rate",
        "game.mechanics.mean_reward",
        "game.player_behavior.skill_progression"
    ],
    "success_criteria": {
        "game.mechanics.win_rate": 0.5,
        "game.mechanics.mean_reward": 0.7
    }
}

ML_TEST_CONFIG = {
    "name": "ml_environment_test",
    "environments": ["ml"],
    "test_type": "environment",
    "dependencies": [],
    "timeout": 300,
    "environment_configs": {
        "ml": {
            "dataset_type": "train",
            "size": 10000,
            "feature_config": {
                "type": "mixed",
                "continuous": {"num_features": 5, "distribution": "normal"},
                "categorical": {"num_features": 3, "num_categories": 4}
            },
            "label_config": {
                "type": "classification",
                "num_classes": 3
            },
            "noise_config": {
                "type": "gaussian",
                "scale": 0.1
            }
        }
    },
    "validation_rules": {
        "ml": {
            "model_rules": {
                "feature_rules": {
                    "distribution_type": "normal",
                    "p_threshold": 0.05
                },
                "label_rules": {
                    "class_balance": [0.33, 0.33, 0.34],
                    "balance_tolerance": 0.05
                },
                "correlation_rules": {
                    "max_correlation": 0.7
                }
            }
        }
    },
    "metrics": [
        "ml.model_behavior.feature_distribution_valid",
        "ml.model_behavior.label_distribution_valid",
        "ml.model_behavior.feature_correlation_valid"
    ],
    "success_criteria": {
        "ml.model_behavior.feature_distribution_valid": True,
        "ml.model_behavior.label_distribution_valid": True,
        "ml.model_behavior.feature_correlation_valid": True
    }
}

# Cross-environment test configurations
ADMIN_FORUM_TEST_CONFIG = {
    "name": "admin_forum_interaction_test",
    "environments": ["admin", "forum"],
    "test_type": "cross_environment",
    "dependencies": [],
    "timeout": 600,
    "environment_configs": {
        "admin": ADMIN_TEST_CONFIG["environment_configs"]["admin"],
        "forum": FORUM_TEST_CONFIG["environment_configs"]["forum"]
    },
    "validation_rules": {
        "resource_rules": ADMIN_TEST_CONFIG["validation_rules"]["resource_rules"],
        "user_rules": ADMIN_TEST_CONFIG["validation_rules"]["user_rules"],
        "forum": FORUM_TEST_CONFIG["validation_rules"]["forum"],
        "admin_forum": {
            "resource_activity_correlation": 0.5,
            "user_activity_correlation": 0.4
        }
    },
    "metrics": [
        "admin_forum.resource_activity_correlation",
        "admin_forum.correlation_valid"
    ],
    "success_criteria": {
        "admin_forum.correlation_valid": True
    }
}

FORUM_GAME_TEST_CONFIG = {
    "name": "forum_game_interaction_test",
    "environments": ["forum", "game"],
    "test_type": "cross_environment",
    "dependencies": [],
    "timeout": 600,
    "environment_configs": {
        "forum": FORUM_TEST_CONFIG["environment_configs"]["forum"],
        "game": GAME_TEST_CONFIG["environment_configs"]["game"]
    },
    "validation_rules": {
        "forum": FORUM_TEST_CONFIG["validation_rules"]["forum"],
        "game": GAME_TEST_CONFIG["validation_rules"]["game"],
        "forum_game": {
            "performance_activity_correlation": 0.4,
            "topic_performance_correlation": 0.3
        }
    },
    "metrics": [
        "forum_game.performance_activity_correlation",
        "forum_game.correlation_valid"
    ],
    "success_criteria": {
        "forum_game.correlation_valid": True
    }
}

GAME_ML_TEST_CONFIG = {
    "name": "game_ml_interaction_test",
    "environments": ["game", "ml"],
    "test_type": "cross_environment",
    "dependencies": [],
    "timeout": 600,
    "environment_configs": {
        "game": GAME_TEST_CONFIG["environment_configs"]["game"],
        "ml": ML_TEST_CONFIG["environment_configs"]["ml"]
    },
    "validation_rules": {
        "game": GAME_TEST_CONFIG["validation_rules"]["game"],
        "ml": ML_TEST_CONFIG["validation_rules"]["ml"],
        "game_ml": {
            "prediction_accuracy": 0.8,
            "prediction_latency": 0.1
        }
    },
    "metrics": [
        "game_ml.prediction_accuracy",
        "game_ml.accuracy_valid"
    ],
    "success_criteria": {
        "game_ml.accuracy_valid": True
    }
}

# Performance test configurations
PERFORMANCE_TEST_CONFIG = {
    "name": "full_system_performance_test",
    "environments": ["ml"],
    "test_type": "performance",
    "dependencies": [],
    "timeout": 900,
    "environment_configs": {
        "ml": {
            "environment": "ml",
            "dataset_type": "train",
            "size": 10000,
            "feature_config": {
                "continuous": {
                    "num_features": 5,
                    "range": [-1, 1]
                },
                "categorical": {
                    "num_features": 3,
                    "num_categories": 4
                }
            },
            "label_config": {
                "type": "classification",
                "num_classes": 3
            },
            "noise_config": {
                "type": "gaussian",
                "scale": 0.1
            }
        }
    },
    "validation_rules": {
        "resource_rules": ADMIN_TEST_CONFIG["validation_rules"]["resource_rules"],
        "user_rules": ADMIN_TEST_CONFIG["validation_rules"]["user_rules"],
        "performance": {
            "max_memory_usage": 4e9,
            "max_cpu_usage": 80.0,
            "max_response_time": 500.0,
            "throughput_threshold": 50.0
        }
    },
    "metrics": [
        "performance.memory_usage",
        "performance.cpu_usage",
        "performance.response_time",
        "performance.throughput"
    ],
    "success_criteria": {
        "performance.memory_usage": 4e9,
        "performance.cpu_usage": 80.0,
        "performance.response_time": 500.0,
        "performance.throughput": 50.0
    }
}

# Privacy ML performance test configuration
PRIVACY_ML_TEST_CONFIG = {
    "name": "privacy_ml_performance_test",
    "environments": ["ml"],
    "test_type": "performance",
    "dependencies": [],
    "timeout": 900,
    "environment_configs": {
        "ml": {
            "environment": "ml",
            "dataset_type": "tabular",
            "size": 10000,
            "feature_config": {
                "num_features": 10,
                "feature_types": ["continuous"] * 10
            },
            "label_config": {
                "num_classes": 2,
                "label_type": "binary"
            },
            "model_type": "simple",
            "batch_size": 32,
            "epochs": 10,
            "dataset_size": 10000,
            "privacy_config": {
                "epsilon": 0.1,
                "delta": 1e-5,
                "max_grad_norm": 1.0,
                "noise_multiplier": 0.1
            }
        }
    },
    "validation_rules": {
        "performance": {
            "max_memory_usage": 4e9,
            "max_cpu_usage": 80.0,
            "max_response_time": 500.0,
            "max_training_time": 300.0,
            "throughput_threshold": 50.0,
            "max_privacy_overhead": 30.0
        }
    },
    "metrics": [
        "performance.memory_usage",
        "performance.cpu_usage",
        "performance.response_time",
        "performance.throughput",
        "performance.training_time",
        "performance.privacy_overhead",
        "performance.epsilon_spent"
    ],
    "success_criteria": {
        "performance.memory_usage": 4e9,
        "performance.cpu_usage": 80.0,
        "performance.response_time": 500.0,
        "performance.training_time": 300.0,
        "performance.throughput": 50.0,
        "performance.privacy_overhead": 30.0,
        "performance.epsilon_spent": 0.1
    }
} 