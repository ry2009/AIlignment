from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
import json
from datetime import datetime
import asyncio
from ..data.environment_generators import (
    AdminDataGenerator,
    ForumDataGenerator,
    GameDataGenerator,
    AdminDataConfig,
    ForumDataConfig,
    GameDataConfig
)
from ..data.validation_generator import ValidationDataGenerator, ValidationDataConfig

@dataclass
class IntegrationTestConfig:
    name: str
    environments: List[str]
    test_type: str  # end_to_end, cross_environment, performance
    dependencies: List[str]
    timeout: int
    environment_configs: Dict[str, Any]
    validation_rules: Dict[str, Dict[str, Any]]
    metrics: List[str]
    success_criteria: Dict[str, float]

class IntegrationTestFramework:
    """Framework for running integration tests across environments."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.logger = logging.getLogger('IntegrationTest')
        self._setup_logging()
        
        # Initialize generators
        self.generators = {
            'admin': AdminDataGenerator(base_path),
            'forum': ForumDataGenerator(base_path),
            'game': GameDataGenerator(base_path),
            'ml': ValidationDataGenerator(base_path)
        }

    def _setup_logging(self):
        log_dir = self.base_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / 'integration_tests.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    async def run_test(self, config: IntegrationTestConfig) -> Dict[str, Any]:
        """Run integration test according to configuration."""
        try:
            self.logger.info(f"Starting integration test: {config.name}")
            start_time = datetime.now()
            
            # Generate data for each environment
            env_data = {}
            for env in config.environments:
                env_data[env] = await self._generate_environment_data(
                    env,
                    config.environment_configs[env]
                )
            
            # Run environment-specific validations
            validation_results = await self._validate_environments(
                env_data,
                config.validation_rules
            )
            
            # Run cross-environment validations
            if config.test_type == 'cross_environment':
                cross_env_results = await self._validate_cross_environment(
                    env_data,
                    config.validation_rules
                )
                validation_results.update(cross_env_results)
            
            # Check success criteria
            success = self._check_success_criteria(
                validation_results,
                config.success_criteria
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'test_name': config.name,
                'status': 'success' if success else 'failure',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': duration,
                'validation_results': validation_results,
                'metrics': self._compute_metrics(validation_results, config.metrics)
            }
            
            self.logger.info(f"Integration test completed: {config.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in integration test {config.name}: {str(e)}")
            raise

    async def _generate_environment_data(
        self,
        environment: str,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Any]:
        """Generate data for a specific environment."""
        generator = self.generators[environment]
        
        if environment == 'admin':
            return await asyncio.to_thread(
                generator.generate_data,
                AdminDataConfig(**config)
            )
        elif environment == 'forum':
            return await asyncio.to_thread(
                generator.generate_data,
                ForumDataConfig(**config)
            )
        elif environment == 'game':
            return await asyncio.to_thread(
                generator.generate_data,
                GameDataConfig(**config)
            )
        elif environment == 'ml':
            return await asyncio.to_thread(
                generator.generate_validation_data,
                ValidationDataConfig(**config)
            )
        else:
            raise ValueError(f"Unsupported environment: {environment}")

    async def _validate_environments(
        self,
        env_data: Dict[str, Tuple[Dict[str, Any], Any]],
        validation_rules: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run validation checks for each environment."""
        results = {}
        
        for env, (data, version) in env_data.items():
            if env in validation_rules:
                env_results = await self._validate_single_environment(
                    env,
                    data,
                    validation_rules[env]
                )
                results[env] = env_results
        
        return results

    async def _validate_single_environment(
        self,
        environment: str,
        data: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data for a single environment."""
        results = {}
        
        if environment == 'admin':
            # Validate resource usage patterns
            results['resource_usage'] = self._validate_resource_usage(
                data['resources'],
                rules.get('resource_rules', {})
            )
            # Validate user activity patterns
            results['user_activity'] = self._validate_user_activity(
                data['users'],
                rules.get('user_rules', {})
            )
            
        elif environment == 'forum':
            # Validate content patterns
            results['content'] = self._validate_forum_content(
                data['posts'],
                rules.get('content_rules', {})
            )
            # Validate user behavior
            results['user_behavior'] = self._validate_forum_users(
                data['users'],
                data['posts'],
                rules.get('user_rules', {})
            )
            
        elif environment == 'game':
            # Validate game mechanics
            results['mechanics'] = self._validate_game_mechanics(
                data['episodes'],
                rules.get('mechanics_rules', {})
            )
            # Validate player behavior
            results['player_behavior'] = self._validate_player_behavior(
                data['players'],
                data['episodes'],
                rules.get('player_rules', {})
            )
            
        elif environment == 'ml':
            # Validate ML model behavior
            results['model_behavior'] = self._validate_ml_behavior(
                data['features'],
                data['labels'],
                rules.get('model_rules', {})
            )
        
        return results

    async def _validate_cross_environment(
        self,
        env_data: Dict[str, Tuple[Dict[str, Any], Any]],
        validation_rules: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate interactions between environments."""
        results = {}
        
        # Validate admin-forum interactions
        if 'admin' in env_data and 'forum' in env_data:
            results['admin_forum'] = self._validate_admin_forum_interaction(
                env_data['admin'][0],
                env_data['forum'][0],
                validation_rules.get('admin_forum', {})
            )
        
        # Validate forum-game interactions
        if 'forum' in env_data and 'game' in env_data:
            results['forum_game'] = self._validate_forum_game_interaction(
                env_data['forum'][0],
                env_data['game'][0],
                validation_rules.get('forum_game', {})
            )
        
        # Validate game-ml interactions
        if 'game' in env_data and 'ml' in env_data:
            results['game_ml'] = self._validate_game_ml_interaction(
                env_data['game'][0],
                env_data['ml'][0],
                validation_rules.get('game_ml', {})
            )
        
        return results

    def _validate_resource_usage(
        self,
        resources: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate resource usage patterns."""
        results = {}
        for resource_type, data in resources.items():
            usage = data['usage']
            alerts = data['alerts']
            
            results[resource_type] = {
                'mean_usage': float(np.mean(usage)),
                'max_usage': float(np.max(usage)),
                'alert_rate': float(np.mean(alerts)),
                'usage_pattern_valid': self._check_usage_pattern(
                    usage,
                    rules.get('pattern_rules', {})
                )
            }
        return results

    def _validate_forum_content(
        self,
        posts: List[Dict[str, Any]],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate forum content patterns."""
        toxic_scores = [p['toxicity_score'] for p in posts]
        engagement_scores = [p['engagement_score'] for p in posts]
        
        return {
            'mean_toxicity': float(np.mean(toxic_scores)),
            'high_toxicity_rate': float(np.mean([s > 0.7 for s in toxic_scores])),
            'mean_engagement': float(np.mean(engagement_scores)),
            'content_distribution_valid': self._check_content_distribution(
                posts,
                rules.get('distribution_rules', {})
            )
        }

    def _validate_game_mechanics(
        self,
        episodes: List[Dict[str, Any]],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate game mechanics and balance."""
        return {
            'win_rate': float(np.mean([e['won'] for e in episodes])),
            'mean_completion_time': float(np.mean([e['completion_time'] for e in episodes])),
            'mean_reward': float(np.mean([e['rewards'].mean() for e in episodes])),
            'balance_valid': self._check_game_balance(
                episodes,
                rules.get('balance_rules', {})
            )
        }

    def _validate_ml_behavior(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate ML model behavior and characteristics."""
        return {
            'feature_distribution_valid': self._check_feature_distribution(
                features,
                rules.get('feature_rules', {})
            ),
            'label_distribution_valid': self._check_label_distribution(
                labels,
                rules.get('label_rules', {})
            ),
            'feature_correlation_valid': self._check_feature_correlation(
                features,
                rules.get('correlation_rules', {})
            )
        }

    def _check_success_criteria(
        self,
        results: Dict[str, Any],
        criteria: Dict[str, float]
    ) -> bool:
        """Check if test results meet success criteria."""
        for metric, threshold in criteria.items():
            parts = metric.split('.')
            value = results
            for part in parts:
                value = value.get(part, {})
            if not isinstance(value, (int, float)) or value < threshold:
                return False
        return True

    def _compute_metrics(
        self,
        results: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, float]:
        """Compute specified metrics from test results."""
        computed_metrics = {}
        for metric in metrics:
            parts = metric.split('.')
            value = results
            for part in parts:
                value = value.get(part, {})
            if isinstance(value, (int, float)):
                computed_metrics[metric] = float(value)
        return computed_metrics 

    def _check_usage_pattern(
        self,
        usage: np.ndarray,
        rules: Dict[str, Any]
    ) -> bool:
        """Check if the usage pattern meets the defined rules."""
        if not rules:
            return True
        
        # Check for periodic patterns if specified
        if rules.get('periodic_check', False):
            # Use FFT to detect periodic patterns
            fft = np.fft.fft(usage)
            freqs = np.fft.fftfreq(len(usage))
            main_freq = np.abs(freqs[np.argmax(np.abs(fft[1:]))+1])
            if main_freq > rules.get('max_frequency', 0.5):
                return False
            
        # Check for spikes
        if 'max_spike' in rules:
            rolling_mean = np.convolve(usage, np.ones(5)/5, mode='valid')
            spikes = np.abs(usage[2:-2] - rolling_mean)
            if np.any(spikes > rules['max_spike']):
                return False
            
        # Check for sustained high usage
        if 'max_sustained' in rules:
            window_size = min(len(usage), 10)
            rolling_mean = np.convolve(usage, np.ones(window_size)/window_size, mode='valid')
            if np.any(rolling_mean > rules['max_sustained']):
                return False
            
        return True 

    def _validate_user_activity(
        self,
        users: Dict[str, Any],
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate user activity patterns."""
        results = {}
        
        # Check concurrent users
        if 'max_concurrent_users' in rules:
            concurrent_users = len(set(users.get('user_ids', [])))
            results['concurrent_users_valid'] = concurrent_users <= rules['max_concurrent_users']
            
        # Check session durations
        if 'max_session_duration' in rules and 'session_durations' in users:
            durations = users['session_durations']
            results['session_duration_valid'] = all(d <= rules['max_session_duration'] for d in durations)
            
        # Check for suspicious activity
        if 'suspicious_activity_threshold' in rules and 'activity_scores' in users:
            scores = users['activity_scores']
            suspicious_count = sum(1 for s in scores if s > rules['suspicious_activity_threshold'])
            results['suspicious_activity_valid'] = suspicious_count == 0
            
        return results 