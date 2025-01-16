from typing import Dict, List, Optional, Union, Any, Callable
import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
from enum import Enum

class ValidationEnvironment(Enum):
    ADMIN = "admin"
    FORUM = "forum"
    GAME = "game"
    CODE = "code"
    ML = "ml"

@dataclass
class ValidationConfig:
    environments: List[ValidationEnvironment]
    metrics: List[str]
    thresholds: Dict[str, float]
    timeout: int
    parallel: bool = True
    stop_on_failure: bool = True

@dataclass
class ValidationResult:
    environment: ValidationEnvironment
    metrics: Dict[str, float]
    passed: bool
    error: Optional[str]
    duration: float
    timestamp: float

class CrossEnvironmentValidator:
    def __init__(self, base_path: Path, session_id: str):
        self.base_path = base_path
        self.session_id = session_id
        self.logger = logging.getLogger(f'Validator-{session_id}')
        self._setup_logging()
        self._validation_lock = threading.Lock()
        self.results_dir = base_path / 'validation_results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        log_dir = self.base_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / f'validation_{self.session_id}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def validate_model(self, model: nn.Module, config: ValidationConfig,
                      environment_validators: Dict[ValidationEnvironment, Callable]) -> List[ValidationResult]:
        """Run validation across multiple environments."""
        try:
            if config.parallel:
                return self._parallel_validate(model, config, environment_validators)
            return self._sequential_validate(model, config, environment_validators)
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            return []

    def _parallel_validate(self, model: nn.Module, config: ValidationConfig,
                          environment_validators: Dict[ValidationEnvironment, Callable]) -> List[ValidationResult]:
        """Run validation in parallel across environments."""
        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for env in config.environments:
                if env in environment_validators:
                    futures.append(executor.submit(
                        self._validate_single_environment,
                        model, env, environment_validators[env],
                        config.thresholds, config.timeout
                    ))

            for future in futures:
                try:
                    result = future.result(timeout=config.timeout)
                    results.append(result)
                    if not result.passed and config.stop_on_failure:
                        break
                except Exception as e:
                    self.logger.error(f"Error in parallel validation: {str(e)}")

        return results

    def _sequential_validate(self, model: nn.Module, config: ValidationConfig,
                           environment_validators: Dict[ValidationEnvironment, Callable]) -> List[ValidationResult]:
        """Run validation sequentially across environments."""
        results = []
        for env in config.environments:
            if env in environment_validators:
                result = self._validate_single_environment(
                    model, env, environment_validators[env],
                    config.thresholds, config.timeout
                )
                results.append(result)
                if not result.passed and config.stop_on_failure:
                    break
        return results

    def _validate_single_environment(self, model: nn.Module,
                                   environment: ValidationEnvironment,
                                   validator: Callable,
                                   thresholds: Dict[str, float],
                                   timeout: int) -> ValidationResult:
        """Validate model in a single environment."""
        start_time = time.time()
        try:
            metrics = validator(model)
            duration = time.time() - start_time
            
            # Check if metrics meet thresholds
            passed = all(
                metrics.get(metric, 0) >= threshold
                for metric, threshold in thresholds.items()
            )
            
            return ValidationResult(
                environment=environment,
                metrics=metrics,
                passed=passed,
                error=None,
                duration=duration,
                timestamp=time.time()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                environment=environment,
                metrics={},
                passed=False,
                error=str(e),
                duration=duration,
                timestamp=time.time()
            )

    def save_validation_results(self, results: List[ValidationResult],
                              model_version: str) -> str:
        """Save validation results to disk."""
        with self._validation_lock:
            try:
                timestamp = int(time.time())
                result_id = f"validation_{model_version}_{timestamp}"
                result_path = self.results_dir / f"{result_id}.json"
                
                result_data = {
                    'model_version': model_version,
                    'timestamp': timestamp,
                    'results': [
                        {
                            'environment': r.environment.value,
                            'metrics': r.metrics,
                            'passed': r.passed,
                            'error': r.error,
                            'duration': r.duration,
                            'timestamp': r.timestamp
                        }
                        for r in results
                    ]
                }
                
                with open(result_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                self.logger.info(f"Saved validation results: {result_id}")
                return result_id
                
            except Exception as e:
                self.logger.error(f"Error saving validation results: {str(e)}")
                raise

    def load_validation_results(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Load validation results from disk."""
        try:
            result_path = self.results_dir / f"{result_id}.json"
            if not result_path.exists():
                return None
                
            with open(result_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading validation results: {str(e)}")
            return None

    def get_validation_history(self, model_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get validation history, optionally filtered by model version."""
        try:
            results = []
            for result_file in self.results_dir.glob('validation_*.json'):
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    if model_version is None or result_data['model_version'] == model_version:
                        results.append(result_data)
            return sorted(results, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            self.logger.error(f"Error getting validation history: {str(e)}")
            return []

    def compare_validation_results(self, result_id1: str, result_id2: str) -> Dict[str, Any]:
        """Compare two validation results."""
        try:
            result1 = self.load_validation_results(result_id1)
            result2 = self.load_validation_results(result_id2)
            
            if not result1 or not result2:
                return {}
            
            comparison = {
                'timestamp_diff': result2['timestamp'] - result1['timestamp'],
                'environments': {}
            }
            
            # Compare results for each environment
            env_results1 = {r['environment']: r for r in result1['results']}
            env_results2 = {r['environment']: r for r in result2['results']}
            
            for env in set(env_results1.keys()) | set(env_results2.keys()):
                if env not in env_results1:
                    comparison['environments'][env] = {'added': env_results2[env]}
                elif env not in env_results2:
                    comparison['environments'][env] = {'removed': env_results1[env]}
                else:
                    r1, r2 = env_results1[env], env_results2[env]
                    comparison['environments'][env] = {
                        'metrics_diff': {
                            k: r2['metrics'].get(k, 0) - r1['metrics'].get(k, 0)
                            for k in set(r1['metrics']) | set(r2['metrics'])
                        },
                        'passed_changed': r1['passed'] != r2['passed'],
                        'duration_diff': r2['duration'] - r1['duration']
                    }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing validation results: {str(e)}")
            return {} 