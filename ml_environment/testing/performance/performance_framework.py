from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import logging
import psutil
import time
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from ..integration.test_framework import IntegrationTestFramework, IntegrationTestConfig

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    memory_usage: float  # in bytes
    cpu_usage: float  # percentage
    response_time: float  # seconds
    throughput: float  # requests/second
    timestamp: str
    environment: str
    operation: str
    training_time: float = 0.0  # seconds
    privacy_overhead: float = 0.0  # percentage overhead from privacy mechanisms
    epsilon_spent: float = 0.0  # privacy budget spent
    noise_scale: float = 0.0  # scale of noise added
    batch_processing_time: float = 0.0  # seconds per batch

class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.operation_counts: Dict[str, int] = {}
        self.metrics_history: List[PerformanceMetrics] = []
        self.training_start_time: Optional[float] = None
        self.baseline_batch_time: Optional[float] = None
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def get_throughput(self, operation: str) -> float:
        """Calculate throughput for a specific operation."""
        count = self.operation_counts.get(operation, 0)
        duration = time.time() - self.start_time
        return count / duration if duration > 0 else 0
    
    def record_operation(self, operation: str):
        """Record an operation for throughput calculation."""
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
    
    def start_training(self):
        """Mark the start of training for timing purposes."""
        self.training_start_time = time.time()
    
    def end_training(self) -> float:
        """Mark the end of training and return total training time."""
        if self.training_start_time is None:
            return 0.0
        training_time = time.time() - self.training_start_time
        self.training_start_time = None
        return training_time
    
    def set_baseline_batch_time(self, time_per_batch: float):
        """Set baseline batch processing time without privacy mechanisms."""
        self.baseline_batch_time = time_per_batch
    
    def calculate_privacy_overhead(self, current_batch_time: float) -> float:
        """Calculate overhead percentage from privacy mechanisms."""
        if self.baseline_batch_time is None or self.baseline_batch_time == 0:
            return 0.0
        return ((current_batch_time - self.baseline_batch_time) / self.baseline_batch_time) * 100
    
    def record_metrics(
        self,
        environment: str,
        operation: str,
        response_time: float,
        training_time: float = 0.0,
        privacy_overhead: float = 0.0,
        epsilon_spent: float = 0.0,
        noise_scale: float = 0.0,
        batch_processing_time: float = 0.0
    ):
        """Record performance metrics."""
        metrics = PerformanceMetrics(
            memory_usage=self.get_memory_usage(),
            cpu_usage=self.get_cpu_usage(),
            response_time=response_time,
            throughput=self.get_throughput(operation),
            timestamp=datetime.now().isoformat(),
            environment=environment,
            operation=operation,
            training_time=training_time,
            privacy_overhead=privacy_overhead,
            epsilon_spent=epsilon_spent,
            noise_scale=noise_scale,
            batch_processing_time=batch_processing_time
        )
        self.metrics_history.append(metrics)
        return metrics

class PerformanceTestFramework:
    """Framework for running performance tests."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.logger = logging.getLogger('PerformanceTest')
        self._setup_logging()
        self.monitor = PerformanceMonitor()
        self.integration_framework = IntegrationTestFramework(base_path)
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.base_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / 'performance_tests.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
    
    async def run_performance_test(
        self,
        config: Union[Dict[str, Any], IntegrationTestConfig]
    ) -> Dict[str, Any]:
        """Run a performance test with the given configuration."""
        try:
            # Convert dict to IntegrationTestConfig if needed
            if isinstance(config, dict):
                # Ensure all required fields are present
                required_fields = {
                    'name', 'environments', 'test_type', 'dependencies', 'timeout',
                    'environment_configs', 'validation_rules', 'metrics', 'success_criteria'
                }
                missing_fields = required_fields - set(config.keys())
                if missing_fields:
                    raise ValueError(f"Missing required fields in config: {missing_fields}")
                
                config = IntegrationTestConfig(
                    name=config['name'],
                    environments=config['environments'],
                    test_type=config['test_type'],
                    dependencies=config['dependencies'],
                    timeout=config['timeout'],
                    environment_configs=config['environment_configs'],
                    validation_rules=config['validation_rules'],
                    metrics=config['metrics'],
                    success_criteria=config['success_criteria']
                )
            
            self.logger.info(f"Starting performance test: {config.name}")
            start_time = time.time()
            
            # Run the integration test while monitoring performance
            test_start = time.time()
            test_result = await self.integration_framework.run_test(config)
            test_duration = time.time() - test_start
            
            # Record final metrics
            final_metrics = self.monitor.record_metrics(
                environment="all",
                operation="full_test",
                response_time=test_duration
            )
            
            # Validate performance metrics against rules
            performance_valid = self._validate_performance(
                final_metrics,
                config.validation_rules.get('performance', {})
            )
            
            result = {
                'test_name': config.name,
                'status': 'success' if performance_valid else 'failure',
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration': test_duration,
                'metrics': {
                    'memory_usage': final_metrics.memory_usage,
                    'cpu_usage': final_metrics.cpu_usage,
                    'response_time': final_metrics.response_time,
                    'throughput': final_metrics.throughput,
                    'training_time': final_metrics.training_time,
                    'privacy_overhead': final_metrics.privacy_overhead,
                    'epsilon_spent': final_metrics.epsilon_spent
                },
                'metrics_history': [
                    {
                        'timestamp': m.timestamp,
                        'environment': m.environment,
                        'operation': m.operation,
                        'memory_usage': m.memory_usage,
                        'cpu_usage': m.cpu_usage,
                        'response_time': m.response_time,
                        'throughput': m.throughput,
                        'training_time': m.training_time,
                        'privacy_overhead': m.privacy_overhead,
                        'epsilon_spent': m.epsilon_spent
                    }
                    for m in self.monitor.metrics_history
                ],
                'validation_results': test_result.get('validation_results', {})
            }
            
            # Save detailed metrics to file
            self._save_metrics(result)
            
            self.logger.info(f"Performance test completed: {config.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in performance test {config.name if hasattr(config, 'name') else 'unknown'}: {str(e)}")
            raise
    
    def _validate_performance(
        self,
        metrics: PerformanceMetrics,
        rules: Dict[str, Any]
    ) -> bool:
        """Validate performance metrics against rules."""
        if not rules:
            return True
        
        if 'max_memory_usage' in rules and metrics.memory_usage > rules['max_memory_usage']:
            self.logger.warning(f"Memory usage {metrics.memory_usage} exceeds limit {rules['max_memory_usage']}")
            return False
        
        if 'max_cpu_usage' in rules and metrics.cpu_usage > rules['max_cpu_usage']:
            self.logger.warning(f"CPU usage {metrics.cpu_usage} exceeds limit {rules['max_cpu_usage']}")
            return False
        
        if 'max_response_time' in rules and metrics.response_time > rules['max_response_time']:
            self.logger.warning(f"Response time {metrics.response_time} exceeds limit {rules['max_response_time']}")
            return False
        
        if 'throughput_threshold' in rules and metrics.throughput < rules['throughput_threshold']:
            self.logger.warning(f"Throughput {metrics.throughput} below threshold {rules['throughput_threshold']}")
            return False
        
        return True
    
    def _save_metrics(self, result: Dict[str, Any]):
        """Save performance metrics to file."""
        metrics_dir = self.base_path / 'metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = metrics_dir / f"performance_metrics_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        self.logger.info(f"Saved performance metrics to {metrics_file}")

async def run_load_test(
    framework: PerformanceTestFramework,
    config: IntegrationTestConfig,
    num_concurrent: int = 5,
    duration: int = 300
) -> Dict[str, Any]:
    """Run a load test with multiple concurrent operations."""
    start_time = time.time()
    tasks = []
    
    async def run_single_test():
        while time.time() - start_time < duration:
            await framework.run_performance_test(config)
    
    # Create concurrent tasks
    for _ in range(num_concurrent):
        tasks.append(asyncio.create_task(run_single_test()))
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    
    # Return final metrics
    return {
        'duration': time.time() - start_time,
        'num_concurrent': num_concurrent,
        'metrics_history': framework.monitor.metrics_history
    }

async def run_stress_test(
    framework: PerformanceTestFramework,
    config: IntegrationTestConfig,
    start_concurrent: int = 1,
    max_concurrent: int = 20,
    step: int = 2,
    duration_per_step: int = 60
) -> Dict[str, Any]:
    """Run a stress test with increasing concurrent operations."""
    results = []
    
    for num_concurrent in range(start_concurrent, max_concurrent + 1, step):
        framework.logger.info(f"Running stress test with {num_concurrent} concurrent operations")
        
        load_result = await run_load_test(
            framework,
            config,
            num_concurrent=num_concurrent,
            duration=duration_per_step
        )
        
        results.append({
            'num_concurrent': num_concurrent,
            'metrics': load_result
        })
    
    return {
        'stress_test_results': results,
        'max_concurrent': max_concurrent,
        'duration_per_step': duration_per_step
    } 