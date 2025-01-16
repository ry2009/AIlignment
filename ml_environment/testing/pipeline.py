from typing import Dict, List, Optional, Any, Callable
import threading
import schedule
import time
from pathlib import Path
import logging
import json
from datetime import datetime
from dataclasses import dataclass
import queue
from concurrent.futures import ThreadPoolExecutor
import traceback

@dataclass
class TestConfig:
    test_name: str
    test_type: str  # unit, integration, end_to_end
    environment: str
    schedule: str  # cron format
    timeout: int
    retries: int
    dependencies: List[str]
    parameters: Dict[str, Any]

@dataclass
class TestResult:
    test_id: str
    test_name: str
    status: str  # success, failure, error, skipped
    start_time: float
    end_time: float
    duration: float
    environment: str
    error_message: Optional[str]
    stack_trace: Optional[str]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]

class TestScheduler:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.logger = logging.getLogger('TestScheduler')
        self._setup_logging()
        self.test_queue = queue.PriorityQueue()
        self.running_tests = {}
        self.test_results = {}
        self._scheduler_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=10)

    def _setup_logging(self):
        log_dir = self.base_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / 'test_scheduler.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def schedule_test(self, config: TestConfig, test_func: Callable) -> str:
        """Schedule a test to run according to its configuration."""
        try:
            test_id = f"{config.test_name}_{int(time.time())}"
            
            # Schedule the test
            if config.schedule == "immediate":
                self.test_queue.put((0, test_id, config, test_func))
            else:
                schedule.every().day.at(config.schedule).do(
                    self._add_to_queue, test_id, config, test_func
                )
            
            self.logger.info(f"Scheduled test: {test_id}")
            return test_id
            
        except Exception as e:
            self.logger.error(f"Error scheduling test: {str(e)}")
            raise

    def _add_to_queue(self, test_id: str, config: TestConfig, test_func: Callable):
        """Add a test to the execution queue."""
        priority = time.time()
        self.test_queue.put((priority, test_id, config, test_func))

    def run_test(self, test_id: str, config: TestConfig, test_func: Callable) -> TestResult:
        """Run a single test and return its result."""
        start_time = time.time()
        try:
            # Check dependencies
            for dep in config.dependencies:
                if dep not in self.test_results or \
                   self.test_results[dep].status != 'success':
                    return TestResult(
                        test_id=test_id,
                        test_name=config.test_name,
                        status='skipped',
                        start_time=start_time,
                        end_time=time.time(),
                        duration=0,
                        environment=config.environment,
                        error_message="Dependencies not met",
                        stack_trace=None,
                        metrics={},
                        artifacts={}
                    )

            # Run the test
            metrics = test_func(**config.parameters)
            end_time = time.time()
            
            return TestResult(
                test_id=test_id,
                test_name=config.test_name,
                status='success',
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                environment=config.environment,
                error_message=None,
                stack_trace=None,
                metrics=metrics,
                artifacts={}
            )
            
        except Exception as e:
            end_time = time.time()
            return TestResult(
                test_id=test_id,
                test_name=config.test_name,
                status='error',
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                environment=config.environment,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                metrics={},
                artifacts={}
            )

    def start(self):
        """Start the test scheduler."""
        self.logger.info("Starting test scheduler")
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.start()

    def stop(self):
        """Stop the test scheduler."""
        self.logger.info("Stopping test scheduler")
        self._running = False
        self._scheduler_thread.join()
        self.executor.shutdown()

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                schedule.run_pending()
                
                # Check for tests in queue
                if not self.test_queue.empty():
                    _, test_id, config, test_func = self.test_queue.get()
                    
                    # Submit test to executor
                    future = self.executor.submit(
                        self.run_test, test_id, config, test_func
                    )
                    future.add_done_callback(
                        lambda f, tid=test_id: self._handle_test_completion(tid, f)
                    )
                    
                    self.running_tests[test_id] = future
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")

    def _handle_test_completion(self, test_id: str, future):
        """Handle test completion and save results."""
        try:
            result = future.result()
            self.test_results[test_id] = result
            self._save_test_result(result)
            del self.running_tests[test_id]
        except Exception as e:
            self.logger.error(f"Error handling test completion: {str(e)}")

    def _save_test_result(self, result: TestResult):
        """Save test result to disk."""
        try:
            results_dir = self.base_path / 'test_results'
            results_dir.mkdir(exist_ok=True)
            
            result_path = results_dir / f"{result.test_id}.json"
            with open(result_path, 'w') as f:
                json.dump({
                    'test_id': result.test_id,
                    'test_name': result.test_name,
                    'status': result.status,
                    'start_time': result.start_time,
                    'end_time': result.end_time,
                    'duration': result.duration,
                    'environment': result.environment,
                    'error_message': result.error_message,
                    'stack_trace': result.stack_trace,
                    'metrics': result.metrics,
                    'artifacts': result.artifacts
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving test result: {str(e)}")

    def get_test_status(self, test_id: str) -> Optional[TestResult]:
        """Get the status of a specific test."""
        return self.test_results.get(test_id)

    def get_environment_results(self, environment: str) -> List[TestResult]:
        """Get all test results for a specific environment."""
        return [
            result for result in self.test_results.values()
            if result.environment == environment
        ]

    def get_test_metrics(self, test_id: str) -> Dict[str, float]:
        """Get metrics for a specific test."""
        result = self.test_results.get(test_id)
        return result.metrics if result else {} 