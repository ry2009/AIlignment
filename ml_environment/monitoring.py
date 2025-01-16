import time
import psutil
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class MonitoringMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    gpu_utilization: Optional[float]
    network_io: Dict[str, int]
    timestamp: float

class EnvironmentMonitor:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.metrics_path = base_path / 'logs' / 'metrics'
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('EnvironmentMonitor')
        
        # Initialize monitoring state
        self.active_monitors: Dict[str, bool] = {}
        self._setup_logging()

    def _setup_logging(self):
        """Configure monitoring-specific logging."""
        handler = logging.FileHandler(self.base_path / 'logs' / 'monitoring.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def start_monitoring(self, session_id: str):
        """Start monitoring a training session."""
        if session_id in self.active_monitors:
            self.logger.warning(f"Monitoring already active for session {session_id}")
            return

        self.active_monitors[session_id] = True
        self.logger.info(f"Started monitoring session {session_id}")

    def stop_monitoring(self, session_id: str):
        """Stop monitoring a training session."""
        if session_id in self.active_monitors:
            self.active_monitors[session_id] = False
            self.logger.info(f"Stopped monitoring session {session_id}")

    def collect_metrics(self, session_id: str) -> MonitoringMetrics:
        """Collect current system metrics for a session."""
        metrics = MonitoringMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            gpu_utilization=self._get_gpu_metrics(),
            network_io=dict(psutil.net_io_counters()._asdict()),
            timestamp=time.time()
        )
        
        # Save metrics to file
        self._save_metrics(session_id, metrics)
        return metrics

    def _get_gpu_metrics(self) -> Optional[float]:
        """Get GPU utilization if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except (ImportError, pynvml.NVMLError):
            return None

    def _save_metrics(self, session_id: str, metrics: MonitoringMetrics):
        """Save metrics to disk."""
        metrics_file = self.metrics_path / f"{session_id}.json"
        
        # Convert metrics to dictionary
        metrics_dict = {
            'timestamp': metrics.timestamp,
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'disk_usage': metrics.disk_usage,
            'gpu_utilization': metrics.gpu_utilization,
            'network_io': metrics.network_io
        }
        
        # Append metrics to file
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics_dict) + '\n')

    def get_session_metrics(self, session_id: str, start_time: Optional[float] = None) -> list:
        """Retrieve metrics for a session."""
        metrics_file = self.metrics_path / f"{session_id}.json"
        if not metrics_file.exists():
            return []
        
        metrics = []
        with open(metrics_file, 'r') as f:
            for line in f:
                metric = json.loads(line)
                if start_time is None or metric['timestamp'] >= start_time:
                    metrics.append(metric)
        
        return metrics

    def check_resource_violations(self, session_id: str, resource_limits: Dict) -> list:
        """Check for resource limit violations."""
        metrics = self.get_session_metrics(session_id)
        violations = []
        
        for metric in metrics:
            if metric['memory_percent'] > resource_limits.get('max_memory_percent', 90):
                violations.append({
                    'type': 'memory',
                    'value': metric['memory_percent'],
                    'timestamp': metric['timestamp']
                })
            if metric['cpu_percent'] > resource_limits.get('max_cpu_percent', 90):
                violations.append({
                    'type': 'cpu',
                    'value': metric['cpu_percent'],
                    'timestamp': metric['timestamp']
                })
        
        return violations 