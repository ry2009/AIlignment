from dataclasses import dataclass
from typing import Dict, Optional, List
import uuid
import logging
import resource
import threading
from pathlib import Path

@dataclass
class ResourceLimits:
    max_memory_mb: int = 4096  # 4GB default
    max_cpu_time: int = 3600   # 1 hour default
    max_processes: int = 8
    max_disk_space_mb: int = 10240  # 10GB default

@dataclass
class TrainingSession:
    session_id: str
    user_id: str
    resource_limits: ResourceLimits
    start_time: float
    model_artifacts_path: Path
    checkpoints_path: Path
    is_active: bool = True
    current_resource_usage: Dict[str, float] = None

class MLEnvironment:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.active_sessions: Dict[str, TrainingSession] = {}
        self.session_locks: Dict[str, threading.Lock] = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MLEnvironment')
        
        # Initialize directories
        self._initialize_directories()

    def _initialize_directories(self):
        """Create necessary directory structure for ML environment."""
        (self.base_path / 'models').mkdir(parents=True, exist_ok=True)
        (self.base_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.base_path / 'logs').mkdir(parents=True, exist_ok=True)
        (self.base_path / 'data').mkdir(parents=True, exist_ok=True)

    def create_session(self, user_id: str, custom_limits: Optional[ResourceLimits] = None) -> str:
        """Create a new isolated training session."""
        session_id = str(uuid.uuid4())
        resource_limits = custom_limits or ResourceLimits()
        
        # Create session-specific directories
        model_path = self.base_path / 'models' / session_id
        checkpoint_path = self.base_path / 'checkpoints' / session_id
        model_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        session = TrainingSession(
            session_id=session_id,
            user_id=user_id,
            resource_limits=resource_limits,
            start_time=time.time(),
            model_artifacts_path=model_path,
            checkpoints_path=checkpoint_path
        )

        self.active_sessions[session_id] = session
        self.session_locks[session_id] = threading.Lock()
        self.logger.info(f"Created new training session {session_id} for user {user_id}")
        
        return session_id

    def enforce_resource_limits(self, session_id: str):
        """Enforce resource limits for a session."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Invalid session ID: {session_id}")

        limits = session.resource_limits
        
        # Set memory limit
        resource.setrlimit(resource.RLIMIT_AS, (limits.max_memory_mb * 1024 * 1024, -1))
        
        # Set CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (limits.max_cpu_time, -1))
        
        # Set process limit
        resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, -1))

    def cleanup_session(self, session_id: str):
        """Clean up resources associated with a session."""
        with self.session_locks.get(session_id, threading.Lock()):
            session = self.active_sessions.get(session_id)
            if not session:
                return

            session.is_active = False
            
            # Archive model artifacts and checkpoints if needed
            # For now, we'll just log the cleanup
            self.logger.info(f"Cleaning up session {session_id}")
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            del self.session_locks[session_id]

    def get_session_status(self, session_id: str) -> Dict:
        """Get current status of a training session."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Invalid session ID: {session_id}")

        # Get current resource usage
        usage = {
            'memory_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            'cpu_time': resource.getrusage(resource.RUSAGE_SELF).ru_utime,
            'is_active': session.is_active,
            'runtime': time.time() - session.start_time
        }
        
        session.current_resource_usage = usage
        return usage 