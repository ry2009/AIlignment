from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
import json
import hashlib
from pathlib import Path
import time
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import threading
import shutil

@dataclass
class ModelVersion:
    version_id: str
    model_hash: str
    parent_version: Optional[str]
    timestamp: float
    metadata: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    privacy_config: Optional[Dict[str, Any]]
    training_config: Dict[str, Any]

class ModelVersionControl:
    def __init__(self, base_path: Path, session_id: str):
        self.base_path = base_path
        self.session_id = session_id
        self.versions_dir = base_path / 'model_versions'
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f'ModelVersioning-{session_id}')
        self._setup_logging()
        self._version_lock = threading.Lock()

    def _setup_logging(self):
        log_dir = self.base_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / f'model_versioning_{self.session_id}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _compute_model_hash(self, model: nn.Module) -> str:
        """Compute a hash of the model's state dict."""
        state_dict = model.state_dict()
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            param_data = state_dict[key].cpu().numpy().tobytes()
            hasher.update(param_data)
        return hasher.hexdigest()

    def save_version(self, model: nn.Module, metadata: Dict[str, Any],
                    metrics: Dict[str, float], tags: List[str],
                    privacy_config: Optional[Dict[str, Any]] = None,
                    training_config: Optional[Dict[str, Any]] = None,
                    parent_version: Optional[str] = None) -> str:
        """Save a new version of the model with metadata."""
        with self._version_lock:
            try:
                # Generate version ID and compute model hash
                timestamp = time.time()
                model_hash = self._compute_model_hash(model)
                version_id = f"v_{int(timestamp)}_{model_hash[:8]}"

                # Create version object
                version = ModelVersion(
                    version_id=version_id,
                    model_hash=model_hash,
                    parent_version=parent_version,
                    timestamp=timestamp,
                    metadata=metadata,
                    metrics=metrics,
                    tags=tags,
                    privacy_config=privacy_config,
                    training_config=training_config or {}
                )

                # Save model and version info
                version_dir = self.versions_dir / version_id
                version_dir.mkdir(exist_ok=True)

                # Save model state
                torch.save(model.state_dict(), version_dir / 'model.pt')

                # Save version metadata
                with open(version_dir / 'version.json', 'w') as f:
                    json.dump(asdict(version), f, indent=2)

                self.logger.info(f"Saved model version {version_id}")
                return version_id

            except Exception as e:
                self.logger.error(f"Error saving version: {str(e)}")
                raise

    def load_version(self, version_id: str, model: nn.Module) -> Optional[ModelVersion]:
        """Load a specific version of the model."""
        try:
            version_dir = self.versions_dir / version_id
            if not version_dir.exists():
                self.logger.error(f"Version {version_id} not found")
                return None

            # Load version metadata
            with open(version_dir / 'version.json', 'r') as f:
                version_data = json.load(f)
            version = ModelVersion(**version_data)

            # Load model state
            state_dict = torch.load(version_dir / 'model.pt')
            model.load_state_dict(state_dict)

            # Verify model hash
            current_hash = self._compute_model_hash(model)
            if current_hash != version.model_hash:
                self.logger.warning(f"Model hash mismatch for version {version_id}")

            return version

        except Exception as e:
            self.logger.error(f"Error loading version {version_id}: {str(e)}")
            return None

    def get_version_history(self) -> List[ModelVersion]:
        """Get the history of all model versions."""
        try:
            versions = []
            for version_dir in sorted(self.versions_dir.glob('v_*')):
                with open(version_dir / 'version.json', 'r') as f:
                    version_data = json.load(f)
                versions.append(ModelVersion(**version_data))
            return versions
        except Exception as e:
            self.logger.error(f"Error getting version history: {str(e)}")
            return []

    def get_version_by_tag(self, tag: str) -> Optional[ModelVersion]:
        """Find a version by tag."""
        try:
            for version in self.get_version_history():
                if tag in version.tags:
                    return version
            return None
        except Exception as e:
            self.logger.error(f"Error finding version by tag: {str(e)}")
            return None

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two model versions."""
        try:
            v1 = self.get_version_info(version_id1)
            v2 = self.get_version_info(version_id2)
            
            if not v1 or not v2:
                return {}

            return {
                'metrics_diff': {
                    k: v2.metrics.get(k, 0) - v1.metrics.get(k, 0)
                    for k in set(v1.metrics) | set(v2.metrics)
                },
                'timestamp_diff': v2.timestamp - v1.timestamp,
                'privacy_changes': self._compare_privacy_configs(
                    v1.privacy_config,
                    v2.privacy_config
                ),
                'training_changes': self._compare_configs(
                    v1.training_config,
                    v2.training_config
                )
            }
        except Exception as e:
            self.logger.error(f"Error comparing versions: {str(e)}")
            return {}

    def _compare_privacy_configs(self, config1: Optional[Dict], 
                               config2: Optional[Dict]) -> Dict[str, Any]:
        """Compare privacy configurations between versions."""
        if not config1 or not config2:
            return {'privacy_config_changed': config1 != config2}
            
        changes = {}
        for key in set(config1) | set(config2):
            if key not in config1:
                changes[key] = {'added': config2[key]}
            elif key not in config2:
                changes[key] = {'removed': config1[key]}
            elif config1[key] != config2[key]:
                changes[key] = {
                    'from': config1[key],
                    'to': config2[key]
                }
        return changes

    def _compare_configs(self, config1: Dict, config2: Dict) -> Dict[str, Any]:
        """Compare general configurations between versions."""
        changes = {}
        for key in set(config1) | set(config2):
            if key not in config1:
                changes[key] = {'added': config2[key]}
            elif key not in config2:
                changes[key] = {'removed': config1[key]}
            elif config1[key] != config2[key]:
                changes[key] = {
                    'from': config1[key],
                    'to': config2[key]
                }
        return changes

    def get_version_info(self, version_id: str) -> Optional[ModelVersion]:
        """Get information about a specific version."""
        try:
            version_dir = self.versions_dir / version_id
            if not version_dir.exists():
                return None
                
            with open(version_dir / 'version.json', 'r') as f:
                version_data = json.load(f)
            return ModelVersion(**version_data)
        except Exception as e:
            self.logger.error(f"Error getting version info: {str(e)}")
            return None

    def delete_version(self, version_id: str) -> bool:
        """Delete a specific version."""
        with self._version_lock:
            try:
                version_dir = self.versions_dir / version_id
                if not version_dir.exists():
                    return False
                    
                shutil.rmtree(version_dir)
                self.logger.info(f"Deleted version {version_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error deleting version {version_id}: {str(e)}")
                return False 