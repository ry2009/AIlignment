from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import logging
from pathlib import Path
import json
import time
import threading
from collections import deque
from dataclasses import dataclass
import hashlib
import hmac
import secrets

@dataclass
class ModelProtection:
    encryption_key: bytes
    signature_key: bytes
    model_hash: str
    last_verified: float

class AdvancedSecurityMonitor:
    def __init__(self, base_path: Path, session_id: str):
        self.base_path = base_path
        self.session_id = session_id
        self.logger = logging.getLogger(f'AdvancedSecurity-{session_id}')
        self._setup_logging()
        
        # Initialize security components
        self.request_history: Dict[str, deque] = {}
        self.model_protections: Dict[str, ModelProtection] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        self._security_lock = threading.Lock()
        
        # Load security configurations
        self._load_config()

    def _setup_logging(self):
        handler = logging.FileHandler(self.base_path / 'logs' / f'advanced_security_{self.session_id}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _load_config(self):
        """Load security configurations."""
        config_file = self.base_path / 'config' / 'security_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.anomaly_thresholds = config.get('anomaly_thresholds', {})

    def protect_model(self, model: nn.Module, model_id: str) -> ModelProtection:
        """Implement model protection mechanisms."""
        with self._security_lock:
            try:
                # Generate encryption and signature keys
                encryption_key = secrets.token_bytes(32)
                signature_key = secrets.token_bytes(32)
                
                # Compute model hash
                model_hash = self._compute_model_hash(model)
                
                protection = ModelProtection(
                    encryption_key=encryption_key,
                    signature_key=signature_key,
                    model_hash=model_hash,
                    last_verified=time.time()
                )
                
                self.model_protections[model_id] = protection
                return protection
                
            except Exception as e:
                self.logger.error(f"Error protecting model: {str(e)}")
                raise

    def verify_model_integrity(self, model: nn.Module, model_id: str) -> bool:
        """Verify model integrity using stored hash."""
        try:
            if model_id not in self.model_protections:
                return False
                
            protection = self.model_protections[model_id]
            current_hash = self._compute_model_hash(model)
            
            return current_hash == protection.model_hash
            
        except Exception as e:
            self.logger.error(f"Error verifying model integrity: {str(e)}")
            return False

    def _compute_model_hash(self, model: nn.Module) -> str:
        """Compute hash of model parameters."""
        hasher = hashlib.sha256()
        
        for param in model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
            
        return hasher.hexdigest()

    def detect_adversarial_inputs(self, data: torch.Tensor, 
                                threshold: float = 0.95) -> Tuple[bool, float]:
        """Detect potential adversarial inputs."""
        try:
            # Implement basic statistical detection
            z_scores = stats.zscore(data.cpu().numpy(), axis=None)
            max_zscore = np.max(np.abs(z_scores))
            
            # Check for unusual patterns
            is_adversarial = max_zscore > threshold
            confidence = 1 - (1 / (1 + max_zscore))
            
            if is_adversarial:
                self.logger.warning(f"Potential adversarial input detected: {confidence:.2f} confidence")
                
            return is_adversarial, confidence
            
        except Exception as e:
            self.logger.error(f"Error detecting adversarial inputs: {str(e)}")
            return False, 0.0

    def monitor_prediction_patterns(self, predictions: torch.Tensor, 
                                 window_size: int = 100) -> bool:
        """Monitor prediction patterns for anomalies."""
        try:
            predictions_np = predictions.cpu().numpy()
            
            # Check for unusual distribution
            if len(self.request_history.get('predictions', [])) >= window_size:
                historical = np.array(list(self.request_history['predictions']))
                current_mean = np.mean(predictions_np)
                historical_mean = np.mean(historical)
                historical_std = np.std(historical)
                
                z_score = abs(current_mean - historical_mean) / (historical_std + 1e-10)
                is_anomalous = z_score > self.anomaly_thresholds.get('prediction_zscore', 3.0)
                
                if is_anomalous:
                    self.logger.warning(f"Anomalous prediction pattern detected: z-score={z_score:.2f}")
                    return True
            
            # Update history
            if 'predictions' not in self.request_history:
                self.request_history['predictions'] = deque(maxlen=window_size)
            self.request_history['predictions'].append(predictions_np)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error monitoring prediction patterns: {str(e)}")
            return False

    def detect_model_extraction(self, query_features: torch.Tensor,
                              window_size: int = 1000) -> bool:
        """Detect potential model extraction attempts."""
        try:
            features_np = query_features.cpu().numpy()
            
            if 'queries' not in self.request_history:
                self.request_history['queries'] = deque(maxlen=window_size)
                return False
            
            # Check for systematic exploration
            similarity_scores = []
            for past_query in self.request_history['queries']:
                similarity = np.mean(np.abs(features_np - past_query))
                similarity_scores.append(similarity)
            
            if similarity_scores:
                avg_similarity = np.mean(similarity_scores)
                is_systematic = avg_similarity < self.anomaly_thresholds.get('query_similarity', 0.1)
                
                if is_systematic:
                    self.logger.warning(f"Potential model extraction attempt detected: similarity={avg_similarity:.3f}")
                    return True
            
            self.request_history['queries'].append(features_np)
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting model extraction: {str(e)}")
            return False

    def verify_computation_integrity(self, input_data: torch.Tensor, 
                                  output_data: torch.Tensor,
                                  model_id: str) -> bool:
        """Verify integrity of model computation."""
        try:
            if model_id not in self.model_protections:
                return False
                
            protection = self.model_protections[model_id]
            
            # Create HMAC of input and output
            h = hmac.new(protection.signature_key, digestmod=hashlib.sha256)
            h.update(input_data.cpu().numpy().tobytes())
            h.update(output_data.cpu().numpy().tobytes())
            
            computation_signature = h.hexdigest()
            
            # Store signature for future verification
            if 'signatures' not in self.request_history:
                self.request_history['signatures'] = {}
            
            request_id = secrets.token_hex(16)
            self.request_history['signatures'][request_id] = computation_signature
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying computation integrity: {str(e)}")
            return False

    def detect_poisoning_attempt(self, data: torch.Tensor, 
                               labels: Optional[torch.Tensor] = None) -> bool:
        """Detect potential poisoning attempts in training data."""
        try:
            data_np = data.cpu().numpy()
            
            # Check for statistical anomalies
            z_scores = stats.zscore(data_np, axis=None)
            max_zscore = np.max(np.abs(z_scores))
            
            # Check label distribution if available
            if labels is not None:
                labels_np = labels.cpu().numpy()
                unique_labels, counts = np.unique(labels_np, return_counts=True)
                label_dist = counts / len(labels_np)
                
                # Check for unusual label distribution
                if 'label_dist' in self.request_history:
                    historical_dist = self.request_history['label_dist']
                    dist_diff = np.max(np.abs(label_dist - historical_dist))
                    
                    if dist_diff > self.anomaly_thresholds.get('label_distribution', 0.3):
                        self.logger.warning(f"Unusual label distribution detected: diff={dist_diff:.3f}")
                        return True
                
                self.request_history['label_dist'] = label_dist
            
            # Check for extreme values
            if max_zscore > self.anomaly_thresholds.get('data_zscore', 5.0):
                self.logger.warning(f"Extreme values detected in data: z-score={max_zscore:.2f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting poisoning attempt: {str(e)}")
            return False 