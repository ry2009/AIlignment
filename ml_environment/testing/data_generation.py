from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch
from pathlib import Path
import json
import logging
from dataclasses import dataclass
import random
from datetime import datetime
import hashlib
from abc import ABC, abstractmethod

@dataclass
class DataGenerationConfig:
    environment: str
    data_size: int
    noise_level: float
    seed: int
    parameters: Dict[str, Any]
    validation_rules: Dict[str, Any]

@dataclass
class DatasetMetadata:
    dataset_id: str
    environment: str
    generation_time: float
    size: int
    features: Dict[str, str]
    statistics: Dict[str, float]
    validation_results: Dict[str, bool]
    hash: str

class DataValidator:
    """Validates generated data against defined rules."""
    
    def __init__(self):
        self.validators = {
            'range': self._validate_range,
            'uniqueness': self._validate_uniqueness,
            'completeness': self._validate_completeness,
            'distribution': self._validate_distribution,
            'correlation': self._validate_correlation
        }

    def validate(self, data: Any, rules: Dict[str, Any]) -> Dict[str, bool]:
        """Validate data against all specified rules."""
        results = {}
        for rule_name, rule_params in rules.items():
            if rule_name in self.validators:
                results[rule_name] = self.validators[rule_name](data, rule_params)
        return results

    def _validate_range(self, data: np.ndarray, params: Dict) -> bool:
        """Validate data falls within specified range."""
        min_val, max_val = params.get('min', -np.inf), params.get('max', np.inf)
        return np.all((data >= min_val) & (data <= max_val))

    def _validate_uniqueness(self, data: np.ndarray, params: Dict) -> bool:
        """Validate uniqueness constraints."""
        threshold = params.get('threshold', 1.0)
        unique_ratio = len(np.unique(data)) / len(data)
        return unique_ratio >= threshold

    def _validate_completeness(self, data: np.ndarray, params: Dict) -> bool:
        """Validate data completeness."""
        threshold = params.get('threshold', 0.99)
        completeness_ratio = 1 - np.isnan(data).sum() / data.size
        return completeness_ratio >= threshold

    def _validate_distribution(self, data: np.ndarray, params: Dict) -> bool:
        """Validate data follows expected distribution."""
        dist_type = params.get('type', 'normal')
        if dist_type == 'normal':
            _, p_value = scipy.stats.normaltest(data)
            return p_value >= params.get('p_threshold', 0.05)
        return True

    def _validate_correlation(self, data: np.ndarray, params: Dict) -> bool:
        """Validate correlation constraints."""
        threshold = params.get('threshold', 0.8)
        if data.ndim == 2:
            corr_matrix = np.corrcoef(data.T)
            return np.all(np.abs(corr_matrix[np.triu_indices(len(corr_matrix), 1)]) <= threshold)
        return True

class BaseDataGenerator(ABC):
    """Base class for environment-specific data generators."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validator = DataValidator()
        self._setup_logging()

    def _setup_logging(self):
        log_dir = self.base_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / f'{self.__class__.__name__}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    @abstractmethod
    def generate(self, config: DataGenerationConfig) -> tuple[Any, DatasetMetadata]:
        """Generate synthetic data according to configuration."""
        pass

    def _compute_hash(self, data: Any) -> str:
        """Compute hash of the generated data."""
        if isinstance(data, np.ndarray):
            return hashlib.sha256(data.tobytes()).hexdigest()
        elif isinstance(data, torch.Tensor):
            return hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()
        else:
            return hashlib.sha256(str(data).encode()).hexdigest()

class AdminDataGenerator(BaseDataGenerator):
    """Generates synthetic data for administrative environment."""
    
    def generate(self, config: DataGenerationConfig) -> tuple[np.ndarray, DatasetMetadata]:
        try:
            np.random.seed(config.seed)
            
            # Generate synthetic admin data
            data = {
                'resource_usage': np.random.uniform(0, 1, (config.data_size, 1)),
                'response_times': np.random.exponential(50, (config.data_size, 1)),
                'error_rates': np.random.beta(2, 10, (config.data_size, 1))
            }
            
            # Add noise
            if config.noise_level > 0:
                for key in data:
                    noise = np.random.normal(0, config.noise_level, data[key].shape)
                    data[key] = np.clip(data[key] + noise, 0, 1)
            
            # Validate data
            validation_results = self.validator.validate(
                data,
                config.validation_rules
            )
            
            # Create metadata
            metadata = DatasetMetadata(
                dataset_id=f"admin_{int(time.time())}",
                environment="admin",
                generation_time=time.time(),
                size=config.data_size,
                features={'type': 'continuous', 'shape': str(data['resource_usage'].shape)},
                statistics={
                    'mean_resource_usage': float(np.mean(data['resource_usage'])),
                    'mean_response_time': float(np.mean(data['response_times'])),
                    'mean_error_rate': float(np.mean(data['error_rates']))
                },
                validation_results=validation_results,
                hash=self._compute_hash(data)
            )
            
            return data, metadata
            
        except Exception as e:
            self.logger.error(f"Error generating admin data: {str(e)}")
            raise

class ForumDataGenerator(BaseDataGenerator):
    """Generates synthetic data for forum environment."""
    
    def generate(self, config: DataGenerationConfig) -> tuple[Dict, DatasetMetadata]:
        try:
            random.seed(config.seed)
            
            # Generate synthetic forum data
            data = {
                'posts': [
                    {
                        'content': f"Post content {i}",
                        'toxicity_score': random.uniform(0, 0.3),
                        'quality_score': random.uniform(0.7, 1.0)
                    }
                    for i in range(config.data_size)
                ]
            }
            
            # Validate data
            validation_results = self.validator.validate(
                np.array([p['toxicity_score'] for p in data['posts']]),
                config.validation_rules
            )
            
            # Create metadata
            metadata = DatasetMetadata(
                dataset_id=f"forum_{int(time.time())}",
                environment="forum",
                generation_time=time.time(),
                size=config.data_size,
                features={'type': 'text_with_scores'},
                statistics={
                    'mean_toxicity': np.mean([p['toxicity_score'] for p in data['posts']]),
                    'mean_quality': np.mean([p['quality_score'] for p in data['posts']])
                },
                validation_results=validation_results,
                hash=self._compute_hash(str(data))
            )
            
            return data, metadata
            
        except Exception as e:
            self.logger.error(f"Error generating forum data: {str(e)}")
            raise

class GameDataGenerator(BaseDataGenerator):
    """Generates synthetic data for game environment."""
    
    def generate(self, config: DataGenerationConfig) -> tuple[Dict, DatasetMetadata]:
        try:
            np.random.seed(config.seed)
            
            # Generate synthetic game data
            data = {
                'episodes': [
                    {
                        'actions': np.random.randint(0, 4, size=20),
                        'rewards': np.random.normal(0, 1, size=20),
                        'win': random.choice([True, False])
                    }
                    for _ in range(config.data_size)
                ]
            }
            
            # Validate data
            validation_results = self.validator.validate(
                np.array([e['rewards'].mean() for e in data['episodes']]),
                config.validation_rules
            )
            
            # Create metadata
            metadata = DatasetMetadata(
                dataset_id=f"game_{int(time.time())}",
                environment="game",
                generation_time=time.time(),
                size=config.data_size,
                features={'type': 'sequential'},
                statistics={
                    'mean_reward': float(np.mean([e['rewards'].mean() for e in data['episodes']])),
                    'win_rate': float(np.mean([e['win'] for e in data['episodes']]))
                },
                validation_results=validation_results,
                hash=self._compute_hash(str(data))
            )
            
            return data, metadata
            
        except Exception as e:
            self.logger.error(f"Error generating game data: {str(e)}")
            raise

class DataGenerationPipeline:
    """Manages data generation across all environments."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.logger = logging.getLogger('DataGeneration')
        self._setup_logging()
        
        # Initialize generators
        self.generators = {
            'admin': AdminDataGenerator(base_path),
            'forum': ForumDataGenerator(base_path),
            'game': GameDataGenerator(base_path)
        }

    def _setup_logging(self):
        log_dir = self.base_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / 'data_generation.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def generate_data(self, config: DataGenerationConfig) -> tuple[Any, DatasetMetadata]:
        """Generate data for specified environment."""
        try:
            if config.environment not in self.generators:
                raise ValueError(f"Unknown environment: {config.environment}")
                
            generator = self.generators[config.environment]
            data, metadata = generator.generate(config)
            
            # Save data and metadata
            self._save_dataset(data, metadata)
            
            return data, metadata
            
        except Exception as e:
            self.logger.error(f"Error in data generation: {str(e)}")
            raise

    def _save_dataset(self, data: Any, metadata: DatasetMetadata):
        """Save generated dataset and its metadata."""
        try:
            dataset_dir = self.base_path / 'datasets' / metadata.environment
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Save data
            data_path = dataset_dir / f"{metadata.dataset_id}_data.npz"
            if isinstance(data, np.ndarray):
                np.savez_compressed(data_path, data=data)
            else:
                with open(data_path.with_suffix('.json'), 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Save metadata
            metadata_path = dataset_dir / f"{metadata.dataset_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.__dict__, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving dataset: {str(e)}")
            raise 