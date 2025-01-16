import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class DataValidationRule:
    name: str
    validation_type: str  # 'range', 'categorical', 'pattern', 'custom'
    parameters: Dict
    error_message: str

@dataclass
class DatasetMetadata:
    name: str
    hash: str
    row_count: int
    column_types: Dict[str, str]
    validation_rules: List[DataValidationRule]
    creation_timestamp: float
    last_modified: float

class SecureDataPipeline:
    def __init__(self, base_path: Path, session_id: str):
        self.base_path = base_path
        self.session_id = session_id
        self.data_path = base_path / 'data' / session_id
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f'DataPipeline-{session_id}')
        self._setup_logging()
        
        self.metadata: Dict[str, DatasetMetadata] = {}
        self._load_metadata()
        
        self._data_lock = threading.Lock()

    def _setup_logging(self):
        handler = logging.FileHandler(self.base_path / 'logs' / f'data_pipeline_{self.session_id}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _load_metadata(self):
        metadata_file = self.data_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                for name, meta in data.items():
                    self.metadata[name] = DatasetMetadata(**meta)

    def _save_metadata(self):
        metadata_file = self.data_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({name: vars(meta) for name, meta in self.metadata.items()}, f)

    def _compute_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """Compute a hash of the data for integrity checking."""
        if isinstance(data, pd.DataFrame):
            data_bytes = data.to_csv(index=False).encode()
        else:
            data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def validate_data(self, data: pd.DataFrame, rules: List[DataValidationRule]) -> Tuple[bool, List[str]]:
        """Validate data against defined rules."""
        errors = []
        for rule in rules:
            if rule.validation_type == 'range':
                col = rule.parameters['column']
                min_val = rule.parameters.get('min')
                max_val = rule.parameters.get('max')
                
                if min_val is not None and data[col].min() < min_val:
                    errors.append(f"{rule.name}: {rule.error_message}")
                if max_val is not None and data[col].max() > max_val:
                    errors.append(f"{rule.name}: {rule.error_message}")
                    
            elif rule.validation_type == 'categorical':
                col = rule.parameters['column']
                allowed_values = set(rule.parameters['values'])
                if not set(data[col].unique()).issubset(allowed_values):
                    errors.append(f"{rule.name}: {rule.error_message}")
                    
            elif rule.validation_type == 'pattern':
                col = rule.parameters['column']
                pattern = rule.parameters['regex']
                if not data[col].str.match(pattern).all():
                    errors.append(f"{rule.name}: {rule.error_message}")
                    
            elif rule.validation_type == 'custom':
                validation_func = rule.parameters['function']
                if not validation_func(data):
                    errors.append(f"{rule.name}: {rule.error_message}")
        
        return len(errors) == 0, errors

    def ingest_data(self, name: str, data: Union[pd.DataFrame, np.ndarray], 
                   validation_rules: Optional[List[DataValidationRule]] = None) -> bool:
        """Ingest data with validation and security checks."""
        with self._data_lock:
            try:
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data)
                
                # Validate data if rules provided
                if validation_rules:
                    is_valid, errors = self.validate_data(data, validation_rules)
                    if not is_valid:
                        self.logger.error(f"Data validation failed for {name}: {errors}")
                        return False
                
                # Compute hash and create metadata
                data_hash = self._compute_hash(data)
                metadata = DatasetMetadata(
                    name=name,
                    hash=data_hash,
                    row_count=len(data),
                    column_types={col: str(dtype) for col, dtype in data.dtypes.items()},
                    validation_rules=validation_rules or [],
                    creation_timestamp=time.time(),
                    last_modified=time.time()
                )
                
                # Save data and metadata
                data_file = self.data_path / f"{name}.parquet"
                data.to_parquet(data_file, index=False)
                
                self.metadata[name] = metadata
                self._save_metadata()
                
                self.logger.info(f"Successfully ingested dataset {name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error ingesting dataset {name}: {str(e)}")
                return False

    def get_data(self, name: str) -> Optional[pd.DataFrame]:
        """Retrieve data with integrity checking."""
        with self._data_lock:
            try:
                if name not in self.metadata:
                    self.logger.error(f"Dataset {name} not found")
                    return None
                
                data_file = self.data_path / f"{name}.parquet"
                if not data_file.exists():
                    self.logger.error(f"Data file for {name} not found")
                    return None
                
                data = pd.read_parquet(data_file)
                current_hash = self._compute_hash(data)
                
                if current_hash != self.metadata[name].hash:
                    self.logger.error(f"Data integrity check failed for {name}")
                    return None
                
                return data
                
            except Exception as e:
                self.logger.error(f"Error retrieving dataset {name}: {str(e)}")
                return None

    def preprocess_data(self, name: str, operations: List[Dict]) -> Optional[pd.DataFrame]:
        """Apply preprocessing operations to data."""
        data = self.get_data(name)
        if data is None:
            return None
            
        try:
            for op in operations:
                op_type = op['type']
                if op_type == 'normalize':
                    cols = op.get('columns', data.select_dtypes(include=[np.number]).columns)
                    data[cols] = (data[cols] - data[cols].mean()) / data[cols].std()
                    
                elif op_type == 'encode_categorical':
                    cols = op['columns']
                    for col in cols:
                        data[f"{col}_encoded"] = pd.Categorical(data[col]).codes
                        
                elif op_type == 'fill_missing':
                    strategy = op.get('strategy', 'mean')
                    cols = op.get('columns', data.columns)
                    
                    if strategy == 'mean':
                        data[cols] = data[cols].fillna(data[cols].mean())
                    elif strategy == 'median':
                        data[cols] = data[cols].fillna(data[cols].median())
                    elif strategy == 'mode':
                        data[cols] = data[cols].fillna(data[cols].mode().iloc[0])
                        
            return data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing dataset {name}: {str(e)}")
            return None 