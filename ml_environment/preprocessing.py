from typing import Dict, List, Optional, Union, Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import threading

@dataclass
class PreprocessingConfig:
    operation: str
    parameters: Dict
    target_columns: Optional[List[str]] = None
    output_format: str = 'dataframe'

class DataPreprocessor:
    def __init__(self, base_path: Path, session_id: str):
        self.base_path = base_path
        self.session_id = session_id
        self.logger = logging.getLogger(f'Preprocessor-{session_id}')
        self._setup_logging()
        
        self.scalers: Dict[str, Dict] = {}
        self.transformers: Dict[str, Dict] = {}
        self._preprocessing_lock = threading.Lock()

    def _setup_logging(self):
        handler = logging.FileHandler(self.base_path / 'logs' / f'preprocessing_{self.session_id}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def scale_data(self, data: pd.DataFrame, method: str = 'standard',
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale data using various methods."""
        with self._preprocessing_lock:
            try:
                columns = columns or data.select_dtypes(include=[np.number]).columns
                scaler_key = f"{method}_{','.join(columns)}"
                
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                elif method == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unsupported scaling method: {method}")
                
                scaled_data = data.copy()
                scaled_data[columns] = scaler.fit_transform(data[columns])
                self.scalers[scaler_key] = {'scaler': scaler, 'columns': columns}
                
                return scaled_data
                
            except Exception as e:
                self.logger.error(f"Error scaling data: {str(e)}")
                raise

    def reduce_dimensions(self, data: pd.DataFrame, method: str = 'pca',
                         n_components: int = 2) -> pd.DataFrame:
        """Reduce dimensionality of data."""
        with self._preprocessing_lock:
            try:
                if method == 'pca':
                    transformer = PCA(n_components=n_components)
                else:
                    raise ValueError(f"Unsupported dimension reduction method: {method}")
                
                transformed = transformer.fit_transform(data)
                result = pd.DataFrame(
                    transformed,
                    columns=[f"{method}_{i+1}" for i in range(n_components)]
                )
                
                self.transformers[method] = transformer
                return result
                
            except Exception as e:
                self.logger.error(f"Error reducing dimensions: {str(e)}")
                raise

    def select_features(self, data: pd.DataFrame, target: pd.Series,
                       n_features: int = 10) -> pd.DataFrame:
        """Select most important features."""
        with self._preprocessing_lock:
            try:
                selector = SelectKBest(score_func=f_classif, k=n_features)
                selected = selector.fit_transform(data, target)
                
                # Get selected feature names
                mask = selector.get_support()
                selected_features = data.columns[mask]
                
                result = pd.DataFrame(selected, columns=selected_features)
                self.transformers['feature_selector'] = selector
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error selecting features: {str(e)}")
                raise

    def handle_outliers(self, data: pd.DataFrame, method: str = 'zscore',
                       threshold: float = 3.0) -> pd.DataFrame:
        """Handle outliers using various methods."""
        with self._preprocessing_lock:
            try:
                result = data.copy()
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                if method == 'zscore':
                    z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / 
                                    data[numeric_cols].std())
                    outliers = z_scores > threshold
                    result[numeric_cols] = data[numeric_cols].mask(outliers, data[numeric_cols].mean())
                    
                elif method == 'iqr':
                    Q1 = data[numeric_cols].quantile(0.25)
                    Q3 = data[numeric_cols].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                               (data[numeric_cols] > (Q3 + 1.5 * IQR)))
                    result[numeric_cols] = data[numeric_cols].mask(outliers, data[numeric_cols].median())
                    
                else:
                    raise ValueError(f"Unsupported outlier handling method: {method}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error handling outliers: {str(e)}")
                raise

    def encode_categorical(self, data: pd.DataFrame, method: str = 'onehot',
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical variables."""
        with self._preprocessing_lock:
            try:
                result = data.copy()
                columns = columns or data.select_dtypes(include=['object', 'category']).columns
                
                if method == 'onehot':
                    result = pd.get_dummies(data, columns=columns)
                elif method == 'label':
                    for col in columns:
                        result[f"{col}_encoded"] = pd.Categorical(data[col]).codes
                elif method == 'target':
                    raise NotImplementedError("Target encoding not implemented yet")
                else:
                    raise ValueError(f"Unsupported encoding method: {method}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error encoding categorical variables: {str(e)}")
                raise

    def handle_missing_values(self, data: pd.DataFrame, method: str = 'mean',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values using various methods."""
        with self._preprocessing_lock:
            try:
                result = data.copy()
                columns = columns or data.columns
                
                if method == 'mean':
                    result[columns] = result[columns].fillna(result[columns].mean())
                elif method == 'median':
                    result[columns] = result[columns].fillna(result[columns].median())
                elif method == 'mode':
                    result[columns] = result[columns].fillna(result[columns].mode().iloc[0])
                elif method == 'interpolate':
                    result[columns] = result[columns].interpolate(method='linear')
                elif method == 'drop':
                    result = result.dropna(subset=columns)
                else:
                    raise ValueError(f"Unsupported missing value handling method: {method}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error handling missing values: {str(e)}")
                raise

    def create_features(self, data: pd.DataFrame, operations: List[Dict]) -> pd.DataFrame:
        """Create new features based on existing ones."""
        with self._preprocessing_lock:
            try:
                result = data.copy()
                
                for op in operations:
                    op_type = op['type']
                    if op_type == 'arithmetic':
                        col1, col2 = op['columns']
                        operator = op['operator']
                        new_col = op['new_column']
                        
                        if operator == '+':
                            result[new_col] = result[col1] + result[col2]
                        elif operator == '-':
                            result[new_col] = result[col1] - result[col2]
                        elif operator == '*':
                            result[new_col] = result[col1] * result[col2]
                        elif operator == '/':
                            result[new_col] = result[col1] / result[col2]
                            
                    elif op_type == 'aggregate':
                        cols = op['columns']
                        method = op['method']
                        new_col = op['new_column']
                        
                        if method == 'sum':
                            result[new_col] = result[cols].sum(axis=1)
                        elif method == 'mean':
                            result[new_col] = result[cols].mean(axis=1)
                        elif method == 'max':
                            result[new_col] = result[cols].max(axis=1)
                        elif method == 'min':
                            result[new_col] = result[cols].min(axis=1)
                            
                    elif op_type == 'transform':
                        col = op['column']
                        transform = op['transform']
                        new_col = op['new_column']
                        
                        if transform == 'log':
                            result[new_col] = np.log1p(result[col])
                        elif transform == 'sqrt':
                            result[new_col] = np.sqrt(result[col])
                        elif transform == 'square':
                            result[new_col] = np.square(result[col])
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error creating features: {str(e)}")
                raise

    def apply_pipeline(self, data: pd.DataFrame, pipeline: List[PreprocessingConfig]) -> pd.DataFrame:
        """Apply a sequence of preprocessing operations."""
        result = data.copy()
        
        for step in pipeline:
            if step.operation == 'scale':
                result = self.scale_data(
                    result, 
                    method=step.parameters.get('method', 'standard'),
                    columns=step.target_columns
                )
            elif step.operation == 'reduce_dimensions':
                result = self.reduce_dimensions(
                    result,
                    method=step.parameters.get('method', 'pca'),
                    n_components=step.parameters.get('n_components', 2)
                )
            elif step.operation == 'handle_outliers':
                result = self.handle_outliers(
                    result,
                    method=step.parameters.get('method', 'zscore'),
                    threshold=step.parameters.get('threshold', 3.0)
                )
            elif step.operation == 'encode_categorical':
                result = self.encode_categorical(
                    result,
                    method=step.parameters.get('method', 'onehot'),
                    columns=step.target_columns
                )
            elif step.operation == 'handle_missing':
                result = self.handle_missing_values(
                    result,
                    method=step.parameters.get('method', 'mean'),
                    columns=step.target_columns
                )
            elif step.operation == 'create_features':
                result = self.create_features(
                    result,
                    operations=step.parameters.get('operations', [])
                )
        
        return result 