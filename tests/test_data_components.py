import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
from datetime import datetime
from ml_environment.testing.data.versioning import DatasetVersionControl, DatasetVersion
from ml_environment.testing.data.quality import DataQualityValidator, QualityMetric
from ml_environment.testing.data.validation_generator import ValidationDataGenerator, ValidationDataConfig

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def version_control(temp_path):
    return DatasetVersionControl(temp_path)

@pytest.fixture
def quality_validator(temp_path):
    return DataQualityValidator(temp_path)

@pytest.fixture
def validation_generator(temp_path):
    return ValidationDataGenerator(temp_path)

# Version Control Tests
def test_version_creation(version_control):
    """Test creating and retrieving a dataset version."""
    data = np.random.normal(0, 1, (100, 10))
    metadata = {
        'features': {'shape': data.shape, 'dtype': str(data.dtype)},
        'description': 'Test dataset'
    }
    
    version = version_control.create_version(
        dataset_id="test_dataset",
        environment="ml",
        data=data,
        metadata=metadata,
        tags=['test']
    )
    
    assert version.dataset_id == "test_dataset"
    assert version.environment == "ml"
    assert 'test' in version.tags
    
    # Test retrieval
    retrieved = version_control.get_version(version.version_id)
    assert retrieved is not None
    assert retrieved.version_id == version.version_id
    assert retrieved.hash == version.hash

def test_version_comparison(version_control):
    """Test comparing two dataset versions."""
    data1 = np.random.normal(0, 1, (100, 10))
    data2 = np.random.normal(0.5, 1.2, (100, 10))
    
    v1 = version_control.create_version(
        dataset_id="test_dataset",
        environment="ml",
        data=data1,
        metadata={'version': 1}
    )
    
    v2 = version_control.create_version(
        dataset_id="test_dataset",
        environment="ml",
        data=data2,
        metadata={'version': 2}
    )
    
    comparison = version_control.compare_versions(v1.version_id, v2.version_id)
    assert 'statistics_diff' in comparison
    assert 'metadata_diff' in comparison
    assert comparison['metadata_diff']['version'] == 'changed'

def test_version_tagging(version_control):
    """Test adding and retrieving versions by tags."""
    data = np.random.normal(0, 1, (100, 10))
    version = version_control.create_version(
        dataset_id="test_dataset",
        environment="ml",
        data=data,
        metadata={},
        tags=['initial']
    )
    
    version_control.tag_version(version.version_id, ['validated', 'production'])
    tagged_versions = version_control.get_versions_by_tag('production')
    assert len(tagged_versions) == 1
    assert tagged_versions[0].version_id == version.version_id

# Quality Validation Tests
def test_completeness_check(quality_validator):
    """Test data completeness validation."""
    data = np.random.normal(0, 1, (100, 10))
    data[10:20, 0] = np.nan
    
    rules = {
        'completeness': {'threshold': 0.8}
    }
    
    metrics = quality_validator.validate_quality(data, rules)
    assert 'completeness' in metrics
    assert metrics['completeness'].passed
    assert metrics['completeness'].value > 0.8

def test_distribution_check(quality_validator):
    """Test distribution validation."""
    # Test normal distribution
    data = np.random.normal(0, 1, 1000)
    rules = {
        'distribution': {'type': 'normal', 'p_threshold': 0.05}
    }
    
    metrics = quality_validator.validate_quality(data, rules)
    assert 'distribution' in metrics
    assert metrics['distribution'].passed

def test_correlation_check(quality_validator):
    """Test correlation validation."""
    data = np.random.normal(0, 1, (100, 5))
    # Create high correlation between two features
    data[:, 1] = data[:, 0] * 0.9 + np.random.normal(0, 0.1, 100)
    
    rules = {
        'correlation': {'threshold': 0.8}
    }
    
    metrics = quality_validator.validate_quality(data, rules)
    assert 'correlation' in metrics
    assert not metrics['correlation'].passed  # Should fail due to high correlation

# Validation Generator Tests
def test_feature_generation(validation_generator):
    """Test generating features with different configurations."""
    config = ValidationDataConfig(
        environment="ml",
        dataset_type="train",
        size=100,
        feature_config={
            'type': 'continuous',
            'num_features': 5,
            'distribution': 'normal'
        },
        label_config={
            'type': 'classification',
            'num_classes': 3
        },
        seed=42
    )
    
    data, version = validation_generator.generate_validation_data(config)
    assert 'features' in data
    assert data['features'].shape == (100, 5)
    assert 'labels' in data
    assert data['labels'].shape == (100, 1)

def test_mixed_feature_generation(validation_generator):
    """Test generating mixed type features."""
    config = ValidationDataConfig(
        environment="ml",
        dataset_type="train",
        size=100,
        feature_config={
            'type': 'mixed',
            'continuous': {
                'num_features': 3,
                'distribution': 'normal'
            },
            'categorical': {
                'num_features': 2,
                'num_categories': 4
            }
        },
        label_config={
            'type': 'regression'
        },
        seed=42
    )
    
    data, version = validation_generator.generate_validation_data(config)
    assert data['features'].shape == (100, 5)  # 3 continuous + 2 categorical

def test_noise_injection(validation_generator):
    """Test noise injection in generated data."""
    config = ValidationDataConfig(
        environment="ml",
        dataset_type="train",
        size=100,
        feature_config={
            'type': 'continuous',
            'num_features': 4
        },
        label_config={
            'type': 'classification',
            'num_classes': 2
        },
        noise_config={
            'type': 'gaussian',
            'scale': 0.1
        },
        seed=42
    )
    
    data1, _ = validation_generator.generate_validation_data(config)
    
    # Generate same data without noise
    config.noise_config = None
    data2, _ = validation_generator.generate_validation_data(config)
    
    # Check that noise was added
    assert not np.allclose(data1['features'], data2['features'])

def test_data_versioning_integration(validation_generator):
    """Test integration of data generation with versioning."""
    config = ValidationDataConfig(
        environment="ml",
        dataset_type="validation",
        size=100,
        feature_config={'type': 'continuous', 'num_features': 3},
        label_config={'type': 'classification', 'num_classes': 2},
        quality_rules={
            'completeness': {'threshold': 0.99},
            'distribution': {'type': 'normal'}
        },
        seed=42
    )
    
    _, version = validation_generator.generate_validation_data(config)
    
    # Load the data back
    loaded_data, loaded_version = validation_generator.load_validation_data(version.version_id)
    assert loaded_version.version_id == version.version_id
    assert 'features' in loaded_data
    assert 'labels' in loaded_data 