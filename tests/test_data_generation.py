import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from ml_environment.testing.data_generation import (
    DataGenerationConfig,
    DatasetMetadata,
    DataValidator,
    AdminDataGenerator,
    ForumDataGenerator,
    GameDataGenerator,
    DataGenerationPipeline
)

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def data_validator():
    return DataValidator()

@pytest.fixture
def generation_config():
    return DataGenerationConfig(
        environment="admin",
        data_size=100,
        noise_level=0.1,
        seed=42,
        parameters={},
        validation_rules={
            'range': {'min': 0, 'max': 1},
            'completeness': {'threshold': 0.99}
        }
    )

def test_data_validator(data_validator):
    """Test data validation functionality."""
    data = np.random.uniform(0, 1, 100)
    rules = {
        'range': {'min': 0, 'max': 1},
        'uniqueness': {'threshold': 0.8},
        'completeness': {'threshold': 0.99}
    }
    
    results = data_validator.validate(data, rules)
    assert 'range' in results
    assert 'uniqueness' in results
    assert 'completeness' in results
    assert all(isinstance(v, bool) for v in results.values())

def test_admin_data_generation(temp_path, generation_config):
    """Test administrative data generation."""
    generator = AdminDataGenerator(temp_path)
    data, metadata = generator.generate(generation_config)
    
    assert isinstance(data, dict)
    assert 'resource_usage' in data
    assert 'response_times' in data
    assert 'error_rates' in data
    
    assert isinstance(metadata, DatasetMetadata)
    assert metadata.environment == "admin"
    assert metadata.size == generation_config.data_size
    assert len(metadata.validation_results) > 0

def test_forum_data_generation(temp_path):
    """Test forum data generation."""
    config = DataGenerationConfig(
        environment="forum",
        data_size=50,
        noise_level=0.1,
        seed=42,
        parameters={},
        validation_rules={
            'range': {'min': 0, 'max': 1}
        }
    )
    
    generator = ForumDataGenerator(temp_path)
    data, metadata = generator.generate(config)
    
    assert isinstance(data, dict)
    assert 'posts' in data
    assert len(data['posts']) == config.data_size
    assert all('toxicity_score' in post for post in data['posts'])
    
    assert isinstance(metadata, DatasetMetadata)
    assert metadata.environment == "forum"
    assert metadata.size == config.data_size

def test_game_data_generation(temp_path):
    """Test game data generation."""
    config = DataGenerationConfig(
        environment="game",
        data_size=30,
        noise_level=0.1,
        seed=42,
        parameters={},
        validation_rules={
            'range': {'min': -5, 'max': 5}
        }
    )
    
    generator = GameDataGenerator(temp_path)
    data, metadata = generator.generate(config)
    
    assert isinstance(data, dict)
    assert 'episodes' in data
    assert len(data['episodes']) == config.data_size
    assert all('actions' in episode for episode in data['episodes'])
    
    assert isinstance(metadata, DatasetMetadata)
    assert metadata.environment == "game"
    assert metadata.size == config.data_size

def test_data_generation_pipeline(temp_path):
    """Test the complete data generation pipeline."""
    pipeline = DataGenerationPipeline(temp_path)
    
    configs = {
        'admin': DataGenerationConfig(
            environment="admin",
            data_size=100,
            noise_level=0.1,
            seed=42,
            parameters={},
            validation_rules={'range': {'min': 0, 'max': 1}}
        ),
        'forum': DataGenerationConfig(
            environment="forum",
            data_size=50,
            noise_level=0.1,
            seed=42,
            parameters={},
            validation_rules={'range': {'min': 0, 'max': 1}}
        ),
        'game': DataGenerationConfig(
            environment="game",
            data_size=30,
            noise_level=0.1,
            seed=42,
            parameters={},
            validation_rules={'range': {'min': -5, 'max': 5}}
        )
    }
    
    for env, config in configs.items():
        data, metadata = pipeline.generate_data(config)
        assert data is not None
        assert metadata.environment == env
        
        # Check if files were saved
        dataset_dir = temp_path / 'datasets' / env
        assert dataset_dir.exists()
        assert any(dataset_dir.glob(f"{metadata.dataset_id}*"))

def test_data_validation_rules(temp_path, generation_config):
    """Test data validation with different rules."""
    generator = AdminDataGenerator(temp_path)
    
    # Test with strict rules
    strict_config = generation_config
    strict_config.validation_rules = {
        'range': {'min': 0, 'max': 0.5},
        'uniqueness': {'threshold': 0.9},
        'completeness': {'threshold': 1.0}
    }
    
    data, metadata = generator.generate(strict_config)
    assert isinstance(metadata.validation_results, dict)
    
    # Test with relaxed rules
    relaxed_config = generation_config
    relaxed_config.validation_rules = {
        'range': {'min': -1, 'max': 2},
        'uniqueness': {'threshold': 0.1},
        'completeness': {'threshold': 0.8}
    }
    
    data, metadata = generator.generate(relaxed_config)
    assert isinstance(metadata.validation_results, dict)

def test_data_reproducibility(temp_path):
    """Test data generation reproducibility with same seed."""
    config1 = DataGenerationConfig(
        environment="admin",
        data_size=100,
        noise_level=0.1,
        seed=42,
        parameters={},
        validation_rules={}
    )
    
    config2 = DataGenerationConfig(
        environment="admin",
        data_size=100,
        noise_level=0.1,
        seed=42,
        parameters={},
        validation_rules={}
    )
    
    generator = AdminDataGenerator(temp_path)
    data1, _ = generator.generate(config1)
    data2, _ = generator.generate(config2)
    
    for key in data1:
        assert np.allclose(data1[key], data2[key])

def test_invalid_environment(temp_path):
    """Test handling of invalid environment."""
    pipeline = DataGenerationPipeline(temp_path)
    config = DataGenerationConfig(
        environment="invalid",
        data_size=100,
        noise_level=0.1,
        seed=42,
        parameters={},
        validation_rules={}
    )
    
    with pytest.raises(ValueError):
        pipeline.generate_data(config)

def test_data_persistence(temp_path):
    """Test data and metadata persistence."""
    pipeline = DataGenerationPipeline(temp_path)
    config = DataGenerationConfig(
        environment="admin",
        data_size=100,
        noise_level=0.1,
        seed=42,
        parameters={},
        validation_rules={}
    )
    
    _, metadata = pipeline.generate_data(config)
    
    # Check data file
    data_path = temp_path / 'datasets' / 'admin' / f"{metadata.dataset_id}_data.npz"
    assert data_path.exists() or data_path.with_suffix('.json').exists()
    
    # Check metadata file
    metadata_path = temp_path / 'datasets' / 'admin' / f"{metadata.dataset_id}_metadata.json"
    assert metadata_path.exists()
    
    # Verify metadata content
    with open(metadata_path, 'r') as f:
        loaded_metadata = json.load(f)
    assert loaded_metadata['environment'] == "admin"
    assert loaded_metadata['size'] == config.data_size 