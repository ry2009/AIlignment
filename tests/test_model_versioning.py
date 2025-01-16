import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import time
from ml_environment.model_versioning import ModelVersionControl, ModelVersion

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def version_control(temp_path):
    return ModelVersionControl(temp_path, 'test_session')

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def test_metadata():
    return {
        'description': 'Test model',
        'dataset': 'test_data',
        'architecture': 'simple'
    }

@pytest.fixture
def test_metrics():
    return {
        'accuracy': 0.85,
        'loss': 0.15
    }

def test_save_version(version_control, model, test_metadata, test_metrics):
    """Test saving a model version."""
    version_id = version_control.save_version(
        model=model,
        metadata=test_metadata,
        metrics=test_metrics,
        tags=['test'],
        privacy_config={'epsilon': 1.0},
        training_config={'epochs': 10}
    )
    
    assert version_id is not None
    assert (version_control.versions_dir / version_id).exists()
    assert (version_control.versions_dir / version_id / 'model.pt').exists()
    assert (version_control.versions_dir / version_id / 'version.json').exists()

def test_load_version(version_control, model, test_metadata, test_metrics):
    """Test loading a model version."""
    version_id = version_control.save_version(
        model=model,
        metadata=test_metadata,
        metrics=test_metrics,
        tags=['test']
    )
    
    new_model = SimpleModel()
    version = version_control.load_version(version_id, new_model)
    
    assert version is not None
    assert version.version_id == version_id
    assert version.metadata == test_metadata
    assert version.metrics == test_metrics

def test_get_version_history(version_control, model, test_metadata, test_metrics):
    """Test retrieving version history."""
    version_ids = []
    for i in range(3):
        version_id = version_control.save_version(
            model=model,
            metadata=test_metadata,
            metrics=test_metrics,
            tags=[f'test_{i}']
        )
        version_ids.append(version_id)
        time.sleep(0.1)  # Ensure different timestamps
    
    history = version_control.get_version_history()
    assert len(history) == 3
    assert all(v.version_id in version_ids for v in history)

def test_get_version_by_tag(version_control, model, test_metadata, test_metrics):
    """Test finding a version by tag."""
    version_id = version_control.save_version(
        model=model,
        metadata=test_metadata,
        metrics=test_metrics,
        tags=['special_tag']
    )
    
    version = version_control.get_version_by_tag('special_tag')
    assert version is not None
    assert version.version_id == version_id

def test_compare_versions(version_control, model, test_metadata):
    """Test comparing two model versions."""
    # Save first version
    v1_id = version_control.save_version(
        model=model,
        metadata=test_metadata,
        metrics={'accuracy': 0.8},
        tags=['v1'],
        privacy_config={'epsilon': 1.0},
        training_config={'epochs': 10}
    )
    
    # Modify model and save second version
    model.fc.weight.data *= 1.1
    v2_id = version_control.save_version(
        model=model,
        metadata=test_metadata,
        metrics={'accuracy': 0.85},
        tags=['v2'],
        privacy_config={'epsilon': 0.8},
        training_config={'epochs': 15}
    )
    
    comparison = version_control.compare_versions(v1_id, v2_id)
    assert 'metrics_diff' in comparison
    assert 'privacy_changes' in comparison
    assert 'training_changes' in comparison
    assert comparison['metrics_diff']['accuracy'] == pytest.approx(0.05)

def test_delete_version(version_control, model, test_metadata, test_metrics):
    """Test deleting a model version."""
    version_id = version_control.save_version(
        model=model,
        metadata=test_metadata,
        metrics=test_metrics,
        tags=['test']
    )
    
    assert version_control.delete_version(version_id)
    assert not (version_control.versions_dir / version_id).exists()
    assert version_control.get_version_info(version_id) is None

def test_invalid_version_operations(version_control, model):
    """Test handling of invalid version operations."""
    # Test loading non-existent version
    assert version_control.load_version('non_existent', model) is None
    
    # Test comparing non-existent versions
    assert version_control.compare_versions('v1', 'v2') == {}
    
    # Test deleting non-existent version
    assert not version_control.delete_version('non_existent')

def test_version_with_parent(version_control, model, test_metadata, test_metrics):
    """Test version with parent reference."""
    parent_id = version_control.save_version(
        model=model,
        metadata=test_metadata,
        metrics=test_metrics,
        tags=['parent']
    )
    
    child_id = version_control.save_version(
        model=model,
        metadata=test_metadata,
        metrics=test_metrics,
        tags=['child'],
        parent_version=parent_id
    )
    
    child_version = version_control.get_version_info(child_id)
    assert child_version is not None
    assert child_version.parent_version == parent_id 