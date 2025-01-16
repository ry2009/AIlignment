import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import tempfile
from ml_environment.privacy import (
    PrivacyPreserver,
    PrivacyConfig,
    PrivateDataset,
    PrivateDataLoader,
    PrivacyCallback
)

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
def privacy_preserver():
    return PrivacyPreserver()

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)

@pytest.fixture
def privacy_config():
    return PrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        secure_mode=True,
        batch_size=32,
        sample_size=1000
    )

@pytest.fixture
def dummy_dataset():
    data = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))
    return TensorDataset(data, targets)

def test_make_model_private(privacy_preserver, model, optimizer, privacy_config):
    """Test converting a model to its private version."""
    private_model, private_optimizer, privacy_engine = privacy_preserver.make_model_private(
        model, optimizer, privacy_config
    )
    
    assert hasattr(private_model, 'privacy_engine')
    assert str(id(private_model)) in privacy_preserver.privacy_engines

def test_add_noise_to_data(privacy_preserver):
    """Test adding Laplace noise to data."""
    data = torch.ones(10, 5)
    noisy_data = privacy_preserver.add_noise_to_data(data, sensitivity=1.0, epsilon=1.0)
    
    assert data.shape == noisy_data.shape
    assert not torch.allclose(data, noisy_data)  # Data should be different due to noise

def test_privacy_budget_tracking(privacy_preserver, model, optimizer, privacy_config):
    """Test tracking privacy budget usage."""
    private_model, private_optimizer, privacy_engine = privacy_preserver.make_model_private(
        model, optimizer, privacy_config
    )
    
    epsilon, delta = privacy_preserver.get_privacy_spent(private_model)
    assert epsilon >= 0
    assert delta > 0

def test_private_dataset(dummy_dataset):
    """Test privacy-preserving dataset wrapper."""
    private_dataset = PrivateDataset(dummy_dataset, epsilon=1.0, sensitivity=1.0)
    
    assert len(private_dataset) == len(dummy_dataset)
    data, target = private_dataset[0]
    orig_data, orig_target = dummy_dataset[0]
    
    assert not torch.allclose(data, orig_data)  # Data should be noisy
    assert torch.allclose(target, orig_target)  # Labels should be unchanged

def test_private_dataloader(dummy_dataset):
    """Test privacy-preserving dataloader."""
    batch_size = 32
    dataloader = PrivateDataLoader(
        dummy_dataset,
        batch_size=batch_size,
        epsilon=1.0,
        sensitivity=1.0
    )
    
    batch = next(iter(dataloader))
    assert len(batch) == 2  # (data, targets)
    assert batch[0].shape[0] == batch_size

def test_privacy_callback(privacy_preserver, model, optimizer, privacy_config):
    """Test privacy budget monitoring callback."""
    private_model, private_optimizer, privacy_engine = privacy_preserver.make_model_private(
        model, optimizer, privacy_config
    )
    
    callback = PrivacyCallback(privacy_engine, privacy_config.epsilon, privacy_config.delta, privacy_config)
    assert callback.on_epoch_end(0)  # Should return True if budget not exceeded

def test_privacy_guarantee_check(privacy_preserver, model, optimizer, privacy_config):
    """Test checking privacy guarantees."""
    private_model, private_optimizer, privacy_engine = privacy_preserver.make_model_private(
        model, optimizer, privacy_config
    )
    
    assert privacy_preserver.check_privacy_guarantee(
        private_model,
        privacy_config.epsilon,
        privacy_config.delta
    )

def test_privacy_with_training(privacy_preserver, model, optimizer, privacy_config, dummy_dataset):
    """Test privacy preservation during training."""
    private_model, private_optimizer, privacy_engine = privacy_preserver.make_model_private(
        model, optimizer, privacy_config
    )
    
    dataloader = DataLoader(dummy_dataset, batch_size=privacy_config.batch_size)
    criterion = nn.CrossEntropyLoss()
    
    # Single training step
    for data, target in dataloader:
        private_optimizer.zero_grad()
        output = private_model(data)
        loss = criterion(output, target)
        loss.backward()
        private_optimizer.step()
        break
    
    epsilon, delta = privacy_preserver.get_privacy_spent(private_model)
    assert epsilon >= 0
    assert delta > 0

def test_invalid_privacy_config():
    """Test handling of invalid privacy configurations."""
    with pytest.raises(ValueError):
        PrivacyConfig(epsilon=-1.0)  # Invalid privacy budget

def test_excessive_privacy_budget(privacy_preserver, model, optimizer):
    """Test handling of excessive privacy budget usage."""
    strict_config = PrivacyConfig(
        epsilon=0.1,  # Very strict privacy budget
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.1,  # Small noise (will exceed budget quickly)
        batch_size=32,
        sample_size=1000
    )
    
    private_model, private_optimizer, privacy_engine = privacy_preserver.make_model_private(
        model, optimizer, strict_config
    )
    
    callback = PrivacyCallback(privacy_engine, strict_config.epsilon, strict_config.delta, strict_config)
    
    # Simulate multiple epochs
    epoch = 0
    while callback.on_epoch_end(epoch):
        epoch += 1
        if epoch > 100:  # Safety limit
            break
    
    assert epoch > 0  # Should have completed at least one epoch
    assert not callback.on_epoch_end(epoch)  # Should return False when budget is exceeded 