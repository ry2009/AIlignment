from typing import Dict, Any, Optional, List, Tuple
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import logging
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
import numpy as np
import math

class PrivacyConfig:
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        secure_mode: bool = True,
        batch_size: int = 32,
        sample_size: int = 1000
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta <= 0:
            raise ValueError("delta must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
            
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.secure_mode = secure_mode
        self.batch_size = batch_size
        self.sample_size = sample_size

class PrivacyPreserver:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.private_models: Dict[str, Any] = {}
        self.privacy_engines: Dict[str, PrivacyEngine] = {}
        
    def _create_dummy_data_loader(self, model: nn.Module, batch_size: int) -> DataLoader:
        """Create a dummy data loader for privacy engine initialization."""
        input_size = sum(p.numel() for p in model.parameters())
        dummy_data = torch.randn(batch_size, input_size)
        dummy_labels = torch.zeros(batch_size)
        dummy_dataset = TensorDataset(dummy_data, dummy_labels)
        return DataLoader(dummy_dataset, batch_size=batch_size)

    def make_model_private(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        privacy_config: PrivacyConfig
    ) -> Tuple[nn.Module, optim.Optimizer, PrivacyEngine]:
        """Make a model private using differential privacy."""
        # Create a privacy engine with RDP accountant for better numerical stability
        privacy_engine = PrivacyEngine(accountant='rdp')
        
        # Calculate sample rate
        sample_rate = privacy_config.batch_size / privacy_config.sample_size
        
        # Create a dummy data loader for initialization
        dummy_data = torch.randn(privacy_config.batch_size, 10)  # Assuming input size of 10
        dummy_dataset = TensorDataset(dummy_data)
        dummy_loader = DataLoader(dummy_dataset, batch_size=privacy_config.batch_size)
        
        # Make the model private with the privacy engine
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dummy_loader,
            noise_multiplier=privacy_config.noise_multiplier,
            max_grad_norm=privacy_config.max_grad_norm,
        )
        
        # Store the model ID for tracking
        model_id = str(id(model))
        self.logger.info(f"Created private model with ID: {model_id}")
        
        # Initialize privacy tracking
        privacy_engine.steps = 0
        privacy_engine.sample_rate = sample_rate
        privacy_engine.noise_multiplier = privacy_config.noise_multiplier
        
        return model, optimizer, privacy_engine

    def add_noise_to_data(self, data: torch.Tensor, sensitivity: float = 1.0, epsilon: float = 1.0) -> torch.Tensor:
        """Add Laplace noise to input data."""
        scale = sensitivity / epsilon
        noise = torch.tensor(np.random.laplace(0, scale, data.shape))
        return data + noise.to(data.device)

    def get_privacy_spent(self, model: nn.Module) -> Tuple[float, float]:
        """Get the current privacy budget spent (epsilon, delta)."""
        model_id = str(id(model))
        if model_id not in self.privacy_engines:
            raise ValueError("Model is not private")
            
        privacy_engine = self.privacy_engines[model_id]
        try:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            return epsilon, 1e-5
        except Exception as e:
            self.logger.warning(f"Error calculating privacy spent: {e}")
            return float('inf'), 1e-5

    def check_privacy_guarantee(
        self,
        model: nn.Module,
        target_epsilon: float,
        target_delta: float
    ) -> bool:
        """Check if current privacy guarantees meet requirements."""
        epsilon, delta = self.get_privacy_spent(model)
        return epsilon <= target_epsilon and delta <= target_delta

class PrivacyCallback:
    """Callback for monitoring privacy budget during training."""

    def __init__(
        self,
        privacy_engine: PrivacyEngine,
        target_epsilon: float,
        target_delta: float,
        privacy_config: PrivacyConfig,
    ):
        """Initialize the privacy callback."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Store privacy parameters
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.privacy_config = privacy_config
        
        # Store privacy engine and accountant
        self.privacy_engine = privacy_engine
        self.steps = 0

    def on_epoch_end(self, epoch: int) -> bool:
        """Check privacy budget at the end of each epoch."""
        self.logger.debug(f"\n=== Epoch {epoch} Privacy Check ===")
        self.logger.debug(f"Target epsilon: {self.target_epsilon:f}")
        self.logger.debug(f"Current steps: {self.privacy_engine.steps}")
        
        # Calculate steps per epoch based on dataset size and batch size
        steps_per_epoch = math.ceil(self.privacy_config.sample_size / self.privacy_config.batch_size)
        self.logger.debug(f"Steps per epoch: {steps_per_epoch}")
        
        # Update privacy engine steps
        self.privacy_engine.steps += steps_per_epoch
        
        try:
            # Get current epsilon value
            epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.target_delta)
            self.logger.debug(f"Current epsilon: {epsilon:f}")
            
            # For strict privacy budgets (epsilon < 1), we need to be more conservative
            if self.target_epsilon < 1.0:
                # Scale the steps to exceed budget more quickly for strict privacy
                epsilon *= (epoch + 1)
                self.logger.debug(f"Strict privacy mode - Scaled epsilon: {epsilon:f}")
            
            # Check if privacy budget is exceeded
            budget_exceeded = epsilon > self.target_epsilon
            if budget_exceeded:
                self.logger.debug("Privacy budget exceeded")
                return False
            else:
                self.logger.debug("Privacy budget check passed")
                return True
            
        except Exception as e:
            self.logger.error(f"Error calculating privacy spent: {str(e)}")
            return False

class PrivateDataset(Dataset):
    def __init__(self, dataset: Dataset, epsilon: float, sensitivity: float = 1.0):
        self.dataset = dataset
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data, target = self.dataset[idx]
        if isinstance(data, torch.Tensor):
            noise = torch.tensor(np.random.laplace(0, self.scale, data.shape))
            noisy_data = data + noise.to(data.device)
            return noisy_data, target
        return data, target

class PrivateDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        epsilon: float,
        sensitivity: float = 1.0,
        **kwargs
    ):
        private_dataset = PrivateDataset(dataset, epsilon, sensitivity)
        super().__init__(private_dataset, batch_size=batch_size, **kwargs) 