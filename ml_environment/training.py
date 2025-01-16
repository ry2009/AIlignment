import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, List, Callable, Tuple, Any
from pathlib import Path
import logging
import json
import time
import numpy as np
from dataclasses import dataclass
import threading
from .monitoring import EnvironmentMonitor

@dataclass
class TrainingConfig:
    model_type: str
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    early_stopping_patience: int = 3
    validation_split: float = 0.2
    optimizer: str = 'adam'
    loss_function: str = 'mse'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    timestamp: float
    custom_metrics: Dict[str, float]

class SecureModelTrainer:
    def __init__(self, base_path: Path, session_id: str, monitor: EnvironmentMonitor):
        self.base_path = base_path
        self.session_id = session_id
        self.monitor = monitor
        
        self.model_path = base_path / 'models' / session_id
        self.checkpoint_path = base_path / 'checkpoints' / session_id
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f'ModelTrainer-{session_id}')
        self._setup_logging()
        
        self._training_lock = threading.Lock()
        self.current_training: Optional[Dict] = None

    def _setup_logging(self):
        handler = logging.FileHandler(self.base_path / 'logs' / f'training_{self.session_id}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _get_optimizer(self, model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
        """Get optimizer based on configuration."""
        if config.optimizer.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer.lower() == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    def _get_loss_function(self, config: TrainingConfig) -> Callable:
        """Get loss function based on configuration."""
        if config.loss_function.lower() == 'mse':
            return nn.MSELoss()
        elif config.loss_function.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {config.loss_function}")

    def _save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                        epoch: int, metrics: TrainingMetrics):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': vars(metrics)
        }
        
        checkpoint_file = self.checkpoint_path / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_file)
        self.logger.info(f"Saved checkpoint for epoch {epoch}")

    def _load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                        epoch: int) -> Tuple[nn.Module, torch.optim.Optimizer, Dict]:
        """Load training checkpoint."""
        checkpoint_file = self.checkpoint_path / f"checkpoint_epoch_{epoch}.pt"
        if not checkpoint_file.exists():
            raise ValueError(f"Checkpoint for epoch {epoch} not found")
            
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model, optimizer, checkpoint['metrics']

    def train_model(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor],
                   config: TrainingConfig, custom_metrics: Optional[Dict[str, Callable]] = None,
                   validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> List[TrainingMetrics]:
        """Train model with security monitoring and checkpointing."""
        with self._training_lock:
            try:
                X_train, y_train = train_data
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                
                if validation_data:
                    X_val, y_val = validation_data
                    val_dataset = TensorDataset(X_val, y_val)
                    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
                
                optimizer = self._get_optimizer(model, config)
                criterion = self._get_loss_function(config)
                
                model = model.to(config.device)
                metrics_history = []
                best_val_loss = float('inf')
                patience_counter = 0
                
                self.current_training = {
                    'start_time': time.time(),
                    'config': config,
                    'current_epoch': 0
                }
                
                for epoch in range(config.epochs):
                    self.current_training['current_epoch'] = epoch
                    model.train()
                    total_loss = 0
                    batch_count = 0
                    
                    # Training loop
                    for batch_X, batch_y in train_loader:
                        batch_X = batch_X.to(config.device)
                        batch_y = batch_y.to(config.device)
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        batch_count += 1
                        
                        # Check resource usage
                        if batch_count % 10 == 0:
                            self.monitor.collect_metrics(self.session_id)
                    
                    avg_train_loss = total_loss / batch_count
                    
                    # Validation
                    val_loss = None
                    if validation_data:
                        model.eval()
                        val_loss = 0
                        val_batch_count = 0
                        
                        with torch.no_grad():
                            for batch_X, batch_y in val_loader:
                                batch_X = batch_X.to(config.device)
                                batch_y = batch_y.to(config.device)
                                
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                                val_loss += loss.item()
                                val_batch_count += 1
                        
                        val_loss = val_loss / val_batch_count
                        
                        # Early stopping check
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= config.early_stopping_patience:
                            self.logger.info("Early stopping triggered")
                            break
                    
                    # Calculate custom metrics
                    custom_metric_values = {}
                    if custom_metrics:
                        model.eval()
                        with torch.no_grad():
                            for metric_name, metric_fn in custom_metrics.items():
                                custom_metric_values[metric_name] = metric_fn(model, validation_data)
                    
                    # Save metrics
                    metrics = TrainingMetrics(
                        epoch=epoch,
                        train_loss=avg_train_loss,
                        val_loss=val_loss,
                        learning_rate=optimizer.param_groups[0]['lr'],
                        timestamp=time.time(),
                        custom_metrics=custom_metric_values
                    )
                    metrics_history.append(metrics)
                    
                    # Save checkpoint
                    self._save_checkpoint(model, optimizer, epoch, metrics)
                    
                    self.logger.info(
                        f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                        f"val_loss={val_loss:.4f if val_loss else 'N/A'}"
                    )
                
                # Save final model
                model_file = self.model_path / "final_model.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'metrics_history': [vars(m) for m in metrics_history]
                }, model_file)
                
                self.current_training = None
                return metrics_history
                
            except Exception as e:
                self.logger.error(f"Error during training: {str(e)}")
                self.current_training = None
                raise

    def get_training_status(self) -> Optional[Dict]:
        """Get current training status."""
        return self.current_training

    def stop_training(self):
        """Stop current training session."""
        self.logger.info("Stopping training...")
        self.current_training = None 