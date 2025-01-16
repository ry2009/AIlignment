import click
import logging
from pathlib import Path
import json
import torch
import pandas as pd
from typing import Dict, Optional
from .environment import MLEnvironment
from .data_pipeline import SecureDataPipeline, DataValidationRule
from .training import SecureModelTrainer, TrainingConfig
from .security import SecurityMonitor
from .advanced_security import AdvancedSecurityMonitor
from .preprocessing import DataPreprocessor, PreprocessingConfig
from .monitoring import EnvironmentMonitor

class MLEnvironmentCLI:
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd() / 'ml_environment'
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.environment = MLEnvironment(self.base_path)
        self.monitor = EnvironmentMonitor(self.base_path)
        self.security = SecurityMonitor(self.base_path)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.base_path / 'logs' / 'cli.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MLEnvironmentCLI')

@click.group()
@click.option('--base-path', type=click.Path(), help='Base path for ML environment')
@click.pass_context
def cli(ctx, base_path):
    """ML Environment Management CLI"""
    ctx.obj = MLEnvironmentCLI(Path(base_path) if base_path else None)

@cli.group()
def session():
    """Manage training sessions"""
    pass

@session.command()
@click.argument('user_id')
@click.option('--memory-limit', type=int, help='Memory limit in MB')
@click.option('--cpu-limit', type=int, help='CPU time limit in seconds')
@click.pass_obj
def create(cli_obj: MLEnvironmentCLI, user_id: str, memory_limit: Optional[int],
           cpu_limit: Optional[int]):
    """Create a new training session"""
    try:
        session = cli_obj.environment.create_session(
            user_id,
            custom_limits={'max_memory_mb': memory_limit} if memory_limit else None
        )
        click.echo(f"Created session: {session}")
    except Exception as e:
        click.echo(f"Error creating session: {str(e)}", err=True)

@session.command()
@click.argument('session_id')
@click.pass_obj
def status(cli_obj: MLEnvironmentCLI, session_id: str):
    """Get session status"""
    try:
        status = cli_obj.environment.get_session_status(session_id)
        click.echo(json.dumps(status, indent=2))
    except Exception as e:
        click.echo(f"Error getting status: {str(e)}", err=True)

@cli.group()
def data():
    """Manage datasets"""
    pass

@data.command()
@click.argument('session_id')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--name', help='Dataset name')
@click.option('--validation-rules', type=click.Path(), help='Path to validation rules JSON')
@click.pass_obj
def ingest(cli_obj: MLEnvironmentCLI, session_id: str, data_path: str,
          name: Optional[str], validation_rules: Optional[str]):
    """Ingest a dataset"""
    try:
        pipeline = SecureDataPipeline(cli_obj.base_path, session_id)
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Load validation rules if provided
        rules = None
        if validation_rules:
            with open(validation_rules, 'r') as f:
                rules_data = json.load(f)
                rules = [DataValidationRule(**rule) for rule in rules_data]
        
        # Ingest data
        success = pipeline.ingest_data(
            name or Path(data_path).stem,
            data,
            validation_rules=rules
        )
        
        if success:
            click.echo("Data ingested successfully")
        else:
            click.echo("Failed to ingest data", err=True)
            
    except Exception as e:
        click.echo(f"Error ingesting data: {str(e)}", err=True)

@data.command()
@click.argument('session_id')
@click.argument('dataset_name')
@click.argument('operations_file', type=click.Path(exists=True))
@click.pass_obj
def preprocess(cli_obj: MLEnvironmentCLI, session_id: str, dataset_name: str,
               operations_file: str):
    """Preprocess a dataset"""
    try:
        pipeline = SecureDataPipeline(cli_obj.base_path, session_id)
        preprocessor = DataPreprocessor(cli_obj.base_path, session_id)
        
        # Load operations
        with open(operations_file, 'r') as f:
            operations = json.load(f)
        
        # Get data
        data = pipeline.get_data(dataset_name)
        if data is None:
            click.echo("Dataset not found", err=True)
            return
        
        # Apply preprocessing
        processed_data = preprocessor.apply_pipeline(
            data,
            [PreprocessingConfig(**op) for op in operations]
        )
        
        # Save processed data
        success = pipeline.ingest_data(
            f"{dataset_name}_processed",
            processed_data
        )
        
        if success:
            click.echo("Data preprocessed successfully")
        else:
            click.echo("Failed to save preprocessed data", err=True)
            
    except Exception as e:
        click.echo(f"Error preprocessing data: {str(e)}", err=True)

@cli.group()
def model():
    """Manage models"""
    pass

@model.command()
@click.argument('session_id')
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('dataset_name')
@click.pass_obj
def train(cli_obj: MLEnvironmentCLI, session_id: str, config_file: str,
          dataset_name: str):
    """Train a model"""
    try:
        # Load training configuration
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        config = TrainingConfig(**config_data)
        
        # Get data
        pipeline = SecureDataPipeline(cli_obj.base_path, session_id)
        data = pipeline.get_data(dataset_name)
        if data is None:
            click.echo("Dataset not found", err=True)
            return
        
        # Initialize components
        trainer = SecureModelTrainer(
            cli_obj.base_path,
            session_id,
            cli_obj.monitor
        )
        
        # Start training
        click.echo("Starting training...")
        metrics = trainer.train_model(
            model=None,  # Model should be defined based on config
            train_data=(torch.tensor(data.values), None),  # Labels should be handled properly
            config=config
        )
        
        click.echo("Training completed")
        click.echo(json.dumps([vars(m) for m in metrics], indent=2))
        
    except Exception as e:
        click.echo(f"Error training model: {str(e)}", err=True)

@model.command()
@click.argument('session_id')
@click.argument('model_id')
@click.pass_obj
def protect(cli_obj: MLEnvironmentCLI, session_id: str, model_id: str):
    """Protect a trained model"""
    try:
        security = AdvancedSecurityMonitor(cli_obj.base_path, session_id)
        
        # Load model (implementation needed)
        model = None  # Should load model based on model_id
        
        protection = security.protect_model(model, model_id)
        click.echo(f"Model protected successfully: {model_id}")
        
    except Exception as e:
        click.echo(f"Error protecting model: {str(e)}", err=True)

@cli.group()
def security():
    """Security operations"""
    pass

@security.command()
@click.argument('session_id')
@click.option('--severity', multiple=True, help='Filter by severity level')
@click.pass_obj
def events(cli_obj: MLEnvironmentCLI, session_id: str, severity):
    """View security events"""
    try:
        events = cli_obj.security.get_security_events(
            session_id,
            severity_filter=list(severity) if severity else None
        )
        
        for event in events:
            click.echo(json.dumps(vars(event), indent=2))
            
    except Exception as e:
        click.echo(f"Error retrieving security events: {str(e)}", err=True)

@cli.group()
def monitor():
    """Monitoring operations"""
    pass

@monitor.command()
@click.argument('session_id')
@click.pass_obj
def metrics(cli_obj: MLEnvironmentCLI, session_id: str):
    """View monitoring metrics"""
    try:
        metrics = cli_obj.monitor.get_session_metrics(session_id)
        click.echo(json.dumps(metrics, indent=2))
    except Exception as e:
        click.echo(f"Error retrieving metrics: {str(e)}", err=True)

if __name__ == '__main__':
    cli() 