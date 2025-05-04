import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from torch.cuda.amp import GradScaler
from typing import Dict, Optional, Tuple


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int
) -> optim.lr_scheduler._LRScheduler:
    """Create a learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """Save a checkpoint of the model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get dimensions in a way that works with both encoder types
    if hasattr(model.tabular_encoder, 'input_projection'):
        # For TabularTransformerEncoder
        tabular_input_dim = model.tabular_encoder.input_projection.in_features
    else:
        # For sequential tabular encoder
        tabular_input_dim = model.tabular_encoder[0].in_features
        
    # Get embedding dimension based on model architecture
    if hasattr(model.image_encoder, 'projection'):
        for layer in model.image_encoder.projection:
            if isinstance(layer, nn.Linear):
                embedding_dim = layer.out_features
                break
        else:
            embedding_dim = 256  # Default embedding dimension
    else:
        # For sequential image encoder
        embedding_dim = model.image_encoder[-2].out_features
    
    # Only save the state dict, not the entire model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'epoch': epoch,
        'metrics': metrics,
        'torch_version': torch.__version__,
        'model_class': model.__class__.__name__,
        'tabular_input_dim': tabular_input_dim,  # Store dimensions
        'embedding_dim': embedding_dim
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
    
    # Save best checkpoint if this is the best model so far
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path, _use_new_zipfile_serialization=True)
        
        # Save metrics separately for easy access
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(checkpoint_dir, 'best_metrics.csv'), index=False)

def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    checkpoint_path: str,
    device: torch.device
) -> Tuple[nn.Module, Dict[str, float], int]:
    """Load a model checkpoint with improved error handling."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if we have a state dict
        if 'model_state_dict' in checkpoint:
            # Verify the model structure matches
            if 'tabular_input_dim' in checkpoint and 'embedding_dim' in checkpoint:
                expected_in_features = checkpoint['tabular_input_dim']
                
                # Get actual in_features based on model architecture
                if hasattr(model.tabular_encoder, 'input_projection'):
                    actual_in_features = model.tabular_encoder.input_projection.in_features
                else:
                    actual_in_features = model.tabular_encoder[0].in_features
                
                # Get expected embedding dimension
                expected_embedding_dim = checkpoint.get('embedding_dim', 256)
                
                # Get actual embedding dimension
                if hasattr(model.image_encoder, 'projection'):
                    # For ViTImageEncoder - find a Linear layer in the projection
                    for layer in model.image_encoder.projection:
                        if isinstance(layer, nn.Linear):
                            actual_embedding_dim = layer.out_features
                            break
                    else:
                        # Fallback if no Linear layer is found
                        actual_embedding_dim = 256
                else:
                    # For sequential image encoder
                    actual_embedding_dim = model.image_encoder[-2].out_features
                
                # Check for dimension mismatches
                if expected_in_features != actual_in_features:
                    print(f"Warning: Model architecture mismatch in tabular input! " 
                          f"Expected: {expected_in_features}, Got: {actual_in_features}. "
                          f"Attempting to load compatible weights...")
                
                if expected_embedding_dim != actual_embedding_dim:
                    print(f"Warning: Model architecture mismatch in embedding dimension! " 
                          f"Expected: {expected_embedding_dim}, Got: {actual_embedding_dim}. "
                          f"Attempting to load compatible weights...")
            
            # Filter state dict to only include keys that exist in current model
            current_model_keys = set(model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                   if k in current_model_keys}
            
            missing_keys = current_model_keys - set(filtered_state_dict.keys())
            if missing_keys:
                print(f"Warning: {len(missing_keys)} keys missing from checkpoint: {list(missing_keys)[:5]}...")
            
            # Load the filtered state dict
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded model weights with {len(filtered_state_dict)}/{len(current_model_keys)} keys")
        else:
            raise ValueError("No model state dict found in checkpoint")
 
        # Load optimizer state if available and requested
        if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Load scheduler state if available and requested
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Loaded scheduler state")
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
        
        # Load scaler state if available and requested
        if scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("Loaded scaler state")
            except Exception as e:
                print(f"Warning: Could not load scaler state: {e}")
        
        # Get metrics and epoch or use defaults
        metrics = checkpoint.get('metrics', {})
        epoch = checkpoint.get('epoch', 0)
        
        return model, metrics, epoch
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating fresh model without loading weights...")
        metrics = {}
        epoch = 0
        
        return model, metrics, epoch
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating fresh model without loading weights...")
        metrics = {}
        epoch = 0
        
        return model, metrics, epoch


def count_tabular_features(csv_file: str) -> int:
    """Count the number of tabular features in the dataset."""
    data = pd.read_csv(csv_file)
    tabular_cols = [col for col in data.columns if col.startswith(('MACRO_', 'SMA_', 'RSI', 'EMA_', 'MACD', 'BB_', 
                                              'Volatility', 'Open', 'High', 'Close', 'Volume', 'Lagged_'))]
    return len(tabular_cols)


def train_with_early_stopping(
    model, train_loader, val_loader, optimizer, scaler, 
    scheduler, device, config, checkpoint_dir
):
    """Training loop with proper early stopping"""
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_losses = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            grad_accum_steps=4  # Add gradient accumulation
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device, split='val')
        
        # Update scheduler
        scheduler.step()
        
        # Check if this is the best model so far
        is_best = val_metrics['avg'] < best_val_loss
        
        if is_best:
            best_val_loss = val_metrics['avg']
            epochs_without_improvement = 0
            
            # Save best checkpoint
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, val_metrics, checkpoint_dir,
                is_best=True
            )
        else:
            epochs_without_improvement += 1
            
            # Save regular checkpoint
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, val_metrics, checkpoint_dir,
                is_best=False
            )
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train loss: {train_losses['avg']:.4f}")
        print(f"Val loss: {val_metrics['avg']:.4f}")
        print(f"Val MSE (1d/5d/10d): {val_metrics['return_1d_mse']:.4f}/{val_metrics['return_5d_mse']:.4f}/{val_metrics['return_10d_mse']:.4f}")
        print(f"Val MSE (1d/5d/10d): {val_metrics['return_1d_mse']:.4f}/{val_metrics['return_5d_mse']:.4f}/{val_metrics['return_10d_mse']:.4f}")
        print(f"Epochs without improvement: {epochs_without_improvement}/{config.patience}")
        
        # Early stopping
        if config.early_stopping and epochs_without_improvement >= config.patience:
            print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    
    return best_val_loss
