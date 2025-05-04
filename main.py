import os
import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from configs import Config
from data import StockDataset
from models import StockReturnPredictor
from training import (
    train_epoch, 
    evaluate, 
)

from training_utils import (
    get_lr_scheduler,
    save_checkpoint,
    load_checkpoint,
    count_tabular_features,
    train_with_early_stopping
)

def main():
    # Create the configuration
    config = Config()
        
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Get data transforms
    train_transform = StockDataset.get_transforms(is_training=True)
    val_transform = StockDataset.get_transforms(is_training=False)
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = StockDataset(
        csv_file=config.train_csv,
        base_dir=config.data_dir,
        image_transform=train_transform,
        max_seq_len=config.seq_length,
        standardize_targets=False  # Set to False to use raw values
    )
    
    print("Loading validation dataset...")
    val_dataset = StockDataset(
        csv_file=config.val_csv,
        base_dir=config.data_dir,
        image_transform=val_transform,
        max_seq_len=config.seq_length,
        standardize_targets=False  # Set to False to use raw values
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        collate_fn=StockDataset.collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        collate_fn=StockDataset.collate_fn
    )
    
    # Count tabular features
    tabular_input_dim = count_tabular_features(config.train_csv)
    print(f"Number of tabular features: {tabular_input_dim}")
    
    # Create model
    model = StockReturnPredictor(
        tabular_input_dim=tabular_input_dim,
        embedding_dim=config.image_embedding_dim
    )
    model = model.to(device)
    
    # Print model structure
    print(f"Model architecture:\n{model}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate * 0.1,  # Reduced learning rate for binary classification
        weight_decay=config.weight_decay
    )
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Calculate total training steps for scheduler
    total_steps = len(train_loader) * config.epochs
    warmup_steps = config.warmup_steps
    
    # Create learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    if config.resume and os.path.isfile(config.resume):
        print(f"Loading checkpoint from {config.resume}")
        model, metrics, start_epoch = load_checkpoint(
            model, optimizer, scheduler, scaler, config.resume, device
        )
        if 'avg' in metrics:
            best_val_loss = metrics['avg']
        print(f"Resumed from epoch {start_epoch}")
    
    # Initialize metrics tracking for directional prediction
    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_direction_1d_acc': [],
        'val_direction_5d_acc': [],
        'val_direction_10d_acc': [],
        'val_direction_1d_precision': [],
        'val_direction_5d_precision': [],
        'val_direction_10d_precision': [],
        'val_direction_1d_recall': [],
        'val_direction_5d_recall': [],
        'val_direction_10d_recall': [],
        'val_direction_1d_f1': [],
        'val_direction_5d_f1': [],
        'val_direction_10d_f1': [],
    }
    
    # Training loop
    print(f"Starting training for {config.epochs} epochs (directional prediction model)")
    start_time = time.time()
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch with gradient accumulation
        train_losses = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            scaler, 
            device, 
            epoch,
            grad_accum_steps=4
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device, split='val')
        
        # Update scheduler
        scheduler.step()
        
        # Track metrics history for directional prediction
        metrics_history['epoch'].append(epoch)
        metrics_history['train_loss'].append(train_losses['avg'])
        metrics_history['val_loss'].append(val_metrics['avg'])
        
        # Track directional accuracy metrics
        for horizon in ['1d', '5d', '10d']:
            metrics_history[f'val_direction_{horizon}_acc'].append(val_metrics[f'direction_{horizon}_acc'])
            metrics_history[f'val_direction_{horizon}_precision'].append(val_metrics[f'direction_{horizon}_precision'])
            metrics_history[f'val_direction_{horizon}_recall'].append(val_metrics[f'direction_{horizon}_recall'])
            metrics_history[f'val_direction_{horizon}_f1'].append(val_metrics[f'direction_{horizon}_f1'])
        
        # Check if this is the best model based on validation loss
        is_best = val_metrics['avg'] < best_val_loss
        
        if is_best:
            best_val_loss = val_metrics['avg']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint
        save_checkpoint(
            model, 
            optimizer, 
            scheduler, 
            scaler,
            epoch, 
            val_metrics, 
            config.output_dir,
            is_best=is_best
        )
        
        # Print epoch summary with directional metrics
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train loss: {train_losses['avg']:.4f}")
        print(f"Val loss: {val_metrics['avg']:.4f}")
        print(f"Val Direction Accuracy (1d/5d/10d):{val_metrics['direction_1d_acc']:.4f}/{val_metrics['direction_5d_acc']:.4f}/{val_metrics['direction_10d_acc']:.4f}")
        print(f"Val Direction F1 (1d/5d/10d): {val_metrics['direction_1d_f1']:.4f}/{val_metrics['direction_5d_f1']:.4f}/{val_metrics['direction_10d_f1']:.4f}")
        print(f"Epochs without improvement: {epochs_without_improvement}/{config.patience}")
        
        # Early stopping
        if config.early_stopping and epochs_without_improvement >= config.patience:
            print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Convert metrics history to DataFrame for easier handling
    metrics_df = pd.DataFrame(metrics_history)
    
    # Save metrics to CSV
    metrics_csv_path = os.path.join(config.output_dir, 'training_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Training metrics saved to {metrics_csv_path}")
    
    # Plot training curves with directional metrics
    plt.figure(figsize=(15, 10))
    
    # Plot loss curves
    plt.subplot(2, 3, 1)
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot directional accuracy curves
    plt.subplot(2, 3, 2)
    plt.plot(metrics_df['epoch'], metrics_df['val_direction_1d_acc'], label='1-day')
    plt.plot(metrics_df['epoch'], metrics_df['val_direction_5d_acc'], label='5-day')
    plt.plot(metrics_df['epoch'], metrics_df['val_direction_10d_acc'], label='10-day')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Directional Prediction Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 scores
    plt.subplot(2, 3, 3)
    plt.plot(metrics_df['epoch'], metrics_df['val_direction_1d_f1'], label='1-day')
    plt.plot(metrics_df['epoch'], metrics_df['val_direction_5d_f1'], label='5-day')
    plt.plot(metrics_df['epoch'], metrics_df['val_direction_10d_f1'], label='10-day')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Directional F1 Scores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'training_curves.png'))
    print(f"Training curves saved to {os.path.join(config.output_dir, 'training_curves.png')}")

if __name__ == "__main__":
    main()