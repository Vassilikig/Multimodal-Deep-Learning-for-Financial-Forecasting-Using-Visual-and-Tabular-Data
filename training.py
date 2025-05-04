import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import time
import os
import pandas as pd

from training_loss import regression_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 4,
    log_interval: int = 10
) -> Dict[str, float]:
    """Train the model for one epoch with gradient accumulation for directional prediction."""
    model.train()
    total_samples = 0
    running_losses = {'avg': 0.0} 
    
    # Update metrics for directional prediction
    for metric in ['direction_1d_bce', 'direction_5d_bce', 'direction_10d_bce']:
        running_losses[metric] = 0.0
    
    optimizer.zero_grad()  # Zero gradients at the start
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        tabular = batch['tabular'].to(device)
        targets = {k: v.to(device) for k, v in batch['targets'].items()}
        
        # Use mixed precision for forward pass
        if device.type == 'cuda':
            with autocast():
                outputs = model(images, tabular)
                losses = regression_loss(outputs, targets)
                loss = losses['avg'] / grad_accum_steps
        else:
            outputs = model(images, tabular)
            losses = regression_loss(outputs, targets)
            loss = losses['avg'] / grad_accum_steps
        
        # Scale gradients and perform backward pass
        scaler.scale(loss).backward()
        
        # Update weights after accumulating gradients
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update running losses
        batch_size = images.size(0)
        total_samples += batch_size
        for k, v in losses.items():
            running_losses[k] = running_losses.get(k, 0.0) + (v.item() * batch_size)
        
        # Update progress bar with directional metrics
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': losses['avg'].item(),
                'dir_1d': losses.get('direction_1d_bce', 0.0),
                'dir_5d': losses.get('direction_5d_bce', 0.0),
            })
    
    # Calculate average losses
    avg_losses = {k: v / total_samples for k, v in running_losses.items()}
    
    return avg_losses


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    split: str = 'val'
) -> Dict[str, float]:
    """Evaluate the model on validation or test set for directional prediction."""
    model.eval()
    total_samples = 0
    running_losses = {'avg': 0.0}
    
    # Update metric keys for directional prediction
    bce_metrics = ['direction_1d_bce', 'direction_5d_bce', 'direction_10d_bce']
    for metric in bce_metrics:
        running_losses[metric] = 0.0
    
    # Update prediction keys
    pred_keys = ['direction_1d', 'direction_5d', 'direction_10d']
    target_keys = ['return_1d', 'return_5d', 'return_10d']
    
    all_preds = {k: [] for k in pred_keys}
    all_targets = {k: [] for k in target_keys}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating on {split}"):
            images = batch['images'].to(device)
            tabular = batch['tabular'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            
            outputs = model(images, tabular)
            
            # Store predictions and targets
            for pred_key, target_key in zip(pred_keys, target_keys):
                all_preds[pred_key].append(outputs[pred_key].cpu().numpy())
                all_targets[target_key].append(targets[target_key].cpu().numpy())
            
            losses = regression_loss(outputs, targets)
            
            # Update running losses
            batch_size = images.size(0)
            total_samples += batch_size
            for k, v in losses.items():
                running_losses[k] = running_losses.get(k, 0.0) + (v.item() * batch_size)
    
    # Calculate average losses
    avg_losses = {k: v / total_samples for k, v in running_losses.items()}
    
    # Calculate additional metrics specifically focused on directional accuracy
    metrics = compute_metrics(all_preds, all_targets)
    metrics.update(avg_losses)
    
    return metrics

def compute_metrics(
    all_preds: Dict[str, List[np.ndarray]],
    all_targets: Dict[str, List[np.ndarray]]
) -> Dict[str, float]:
    """Compute evaluation metrics focusing on directional accuracy."""
    metrics = {}
    
    # Mapping between prediction keys and target keys
    pred_target_mapping = {
        'direction_1d': 'return_1d',
        'direction_5d': 'return_5d',
        'direction_10d': 'return_10d'
    }
    
    for pred_key, target_key in pred_target_mapping.items():
        # Concatenate all batches
        preds = np.concatenate(all_preds[pred_key])
        targets = np.concatenate(all_targets[target_key])
        
        # Convert predicted probabilities to binary predictions
        pred_direction = (preds > 0.5).astype(np.float32)
        # Convert actual returns to binary (up/down)
        target_direction = (targets > 0).astype(np.float32)
        
        # Directional Accuracy
        dir_acc = np.mean(pred_direction == target_direction)
        metrics[f'{pred_key}_acc'] = dir_acc
        
        # True/False Positives/Negatives
        true_pos = np.sum((pred_direction == 1) & (target_direction == 1))
        false_pos = np.sum((pred_direction == 1) & (target_direction == 0))
        true_neg = np.sum((pred_direction == 0) & (target_direction == 0))
        false_neg = np.sum((pred_direction == 0) & (target_direction == 1))
        
        # Precision, Recall, F1 Score
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'{pred_key}_precision'] = precision
        metrics[f'{pred_key}_recall'] = recall
        metrics[f'{pred_key}_f1'] = f1
    
    return metrics