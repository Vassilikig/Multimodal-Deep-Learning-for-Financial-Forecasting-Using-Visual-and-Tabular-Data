import torch
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F

def regression_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    confidence_weight: float = 0.2
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted loss for directional prediction.
    
    Args:
        predictions: Dictionary of prediction tensors (probabilities)
        targets: Dictionary of target tensors (actual returns)
        confidence_weight: Weight factor for the confidence term
    
    Returns:
        Dictionary of loss values
    """
    losses = {}
    
    # Initialize combined loss
    total_loss = 0.0
    
    pred_keys = ['direction_1d', 'direction_5d', 'direction_10d']
    
    target_keys = ['return_1d', 'return_5d', 'return_10d']
    
    for pred_key, target_key in zip(pred_keys, target_keys):
        pred = predictions[pred_key]
        
        # Convert actual returns to binary labels (1 for positive, 0 for negative)
        direction = (targets[target_key] > 0).float()
        
        # Calculate magnitude of the returns (for confidence weighting)
        magnitude = torch.abs(targets[target_key])
        
        # Normalize magnitude to [0, 1] range for the batch
        if torch.max(magnitude) > 0:
            normalized_magnitude = magnitude / torch.max(magnitude)
        else:
            normalized_magnitude = magnitude
        
        # Add small constant to prevent zero weights
        confidence = normalized_magnitude + 0.5
        
        # Calculate weighted BCE loss
        sample_weights = 1.0 + confidence_weight * confidence
        
        # Apply weighted BCE
        bce_loss = F.binary_cross_entropy(
            pred, direction, reduction='none'
        ) * sample_weights
        
        # Average over batch
        weighted_loss = bce_loss.mean()
        
        losses[f"{pred_key}_bce"] = weighted_loss
        total_loss += weighted_loss
    
    # Store average loss
    losses['avg'] = total_loss / len(pred_keys)
    
    return losses
