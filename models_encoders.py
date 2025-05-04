import torch
import torch.nn as nn
import timm
from typing import Optional


class FeatureStandardizer(nn.Module):
    """Standardizes features along batch dimension."""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        # Standardize along feature dimension (last dim)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.epsilon
        return (x - mean) / std


class ViTImageEncoder(nn.Module):
    """Image encoder using Vision Transformer (ViT) backbone with reduced size."""
    def __init__(self, embedding_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        
        # Get the dimension of the ViT output
        backbone_dim = self.backbone.head.in_features
        
        # Replace the classification head with identity
        self.backbone.head = nn.Identity()
        
        # Add a projection layer
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings


class TabularTransformerEncoder(nn.Module):
    """Transformer encoder for tabular time-series data with reduced size."""
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256, 
        num_layers: int = 2,       
        num_heads: int = 8,
        hidden_dim: int = 512,    
        dropout: float = 0.1,
        max_seq_len: int = 14,
    ):
        super().__init__()
        
        # Feature standardization layer
        self.feature_standardizer = FeatureStandardizer()
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, embedding_dim)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # For better gradient flow
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        # Final projection layer
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Standardize features
        x = self.feature_standardizer(x)
        
        # Project inputs to embedding dimension
        x = self.input_projection(x)
        
        # Add positional encodings
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask if mask is not None else None)
        
        # Extract CLS token representation
        cls_representation = x[:, 0]
        
        # Final projection
        output = self.output_projection(cls_representation)
        
        return output