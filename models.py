import torch
import torch.nn as nn
import timm
from models_fusion import CrossModalAttention
from models_encoders import ViTImageEncoder, TabularTransformerEncoder, FeatureStandardizer

class StockReturnPredictor(nn.Module):
    def __init__(self, tabular_input_dim, embedding_dim=256, transformer_layers=2, 
                 transformer_heads=4, transformer_hidden_dim=512, dropout=0.2, max_seq_len=14):
        super().__init__()
        
        # Image encoder 
        self.image_encoder = ViTImageEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Tabular encoder 
        self.tabular_encoder = TabularTransformerEncoder(
            input_dim=tabular_input_dim,
            embedding_dim=embedding_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,  
            hidden_dim=transformer_hidden_dim,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Enhanced cross-modal fusion
        self.cross_modal_attention = CrossModalAttention(embedding_dim, num_heads=4)
        
        # Gating mechanism to control information flow (simplified from second example)
        self.gate = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout * 0.5),  # Reduced dropout in later layers
        )
        
        # Separate prediction heads for different time horizons 
        self.head_1d = nn.Linear(embedding_dim, 1)
        self.head_5d = nn.Linear(embedding_dim, 1)
        self.head_10d = nn.Linear(embedding_dim, 1)
        
    def forward(self, images, tabular_data):
        # Process encoders
        v = self.image_encoder(images)         
        t = self.tabular_encoder(tabular_data)  
        
        # Cross-modal attention (v queries t)
        v_star = self.cross_modal_attention(v, t)  
        
        # Concatenate features
        z = torch.cat([v_star, t], dim=1)  
        
        # Apply gating to control information flow 
        g = torch.sigmoid(self.gate(z))  
        z_g = z * g  # (B, 2d)
        
        # Fuse features
        h = self.fusion(z_g)  # (B, d)
        
        # Apply separate prediction heads
        p_1d = torch.sigmoid(self.head_1d(h)).squeeze(-1)
        p_5d = torch.sigmoid(self.head_5d(h)).squeeze(-1)
        p_10d = torch.sigmoid(self.head_10d(h)).squeeze(-1)
        
        return {
            'direction_1d': p_1d,
            'direction_5d': p_5d,
            'direction_10d': p_10d,
        }
