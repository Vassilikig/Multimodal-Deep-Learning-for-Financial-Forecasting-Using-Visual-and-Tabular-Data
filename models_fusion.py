import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Cross-modal attention block where image embedding attends to tabular data."""
    def __init__(self, embedding_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key_value):
        
        # Reshape for attention
        query = query.unsqueeze(1) 
        key_value = key_value.unsqueeze(1)  
        
        # Self attention block
        attn_output, _ = self.attention(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)  
        query = query.squeeze(1)  
        
        # First residual connection
        out1 = self.norm1(query + attn_output)
        
        # Feed forward block
        ff_output = self.ff(out1)
        
        # Second residual connection
        out2 = self.norm2(out1 + ff_output)
        
        return out2
