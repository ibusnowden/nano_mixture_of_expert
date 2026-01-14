import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeSparseAttention(nn.Module):
    """
    Native Sparse Attention (NSA).
    
    Combines local sliding window attention with global attention to anchor tokens.
    This balances efficient local context with long-range dependency modeling.
    
    Inspired by Longformer/BigBird sparse attention patterns.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        window_size: int = 128,
        global_stride: int = 64,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.global_stride = global_stride
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Project to QKV
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, S, H, hd]
        q = q.transpose(1, 2)  # [B, H, S, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, S, S]
        
        # Build sparse attention mask combining:
        # 1. Causal mask (can't look into future)
        # 2. Local window (attend to nearby tokens)
        # 3. Global anchors (attend to every global_stride-th token)
        
        row_idx = torch.arange(S, device=x.device).unsqueeze(1)  # [S, 1]
        col_idx = torch.arange(S, device=x.device).unsqueeze(0)  # [1, S]
        
        # Causal constraint
        causal_mask = col_idx > row_idx
        
        # Local window: attend to tokens within window
        in_window = (col_idx >= (row_idx - self.window_size + 1)) & (col_idx <= row_idx)
        
        # Global anchors: attend to every global_stride-th token (that is causal)
        is_global = (col_idx % self.global_stride == 0) & (col_idx <= row_idx)
        
        # Combined: attend if in window OR global anchor, but never future
        can_attend = (in_window | is_global) & ~causal_mask
        mask = ~can_attend
        
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        
        # Handle potential NaN from all-masked rows (shouldn't happen with proper masking)
        att = torch.nan_to_num(att, nan=0.0)
        
        # Apply attention to values
        y = att @ v  # [B, H, S, hd]
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.proj(y)
