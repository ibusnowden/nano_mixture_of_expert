import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention (SWA).
    
    Each token attends only to a local window of previous tokens plus itself.
    This reduces memory from O(n²) to O(n·w), enabling longer sequences.
    
    Used in architectures like Mistral and Mixtral for efficient long-context.
    """
    
    def __init__(self, d_model: int, n_heads: int, window_size: int = 256):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        
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
        
        # Create sliding window causal mask
        # Each position i can attend to positions max(0, i - window_size + 1) to i
        row_idx = torch.arange(S, device=x.device).unsqueeze(1)  # [S, 1]
        col_idx = torch.arange(S, device=x.device).unsqueeze(0)  # [1, S]
        
        # Causal: can only attend to current and past
        causal_mask = col_idx > row_idx
        # Window: can only attend within window_size
        window_mask = col_idx < (row_idx - self.window_size + 1)
        # Combined mask: positions outside both causal and window bounds
        mask = causal_mask | window_mask
        
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        
        # Apply attention to values
        y = att @ v  # [B, H, S, hd]
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.proj(y)
