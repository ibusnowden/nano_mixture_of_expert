import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyDrivenAttention(nn.Module):
    """
    Key-Driven Attention (KDA).
    
    Sparse attention that dynamically selects the top-k most relevant keys
    for each query position. This learns to focus on the most important
    tokens rather than using fixed sparsity patterns.
    
    Each query attends only to its top-k highest-scoring keys (within causal bounds).
    """
    
    def __init__(self, d_model: int, n_heads: int, top_k_keys: int = 64):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k_keys = top_k_keys
        
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
        
        # Apply causal mask first
        row_idx = torch.arange(S, device=x.device).unsqueeze(1)
        col_idx = torch.arange(S, device=x.device).unsqueeze(0)
        causal_mask = col_idx > row_idx
        att = att.masked_fill(causal_mask, float("-inf"))
        
        # For each query position, keep only top-k keys (within causal bounds)
        # Effective k is min(top_k_keys, number of valid causal positions)
        effective_k = min(self.top_k_keys, S)
        
        # Get top-k values and indices per query position
        topk_vals, topk_idx = att.topk(effective_k, dim=-1, largest=True, sorted=False)
        
        # Create sparse attention: zero out everything except top-k
        sparse_att = torch.full_like(att, float("-inf"))
        sparse_att.scatter_(-1, topk_idx, topk_vals)
        
        # Softmax over the sparse attention
        att = F.softmax(sparse_att, dim=-1)
        
        # Handle positions where all values were -inf (shouldn't normally happen)
        att = torch.nan_to_num(att, nan=0.0)
        
        # Apply attention to values
        y = att @ v  # [B, H, S, hd]
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.proj(y)
