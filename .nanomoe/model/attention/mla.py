import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention (MLA).
    
    Low-rank compression of Key-Value pairs inspired by DeepSeek-V2.
    Projects input to a compressed latent space before generating K, V,
    reducing KV cache memory during inference.
    
    Architecture:
    - Input → Compressed Latent (low-rank) → K, V projections
    - Q is projected directly from input (full rank)
    - Standard scaled dot-product attention with causal mask
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        latent_ratio: float = 0.25,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        
        # Latent dimension for KV compression
        self.latent_dim = max(int(d_model * latent_ratio), n_heads)
        # Ensure latent_dim is divisible by n_heads for clean head splitting
        self.latent_dim = (self.latent_dim // n_heads) * n_heads
        
        # Query projection (full rank)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        # KV compression: input → latent → K, V
        self.kv_down = nn.Linear(d_model, self.latent_dim, bias=False)
        self.k_up = nn.Linear(self.latent_dim, d_model, bias=False)
        self.v_up = nn.Linear(self.latent_dim, d_model, bias=False)
        
        # Output projection
        self.proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Full-rank query projection
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, S, hd]
        
        # Compressed KV pathway
        latent = self.kv_down(x)  # [B, S, latent_dim]
        k = self.k_up(latent).view(B, S, self.n_heads, self.head_dim)
        v = self.v_up(latent).view(B, S, self.n_heads, self.head_dim)
        k = k.transpose(1, 2)  # [B, H, S, hd]
        v = v.transpose(1, 2)
        
        # Standard scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, S, S]
        
        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        
        # Apply attention to values
        y = att @ v  # [B, H, S, hd]
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.proj(y)
