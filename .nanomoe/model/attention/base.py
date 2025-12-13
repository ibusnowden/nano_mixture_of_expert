from torch import nn

from nanomoe.model.attention.mha import CausalSelfAttention


def build_attention(kind: str, d_model: int, n_heads: int) -> nn.Module:
    """
    Factory for attention modules so TrainConfig.attn.kind is respected.
    Currently supports the default causal MHA.
    """
    kind = (kind or "mha").lower()
    if kind in ("mha", "self_attention", "causal_mha"):
        return CausalSelfAttention(d_model=d_model, n_heads=n_heads)
    raise ValueError(f"Unknown attention kind '{kind}'. Available kinds: ['mha'].")
