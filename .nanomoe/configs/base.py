# nanomoe/configs/base.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MoEConfig:
    num_experts: int = 64
    top_k: int = 2
    capacity_factor: float = 1.25

    # IMPORTANT: capacity_mode controls whether capacity accounts for top-k expansion.
    # "token"       => cap ~ T/E
    # "assignment"  => cap ~ (T*K)/E  (recommended for research honesty)
    capacity_mode: str = "assignment"

    # router
    router_type: str = "softmax_topk"  # "sigmoid_topk"
    router_z_loss: float = 0.0
    router_temp: float = 1.0

    # dispatch fairness
    randomize_dispatch: bool = True     # avoids systematic drop bias
    enforce_k_fairness_metrics: bool = True

    # bungee
    use_bungee: bool = True
    per_expert_bungee: bool = True      # NEW: per-expert gain
    bungee_init: float = 2.0


@dataclass
class AttnConfig:
    kind: str = "mha"

@dataclass
class PrecisionConfig:
    amp_bf16: bool = True
    use_fp8_experts: bool = False
    fp8_format: str = "E4M3"
    fp8_delayed_scaling: bool = True

@dataclass
class TrainConfig:
    vocab_size: int = 32000
    seq_len: int = 2048
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    mlp_ratio: float = 4.0

    moe: MoEConfig = field(default_factory=MoEConfig)
    attn: AttnConfig = field(default_factory=AttnConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)

    global_batch_size: int = 256
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    max_steps: int = 20_000

    grad_clip_norm: Optional[float] = None
    loss_scale: Optional[float] = None
    mup_embed_scale: float = 1.0
    mup_logits_scale: float = 1.0

    log_every: int = 20
    save_every: int = 1000
    out_dir: str = "./checkpoints"

    seed: int = 1337
