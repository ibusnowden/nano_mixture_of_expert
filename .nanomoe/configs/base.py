# nanomoe/configs/base.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class MoEConfig:
    num_experts: int = 64
    top_k: int = 2
    capacity_factor: float = 1.25
    router_type: str = "softmax_topk"  # "sigmoid_topk" later
    router_z_loss: float = 0.0         # keep 0.0 for aux-free baseline
    router_temp: float = 1.0
    use_bungee: bool = True
    bungee_init: float = 2.0

@dataclass
class AttnConfig:
    kind: str = "mha"  # "swa", "mla", "nsa", "kda" later

@dataclass
class PrecisionConfig:
    amp_bf16: bool = True
    use_fp8_experts: bool = False
    fp8_format: str = "E4M3"      # placeholder
    fp8_delayed_scaling: bool = True

@dataclass
class TrainConfig:
    # model
    vocab_size: int = 32000
    seq_len: int = 2048
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    mlp_ratio: float = 4.0

    moe: MoEConfig = MoEConfig()
    attn: AttnConfig = AttnConfig()
    precision: PrecisionConfig = PrecisionConfig()

    # training
    global_batch_size: int = 256
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    max_steps: int = 20_000

    # stability knobs (keep explicit)
    grad_clip_norm: Optional[float] = None  # set None for "YOLO"
    loss_scale: Optional[float] = None      # if you want manual scaling
    mup_embed_scale: float = 1.0            # e.g. 10.66 if enabled
    mup_logits_scale: float = 1.0           # e.g. 0.125 if enabled

    # io/logging
    log_every: int = 20
    save_every: int = 1000
    out_dir: str = "./checkpoints"

    # distributed
    seed: int = 1337
