# nanomoe/logging/wandb.py
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def is_enabled() -> bool:
    return bool(int(os.environ.get("WANDB", "0")))


def is_rank0(rank: int) -> bool:
    return rank == 0


def init_wandb(cfg: Any, rank: int) -> None:
    """
    Call once at startup (rank0 only).
    """
    if not (is_enabled() and is_rank0(rank)):
        return
    if wandb is None:
        raise RuntimeError("WANDB=1 but wandb is not installed/importable.")

    project = os.environ.get("WANDB_PROJECT", "nanomoe")
    entity = os.environ.get("WANDB_ENTITY", None)
    name = os.environ.get("WANDB_RUN_NAME", None)

    # Best-effort config serialization (dataclasses -> dict)
    try:
        cfg_dict = asdict(cfg)
    except Exception:
        cfg_dict = {"cfg_repr": repr(cfg)}

    wandb.init(project=project, entity=entity, name=name, config=cfg_dict)


@torch.no_grad()
def _layerwise_mean_vec(moe_stats_layers: list[dict], key: str) -> torch.Tensor:
    # returns [E] float32 on device
    return torch.stack([st[key].float() for st in moe_stats_layers], dim=0).mean(dim=0)


@torch.no_grad()
def log_step(
    *,
    step: int,
    rank: int,
    scalars: Dict[str, float],
    moe_stats_layers: Optional[list[dict]] = None,
    hist_every: int = 200,
) -> None:
    """
    Scalars logged every call; histograms logged every hist_every steps.
    Keep overhead low:
      - rank0 only
      - histograms only every N steps
      - aggregate across layers to [E]
    """
    if not (is_enabled() and is_rank0(rank)):
        return
    if wandb is None:
        return

    log: Dict[str, Any] = dict(scalars)

    if moe_stats_layers is not None and hist_every > 0 and (step % hist_every == 0):
        # Required vectors stored by MoEBlock
        fill_vec = _layerwise_mean_vec(moe_stats_layers, "exec_expert_fill_vec")   # [E]
        mass_vec = _layerwise_mean_vec(moe_stats_layers, "exec_expert_mass_vec")  # [E]
        rcnt_vec = _layerwise_mean_vec(moe_stats_layers, "router_counts_vec")     # [E]

        # Avoid CPU sync until the very end
        fill_np = fill_vec.detach().cpu().numpy()
        mass_np = mass_vec.detach().cpu().numpy()
        rcnt_np = rcnt_vec.detach().cpu().numpy()

        log["hist/expert_fill"] = wandb.Histogram(fill_np)
        log["hist/expert_mass"] = wandb.Histogram(mass_np)
        log["hist/router_counts"] = wandb.Histogram(rcnt_np)

        # Bonus: avg weight per kept assignment (catches “2nd expert never learns”)
        avg_w = mass_vec / (fill_vec + 1e-9)
        log["hist/avg_weight_per_assign"] = wandb.Histogram(avg_w.detach().cpu().numpy())

    wandb.log(log, step=step)
