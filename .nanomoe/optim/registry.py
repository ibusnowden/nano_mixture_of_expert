# nanomoe/optim/registry.py
import torch

def build_optimizer(cfg, model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
