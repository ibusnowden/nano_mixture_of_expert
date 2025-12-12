# nanomoe/model/moe/experts.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertsMLP(nn.Module):
    """
    v0.2: optional per-expert bungee scalars.
    """
    def __init__(self, d_model: int, hidden: int, num_experts: int,
                 use_bungee: bool, per_expert_bungee: bool, bungee_init: float):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.b1 = nn.Parameter(torch.zeros(num_experts, hidden))
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))

        self.use_bungee = use_bungee
        self.per_expert_bungee = per_expert_bungee

        if use_bungee:
            if per_expert_bungee:
                self.bungee = nn.Parameter(torch.full((num_experts,), float(bungee_init)))
            else:
                self.bungee = nn.Parameter(torch.tensor(float(bungee_init)))
        else:
            self.bungee = None

        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x: torch.Tensor):
        # x: [E, C, D]
        h = torch.einsum("ecd,edh->ech", x, self.w1) + self.b1[:, None, :]
        h = F.gelu(h)
        y = torch.einsum("ech,ehd->ecd", h, self.w2) + self.b2[:, None, :]

        if self.use_bungee:
            if self.per_expert_bungee:
                y = y * self.bungee[:, None, None]
            else:
                y = y * self.bungee

        stats = {
            "expert_out_rms": y.pow(2).mean().sqrt().detach(),
        }
        if self.use_bungee:
            if self.per_expert_bungee:
                stats.update({
                    "bungee_mean": self.bungee.mean().detach(),
                    "bungee_min": self.bungee.min().detach(),
                    "bungee_max": self.bungee.max().detach(),
                })
            else:
                stats.update({
                    "bungee_mean": self.bungee.detach(),
                    "bungee_min": self.bungee.detach(),
                    "bungee_max": self.bungee.detach(),
                })
        else:
            z = torch.tensor(0.0, device=x.device)
            stats.update({"bungee_mean": z, "bungee_min": z, "bungee_max": z})

        return y, stats
