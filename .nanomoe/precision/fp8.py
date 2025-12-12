# nanomoe/model/moe/experts.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertsMLP(nn.Module):
    """
    v0.1: E experts, each is a 2-layer MLP.
    expert_in: [E, C, D] -> expert_out: [E, C, D]
    """
    def __init__(self, d_model: int, hidden: int, num_experts: int, use_bungee: bool, bungee_init: float):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        self.b1 = nn.Parameter(torch.zeros(num_experts, hidden))
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))

        # “bungee” scalar pre-output norm hook point (v0.1: scalar per layer; later: per-expert)
        self.use_bungee = use_bungee
        self.bungee = nn.Parameter(torch.tensor(float(bungee_init))) if use_bungee else None

        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x: torch.Tensor):
        # x: [E, C, D]
        E, C, D = x.shape
        # (E,C,D) @ (E,D,H) -> (E,C,H)
        h = torch.einsum("ecd,edh->ech", x, self.w1) + self.b1[:, None, :]
        h = F.gelu(h)
        y = torch.einsum("ech,ehd->ecd", h, self.w2) + self.b2[:, None, :]

        if self.use_bungee:
            y = y * self.bungee  # simple and explicit

        stats = {
            "bungee": self.bungee.detach() if self.use_bungee else torch.tensor(0.0, device=x.device),
            "expert_out_rms": y.pow(2).mean().sqrt().detach(),
        }
        return y, stats
