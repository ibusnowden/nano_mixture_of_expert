# nanomoe/model/moe/router.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """
    Returns:
      topk_idx: [T, K] int64
      topk_w:   [T, K] float, normalized across K
      stats: dict with entropy/cv/logits_rms/top1_margin
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int,
                 router_type: str = "softmax_topk", temp: float = 1.0, z_loss: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_type = router_type
        self.temp = temp
        self.z_loss = z_loss
        self.proj = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        T = B * S
        h = x.reshape(T, D)

        logits = self.proj(h) / max(self.temp, 1e-6)  # [T,E]

        if self.router_type == "softmax_topk":
            dist = F.softmax(logits, dim=-1)  # [T,E]
            topk_w, topk_idx = torch.topk(dist, k=self.top_k, dim=-1)

        elif self.router_type == "sigmoid_topk":
            gates = torch.sigmoid(logits)     # [T,E]
            topk_w, topk_idx = torch.topk(gates, k=self.top_k, dim=-1)
            # for entropy proxy, renormalize gates to a distribution
            dist = gates / (gates.sum(dim=-1, keepdim=True) + 1e-9)

        else:
            raise ValueError(f"Unknown router_type={self.router_type}")

        # normalize weights across selected experts (important for combine)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)

        # stats: entropy & load balance (top1 proxy)
        entropy = (-dist * (dist + 1e-9).log()).sum(dim=-1)  # [T]
        top1 = topk_idx[:, 0]
        counts = torch.bincount(top1, minlength=self.num_experts).float()
        mean = counts.mean()
        cv = counts.std(unbiased=False) / (mean + 1e-9)

        # margin/confidence for K>=2 (helps detect under-specialization)
        if self.top_k >= 2:
            top1_margin = (topk_w[:, 0] - topk_w[:, 1]).mean()
            top1_conf = topk_w[:, 0].mean()
        else:
            top1_margin = torch.tensor(0.0, device=x.device)
            top1_conf = topk_w[:, 0].mean()

        # optional z-loss (default 0)
        zloss = torch.tensor(0.0, device=x.device)
        if self.z_loss > 0.0:
            z = torch.logsumexp(logits, dim=-1)
            zloss = self.z_loss * (z ** 2).mean()

        stats = {
            "router_entropy_mean": entropy.mean().detach(),
            "router_entropy_min": entropy.min().detach(),
            "router_cv": cv.detach(),
            "router_counts": counts.detach(),
            "router_zloss": zloss.detach(),
            "router_logits_rms": logits.pow(2).mean().sqrt().detach(),
            "router_top1_margin": top1_margin.detach(),
            "router_top1_conf": top1_conf.detach(),
        }
        return topk_idx.to(torch.int64), topk_w, stats
