# nanomoe/model/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanomoe.model.moe.router import Router
from nanomoe.model.moe.experts import ExpertsMLP
from nanomoe.model.moe.dispatch import compute_capacity, dispatch_tokens_vectorized, combine_expert_outputs_vectorized
from nanomoe.precision.fp8 import fp8_expert_context
from nanomoe.model.attention.base import build_attention

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.w

class MoEBlock(nn.Module):
    def __init__(self, d_model, hidden, moe_cfg, precision_cfg):
        super().__init__()
        self.router = Router(
            d_model=d_model,
            num_experts=moe_cfg.num_experts,
            top_k=moe_cfg.top_k,
            router_type=moe_cfg.router_type,          # NEW
            temp=moe_cfg.router_temp,
            z_loss=moe_cfg.router_z_loss,
        )
        self.experts = ExpertsMLP(
            d_model=d_model,
            hidden=hidden,
            num_experts=moe_cfg.num_experts,
            use_bungee=moe_cfg.use_bungee,
            per_expert_bungee=moe_cfg.per_expert_bungee,  # NEW
            bungee_init=moe_cfg.bungee_init,
        )
        self.moe_cfg = moe_cfg
        self.precision_cfg = precision_cfg

    def forward(self, x):
        B, S, D = x.shape
        T = B * S

        x_flat = x.reshape(T, D)
        topk_idx, topk_w, r_stats = self.router(x)

        cap = compute_capacity(
            T=T,
            K=self.moe_cfg.top_k,
            num_experts=self.moe_cfg.num_experts,
            capacity_factor=self.moe_cfg.capacity_factor,
            mode=self.moe_cfg.capacity_mode,
        )

        expert_in, map_, aux = dispatch_tokens_vectorized(
            x_flat=x_flat,
            topk_idx=topk_idx,
            topk_w=topk_w,
            num_experts=self.moe_cfg.num_experts,
            cap=cap,
            randomize=self.moe_cfg.randomize_dispatch,
            return_k_stats=self.moe_cfg.enforce_k_fairness_metrics,
        )

        with fp8_expert_context(self.precision_cfg.use_fp8_experts, fmt=self.precision_cfg.fp8_format):
            expert_out, e_stats = self.experts(expert_in)

        y_flat = combine_expert_outputs_vectorized(expert_out, map_)
        y = y_flat.view(B, S, D)

        stats = {**r_stats, **e_stats}
        stats["router_counts_vec"] = r_stats["router_counts"]

        # executed routing metrics (THIS prevents fake stability)
        stats["expert_cap"] = aux["cap"].detach()
        stats["dropped"] = aux["dropped"].detach()
        stats["drop_rate"] = aux["drop_rate"].detach()
        stats["expert_fill_mean"] = aux["expert_fill"].float().mean().detach()
        stats["expert_fill_min"] = aux["expert_fill"].float().min().detach()
        stats["expert_fill_max"] = aux["expert_fill"].float().max().detach()
        stats["exec_expert_fill_vec"] = aux["expert_fill"].detach()

        stats["expert_mass_mean"] = aux["expert_mass"].float().mean().detach()
        stats["expert_mass_min"] = aux["expert_mass"].float().min().detach()
        stats["expert_mass_max"] = aux["expert_mass"].float().max().detach()
        stats["exec_expert_mass_vec"] = aux["expert_mass"].detach()

        if "keep_rate_k" in aux:
            # log first two for K=2; extend if you like
            kr = aux["keep_rate_k"]
            stats["keep_rate_k0"] = kr[0].detach()
            if kr.numel() > 1:
                stats["keep_rate_k1"] = kr[1].detach()

        return y, stats


class Block(nn.Module):
    def __init__(self, d_model, n_heads, hidden, moe_cfg, precision_cfg, attn_cfg):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = build_attention(attn_cfg.kind, d_model, n_heads)
        self.n2 = RMSNorm(d_model)
        self.moe = MoEBlock(d_model, hidden, moe_cfg, precision_cfg)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        moe_out, moe_stats = self.moe(self.n2(x))
        x = x + moe_out
        return x, moe_stats

class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        hidden = int(cfg.d_model * cfg.mlp_ratio)

        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, hidden, cfg.moe, cfg.precision, cfg.attn)
            for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx):
        # idx: [B,S]
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device)
        x = self.tok(idx) * self.cfg.mup_embed_scale + self.pos(pos)[None, :, :]

        moe_stats_layers = []
        for b in self.blocks:
            x, st = b(x)
            moe_stats_layers.append(st)

        x = self.norm(x)
        logits = self.lm_head(x) * self.cfg.mup_logits_scale
        return logits, moe_stats_layers
