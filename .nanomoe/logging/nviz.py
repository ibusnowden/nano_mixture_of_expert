# nanomoe/logging/nviz.py
import torch

@torch.no_grad()
def aggregate_router_stats(moe_stats_layers: list[dict]):
    def stack(key: str):
        return torch.stack([s[key] for s in moe_stats_layers])

    ent = stack("router_entropy_mean")
    ent_min = stack("router_entropy_min")
    cv = stack("router_cv")

    b_mean = stack("bungee_mean")
    out_rms = stack("expert_out_rms")

    drop_rate = stack("drop_rate")
    fill_min = stack("expert_fill_min")
    fill_mean = stack("expert_fill_mean")
    fill_max = stack("expert_fill_max")

    mass_min = stack("expert_mass_min")
    mass_mean = stack("expert_mass_mean")
    mass_max = stack("expert_mass_max")

    top1_margin = stack("router_top1_margin")
    top1_conf = stack("router_top1_conf")

    out = {
        "router/entropy_mean": ent.mean().item(),
        "router/entropy_min": ent_min.min().item(),
        "router/cv_mean": cv.mean().item(),
        "router/cv_max": cv.max().item(),
        "router/top1_margin": top1_margin.mean().item(),
        "router/top1_conf": top1_conf.mean().item(),

        "moe/bungee_mean": b_mean.mean().item(),
        "moe/expert_out_rms_mean": out_rms.mean().item(),

        # executed routing (prevents “fake stability”)
        "exec/drop_rate_mean": drop_rate.mean().item(),
        "exec/fill_min_min": fill_min.min().item(),
        "exec/fill_mean_mean": fill_mean.mean().item(),
        "exec/fill_max_max": fill_max.max().item(),

        "exec/mass_min_min": mass_min.min().item(),
        "exec/mass_mean_mean": mass_mean.mean().item(),
        "exec/mass_max_max": mass_max.max().item(),
    }

    # if k stats exist, aggregate them too
    if "keep_rate_k0" in moe_stats_layers[0]:
        k0 = stack("keep_rate_k0")
        out["exec/keep_rate_k0_mean"] = k0.mean().item()
    if "keep_rate_k1" in moe_stats_layers[0]:
        k1 = stack("keep_rate_k1")
        out["exec/keep_rate_k1_mean"] = k1.mean().item()

    return out
