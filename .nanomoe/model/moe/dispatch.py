# nanomoe/model/moe/dispatch.py
import torch

def compute_capacity(T: int, K: int, num_experts: int, capacity_factor: float, mode: str) -> int:
    """
    mode:
      - "token":      cap ~ T/E
      - "assignment": cap ~ (T*K)/E  (recommended)
    """
    if mode not in ("token", "assignment"):
        raise ValueError(f"capacity_mode must be 'token' or 'assignment', got {mode}")
    base = T if mode == "token" else (T * K)
    return int((base / num_experts) * capacity_factor) + 1


def dispatch_tokens_vectorized(
    x_flat: torch.Tensor,          # [T, D]
    topk_idx: torch.Tensor,        # [T, K] int64
    topk_w: torch.Tensor,          # [T, K] float
    num_experts: int,
    cap: int,
    randomize: bool = True,
    return_k_stats: bool = True,
):
    """
    Vectorized gather/scatter dispatch:
      - expands assignments: N = T*K
      - optional randomization BEFORE sorting to avoid systematic drop bias
      - sorts by expert id (stable) so per-expert segments are contiguous
      - drops assignments with slot>=cap

    Returns:
      expert_in: [E, cap, D]
      map_: dict for combine
      aux: dict with executed metrics (fill/mass/drop/k-survival)
    """
    device = x_flat.device
    T, D = x_flat.shape
    K = topk_idx.shape[1]
    N = T * K

    # assignment-level ids
    token_ids = torch.arange(T, device=device, dtype=torch.int64).repeat_interleave(K)  # [N]
    expert_ids = topk_idx.reshape(-1).to(torch.int64)                                   # [N]
    weights = topk_w.reshape(-1)                                                        # [N]

    # track which k each assignment came from (for k-survival)
    if return_k_stats:
        k_ids = torch.arange(K, device=device, dtype=torch.int64).repeat(T)             # [N]
    else:
        k_ids = None

    # Fairness: randomize assignment order before expert sort, so drops aren't biased
    if randomize:
        perm0 = torch.randperm(N, device=device)
        token_ids = token_ids[perm0]
        expert_ids = expert_ids[perm0]
        weights = weights[perm0]
        if k_ids is not None:
            k_ids = k_ids[perm0]

    # Sort by expert id (stable)
    perm = torch.argsort(expert_ids, stable=True)
    token_ids = token_ids[perm]
    expert_ids = expert_ids[perm]
    weights = weights[perm]
    if k_ids is not None:
        k_ids = k_ids[perm]

    # counts per expert for ALL intended assignments
    counts_all = torch.bincount(expert_ids, minlength=num_experts)  # [E]

    # compute slot within each expert segment
    offsets = torch.cumsum(counts_all, dim=0) - counts_all
    idx = torch.arange(N, device=device, dtype=torch.int64)
    slot = idx - offsets[expert_ids]  # [N], 0..count-1 per expert

    # keep under capacity
    keep = slot < cap
    token_ids_k = token_ids[keep]
    expert_ids_k = expert_ids[keep]
    weights_k = weights[keep]
    slot_k = slot[keep]

    # gather inputs, scatter into expert buffer
    x_gather = x_flat[token_ids_k]  # [Nk, D]
    expert_in = torch.zeros((num_experts, cap, D), device=device, dtype=x_flat.dtype)
    expert_in[expert_ids_k, slot_k] = x_gather

    # executed fill (clipped)
    fill_exec = torch.clamp(counts_all, max=cap)

    # executed mass = sum of kept weights per expert (this catches "2nd expert never learns")
    mass_exec = torch.zeros((num_experts,), device=device, dtype=weights_k.dtype)
    mass_exec.index_add_(0, expert_ids_k, weights_k)

    # drop statistics
    dropped = (counts_all - fill_exec).sum()
    drop_rate = dropped.float() / float(N)

    aux = {
        "counts_all": counts_all.detach(),
        "expert_fill": fill_exec.detach(),
        "expert_mass": mass_exec.detach(),
        "dropped": dropped.detach(),
        "drop_rate": drop_rate.detach(),
        "cap": torch.tensor(cap, device=device),
    }

    # k survival stats (are we silently degenerating top-2 -> top-1?)
    if return_k_stats and k_ids is not None:
        k_kept = k_ids[keep]
        kept_per_k = torch.bincount(k_kept, minlength=K).float()
        total_per_k = torch.full((K,), float(T), device=device)  # each k had exactly T assignments
        keep_rate_k = kept_per_k / (total_per_k + 1e-9)
        aux["keep_rate_k"] = keep_rate_k.detach()

    map_ = {
        "token_ids": token_ids_k,
        "expert_ids": expert_ids_k,
        "slot": slot_k,
        "weights": weights_k,
        "T": T,
        "D": D,
    }
    return expert_in, map_, aux


def combine_expert_outputs_vectorized(
    y_expert: torch.Tensor,  # [E, cap, D]
    map_: dict,
):
    device = y_expert.device
    T = map_["T"]
    D = map_["D"]

    token_ids = map_["token_ids"]
    expert_ids = map_["expert_ids"]
    slot = map_["slot"]
    weights = map_["weights"]

    y = y_expert[expert_ids, slot] * weights[:, None]  # [Nk, D]

    y_flat = torch.zeros((T, D), device=device, dtype=y_expert.dtype)
    y_flat.index_add_(0, token_ids, y)
    return y_flat
