# nanomoe/train.py
#(DDP + explicit step + router stats + fp8 hooks)
# nanomoe/train.py
import os, time, math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from nanomoe.configs.base import TrainConfig
from nanomoe.model.transformer import TransformerLM
from nanomoe.optim.registry import build_optimizer
from nanomoe.logging.nviz import aggregate_router_stats
from nanomoe.data.packed_memmap import PackedMemMapDataset
from nanomoe.logging.wandb import init_wandb, log_step

def ddp_init():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world
    return False, 0, 0, 1

@torch.no_grad()
def allreduce_mean(x: float, device) -> float:
    t = torch.tensor([x], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()

def main():
    cfg = TrainConfig()
    ddp, rank, local_rank, world = ddp_init()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(cfg.seed + rank)

    # data
    ds = PackedMemMapDataset(path=os.environ.get("DATA_MEMMAP", "data_train.memmap"), seq_len=cfg.seq_len)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    dl = DataLoader(ds, batch_size=cfg.micro_batch_size, sampler=sampler, shuffle=(sampler is None),
                    num_workers=2, pin_memory=True, drop_last=True)

    model = TransformerLM(cfg).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)

    opt = build_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.precision.amp_bf16)

    model.train()
    step = 0
    t0 = time.time()

    it = iter(dl)
    while step < cfg.max_steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step)

        opt.zero_grad(set_to_none=True)

        loss_accum = 0.0
        router_log = None

        for micro in range(cfg.grad_accum_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

            batch = batch.to(device, non_blocking=True)  # [B,S]
            x = batch[:, :-1]
            y = batch[:, 1:]

            with torch.cuda.amp.autocast(enabled=cfg.precision.amp_bf16, dtype=torch.bfloat16):
                logits, moe_stats_layers = model(x)  # logits [B,S-1,V]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )

                # add router z-loss (already computed in router stats)
                zloss = sum([st["router_zloss"] for st in moe_stats_layers])
                loss = loss + zloss

            loss_scaled = loss / cfg.grad_accum_steps
            scaler.scale(loss_scaled).backward()

            loss_accum += loss.item()
            router_log = aggregate_router_stats(moe_stats_layers)

        # optional: clip (explicit; default None for “YOLO”)
        if cfg.grad_clip_norm is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        scaler.step(opt)
        scaler.update()

        # logging
        if step % cfg.log_every == 0:
            dt = time.time() - t0
            toks = cfg.micro_batch_size * (cfg.seq_len - 1) * cfg.grad_accum_steps * (world if ddp else 1)
            tok_s = toks / max(dt, 1e-9)
            t0 = time.time()

            if ddp:
                loss_mean = allreduce_mean(loss_accum, device)
                tok_s_mean = allreduce_mean(tok_s, device)
            else:
                loss_mean = loss_accum
                tok_s_mean = tok_s

            if rank == 0:
                msg = {
                    "step": step,
                    "loss": round(loss_mean, 4),
                    "tok/s": round(tok_s_mean, 1),
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in router_log.items()},
                }
                print(msg)

        step += 1

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
