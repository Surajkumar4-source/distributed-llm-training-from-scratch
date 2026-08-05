# Module 10 — GPT with DDP

## Summary
Take the exact GPT from Module 9 and distribute it across **2 → 8 → 16 GPUs** using `DistributedDataParallel`. You will see that the **model code does not change at all** — only the training infrastructure changes.

- **Theory**: what DDP does to GPT specifically, gradient sync cost for transformer, effective batch size math, learning rate scaling
- **From scratch**: convert single-GPU GPT trainer → full DDP trainer
- **Run**: 1 → 2 → 4 → 8 GPUs single node, then 16 GPUs across 2 nodes
- **Measure**: tokens/s, scaling efficiency, communication overhead, memory per GPU
- **Advanced**: gradient accumulation with `no_sync()`, checkpointing in DDP, distributed evaluation
- **Debug drills**: gradient sync bugs, checkpoint save/load on rank 0 only, dataloader race conditions

---

## 10.1 Theory — What Changes When GPT Goes to DDP

### The fundamental contract
DDP makes one promise:

```
After backward():
  every rank has the same averaged gradients
  as if you trained on the full global batch
  on one GPU.
```

Everything else (model architecture, optimizer, loss function, scheduler) is **identical** to single-GPU training.

### What DDP actually does to your GPT training loop

```
Single GPU (Module 9):
  batch = data[0:64]
  logits, loss = model(batch)
  loss.backward()
  optimizer.step()

DDP with 8 GPUs:
  rank 0: batch = data[0:64]    → loss_0 → grads_0
  rank 1: batch = data[64:128]  → loss_1 → grads_1
  rank 2: batch = data[128:192] → loss_2 → grads_2
  ...
  rank 7: batch = data[448:512] → loss_7 → grads_7

  AllReduce: avg(grads_0 ... grads_7)
  → every rank now has the same averaged gradient
  → every rank runs identical optimizer.step()
```

Global batch size = `per_gpu_batch × world_size`
= `64 × 8` = `512`

### What this means for your training dynamics

**More important than speed**: DDP changes your optimization because:

| Setting | Single GPU | DDP 8 GPU |
|---------|-----------|-----------|
| per-GPU batch | 64 | 64 |
| global batch | 64 | 512 |
| gradient update frequency | every 64 samples | every 512 samples |
| steps per epoch | N | N/8 |

A global batch of 512 is **not the same** as 64. You must scale the learning rate or the training becomes suboptimal. 

### Learning Rate Scaling Rules

Two common approaches:

**Linear scaling rule** (Goyal et al., used in vision):
```
lr_new = lr_base × world_size
```

If single-GPU lr=6e-4 and world_size=8:
```
lr_new = 6e-4 × 8 = 4.8e-3
```

**Square root scaling rule** (more conservative, common in NLP):
```
lr_new = lr_base × sqrt(world_size)
```

If single-GPU lr=6e-4 and world_size=8:
```
lr_new = 6e-4 × sqrt(8) ≈ 1.7e-3
```

For LLMs, the **square root rule** tends to work better because large batches can destabilize transformer training if LR is too high.

We'll expose `--lr_scale` as a flag so you can experiment.

### Warmup steps should scale too
```
warmup_iters_ddp = warmup_iters_single × world_size
```

More tokens are seen per step → you need fewer total steps → but warmup relative to total training should stay proportional.

### Communication cost for GPT
Parameter count drives AllReduce cost.

Our GPT (d_model=384, n_layers=6):
- ~10M params × 4 bytes = ~40 MB per AllReduce

With DDP bucketing (25MB buckets):
- 2 AllReduce operations per backward
- Each ~20MB
- On NVLink (300 GB/s): ~0.13ms each = negligible

With GPT-2 small (124M params):
- ~496 MB per AllReduce
- On NVLink: ~1.6ms per step (still small compared to compute)

This is why single-node multi-GPU DDP scales very well for transformers.

---

## 10.2 Theory — Gradient Accumulation + DDP

Gradient accumulation lets you simulate a larger global batch without more GPU memory:

```
Without accumulation:
  per_gpu_batch=64, world_size=8
  global_batch = 64 × 8 = 512

With accumulation (accum_steps=4):
  micro_batch=64, accum_steps=4, world_size=8
  global_batch = 64 × 4 × 8 = 2048
```

With DDP, you must use `model.no_sync()` on all but the last accumulation step to avoid doing AllReduce too early:

```python
for micro_step in range(accum_steps):
    is_last = (micro_step == accum_steps - 1)
    with model.no_sync() if not is_last else contextlib.nullcontext():
        logits, loss = model(x, y)
        (loss / accum_steps).backward()
# AllReduce fires only on last micro_step
optimizer.step()
```

---

## 10.3 Theory — Checkpointing in DDP

Two rules:
1. **Save**: only rank 0 saves checkpoints (`if rank == 0: torch.save(...)`)
2. **Load**: all ranks load the same checkpoint, but must do it **after** `DDP(model)` wrap or load into the underlying model first

```python
# Correct save pattern
if rank == 0:
    torch.save(model.module.state_dict(), "checkpoint.pt")

# Correct load pattern
model = GPT(cfg).to(device)
if checkpoint exists:
    state = torch.load("checkpoint.pt", map_location=device)
    model.load_state_dict(state)
model = DDP(model, device_ids=[local_rank])
# DDP broadcasts rank 0 params to all ranks at construction → consistent
```

If all ranks try to save simultaneously → race condition → corrupted checkpoint.

---

## 10.4 Theory — Distributed Evaluation

Two correct patterns:

**Pattern A: Evaluate on rank 0 only** (simpler):
```python
if rank == 0:
    evaluate(model.module, val_loader)
dist.barrier()
```
Problem: rank 0 may have different data than full val set.

**Pattern B: Distributed eval + AllReduce** (correct):
```python
# Every rank evaluates its own val shard
correct = tensor(0.0)
total   = tensor(0.0)
for x, y in val_loader:  # val_loader uses DistributedSampler
    ...
    correct += right_count
    total   += count

dist.all_reduce(correct)
dist.all_reduce(total)
# rank 0 reports
if rank == 0:
    print(f"val_loss = {correct/total}")
```

We'll use Pattern B — it's always correct regardless of dataset size.

---

## 10.5 Setup

```bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course
mkdir -p gpt_ddp
cp gpt_scratch/tokenizer.py         gpt_ddp/
cp gpt_scratch/attention.py         gpt_ddp/
cp gpt_scratch/multihead_attention.py gpt_ddp/
cp gpt_scratch/feedforward.py       gpt_ddp/
cp gpt_scratch/transformer_block.py gpt_ddp/
cp gpt_scratch/gpt.py               gpt_ddp/
cp gpt_scratch/shakespeare.txt      gpt_ddp/
cd gpt_ddp
```

---

## 10.6 From Scratch — GPT DDP Training Script

This is the complete production-grade DDP trainer. Every line is explained.

```bash
cat > train_gpt_ddp.py <<'PY'
"""
GPT training with DistributedDataParallel.

Supports:
  - Single GPU (world_size=1, functionally identical to Module 9)
  - Multi-GPU single node (torchrun --nproc_per_node=N)
  - Multi-node (torchrun --nnodes=2 --nproc_per_node=8)
  - Gradient accumulation
  - Mixed precision (bf16 on A100)
  - Gradient clipping
  - LR warmup + cosine decay with LR scaling
  - Checkpointing (rank 0 only)
  - Distributed evaluation
  - TensorBoard logging (rank 0 only)
  - Throughput + scaling efficiency measurement

Launch:
  Single GPU:   torchrun --nproc_per_node=1 train_gpt_ddp.py
  8 GPU:        torchrun --nproc_per_node=8 train_gpt_ddp.py
  16 GPU:       see launch_multinode.sh
"""

import os
import sys
import time
import math
import argparse
import contextlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tokenizer import CharTokenizer
from gpt import GPT, GPTConfig


# ─── Configs ──────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size:   int   = 65
    max_seq_len:  int   = 256
    d_model:      int   = 384
    n_heads:      int   = 6
    n_layers:     int   = 6
    dropout:      float = 0.1


@dataclass
class TrainConfig:
    # Data
    data_path:      str   = "shakespeare.txt"
    train_split:    float = 0.9

    # Optimization
    batch_size:     int   = 64        # per-GPU micro batch
    accum_steps:    int   = 1         # gradient accumulation steps
    max_iters:      int   = 5000      # total optimizer steps
    lr:             float = 6e-4      # base learning rate (for 1 GPU)
    lr_scale:       str   = "sqrt"    # "linear", "sqrt", or "none"
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0
    warmup_iters:   int   = 200
    lr_decay_iters: int   = 5000
    min_lr:         float = 6e-5

    # System
    dtype:          str   = "bfloat16"
    compile_model:  bool  = False

    # Logging
    log_interval:   int   = 50
    eval_interval:  int   = 500
    eval_iters:     int   = 50
    gen_interval:   int   = 500
    gen_max_tokens: int   = 200

    # Checkpointing
    checkpoint_dir: str   = "checkpoints_ddp"
    resume:         Optional[str] = None

    # Logging dir
    log_dir:        str   = "runs/gpt_ddp"


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx     : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ─── LR Schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig, effective_lr: float) -> float:
    """
    Cosine LR schedule with linear warmup.
    Uses effective_lr (already scaled for world_size).
    """
    if step < cfg.warmup_iters:
        return effective_lr * (step + 1) / cfg.warmup_iters
    if step > cfg.lr_decay_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / \
               max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (effective_lr - cfg.min_lr)


def scale_lr(base_lr: float, world_size: int, rule: str) -> float:
    """Scale learning rate based on world_size."""
    if rule == "linear":
        return base_lr * world_size
    elif rule == "sqrt":
        return base_lr * math.sqrt(world_size)
    elif rule == "none":
        return base_lr
    else:
        raise ValueError(f"Unknown lr_scale rule: {rule}")


# ─── Distributed Utilities ────────────────────────────────────────────────────

def setup_distributed():
    """Initialize distributed process group."""
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_once(msg: str, rank: int):
    """Print only from rank 0."""
    if rank == 0:
        print(msg, flush=True)


def broadcast_object(obj, rank: int, device):
    """
    Broadcast a Python object from rank 0 to all ranks.
    Useful for sharing tokenizer vocab, config, etc.
    """
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_distributed(
        model:       nn.Module,
        val_loader:  DataLoader,
        device:      torch.device,
        eval_iters:  int,
        ctx,
        rank:        int,
        world_size:  int,
) -> float:
    """
    Evaluate on all ranks simultaneously.
    AllReduce loss across ranks → correct global val loss.
    """
    model.eval()

    total_loss  = torch.zeros(1, device=device)
    total_steps = torch.zeros(1, device=device)

    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with ctx:
            _, loss = model(x, y)
        total_loss  += loss.detach()
        total_steps += 1

    # Sum across all ranks
    dist.all_reduce(total_loss,  op=dist.ReduceOp.SUM)
    dist.all_reduce(total_steps, op=dist.ReduceOp.SUM)

    avg_loss = (total_loss / total_steps).item()
    model.train()
    return avg_loss


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(
        model:      nn.Module,
        optimizer:  torch.optim.Optimizer,
        step:       int,
        val_loss:   float,
        cfg:        TrainConfig,
        model_cfg:  ModelConfig,
        rank:       int,
        path:       str,
):
    """Only rank 0 saves. All other ranks skip."""
    if rank != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # model.module is the underlying GPT (unwrapped from DDP)
    torch.save({
        "step":       step,
        "model":      model.module.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
        "train_cfg":  cfg,
        "model_cfg":  model_cfg,
    }, path)
    print(f"[rank 0] saved checkpoint → {path}  (val_loss={val_loss:.4f})",
          flush=True)


def load_checkpoint(
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        path:      str,
        device:    torch.device,
        rank:      int,
) -> int:
    """
    Load checkpoint into model (before DDP wrap).
    Returns the step number to resume from.
    """
    print_once(f"Loading checkpoint: {path}", rank)
    # map_location ensures checkpoint is loaded onto correct device
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    print_once(f"Resumed from step={step} val_loss={ckpt['val_loss']:.4f}",
               rank)
    return step


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Parse args ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--accum_steps",   type=int,   default=1)
    parser.add_argument("--max_iters",     type=int,   default=5000)
    parser.add_argument("--lr",            type=float, default=6e-4)
    parser.add_argument("--lr_scale",      type=str,   default="sqrt",
                        choices=["linear","sqrt","none"])
    parser.add_argument("--d_model",       type=int,   default=384)
    parser.add_argument("--n_heads",       type=int,   default=6)
    parser.add_argument("--n_layers",      type=int,   default=6)
    parser.add_argument("--max_seq_len",   type=int,   default=256)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--resume",        type=str,   default=None)
    parser.add_argument("--compile",       action="store_true")
    parser.add_argument("--log_interval",  type=int,   default=50)
    parser.add_argument("--eval_interval", type=int,   default=500)
    parser.add_argument("--checkpoint_dir",type=str,   default="checkpoints_ddp")
    args = parser.parse_args()

    # ── Init distributed ─────────────────────────────────────────────────
    rank, world_size, local_rank = setup_distributed()
    device   = torch.device("cuda", local_rank)
    is_main  = is_main_process(rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.manual_seed(1337 + rank)   # different seed per rank for data augmentation

    # ── Build configs ────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        batch_size      = args.batch_size,
        accum_steps     = args.accum_steps,
        max_iters       = args.max_iters,
        lr              = args.lr,
        lr_scale        = args.lr_scale,
        compile_model   = args.compile,
        log_interval    = args.log_interval,
        eval_interval   = args.eval_interval,
        checkpoint_dir  = args.checkpoint_dir,
        resume          = args.resume,
    )
    model_cfg = ModelConfig(
        max_seq_len = args.max_seq_len,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        dropout     = args.dropout,
    )

    # ── LR scaling ───────────────────────────────────────────────────────
    effective_lr = scale_lr(train_cfg.lr, world_size, train_cfg.lr_scale)

    # ── Mixed precision ──────────────────────────────────────────────────
    ptdtype = torch.bfloat16   # A100 supports bf16 natively
    ctx     = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    # ── Data ─────────────────────────────────────────────────────────────
    # Only rank 0 reads file; then we distribute token tensor
    if is_main:
        text       = open(train_cfg.data_path, 'r').read()
        tokenizer  = CharTokenizer(text)
        data       = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n_train    = int(len(data) * train_cfg.train_split)
        train_data = data[:n_train]
        val_data   = data[n_train:]
        vocab_size = tokenizer.vocab_size
        print(f"\nworld_size={world_size}  local_rank={local_rank}")
        print(f"effective_lr={effective_lr:.4e}  "
              f"(base={train_cfg.lr:.4e} × {train_cfg.lr_scale}({world_size}))")
        print(f"global_batch = {train_cfg.batch_size} × "
              f"{train_cfg.accum_steps} accum × "
              f"{world_size} GPUs = "
              f"{train_cfg.batch_size * train_cfg.accum_steps * world_size}")
        print(f"train tokens: {len(train_data):,}  "
              f"val tokens: {len(val_data):,}")
    else:
        train_data = val_data = None
        vocab_size = None

    # Broadcast vocabulary size and token data to all ranks
    vocab_size = broadcast_object(vocab_size, rank, device)
    train_data = broadcast_object(train_data, rank, device)
    val_data   = broadcast_object(val_data,   rank, device)

    model_cfg.vocab_size = vocab_size

    # ── DataLoader ────────────────────────────────────────────────────────
    train_ds = TextDataset(train_data.cpu(), model_cfg.max_seq_len)
    val_ds   = TextDataset(val_data.cpu(),   model_cfg.max_seq_len)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas = world_size,
        rank         = rank,
        shuffle      = True,
        drop_last    = True,
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas = world_size,
        rank         = rank,
        shuffle      = False,
        drop_last    = False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size       = train_cfg.batch_size,
        sampler          = train_sampler,
        num_workers      = 4,
        pin_memory       = True,
        persistent_workers = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size       = train_cfg.batch_size,
        sampler          = val_sampler,
        num_workers      = 4,
        pin_memory       = True,
        persistent_workers = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    gpt_cfg = GPTConfig(
        vocab_size  = model_cfg.vocab_size,
        max_seq_len = model_cfg.max_seq_len,
        d_model     = model_cfg.d_model,
        n_heads     = model_cfg.n_heads,
        n_layers    = model_cfg.n_layers,
        dropout     = model_cfg.dropout,
    )

    # Build model on CPU first, then move to device
    torch.manual_seed(1337)   # same seed on ALL ranks → same init
    model = GPT(gpt_cfg).to(device)

    if args.compile:
        print_once("Compiling model...", rank)
        model = torch.compile(model)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Set up param groups with/without weight decay
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr     = effective_lr,
        betas  = (train_cfg.beta1, train_cfg.beta2),
        fused  = True,
    )

    # ── Resume checkpoint ──────────────────────────────────────────────────
    start_step = 0
    if train_cfg.resume and os.path.exists(train_cfg.resume):
        start_step = load_checkpoint(
            model, optimizer, train_cfg.resume, device, rank
        )

    # ── Wrap with DDP ──────────────────────────────────────────────────────
    # IMPORTANT: wrap AFTER loading checkpoint (checkpoint is for raw model)
    model = DDP(
        model,
        device_ids        = [local_rank],
        output_device     = local_rank,
        find_unused_parameters = False,   # False for performance (True for debug)
        bucket_cap_mb     = 25,           # default 25MB buckets
    )

    # ── TensorBoard (rank 0 only) ──────────────────────────────────────────
    writer = None
    if is_main:
        os.makedirs(train_cfg.log_dir, exist_ok=True)
        writer = SummaryWriter(train_cfg.log_dir)

    # ── Training state ─────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    best_val_loss = float('inf')
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # ── Per-rank throughput tracking ───────────────────────────────────────
    tokens_processed = 0
    t_total_start    = time.perf_counter()

    print_once(f"\n── Training starts ─────────────────────────────────────", rank)
    print_once(f"  model params: {model.module.count_params()/1e6:.2f}M", rank)
    print_once(f"  DDP world_size: {world_size}", rank)
    print_once(f"  effective_lr: {effective_lr:.4e}", rank)
    print_once(f"────────────────────────────────────────────────────────\n", rank)

    # ── Training loop ──────────────────────────────────────────────────────
    train_iter = iter(train_loader)
    t0         = time.perf_counter()

    for step in range(start_step, train_cfg.max_iters):

        # ── LR update ────────────────────────────────────────────────────
        lr = get_lr(step, train_cfg, effective_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Set epoch in sampler (new epoch = new shuffle) ───────────────
        epoch = step * train_cfg.batch_size // max(1, len(train_ds))
        train_sampler.set_epoch(epoch)

        # ── Gradient accumulation loop ────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(train_cfg.accum_steps):
            # Get next batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # no_sync on all but last micro_step
            # This suppresses AllReduce until the last accumulation step
            # saving communication overhead
            is_last_micro = (micro_step == train_cfg.accum_steps - 1)
            sync_ctx = contextlib.nullcontext() if is_last_micro \
                       else model.no_sync()

            with sync_ctx:
                with ctx:
                    _, loss = model(x, y)
                # normalize loss by accum_steps
                (loss / train_cfg.accum_steps).backward()

            accum_loss += loss.item() / train_cfg.accum_steps

        # AllReduce of gradients fires here (at the end of last micro_step backward)

        # ── Gradient clipping ─────────────────────────────────────────────
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg.grad_clip
        )

        # ── Optimizer step ────────────────────────────────────────────────
        optimizer.step()

        # ── Throughput tracking ───────────────────────────────────────────
        tokens_this_step = (train_cfg.batch_size *
                            train_cfg.accum_steps *
                            model_cfg.max_seq_len)
        tokens_processed += tokens_this_step

        # ── Logging ───────────────────────────────────────────────────────
        if step % train_cfg.log_interval == 0:
            torch.cuda.synchronize()
            t1      = time.perf_counter()
            dt_ms   = (t1 - t0) * 1000 / train_cfg.log_interval
            # Global tokens per second across all GPUs
            tok_s   = (tokens_this_step * world_size * train_cfg.log_interval) \
                      / (t1 - t0)
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            alloc_gb= torch.cuda.memory_allocated(device) / 1024**3
            t0      = t1

            if is_main:
                writer.add_scalar("train/loss",       accum_loss, step)
                writer.add_scalar("train/lr",          lr,         step)
                writer.add_scalar("train/grad_norm",   grad_norm,  step)
                writer.add_scalar("perf/tokens_per_s", tok_s,      step)
                writer.add_scalar("perf/step_ms",      dt_ms,      step)
                writer.add_scalar("perf/peak_gb",      peak_gb,    step)

                print(
                    f"step={step:5d}  "
                    f"loss={accum_loss:.4f}  "
                    f"lr={lr:.2e}  "
                    f"grad={grad_norm:.3f}  "
                    f"step_ms={dt_ms:.1f}  "
                    f"tok/s={tok_s:.0f}  "
                    f"alloc={alloc_gb:.2f}GB  "
                    f"peak={peak_gb:.2f}GB",
                    flush=True
                )

        # ── Evaluation ────────────────────────────────────────────────────
        if step % train_cfg.eval_interval == 0 or \
           step == train_cfg.max_iters - 1:

            val_loss = evaluate_distributed(
                model, val_loader, device,
                train_cfg.eval_iters, ctx, rank, world_size
            )

            if is_main:
                writer.add_scalar("eval/val_loss", val_loss, step)
                print(f"\n── eval step={step} "
                      f"val_loss={val_loss:.4f} ──────────────────────────")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, step, val_loss,
                        train_cfg, model_cfg, rank,
                        path=os.path.join(train_cfg.checkpoint_dir, "best.pt")
                    )
                print("──────────────────────────────────────────────────────\n",
                      flush=True)

            # All ranks sync after eval
            dist.barrier()

        # ── Text generation sample (rank 0 only) ──────────────────────────
        if step % train_cfg.gen_interval == 0 and step > 0 and is_main:
            model.eval()
            prompt = torch.zeros(1, 1, dtype=torch.long, device=device)
            with torch.no_grad():
                gen = model.module.generate(
                    prompt,
                    max_new_tokens = train_cfg.gen_max_tokens,
                    temperature    = 0.8,
                    top_k          = 40,
                )
            text       = open(train_cfg.data_path).read()
            tokenizer  = CharTokenizer(text)
            gen_text   = tokenizer.decode(gen[0].tolist())
            writer.add_text("generation", gen_text, step)
            print(f"── generation step={step} ──────────────────────────────")
            print(gen_text)
            print("─────────────────────────────────────────────────────────\n",
                  flush=True)
            model.train()

    # ── Final summary ──────────────────────────────────────────────────────
    if is_main:
        total_time  = time.perf_counter() - t_total_start
        total_toks  = tokens_processed * world_size
        avg_tok_s   = total_toks / total_time
        print(f"\n── Training Complete ────────────────────────────────────")
        print(f"  world_size:      {world_size}")
        print(f"  total_steps:     {train_cfg.max_iters}")
        print(f"  total_time:      {total_time:.1f}s")
        print(f"  total_tokens:    {total_toks:,}")
        print(f"  avg_tok/s:       {avg_tok_s:.0f}")
        print(f"  best_val_loss:   {best_val_loss:.4f}")
        print(f"  peak_memory:     {torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB")
        print("────────────────────────────────────────────────────────────\n",
              flush=True)
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
PY
```

---

## 10.7 Run — 1 → 2 → 4 → 8 GPUs

### 1 GPU (baseline)
```bash
torchrun --nproc_per_node=1 \
  train_gpt_ddp.py \
  --batch_size=64 \
  --max_iters=2000 \
  --lr_scale=none \
  --checkpoint_dir=checkpoints_ddp/run_1gpu
```

### 2 GPUs
```bash
torchrun --nproc_per_node=2 \
  train_gpt_ddp.py \
  --batch_size=64 \
  --max_iters=2000 \
  --lr_scale=sqrt \
  --checkpoint_dir=checkpoints_ddp/run_2gpu
```

### 4 GPUs
```bash
torchrun --nproc_per_node=4 \
  train_gpt_ddp.py \
  --batch_size=64 \
  --max_iters=2000 \
  --lr_scale=sqrt \
  --checkpoint_dir=checkpoints_ddp/run_4gpu
```

### 8 GPUs
```bash
torchrun --nproc_per_node=8 \
  train_gpt_ddp.py \
  --batch_size=64 \
  --max_iters=2000 \
  --lr_scale=sqrt \
  --checkpoint_dir=checkpoints_ddp/run_8gpu
```

Watch all 8 GPUs during training:
```bash
watch -n 1 gpustat
```

---

## 10.8 Run — 16 GPUs (2 Nodes)

### Create SLURM job script
```bash
cat > jobs/job_gpt_ddp_16gpu.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=gpt_ddp_16gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/gpt_ddp_16gpu_%j.out
#SBATCH --error=logs/gpt_ddp_16gpu_%j.err

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/gpt_ddp

mkdir -p logs checkpoints_ddp

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_ADDR MASTER_PORT
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

echo "MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "NODELIST=$SLURM_JOB_NODELIST"

srun bash -c "
  torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_gpt_ddp.py \
    --batch_size=64 \
    --max_iters=2000 \
    --lr_scale=sqrt \
    --checkpoint_dir=checkpoints_ddp/run_16gpu
"
SLURM

mkdir -p jobs logs
sbatch jobs/job_gpt_ddp_16gpu.sh
```

### Or manually (no SLURM)
```bash
# Node 0:
MASTER_ADDR=$(hostname -I | awk '{print $1}')
torchrun \
  --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  train_gpt_ddp.py --batch_size=64 --max_iters=2000 --lr_scale=sqrt

# Node 1 (same time):
torchrun \
  --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=<node0_ip> --master_port=29500 \
  train_gpt_ddp.py --batch_size=64 --max_iters=2000 --lr_scale=sqrt
```

---

## 10.9 Scaling Benchmark Script

Run this to precisely measure comm vs compute breakdown at each GPU count:

```bash
cat > bench_gpt_scaling.py <<'PY'
"""
Precise GPT DDP scaling benchmark.
Measures:
  - forward time
  - forward + backward (no sync)
  - full step (with AllReduce)
  - communication overhead = full - no_sync
  - tokens/s
  - scaling efficiency
"""
import os, time, contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt import GPT, GPTConfig


def bench(fn, device, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    return (time.perf_counter() - t0) * 1000 / iters  # ms


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    # Model config matching training
    cfg = GPTConfig(
        vocab_size  = 65,
        max_seq_len = 256,
        d_model     = 384,
        n_heads     = 6,
        n_layers    = 6,
        dropout     = 0.0,  # no dropout for benchmark
    )

    torch.manual_seed(1337)
    model = GPT(cfg).to(device)
    ddp   = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)
    opt   = torch.optim.AdamW(ddp.parameters(), lr=1e-4, fused=True)

    B  = 64
    T  = cfg.max_seq_len
    x  = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y  = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    # ── 1. Forward only ──────────────────────────────────────────────────
    def fwd():
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(x)

    fwd_ms = bench(fwd, device)

    # ── 2. Forward + Backward (NO AllReduce) ─────────────────────────────
    def fwd_bwd_no_sync():
        with ddp.no_sync():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = ddp(x, y)
            loss.backward()

    fwd_bwd_ms = bench(fwd_bwd_no_sync, device)

    # ── 3. Full step (with AllReduce) ─────────────────────────────────────
    def full_step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = ddp(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        opt.step()

    full_ms = bench(full_step, device)

    # ── 4. Compute metrics ────────────────────────────────────────────────
    comm_ms   = max(0.0, full_ms - fwd_bwd_ms)
    comm_pct  = 100.0 * comm_ms / full_ms
    toks_s    = B * T * world_size / (full_ms / 1000)

    # Param count for AllReduce size estimate
    n_params  = model.count_params()
    ar_mb     = n_params * 4 / 1024**2   # float32 gradients

    if is_main:
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        alloc_gb= torch.cuda.memory_allocated(device) / 1024**3

        print(f"\n{'='*60}")
        print(f"GPT DDP Scaling Benchmark")
        print(f"{'='*60}")
        print(f"world_size:         {world_size}")
        print(f"model params:       {n_params/1e6:.2f}M")
        print(f"allreduce size:     {ar_mb:.1f} MB")
        print(f"batch per GPU:      {B}")
        print(f"seq_len:            {T}")
        print(f"global batch:       {B * world_size}")
        print(f"{'-'*60}")
        print(f"forward only:       {fwd_ms:8.2f} ms")
        print(f"fwd+bwd (no sync):  {fwd_bwd_ms:8.2f} ms")
        print(f"full step:          {full_ms:8.2f} ms")
        print(f"comm overhead:      {comm_ms:8.2f} ms  ({comm_pct:.1f}%)")
        print(f"{'-'*60}")
        print(f"throughput:         {toks_s:,.0f} tokens/s")
        print(f"alloc memory:       {alloc_gb:.2f} GB")
        print(f"peak memory:        {peak_gb:.2f} GB")
        print(f"{'='*60}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run across GPU counts:
```bash
for N in 1 2 4 8; do
    echo "=== $N GPUs ==="
    torchrun --nproc_per_node=$N bench_gpt_scaling.py
done
```

---

## 10.10 Compute Your Scaling Efficiency Table

After running the benchmark, fill this in:

| GPUs | step_ms | tok/s | comm_ms | comm% | peak_GB | efficiency |
|------|---------|-------|---------|-------|---------|------------|
| 1 | | | | | | 100% |
| 2 | | | | | | ? |
| 4 | | | | | | ? |
| 8 | | | | | | ? |
| 16 | | | | | | ? |

Calculate efficiency:
```
efficiency(N) = tok/s(N) / (N × tok/s(1)) × 100%
```

On A100 SXM4 with NVSwitch you should see:
- 8 GPU efficiency: ~85–95%
- 16 GPU efficiency: ~75–85% (cross-node InfiniBand is the bottleneck)

---

## 10.11 Gradient Accumulation Experiment

Test different accumulation configurations that give the same global batch:

```bash
# Global batch = 512

# Option A: 8 GPUs, no accumulation
torchrun --nproc_per_node=8 train_gpt_ddp.py \
  --batch_size=64 --accum_steps=1 --max_iters=1000

# Option B: 4 GPUs, accum=2
torchrun --nproc_per_node=4 train_gpt_ddp.py \
  --batch_size=64 --accum_steps=2 --max_iters=1000

# Option C: 2 GPUs, accum=4
torchrun --nproc_per_node=2 train_gpt_ddp.py \
  --batch_size=64 --accum_steps=4 --max_iters=1000

# Option D: 1 GPU, accum=8
torchrun --nproc_per_node=1 train_gpt_ddp.py \
  --batch_size=64 --accum_steps=8 --max_iters=1000
```

All 4 options use the same global batch size = 512 tokens × seq_len.
- Option A is fastest (parallel compute + AllReduce overlap)
- Option D is slowest (serial, no parallelism)
- Options B/C are intermediate

---

## 10.12 Debug Drills

### Drill 1 — `model.generate()` must use `model.module.generate()`

```bash
python - <<'PY'
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, GPTConfig

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29510")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

dist.init_process_group("nccl")
torch.cuda.set_device(0)

cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                  d_model=128, n_heads=4, n_layers=2)
model = GPT(cfg).cuda()
ddp   = DDP(model, device_ids=[0])

prompt = torch.zeros(1, 1, dtype=torch.long, device="cuda")

# WRONG: DDP wrapper doesn't have .generate()
try:
    out = ddp.generate(prompt, max_new_tokens=5)
except AttributeError as e:
    print(f"WRONG: {e}")

# CORRECT: access underlying model via .module
out = ddp.module.generate(prompt, max_new_tokens=5)
print(f"CORRECT: generated shape={out.shape}")

dist.destroy_process_group()
PY
```

---

### Drill 2 — All ranks must have the same model init

```bash
cat > check_model_sync.py <<'PY'
"""
Verify all ranks start with identical model weights.
DDP constructor broadcasts rank 0 params → all ranks should agree.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, GPTConfig


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Use same seed on all ranks → same init
    torch.manual_seed(1337)

    cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                      d_model=128, n_heads=4, n_layers=2)
    model = GPT(cfg).to(device)
    ddp   = DDP(model, device_ids=[local_rank])

    # Check: sum of first layer weights should be identical across ranks
    param = list(ddp.parameters())[0]
    param_sum = param.sum().item()

    # Gather all sums to rank 0
    sums = [torch.zeros(1, device=device) for _ in range(world_size)]
    my_sum = torch.tensor([param_sum], device=device)
    dist.all_gather(sums, my_sum)

    if rank == 0:
        vals = [s.item() for s in sums]
        print(f"Parameter sum per rank: {vals}")
        all_equal = all(abs(v - vals[0]) < 1e-4 for v in vals)
        print(f"All ranks have identical init: {all_equal}")
        assert all_equal, "FAIL: ranks have different model init!"
        print("Model sync: PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 check_model_sync.py
```

---

### Drill 3 — `find_unused_parameters` bug

```bash
cat > unused_param_demo.py <<'PY'
"""
Demonstrates find_unused_parameters behavior.
If a model has parameters that don't get gradients,
DDP with find_unused_parameters=False will hang.
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class ModelWithUnusedParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.used   = nn.Linear(4, 4)
        self.unused = nn.Linear(4, 4)   # never called in forward!

    def forward(self, x):
        return self.used(x)   # unused is never called


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = ModelWithUnusedParam().to(device)

    # This will HANG because unused param never gets gradient
    # ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # This works:
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    x    = torch.randn(2, 4, device=device)
    loss = ddp(x).mean()
    loss.backward()   # AllReduce fires, unused param handled

    if rank == 0:
        print("find_unused_parameters=True: PASSED")
        print(f"  used.weight grad norm:   {model.used.weight.grad.norm():.4f}")
        print(f"  unused.weight grad:      {model.unused.weight.grad}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=2 unused_param_demo.py
```

---

### Drill 4 — Checkpoint save/load correctness across ranks

```bash
cat > checkpoint_test.py <<'PY'
"""
Verify checkpoint save (rank 0 only) and load (all ranks) works correctly.
After load, all ranks should have identical weights.
"""
import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, GPTConfig


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                      d_model=128, n_heads=4, n_layers=2)

    # ── Phase 1: train 1 step + save ─────────────────────────────────────
    torch.manual_seed(1337)
    model = GPT(cfg).to(device)
    ddp   = DDP(model, device_ids=[local_rank])
    opt   = torch.optim.AdamW(ddp.parameters(), lr=1e-4)

    x = torch.randint(0, 65, (4, 64), device=device)
    y = torch.randint(0, 65, (4, 64), device=device)
    _, loss = ddp(x, y)
    loss.backward()
    opt.step()

    # Save rank 0 checkpoint
    if rank == 0:
        os.makedirs("checkpoints_test", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints_test/test.pt")
        print(f"rank 0 saved checkpoint")

    dist.barrier()   # wait for save to complete

    # Record weight sum for verification
    w0_before = list(model.parameters())[0].sum().item()

    # ── Phase 2: corrupt weights, then reload ────────────────────────────
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(99.0)   # corrupt

    w0_corrupt = list(model.parameters())[0].sum().item()
    assert abs(w0_corrupt - 99.0) > 1e-3   # confirm corruption

    # Reload checkpoint (all ranks load same file)
    state = torch.load("checkpoints_test/test.pt", map_location=device)
    model.load_state_dict(state)

    w0_after = list(model.parameters())[0].sum().item()

    # Verify restoration
    dist.barrier()
    for r in range(dist.get_world_size()):
        dist.barrier()
        if rank == r:
            restored = abs(w0_after - w0_before) < 1e-4
            print(f"rank={rank}  before={w0_before:.6f}  "
                  f"corrupted={w0_corrupt:.2f}  "
                  f"after={w0_after:.6f}  "
                  f"restored={restored}", flush=True)
        dist.barrier()

    if rank == 0:
        print("Checkpoint test: PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 checkpoint_test.py
```

---

### Drill 5 — Loss should be same with/without DDP (correctness check)

```bash
cat > ddp_correctness.py <<'PY'
"""
Verify DDP gives same loss as single-GPU when global batch is the same.

With world_size=1, DDP should give identical result to no DDP.
"""
import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, GPTConfig


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = GPTConfig(vocab_size=65, max_seq_len=32,
                    d_model=64, n_heads=4, n_layers=2, dropout=0.0)

    # Same seed → same init
    torch.manual_seed(42)
    model_ddp = GPT(cfg).to(device)
    torch.manual_seed(42)
    model_ref = GPT(cfg).to(device)   # reference: no DDP

    ddp = DDP(model_ddp, device_ids=[local_rank])

    # Create a fixed batch
    torch.manual_seed(0)
    x = torch.randint(0, 65, (4, 32), device=device)
    y = torch.randint(0, 65, (4, 32), device=device)

    # Forward through DDP model
    _, loss_ddp = ddp(x, y)

    # Forward through reference model (no DDP)
    _, loss_ref = model_ref(x, y)

    if rank == 0:
        print(f"loss_ddp = {loss_ddp.item():.6f}")
        print(f"loss_ref = {loss_ref.item():.6f}")
        print(f"diff     = {abs(loss_ddp.item() - loss_ref.item()):.8f}")
        assert abs(loss_ddp.item() - loss_ref.item()) < 1e-4, \
            "DDP and reference model give different loss!"
        print("DDP correctness: PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

# Run with 1 GPU (DDP should match non-DDP)
torchrun --nproc_per_node=1 ddp_correctness.py
```

---

## 10.13 LR Scaling Experiment

Compare the three LR scaling rules on the same number of steps:

```bash
cat > compare_lr_scaling.sh <<'BASH'
#!/bin/bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/gpt_ddp

for RULE in none sqrt linear; do
    echo "=== lr_scale=$RULE on 4 GPUs ==="
    torchrun --nproc_per_node=4 \
      train_gpt_ddp.py \
      --batch_size=64 \
      --max_iters=1000 \
      --lr_scale=$RULE \
      --eval_interval=200 \
      --checkpoint_dir=checkpoints_ddp/lr_${RULE} \
      --log_dir=runs/lr_${RULE} \
      2>&1 | grep -E "step=|eval"
    echo ""
done
BASH

bash compare_lr_scaling.sh
```

You'll see:
- `none`: same LR as 1 GPU → may converge slowly at large batch
- `sqrt`: moderate scaling → usually best for NLP
- `linear`: aggressive scaling → fast early but can diverge

---

## 10.14 Full Scaling Report Script

```bash
cat > scaling_report.py <<'PY'
"""
Parse training logs and compute scaling report.
Run after all 4 experiments (1/2/4/8 GPU).
"""
import glob, re, os


def parse_log(path: str) -> dict:
    """Parse training log for throughput and loss."""
    tok_rates = []
    losses    = []
    val_losses= []

    with open(path) as f:
        for line in f:
            # parse tok/s from training lines
            m = re.search(r'tok/s=(\d+)', line)
            if m:
                tok_rates.append(int(m.group(1)))

            m = re.search(r'loss=(\d+\.\d+)', line)
            if m and 'eval' not in line:
                losses.append(float(m.group(1)))

            m = re.search(r'val_loss=(\d+\.\d+)', line)
            if m:
                val_losses.append(float(m.group(1)))

    return {
        "avg_tok_s": sum(tok_rates[-20:]) / len(tok_rates[-20:]) if tok_rates else 0,
        "final_loss": losses[-1] if losses else 0,
        "best_val": min(val_losses) if val_losses else 0,
    }


def main():
    log_pattern = "logs/gpt_ddp_*gpu_*.out"
    logs        = sorted(glob.glob(log_pattern))

    if not logs:
        print("No logs found. Run training first.")
        print(f"Expected pattern: {log_pattern}")
        return

    print(f"\n{'GPUs':>6}  {'tok/s':>10}  "
          f"{'efficiency':>12}  {'best_val':>10}")
    print("-" * 50)

    baseline_tok_s = None
    for log in logs:
        n_gpu = int(re.search(r'(\d+)gpu', log).group(1))
        info  = parse_log(log)
        tok_s = info["avg_tok_s"]

        if n_gpu == 1:
            baseline_tok_s = tok_s

        if baseline_tok_s:
            eff = tok_s / (n_gpu * baseline_tok_s) * 100
        else:
            eff = 0

        print(f"{n_gpu:>6}  {tok_s:>10,.0f}  "
              f"{eff:>11.1f}%  {info['best_val']:>10.4f}")


if __name__ == "__main__":
    main()
PY

python scaling_report.py
```

---

## 10.15 Module 10 Pass Criteria

You're done when you can:

1) **Explain clearly**:
   - why the GPT model code is **unchanged** between single-GPU and DDP
   - what `model.module` is and why it's needed for save/generate
   - why `no_sync()` is needed for gradient accumulation
   - how the three LR scaling rules differ and when to use each
   - why `set_epoch()` is critical for proper data shuffling

2) **Run successfully**:
   - `bench_gpt_scaling.py` at 1/2/4/8 GPUs → fill scaling table
   - `train_gpt_ddp.py` at 8 GPUs → training converges
   - `check_model_sync.py` → PASSED
   - `checkpoint_test.py` → PASSED
   - `ddp_correctness.py` → PASSED

3) **Fill in your scaling table** with:
   - tok/s at each GPU count
   - scaling efficiency (%)
   - comm overhead (ms and %)
   - peak memory per GPU

4) **Know the key numbers**:
   - your GPT model parameter count
   - AllReduce tensor size (MB)
   - single GPU tok/s baseline
   - 8 GPU tok/s and efficiency

---

