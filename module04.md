# Module 4 — PyTorch Distributed Data Parallel (DDP)

## Summary
You'll learn exactly why DDP exists, what it does internally, and how to convert your single-GPU CNN from Module 1 into a proper multi-GPU DDP training job.

- **Theory**: why `DataParallel` is slow, how DDP works internally (gradient bucketing, autograd hooks, overlapping comm with backward)
- **Hands-on**: convert CNN → DDP, run on 1→2→4→8 GPUs
- **Measure**: throughput, memory per GPU, scaling efficiency
- **Debug drills**: rank mismatch, port conflict, NCCL timeout, seed mismatch

---

## 4.1 Theory — The story of why DDP exists

### The old way: `DataParallel` (why it's slow)
`nn.DataParallel` works on a single process, multiple GPUs:
```
One process
  → scatter batch to all GPUs
  → forward on each GPU
  → gather outputs to GPU0
  → compute loss on GPU0
  → backward on GPU0
  → scatter gradients back
  → update on GPU0
```

Problems:
- **Single process**: Python GIL limits CPU side.
- **Gather to GPU0**: GPU0 is always bottlenecked and uses more memory.
- **Serial gradient scatter**: no overlap between compute and communication.
- NCCL is designed for multi-process, not single-process multi-GPU in this style.

### The new way: `DistributedDataParallel` (why it's fast)
DDP uses **one process per GPU**:
```
Process 0 (GPU 0)    Process 1 (GPU 1)  ...  Process 7 (GPU 7)
  full model copy       full model copy         full model copy
  own data shard        own data shard          own data shard
  forward               forward                 forward
  backward              backward                backward
  allreduce grads ←――――――――――――――――――――――――――→ allreduce grads
  optimizer step        optimizer step          optimizer step
```

Each GPU:
- Has a **full copy** of the model (parameters + gradients).
- Processes its **own shard** of the batch.
- Computes its **own gradients**.
- Runs **AllReduce** to average gradients across all GPUs.
- Runs the **same optimizer step** (because gradients are now identical across all ranks).

After AllReduce, all GPUs have the same averaged gradients → same model update → models stay in sync.

### Gradient Bucketing (how DDP overlaps comm with backward)
DDP doesn't AllReduce each gradient tensor individually. Instead:
- Parameters are grouped into **buckets** (default 25MB in PyTorch).
- As gradients are computed during backward, DDP checks if a bucket is full.
- When a bucket is full, it fires AllReduce for that bucket **while backward continues** on other layers.

This **overlaps** communication with backward computation.

```
backward:  [layer N grad] [layer N-1 grad] [layer N-2 grad] ...
allreduce:               [bucket1 AR]      [bucket2 AR]      ...
```

This is a major performance win: comm and compute run at the same time.

### Autograd hooks (how DDP triggers bucketing)
DDP registers **autograd hooks** on each parameter. When a gradient is computed, the hook fires, and DDP's internal bookkeeping decides whether to launch an AllReduce for the ready bucket.

You never call AllReduce manually in DDP — it's all automatic.

### What happens at the optimizer step
After backward + AllReduce, all ranks have **identical averaged gradients**. Each rank runs `optimizer.step()` independently but identically → models stay in sync without any extra communication.

### Model initialization sync
When you wrap your model with DDP:
```python
model = DDP(model)
```
DDP **broadcasts** parameters from rank 0 to all other ranks at construction time. This ensures everyone starts with the same weights.

---

## 4.2 Key DDP rules (internals you must know)

| Rule | Why |
|---|---|
| 1 process per GPU | NCCL designed for multi-process; avoids GIL |
| All ranks call same ops | Collective symmetry (or you deadlock) |
| DistributedSampler needed | Without it, all GPUs see same data (covered deeply in Module 5) |
| `set_to_none=True` in zero_grad | Saves memory; sets grad to None instead of zero tensor |
| `torch.cuda.synchronize()` before timing | GPU ops are async; sync gives real timing |
| Seed each rank carefully | Different data augmentation per rank = good; different model init = bad |
| `model.module` to access underlying model | DDP wraps model; `.module` gives you back original |
| `find_unused_parameters=False` default | If some params don't get gradients, set True (adds overhead) |

---

## 4.3 DDP environment variables set by torchrun

When you run `torchrun --nproc_per_node=8 script.py`, it sets:

```
MASTER_ADDR   = "localhost"  (or user-specified)
MASTER_PORT   = auto-chosen  (or user-specified)
RANK          = global rank of this process
LOCAL_RANK    = GPU index on this node
WORLD_SIZE    = total processes
```

Your code reads these via `os.environ` or `dist.get_rank()` etc.

---

# 4.4 Hands-on Lab — Convert CNN to DDP

## Step A — Create the DDP training script
This is the Module 1 CNN + CIFAR10 converted to proper DDP.

```bash
cat > train_cifar10_ddp.py <<'PY'
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from dataclasses import dataclass


# ─── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    data_dir: str   = "./data"
    batch_size: int = 256      # per GPU batch size
    num_workers: int = 4
    epochs: int     = 5
    lr: float       = 0.1
    weight_decay: float = 5e-4
    use_amp: bool   = True
    print_every: int = 50


# ─── Model ─────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, rank, world_size):
    """
    Evaluate across all ranks.
    Each rank evaluates its own subset,
    then we AllReduce correct/total counts.
    """
    model.eval()
    correct_t = torch.zeros(1, device=device)
    total_t   = torch.zeros(1, device=device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct_t += (pred == y).sum()
        total_t   += y.numel()

    # Sum across all ranks
    dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_t,   op=dist.ReduceOp.SUM)

    return (correct_t / total_t).item()


# ─── Utility ───────────────────────────────────────────────────────────────────
def mem_stats(device):
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    resv  = torch.cuda.memory_reserved(device)  / 1024**3
    peak  = torch.cuda.max_memory_allocated(device) / 1024**3
    return alloc, resv, peak


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = Config()

    # ── 1. Init distributed ──────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")

    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Only rank 0 prints status lines
    is_main = (rank == 0)

    if is_main:
        print(f"world_size={world_size}")
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")

    # Reproducibility: same model init across ranks, different augmentation ok
    torch.manual_seed(42)

    # ── 2. Data ──────────────────────────────────────────────────────────────
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010)),
    ])

    # Download only on rank 0 to avoid race condition
    if is_main:
        datasets.CIFAR10(cfg.data_dir, train=True,  download=True)
        datasets.CIFAR10(cfg.data_dir, train=False, download=True)
    dist.barrier()  # wait for rank 0 to finish download

    train_ds = datasets.CIFAR10(cfg.data_dir, train=True,  transform=transform_train)
    test_ds  = datasets.CIFAR10(cfg.data_dir, train=False, transform=transform_test)

    # DistributedSampler: ensures each rank sees different non-overlapping data
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    test_sampler = DistributedSampler(
        test_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,      # no shuffle=True here (sampler handles it)
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── 3. Model ─────────────────────────────────────────────────────────────
    model = SimpleCNN().to(device)

    # DDP wraps model; broadcasts params from rank 0 to all ranks automatically
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # ── 4. Training loop ─────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(cfg.epochs):
        model.train()

        # IMPORTANT: set epoch in sampler so shuffle is different each epoch
        train_sampler.set_epoch(epoch)

        running_loss = 0.0
        step_times   = []
        t_epoch = time.time()

        for step, (x, y) in enumerate(train_loader):
            t0 = time.time()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                logits = model(x)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize()
            step_times.append(time.time() - t0)
            running_loss += loss.item()

            if is_main and (step % cfg.print_every == 0) and step > 0:
                avg_t = sum(step_times[-cfg.print_every:]) / cfg.print_every
                alloc, resv, peak = mem_stats(device)
                imgs_per_sec = (cfg.batch_size * world_size) / avg_t
                print(
                    f"ep={epoch+1} step={step} "
                    f"loss={running_loss/cfg.print_every:.4f} "
                    f"step_ms={avg_t*1000:.1f} "
                    f"imgs/s={imgs_per_sec:.0f} "
                    f"alloc={alloc:.2f}GB "
                    f"peak={peak:.2f}GB "
                    f"lr={optimizer.param_groups[0]['lr']:.4g}",
                    flush=True
                )
                running_loss = 0.0

        scheduler.step()

        acc = evaluate(model, test_loader, device, rank, world_size)
        epoch_time = time.time() - t_epoch

        if is_main:
            print(f"── epoch {epoch+1} acc={acc*100:.2f}% "
                  f"epoch_time={epoch_time:.1f}s ──", flush=True)

    if is_main:
        print("Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

---

## Step B — Run experiments (1 → 2 → 4 → 8 GPUs)

### 1 GPU
```bash
torchrun --nproc_per_node=1 train_cifar10_ddp.py
```

### 2 GPUs
```bash
torchrun --nproc_per_node=2 train_cifar10_ddp.py
```

### 4 GPUs
```bash
torchrun --nproc_per_node=4 train_cifar10_ddp.py
```

### 8 GPUs
```bash
torchrun --nproc_per_node=8 train_cifar10_ddp.py
```

Watch GPU utilization live in another terminal:
```bash
watch -n 0.5 gpustat
```

---

## Step C — Record metrics (build your own scaling table)
After all 4 runs, fill in:

| GPUs | step_ms | imgs/s | peak_GB/GPU | epoch_time (s) | acc (5 ep) |
|------|---------|--------|-------------|----------------|------------|
| 1    |         |        |             |                |            |
| 2    |         |        |             |                |            |
| 4    |         |        |             |                |            |
| 8    |         |        |             |                |            |

**Ideal throughput scaling**: `imgs/s` should be proportional to GPU count (linear).
If imgs/s at 8 GPUs < 8× imgs/s at 1 GPU, the gap is:
- communication overhead
- DataLoader stall
- sync barrier cost

---

# 4.5 Inspecting DDP internals (see gradient bucketing)

Create `inspect_ddp_buckets.py`:

```bash
cat > inspect_ddp_buckets.py <<'PY'
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = SimpleCNN().to(device)

    # Inspect parameter sizes before wrapping
    if rank == 0:
        total = 0
        print("\n── Parameters ──")
        for name, p in model.named_parameters():
            mb = p.numel() * p.element_size() / 1024**2
            print(f"  {name:40s} shape={str(p.shape):25s} {mb:.4f} MB")
            total += p.numel() * p.element_size()
        print(f"  Total param memory: {total/1024**2:.2f} MB\n")

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)

    # Inspect DDP bucket layout
    if rank == 0:
        print("── DDP Buckets ──")
        for i, bucket in enumerate(model.reducer.buckets):
            bucket_size = sum(
                p.numel() * p.element_size()
                for p in bucket.parameters()
            ) / 1024**2
            print(f"  bucket[{i}]: {len(bucket.parameters())} params, "
                  f"{bucket_size:.2f} MB")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=2 inspect_ddp_buckets.py
```

You'll see how DDP groups parameters into buckets and the size of each bucket.

---

# 4.6 Debug Drills

## Drill 1 — Port conflict
If you run two jobs simultaneously you'll see:
```
RuntimeError: Address already in use
```

Fix: set a different port:
```bash
MASTER_PORT=29502 torchrun --nproc_per_node=8 train_cifar10_ddp.py
```

Or kill old processes:
```bash
pkill -f train_cifar10_ddp.py
pkill -f torchrun
```

---

## Drill 2 — Rank mismatch (wrong model output)
Simulate by not calling `dist.barrier()` before dataset download.
If rank 0 is downloading and rank 1 tries to load before it's done → crash/corrupted file.

The script already handles this:
```python
if is_main:
    datasets.CIFAR10(..., download=True)  # only rank 0 downloads
dist.barrier()                            # all wait here
```

Remove the `dist.barrier()` and run to see the race condition failure.

---

## Drill 3 — NCCL timeout (simulate slow rank)
Add an artificial sleep on one rank:

Create `dist_timeout_demo.py`:

```bash
cat > dist_timeout_demo.py <<'PY'
import os, time
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(
        backend="nccl",
        timeout=__import__("datetime").timedelta(seconds=10)
    )
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    x = torch.ones(4, device=device)

    if rank == 0:
        print("rank 0 sleeping 15s (longer than timeout)...", flush=True)
        time.sleep(15)

    # This will timeout because rank 0 is still sleeping
    dist.all_reduce(x)
    print(f"rank {rank} done", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=2 dist_timeout_demo.py
```

You'll see NCCL timeout after ~10 seconds. That's the same error you see in large cluster runs when a GPU stalls.

Cleanup:
```bash
pkill -f dist_timeout_demo.py
pkill -f torchrun
```

---

## Drill 4 — forget `train_sampler.set_epoch(epoch)`
If you remove `train_sampler.set_epoch(epoch)`:
- same shuffle order every epoch
- every GPU sees the same data shards every epoch
- model trains but validation acc suffers

(We'll cover DistributedSampler deeply in Module 5.)

---

## Drill 5 — access underlying model correctly
```bash
python - <<'PY'
import torch, os
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

dist.init_process_group("nccl")
torch.cuda.set_device(0)

model = nn.Linear(4, 2).cuda()
ddp_model = DDP(model, device_ids=[0])

# Wrong: ddp_model.weight -- AttributeError
# Right: ddp_model.module.weight
print("Underlying model type:", type(ddp_model.module))
print("Weight shape:", ddp_model.module.weight.shape)

# Saving model: ALWAYS save .module not ddp_model
# torch.save(ddp_model.module.state_dict(), "model.pt")

dist.destroy_process_group()
PY
```

---

# 4.7 Inspect what's happening in real time

### Watch all 8 GPUs during training
```bash
watch -n 1 gpustat
```

### Check NVLink bandwidth (while training is running)
```bash
nvidia-smi nvlink -g 0 -s
```

### See process tree (all 8 training processes)
```bash
ps -eo pid,ppid,cmd | grep train_cifar10_ddp | head -20
```

### Confirm each process is on a different GPU
```bash
nvidia-smi | grep python
```

---

# 4.8 Module 4 pass criteria
You're done when you can:

1) Explain in your own words:
   - why DDP is faster than DataParallel
   - what gradient bucketing does and why it matters
   - what the autograd hook does in DDP

2) Run `train_cifar10_ddp.py` successfully on 1/2/4/8 GPUs

3) Build your scaling table (imgs/s vs GPU count)

4) Reproduce and fix:
   - port conflict error
   - NCCL timeout error
   - `ddp_model.module` access pattern

---

