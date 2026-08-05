# Module 2 — Why Multiple GPUs?

## Summary (what you’ll learn + do)
- **Why single-GPU training becomes slow** (and/or impossible) as models and batch sizes grow.
- The two fundamental blockers: **compute time** and **memory limits**.
- The third blocker that appears when you scale out: **communication + synchronization**.
- How to reason about scaling with **Amdahl’s Law**, **strong vs weak scaling**, and **bandwidth/latency**.
- Hands-on: run a **controlled benchmark** on **1 → 2 → 4 → 8 GPUs** on your A100 node and measure:
  - time/iteration
  - throughput
  - GPU utilization
  - how much time is “communication-like overhead”

---

## 2.1 Theory — the 3 reasons you need multiple GPUs

### Reason 1: Training time (compute throughput)
If a model takes, say, **200 ms/step** on one GPU, and you need **100,000 steps**, that’s ~5.5 hours (and real models are far worse). Multiple GPUs can reduce wall-clock time by parallelizing work.

But you don’t get “free speed”—you pay coordination overhead (below).

### Reason 2: Memory limits (the “LLMs don’t fit” problem)
Even on an A100 40GB, modern training runs out of memory because training needs:

- **Parameters**
- **Gradients**
- **Optimizer states** (especially Adam)
- **Activations** (often the biggest, batch-size dependent)

Multiple GPUs let you:
- split batch across GPUs (**data parallel**) to increase throughput
- later: shard model states across GPUs (**FSDP / ZeRO**) to fit bigger models

### Reason 3: Communication + synchronization overhead (the scaling tax)
The moment you use multiple GPUs, you must **combine gradients** (or parameters) across GPUs so everyone stays consistent.

In DDP-style training, this is usually **AllReduce** across GPUs each iteration.

NCCL is the standard library that provides these high-performance collectives on NVIDIA GPUs (all-reduce/all-gather/reduce-scatter/broadcast, etc.). 

---

## 2.2 The key mental model: each step becomes Compute + Communication


<img width="924" height="217" alt="Screenshot 2026-08-05 145338" src="https://github.com/user-attachments/assets/a5a29c89-c5e3-42ad-9adf-7071a4ecf876" />

  
### Compute vs Communication (why A100 nodes still don’t scale perfectly)
- GPUs are insanely fast at math.
- GPU-to-GPU communication (even NVLink) is slower than on-chip compute, and cross-node is slower still.
- So scaling depends on whether your workload is **compute-bound** or **communication-bound**.

---

## 2.3 Amdahl’s Law (strong scaling limit)


<img width="926" height="375" alt="Screenshot 2026-08-05 145548" src="https://github.com/user-attachments/assets/6db635d3-a774-485d-b084-cc1fcbd9b2b1" />


---

## 2.4 Strong vs Weak scaling (how to measure honestly)

### Strong scaling
- Fixed total batch size / fixed workload
- Add GPUs and see if time per step drops
- Often disappoints at larger N because communication becomes a bigger fraction.

### Weak scaling
- Increase global batch size proportional to number of GPUs
- Keep **work per GPU** roughly constant
- Often looks better, but changes optimization dynamics (LR/batch scaling).

---

## 2.5 Bandwidth, latency, and collective algorithms (why topology matters)

### Bandwidth vs latency
- **Latency** dominates small messages (lots of tiny syncs are bad).
- **Bandwidth** dominates big messages (like large gradient buckets).

### Ring vs Tree topologies (why you keep hearing “ring allreduce”)
NCCL uses topology-aware algorithms and has historically used **ring** and **tree** families (and more recent variants). 

Intuition:
- **Ring**: great bandwidth efficiency, good for large tensors (common in DL).
- **Tree**: can reduce latency in some cases (often better for smaller messages or certain topologies).

NCCL’s goal is to pick something close to optimal for your actual hardware links.

---

# 2.6 Practical — Benchmark scaling on 1 → 2 → 4 → 8 GPUs (single node)

We will benchmark **two things** separately:

1) **Compute-only-ish** workload (matrix multiplications)  
2) **Communication-heavy** workload (AllReduce on big tensors)

This isolates where time is going.

## Step A — Install what we need (inside your venv)
```bash
source /root/ddl_course/venv/bin/activate
pip install -U torch
```

(You already have CUDA driver working and `nvcc` available; PyTorch wheels include their own CUDA runtime, so it’s fine if your system CUDA is 11.5.)

---

## Step B — Benchmark 1: Single-process matmul scaling (no comm, uses 1 GPU)
Create `bench_matmul_single.py`:

```bash
cat > bench_matmul_single.py <<'PY'
import time, torch

assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda:0"
n = 16384  # adjust if too slow/too much memory
dtype = torch.float16

a = torch.randn(n, n, device=device, dtype=dtype)
b = torch.randn(n, n, device=device, dtype=dtype)

# warmup
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()

iters = 50
t0 = time.time()
for _ in range(iters):
    c = a @ b
torch.cuda.synchronize()
t1 = time.time()

ms = (t1-t0)*1000/iters
print(f"matmul {n}x{n} dtype={dtype} avg {ms:.3f} ms")
print("alloc_GB", torch.cuda.memory_allocated()/1024**3)
PY
```

Run:
```bash
python bench_matmul_single.py
```

This gives you a baseline: “one GPU compute speed”.

---

## Step C — Benchmark 2: NCCL AllReduce scaling (this is the key)
Now we do real multi-GPU comm using PyTorch distributed (NCCL backend).

Create `bench_allreduce.py`:

```bash
cat > bench_allreduce.py <<'PY'
import os, time
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 256MB tensor per rank (adjustable)
    bytes_mb = 256
    numel = (bytes_mb * 1024 * 1024) // 4  # float32 = 4 bytes
    x = torch.ones(numel, device=device, dtype=torch.float32)

    # warmup
    for _ in range(10):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iters = 50
    dist.barrier()
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1000 / iters
    if rank == 0:
        print(f"world_size={world} tensor={bytes_mb}MB allreduce avg={avg_ms:.3f} ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
```

Now run it at different GPU counts:

### 1 GPU
```bash
torchrun --nproc_per_node=1 bench_allreduce.py
```

### 2 GPUs
```bash
torchrun --nproc_per_node=2 bench_allreduce.py
```

### 4 GPUs
```bash
torchrun --nproc_per_node=4 bench_allreduce.py
```

### 8 GPUs
```bash
torchrun --nproc_per_node=8 bench_allreduce.py
```

What you should observe:
- AllReduce time rises as world size increases (not always linearly, but it grows).
- This is the “communication tax” that limits scaling.

NCCL provides the AllReduce primitive optimized for multi-GPU and multi-node; that’s exactly what you’re timing here. 

---

## Step D — Record metrics (make a small table)
For each run (1/2/4/8 GPUs), record:
- avg allreduce ms
- `nvidia-smi` observed link/util (optional)
- any NCCL warnings/errors

---

# 2.7 How to interpret results (what matters for Module 3+)

### If AllReduce is “small”
Good news: your node topology (NVLink) is strong and you’ll scale well.

### If AllReduce is “big”
Then:
- your message is too big or too frequent
- your links are slower (PCIe-only, or topology issues)
- NCCL algorithm choice may not be ideal (rare on HGX A100, but possible)
- you may be hitting CPU affinity / process placement issues (comes up with 8 GPUs)

In later modules:
- DDP reduces comm overhead using **gradient bucketing** and overlapping comm with backward compute. PyTorch explicitly documents bucket-based reduction to overlap comm and compute. 

---

# 2.8 Debugging: common failures you might hit now

### 1) “NCCL error / unhandled system error / connection refused”
On single node, this is often:
- wrong interface selection (rare single node)
- leftover processes hanging
- permissions / driver mismatch

Quick cleanup:
```bash
pkill -f bench_allreduce.py || true
pkill -f torchrun || true
```

### 2) Hang at init / barrier
Usually means one or more ranks died. Check logs and:
```bash
ps aux | grep python
nvidia-smi
```

### 3) Port conflict
Torchrun picks a default port sometimes; set one:
```bash
MASTER_PORT=29501 torchrun --nproc_per_node=8 bench_allreduce.py
```

---

# Module 2 pass criteria
You’re done with Module 2 when you can show:

1) A table of **AllReduce avg ms** for world sizes **1,2,4,8**
2) A statement like:
   - “At 8 GPUs, AllReduce is X ms, so comm is Y% of step time (if step time is Z ms).”
3) You can explain (in your own words):
   - strong vs weak scaling
   - why Amdahl’s law applies
   - why ring/tree choices matter conceptually

---
