# Module 6 — Multi-GPU Single Node (PCIe, NVLink, Topology, P2P, NCCL Rings)

## Summary
This module goes deep into **hardware**: what physically connects your 8 A100s, how data moves between them, why topology matters for AllReduce performance, and how NCCL exploits hardware topology automatically.

- **Theory**: PCIe vs NVLink, GPU topology, peer-to-peer memory, CUDA IPC, NCCL ring/tree algorithms
- **Hardware inspection**: read your actual A100 node topology
- **From scratch**: measure P2P bandwidth between every GPU pair
- **Benchmarks**: AllReduce scaling vs message size, bandwidth saturation
- **Visualization**: heatmap of GPU-GPU bandwidth
- **Debug drills**: topology pitfalls, wrong device placement, P2P not enabled

---

## 6.1 Theory — How GPUs talk to each other

### PCIe (PCI Express)
The default interconnect between GPU and CPU (and between GPUs without NVLink):

```
CPU
 │
 PCIe Root Complex
 ├── PCIe Switch
 │    ├── GPU0
 │    ├── GPU1
 │    ├── GPU2
 │    └── GPU3
 └── PCIe Switch
      ├── GPU4
      ├── GPU5
      ├── GPU6
      └── GPU7
```

PCIe bandwidth:
- PCIe Gen4 x16: ~32 GB/s (bidirectional peak)
- Real-world: ~25 GB/s unidirectional

GPU-to-GPU over PCIe goes through the CPU (unless P2P is enabled):
```
GPU0 → PCIe → CPU memory → PCIe → GPU1
```
This is slow and wastes CPU memory bandwidth.

### NVLink (what your A100 SXM4 actually has)
NVLink is a direct GPU-to-GPU interconnect, much faster than PCIe:

```
A100 SXM4 NVLink specs:
- NVLink 3.0
- 12 NVLink links per GPU
- Each link: ~50 GB/s bidirectional
- Total: ~600 GB/s bidirectional per GPU
```

NVLink creates a **mesh** between GPUs on the same node:
```
GPU0 ←—NVLink—→ GPU1
GPU0 ←—NVLink—→ GPU2
...
(topology depends on specific server config)
```

Data moves GPU-to-GPU directly, without touching CPU:
```
GPU0 → NVLink → GPU1    (fast, ~300+ GB/s)
```

### NVSwitch (what HGX A100 actually uses)
Your server (A100 SXM4) likely has **NVSwitch** — a dedicated chip that creates a full all-to-all NVLink fabric:

```
           NVSwitch
          /  |  |  \
       GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
```

Every GPU can talk to every other GPU at full NVLink bandwidth simultaneously. This is why HGX A100 nodes scale so well for AllReduce.

### PCIe vs NVLink bandwidth comparison

| Link | Bandwidth (unidirectional) |
|------|---------------------------|
| PCIe Gen3 x16 | ~16 GB/s |
| PCIe Gen4 x16 | ~32 GB/s |
| NVLink 3.0 (A100) | ~300 GB/s |
| NVSwitch (full fabric) | ~600 GB/s bidirectional |

For AllReduce of large gradient tensors, NVLink is ~10-20× faster than PCIe.

### NUMA (Non-Uniform Memory Access)
Modern servers have multiple CPU sockets:

```
Socket 0 (NUMA node 0)          Socket 1 (NUMA node 1)
├── CPU cores 0-31              ├── CPU cores 32-63
├── RAM bank 0                  ├── RAM bank 1
├── GPU0, GPU1, GPU2, GPU3      ├── GPU4, GPU5, GPU6, GPU7
└── PCIe root 0                 └── PCIe root 1
```

If GPU0 (NUMA 0) needs to transfer data through CPU to GPU4 (NUMA 1), it crosses the **QPI/UPI inter-socket link** → slower.

For DataLoader workers, pinning CPU cores to the same NUMA node as the GPU they feed improves throughput. NCCL is NUMA-aware and tries to exploit this.

### Peer-to-Peer (P2P) Memory Access
With NVLink/PCIe P2P, one GPU can directly read/write another GPU's memory:

```
Without P2P:
GPU0 → copy to host RAM → copy to GPU1 (2 copies, uses PCIe twice)

With P2P enabled:
GPU0 → directly write to GPU1 memory (1 copy, uses NVLink)
```

P2P requires:
- Both GPUs on same PCIe root complex (for PCIe P2P) OR NVLink
- P2P access explicitly enabled (or handled by NCCL/CUDA runtime)

### CUDA IPC (Inter-Process Communication)
When two processes (rank 0, rank 1) each own a GPU:
- They can't directly share GPU memory pointers (different processes)
- CUDA IPC lets them share a **handle** to a GPU memory allocation
- The other process can map that memory into its own address space
- This is what NCCL uses internally for intra-node communication

### NCCL Ring AllReduce (the classic algorithm)
For N GPUs in a ring:

```
Ring: GPU0 → GPU1 → GPU2 → GPU3 → GPU0

Step 1 (ReduceScatter): 
  Divide gradient tensor into N chunks
  Each GPU sends its chunk to the next GPU
  GPU accumulates (adds) received chunk with own chunk
  After N-1 steps: each GPU has the reduced sum of ONE chunk

Step 2 (AllGather):
  Each GPU broadcasts its reduced chunk around the ring
  After N-1 steps: every GPU has the complete reduced tensor
```

Total data transferred per GPU: `2 * (N-1)/N * tensor_size`

As N grows, this approaches `2 * tensor_size` per GPU — independent of N!

This is why ring AllReduce scales well: communication per GPU doesn't grow with world size.

### NCCL Tree AllReduce
For very large numbers of GPUs or high-latency networks:

```
         GPU0
        /    \
      GPU1   GPU2
      / \    / \
    GPU3 GPU4 GPU5 GPU6
```

Reduce up the tree (ReduceTree), then Broadcast down.

NCCL automatically chooses between ring and tree (and newer algorithms like **NVLS** on NVSwitch systems) based on topology.

### NCCL topology detection
When you call `dist.init_process_group("nccl")`, NCCL:
1. Reads `/sys/class/nvlink/*` and PCIe topology files
2. Detects NVSwitch if present
3. Chooses optimal algorithm (ring / tree / NVLS)
4. Builds a communication graph

You can see this with `NCCL_DEBUG=INFO`.

---

## 6.2 Hardware Inspection — Read your A100 topology

### Step A — Full topology matrix
```bash
nvidia-smi topo -m
```

You'll see a matrix like:
```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity
GPU0     X    NV12  NV12  NV12  NV12  NV12  NV12  NV12  0-31
GPU1    NV12   X    NV12  NV12  NV12  NV12  NV12  NV12  0-31
...
```

Decode the values:

| Symbol | Meaning |
|--------|---------|
| X | Same GPU |
| NV1..NV12 | NVLink (number = number of NVLink connections) |
| SYS | Connected via PCIe + CPU interconnect |
| NODE | Same NUMA node, different PCIe root |
| PHB | Same PCIe host bridge |
| PXB | Connected via PCIe switch |

`NV12` = 12 NVLink connections = maximum bandwidth = ideal.

### Step B — NVLink status per GPU
```bash
nvidia-smi nvlink -s
```

Shows NVLink speed and status for each link.

### Step C — P2P access matrix
```bash
cat > check_p2p.py <<'PY'
import torch

n = torch.cuda.device_count()
print(f"GPUs: {n}")
print("\nP2P Access Matrix (can GPU_i access GPU_j memory?):")
print("      " + "  ".join(f"GPU{j}" for j in range(n)))

for i in range(n):
    row = f"GPU{i}  "
    for j in range(n):
        if i == j:
            row += " X    "
        else:
            ok = torch.cuda.can_device_access_peer(i, j)
            row += f" {'YES' if ok else 'NO '}  "
    print(row)
PY
```

Run:
```bash
python check_p2p.py
```

On A100 SXM4 with NVSwitch, you should see YES for all pairs.

### Step D — NCCL topology detection output
```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
torchrun --nproc_per_node=8 -m torch.distributed.run \
  --no_python bash -c "python -c 'import torch.distributed as dist, os; dist.init_process_group(\"nccl\"); print(\"rank\", dist.get_rank(), \"ok\"); dist.destroy_process_group()'"
```

Simpler version:
```bash
cat > nccl_topo_check.py <<'PY'
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if rank == 0:
        print("NCCL init complete, topology detected", flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
torchrun --nproc_per_node=8 nccl_topo_check.py 2>&1 | head -80
```

Look for lines mentioning:
- `NVLink` — NCCL found NVLink
- `NVSwitch` — NCCL found NVSwitch fabric
- `Trees` / `Rings` — algorithm chosen
- `Ring 0` — ring topology built

---

## 6.3 From Scratch — Measure P2P bandwidth between every GPU pair

This is the most important benchmark in this module.

```bash
cat > p2p_bandwidth.py <<'PY'
"""
Measures GPU-to-GPU copy bandwidth for every pair.
Builds a full bandwidth matrix.
No distributed needed.
"""
import time
import torch

def measure_bandwidth_gb_s(src_gpu: int, dst_gpu: int,
                            size_mb: int = 512,
                            iters: int = 20) -> float:
    """
    Copy a tensor from src_gpu to dst_gpu and measure bandwidth.
    Returns GB/s.
    """
    n = (size_mb * 1024 * 1024) // 4  # float32

    with torch.cuda.device(src_gpu):
        src = torch.randn(n, device=f"cuda:{src_gpu}")

    with torch.cuda.device(dst_gpu):
        dst = torch.empty(n, device=f"cuda:{dst_gpu}")

    # warmup
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize(src_gpu)
    torch.cuda.synchronize(dst_gpu)

    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize(dst_gpu)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    bytes_transferred = n * 4 * iters
    gb_s = bytes_transferred / elapsed / 1e9
    return gb_s


def main():
    assert torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
    print(f"Measuring P2P bandwidth ({n_gpus} GPUs), tensor=512MB\n")

    # Check P2P access
    print("P2P access matrix:")
    for i in range(n_gpus):
        row = ""
        for j in range(n_gpus):
            if i == j:
                row += "  --- "
            else:
                ok = torch.cuda.can_device_access_peer(i, j)
                row += f"  {'YES' if ok else 'NO '}"
        print(f"  GPU{i}: {row}")
    print()

    # Enable P2P where possible
    for i in range(n_gpus):
        for j in range(n_gpus):
            if i != j and torch.cuda.can_device_access_peer(i, j):
                torch.cuda.device(i)

    # Bandwidth matrix
    bw = [[0.0] * n_gpus for _ in range(n_gpus)]

    for i in range(n_gpus):
        for j in range(n_gpus):
            if i == j:
                bw[i][j] = 0.0
                continue
            gb_s = measure_bandwidth_gb_s(i, j, size_mb=512, iters=20)
            bw[i][j] = gb_s
            print(f"  GPU{i} → GPU{j}: {gb_s:6.1f} GB/s", flush=True)

    # Print matrix
    print("\n── Bandwidth Matrix (GB/s) ──────────────────────────────────")
    header = "        " + "".join(f"  GPU{j:1d}" for j in range(n_gpus))
    print(header)
    for i in range(n_gpus):
        row = f"  GPU{i}:  "
        for j in range(n_gpus):
            if i == j:
                row += "   --- "
            else:
                row += f"  {bw[i][j]:5.1f}"
        print(row)

    # Summary stats
    off_diag = [bw[i][j] for i in range(n_gpus) for j in range(n_gpus) if i != j]
    print(f"\n  min: {min(off_diag):.1f} GB/s")
    print(f"  max: {max(off_diag):.1f} GB/s")
    print(f"  avg: {sum(off_diag)/len(off_diag):.1f} GB/s")


if __name__ == "__main__":
    main()
PY
```

Run:
```bash
python p2p_bandwidth.py
```

On A100 SXM4 with NVSwitch you should see:
- ~200–300 GB/s for NVLink pairs
- Roughly uniform across all pairs (NVSwitch full mesh)

If you had a PCIe-only system, you'd see:
- ~25 GB/s for PCIe pairs
- lower for cross-NUMA pairs

---

## 6.4 Benchmark — AllReduce vs Message Size

This tells you where bandwidth saturates and latency dominates.

```bash
cat > bench_allreduce_msgsize.py <<'PY'
"""
Benchmark NCCL AllReduce at different message sizes.
Helps understand: latency-bound (small) vs bandwidth-bound (large).
"""
import os, time
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Message sizes to test: 1KB → 1GB
    sizes_bytes = [
        1024,            # 1 KB
        4 * 1024,        # 4 KB
        16 * 1024,       # 16 KB
        64 * 1024,       # 64 KB
        256 * 1024,      # 256 KB
        1024 * 1024,     # 1 MB
        4 * 1024**2,     # 4 MB
        16 * 1024**2,    # 16 MB
        64 * 1024**2,    # 64 MB
        256 * 1024**2,   # 256 MB
        512 * 1024**2,   # 512 MB
        1024 * 1024**2,  # 1 GB
    ]

    iters  = 50
    warmup = 10

    if rank == 0:
        print(f"world_size={world_size}")
        print(f"{'Size':>12}  {'Avg (ms)':>10}  {'BW (GB/s)':>10}  {'Regime':>12}")
        print("-" * 55)

    for nbytes in sizes_bytes:
        numel = nbytes // 4  # float32
        x = torch.ones(numel, device=device, dtype=torch.float32)

        # warmup
        for _ in range(warmup):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) * 1000 / iters

        # AllReduce traffic per GPU (ring algo): 2*(N-1)/N * size ≈ 2*size
        traffic_bytes = 2 * nbytes * (world_size - 1) / world_size
        bw_gb_s = traffic_bytes / (avg_ms / 1000) / 1e9

        # Rough regime classification
        if avg_ms < 0.1:
            regime = "latency-bound"
        elif avg_ms < 1.0:
            regime = "transition"
        else:
            regime = "bandwidth-bound"

        if rank == 0:
            size_str = (
                f"{nbytes // 1024**2} MB" if nbytes >= 1024**2
                else f"{nbytes // 1024} KB" if nbytes >= 1024
                else f"{nbytes} B"
            )
            print(f"{size_str:>12}  {avg_ms:>10.3f}  {bw_gb_s:>10.1f}  {regime:>12}",
                  flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run across different GPU counts:
```bash
# 2 GPUs
torchrun --nproc_per_node=2 bench_allreduce_msgsize.py

# 4 GPUs
torchrun --nproc_per_node=4 bench_allreduce_msgsize.py

# 8 GPUs
torchrun --nproc_per_node=8 bench_allreduce_msgsize.py
```

What you'll learn from results:
- Small messages (< 64KB): latency dominated, bandwidth looks low
- Large messages (> 64MB): bandwidth dominated, BW approaches NVLink peak
- DDP uses 25MB buckets by default (in the "transition" zone — tunable)

---

## 6.5 Benchmark — Scaling Efficiency (compute + comm together)

```bash
cat > bench_scaling_efficiency.py <<'PY'
"""
Measures realistic training step time across GPU counts.
Separates compute time vs communication time.
"""
import os, time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class BenchModel(nn.Module):
    """Configurable model: controls compute vs param count"""
    def __init__(self, hidden=4096, depth=8):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def time_section(fn, sync_device, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(sync_device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(sync_device)
    return (time.perf_counter() - t0) * 1000 / iters  # ms


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    hidden = 4096
    depth  = 8
    batch  = 64

    model = BenchModel(hidden=hidden, depth=depth).to(device)
    ddp   = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)

    x = torch.randn(batch, hidden, device=device)

    param_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024**2

    if is_main:
        print(f"\nworld_size={world_size}  params={param_mb:.1f}MB  "
              f"hidden={hidden}  depth={depth}  batch={batch}")

    # ── Measure forward only ──────────────────────────────────────────────
    def fwd():
        with torch.no_grad():
            _ = model(x)

    fwd_ms = time_section(fwd, device)

    # ── Measure forward + backward (no DDP comm) ──────────────────────────
    # Use no_sync() context: suppresses gradient AllReduce
    def fwd_bwd_no_comm():
        with ddp.no_sync():
            out = ddp(x)
            loss = out.mean()
            loss.backward()

    fwd_bwd_ms = time_section(fwd_bwd_no_comm, device)

    # ── Measure full step (forward + backward + allreduce) ────────────────
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01)

    def full_step():
        opt.zero_grad(set_to_none=True)
        out  = ddp(x)
        loss = out.mean()
        loss.backward()
        opt.step()

    full_ms = time_section(full_step, device)

    comm_ms    = full_ms - fwd_bwd_ms
    comm_pct   = 100.0 * comm_ms / full_ms

    # ── Print results (rank 0 only) ───────────────────────────────────────
    if is_main:
        print(f"  forward only         : {fwd_ms:7.2f} ms")
        print(f"  fwd+bwd (no comm)    : {fwd_bwd_ms:7.2f} ms")
        print(f"  full step (with comm): {full_ms:7.2f} ms")
        print(f"  comm overhead        : {comm_ms:7.2f} ms  ({comm_pct:.1f}% of step)")
        print(f"  throughput           : {batch * world_size / (full_ms/1000):.0f} samples/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run across GPU counts:
```bash
torchrun --nproc_per_node=1 bench_scaling_efficiency.py
torchrun --nproc_per_node=2 bench_scaling_efficiency.py
torchrun --nproc_per_node=4 bench_scaling_efficiency.py
torchrun --nproc_per_node=8 bench_scaling_efficiency.py
```

Build your table:

| GPUs | fwd_ms | fwd+bwd_ms | full_ms | comm_ms | comm% | samples/s |
|------|--------|------------|---------|---------|-------|-----------|
| 1 | | | | | | |
| 2 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |

Compute scaling efficiency:
```
efficiency(N) = (samples/s at N GPUs) / (N × samples/s at 1 GPU) × 100%
```

Perfect scaling = 100%. Real NVLink A100 nodes typically achieve 85–95%.

---

## 6.6 DDP no_sync() — Gradient Accumulation (bonus)

`no_sync()` is also how you do **gradient accumulation** in DDP:

```bash
cat > ddp_grad_accumulation.py <<'PY'
"""
Gradient accumulation with DDP.
Accumulate gradients for N micro-steps, then sync + step.
This simulates a larger effective batch without more GPU memory.
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    model = nn.Linear(128, 10).to(device)
    ddp   = DDP(model, device_ids=[local_rank])
    opt   = torch.optim.SGD(ddp.parameters(), lr=0.01)

    accum_steps = 4  # accumulate 4 micro-batches before syncing

    for global_step in range(3):
        opt.zero_grad(set_to_none=True)

        for micro_step in range(accum_steps):
            x = torch.randn(32, 128, device=device)
            y = torch.randint(0, 10, (32,), device=device)

            # Suppress AllReduce for all but the last micro-step
            is_last = (micro_step == accum_steps - 1)
            ctx = ddp.no_sync() if not is_last else torch.cuda.amp.autocast()

            if not is_last:
                with ddp.no_sync():
                    loss = nn.functional.cross_entropy(ddp(x), y)
                    loss = loss / accum_steps  # normalize
                    loss.backward()
            else:
                # Last micro-step: let AllReduce fire
                loss = nn.functional.cross_entropy(ddp(x), y)
                loss = loss / accum_steps
                loss.backward()  # AllReduce fires here

            if is_main:
                print(f"  step={global_step} micro={micro_step} "
                      f"{'[SYNC]' if is_last else '[no_sync]'}",
                      flush=True)

        opt.step()
        if is_main:
            print(f"  ── optimizer step {global_step} done ──\n", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=2 ddp_grad_accumulation.py
```

You'll see `[no_sync]` for micro-steps 0-2, then `[SYNC]` at micro-step 3 where AllReduce fires.

---

## 6.7 Debug Drills

### Drill 1 — Wrong device placement (silent wrong result)
```bash
cat > wrong_device.py <<'PY'
import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # BUG: tensor created on cuda:0 for all ranks
    # instead of their own local GPU
    x = torch.ones(4, device="cuda:0")   # WRONG for rank > 0

    # This may hang or give wrong results because
    # rank 1's tensor is on GPU0, not GPU1
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"rank={rank} result={x.tolist()}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=2 wrong_device.py
```

Then fix it:
```python
# CORRECT: each rank uses its own local GPU
device = torch.device("cuda", local_rank)
x = torch.ones(4, device=device)
```

---

### Drill 2 — bucket_cap_mb tuning (performance experiment)

DDP's default bucket size is 25MB. Try different values and observe AllReduce timing:

```bash
cat > ddp_bucket_tuning.py <<'PY'
import os, time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *[nn.Linear(2048, 2048) for _ in range(6)]
        )
    def forward(self, x): return self.net(x)


def bench_bucket(bucket_mb, device, local_rank, world_size, iters=30):
    model = SimpleNet().to(device)
    ddp   = DDP(model, device_ids=[local_rank], bucket_cap_mb=bucket_mb)
    opt   = torch.optim.SGD(ddp.parameters(), lr=0.01)
    x     = torch.randn(64, 2048, device=device)

    # warmup
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        ddp(x).mean().backward()
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        ddp(x).mean().backward()
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()
    ms = (time.perf_counter() - t0) * 1000 / iters

    # cleanup
    del ddp, model, opt
    torch.cuda.empty_cache()
    return ms


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    bucket_sizes = [1, 5, 10, 25, 50, 100, 200]

    if rank == 0:
        print(f"\nworld_size={world_size}  bucket tuning experiment")
        print(f"{'bucket_mb':>12}  {'step_ms':>10}")
        print("-" * 28)

    for bsz in bucket_sizes:
        ms = bench_bucket(bsz, device, local_rank, world_size)
        if rank == 0:
            print(f"{bsz:>12}  {ms:>10.2f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=8 ddp_bucket_tuning.py
```

You'll find a sweet spot. Very small buckets = too many AllReduce calls = high latency cost. Very large buckets = comm doesn't overlap with backward (fires too late).

---

### Drill 3 — NCCL environment variable reference

Keep this as a cheat sheet:

```bash
# Show NCCL topology detection
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH

# Show every collective operation
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL

# Force specific algorithm (for experiments)
NCCL_ALGO=Ring      # force ring
NCCL_ALGO=Tree      # force tree

# Force P2P protocol
NCCL_P2P_LEVEL=NVL  # NVLink only
NCCL_P2P_LEVEL=PIX  # PCIe only (for testing degraded perf)

# Disable NVLink (simulate PCIe-only performance)
NCCL_P2P_DISABLE=1

# Useful for debugging hangs
NCCL_TIMEOUT=10     # seconds before timeout error
```

Experiment: simulate PCIe-only performance on your NVLink machine:
```bash
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=8 bench_allreduce_msgsize.py
```

Compare to normal run — you should see significant bandwidth drop.

---

## 6.8 Full picture — what module 6 taught you

```
Your A100 node:

  ┌─────────────────────────────────────────────────┐
  │                  NVSwitch Fabric                │
  │   GPU0 ←──────────────────────────────→ GPU7   │
  │   GPU1 ←──────────────────────────────→ GPU6   │
  │   GPU2 ←──────────────────────────────→ GPU5   │
  │   GPU3 ←──────────────────────────────→ GPU4   │
  │          All pairs: ~300 GB/s NVLink            │
  └─────────────────────────────────────────────────┘
          │                          │
     PCIe Gen4                  PCIe Gen4
          │                          │
   CPU Socket 0              CPU Socket 1
   (NUMA node 0)             (NUMA node 1)
```

NCCL sees the NVSwitch fabric and uses NVLS (NVLink Switch) algorithm where available, which is more efficient than ring on NVSwitch systems.

DDP sits on top:
- wraps your model
- registers autograd hooks
- groups gradients into buckets
- fires NCCL AllReduce per bucket
- overlaps with backward compute

---

## 6.9 Module 6 Pass Criteria

You're done when you can:

1) Read `nvidia-smi topo -m` and explain every symbol in the matrix

2) Explain:
   - why NVLink is 10× faster than PCIe for GPU-GPU transfers
   - what ring AllReduce does step by step
   - why bucket size affects DDP performance
   - what `no_sync()` does and when to use it

3) Run and interpret:
   - `python p2p_bandwidth.py` → bandwidth matrix
   - `bench_allreduce_msgsize.py` at 2/4/8 GPUs → latency vs bandwidth regimes
   - `bench_scaling_efficiency.py` → compute vs comm breakdown

4) Fill in your scaling efficiency table and compute % efficiency

5) Reproduce and fix:
   - wrong device placement bug
   - simulate PCIe-only performance with `NCCL_P2P_DISABLE=1`

---

