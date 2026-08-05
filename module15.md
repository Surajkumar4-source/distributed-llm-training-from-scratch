# Module 15 — Profiling

## Summary
You cannot optimize what you cannot measure. Profiling tells you exactly where time and memory go — not what you guess, but what hardware evidence shows. This module teaches you to use every major profiling tool on your A100 cluster, from timeline-level (Nsight Systems) to individual kernel analysis (Nsight Compute), to PyTorch op-level tracing.

- **Theory**: GPU execution model, roofline model, memory bandwidth vs compute bound, what "GPU utilization" actually means
- **Nsight Systems**: full timeline profiling, CPU/GPU overlap, NCCL visibility
- **Nsight Compute**: kernel-level analysis, occupancy, memory bandwidth, Tensor Core utilization
- **PyTorch Profiler**: op-level traces, Chrome viewer, TensorBoard integration
- **NCCL Debug**: communication bottleneck diagnosis
- **Memory Profiler**: allocation traces, snapshot visualization
- **End-to-end**: profile your GPT DDP training, find top-3 bottlenecks, fix them

---

## 15.1 Theory — GPU Execution and the Roofline Model

### What "GPU utilization" actually means

`nvidia-smi` shows "GPU Utilization %" — this is the fraction of time the GPU had at least one kernel running. It says **nothing** about efficiency.

```
Example:
  nvidia-smi shows: 99% utilization

Reality could be:
  A) GPU running kernels at full Tensor Core efficiency   ← great
  B) GPU running a tiny kernel that takes the whole time  ← terrible
  C) GPU mostly waiting for memory                        ← memory bound

You need Nsight Compute to know which is true.
```

### The Roofline Model

Two fundamental limits on GPU performance:

```
Peak Compute:       ~312 TFLOPS (A100 bf16 with Tensor Cores)
Peak Memory BW:     ~2 TB/s    (A100 HBM2e)

Arithmetic Intensity (AI) = FLOPs / bytes_moved

If AI < Peak_Compute / Peak_MemBW  → memory-bound
If AI > Peak_Compute / Peak_MemBW  → compute-bound

A100 "ridge point":
  312e12 / 2e12 = 156 FLOPs/byte

Large matmuls (transformer):
  GEMM(4096, 4096, 4096): AI ≈ 1365 FLOPs/byte → compute-bound ✓
  LayerNorm(B×T, d):      AI ≈ 5 FLOPs/byte    → memory-bound ✗
```

### Common bottleneck types

```
1. Compute-bound:
   - Large matrix multiplications
   - Fix: use Tensor Cores (bf16/fp16), increase batch size

2. Memory-bandwidth-bound:
   - Elementwise ops (ReLU, add, norm)
   - Small matrix multiplications
   - Fix: kernel fusion (torch.compile), FlashAttention

3. Memory-capacity-bound:
   - OOM errors, forced to reduce batch
   - Fix: Module 11 techniques (ckpt, AMP, offload)

4. Communication-bound:
   - AllReduce / AllGather / ReduceScatter
   - Fix: overlap comm with compute, reduce model size, faster interconnect

5. CPU-bound / DataLoader-bound:
   - GPU idle waiting for data
   - Fix: more workers, pin_memory, prefetch, faster storage

6. Launch-overhead-bound:
   - Too many small kernels (Python loop over ops)
   - Fix: torch.compile, fused kernels
```

### CUDA Stream model

```
GPU executes operations asynchronously on streams.
Multiple streams can run concurrently (if resources allow).

Default stream:
  All ops go to stream 0 by default
  Sequential within a stream

DDP uses separate streams for:
  - Computation (forward/backward)
  - Communication (AllReduce)
  → True overlap of comm and compute

To profile accurately: always torch.cuda.synchronize() before timing.
```

---

## 15.2 Setup — Install Profiling Tools

```bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course
mkdir -p profiling_module
cp gpt_scratch/tokenizer.py          profiling_module/
cp gpt_scratch/shakespeare.txt       profiling_module/
cp fsdp_module/model.py              profiling_module/
cd profiling_module

# Install profiling dependencies
pip install tensorboard
pip install torch-tb-profiler
pip install matplotlib pandas

# Check NVIDIA profiling tools
nsys --version 2>/dev/null || echo "nsys not in PATH — may need: export PATH=/usr/local/cuda/bin:$PATH"
ncu  --version 2>/dev/null || echo "ncu not in PATH"

# Add CUDA tools to PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
export PATH=/usr/local/bin:$PATH

# Verify
nvidia-smi
nsys --version
ncu  --version

mkdir -p profiles logs traces
```

---

## 15.3 Lab 1 — PyTorch Profiler (Start Here)

The most accessible profiler. No extra tools needed. Output viewable in browser.

```bash
cat > profile_basic.py <<'PY'
"""
PyTorch Profiler — basic usage.

Shows:
  - Time per operation (CPU + CUDA side by side)
  - Memory allocated per op
  - Chrome trace (interactive timeline)
  - TensorBoard integration

Run: python profile_basic.py
View: tensorboard --logdir traces/ --port 6006
      then open Chrome trace at: http://localhost:6006/#profile
"""
import torch
import torch.nn as nn
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
    record_function,
)

from model import GPT, GPTConfig


def main():
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"

    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=256,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0
    )
    model = GPT(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ctx   = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    B, T  = 32, 256
    x     = torch.randint(0, 65, (B, T), device=device)
    y     = torch.randint(0, 65, (B, T), device=device)

    # ── Method 1: Simple one-shot profile ─────────────────────────────────
    print("Method 1: Simple profile (3 steps)")
    with profile(
        activities      = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes   = True,
        profile_memory  = True,
        with_stack      = False,
        with_flops      = True,
    ) as prof:
        for step in range(3):
            with record_function("forward"):
                with ctx:
                    _, loss = model(x, y)

            with record_function("backward"):
                loss.backward()

            with record_function("optimizer"):
                opt.step()
                opt.zero_grad(set_to_none=True)

    # Print top ops by CUDA time
    print("\n── Top 20 ops by CUDA time ──────────────────────────────────────")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))

    # Print top ops by memory
    print("\n── Top 10 ops by memory ─────────────────────────────────────────")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10
    ))

    # Export Chrome trace
    trace_path = "traces/simple_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved: {trace_path}")
    print("View: chrome://tracing → load file")

    # ── Method 2: Schedule-based (production pattern) ─────────────────────
    print("\n\nMethod 2: Schedule-based profile (warm up, then capture)")

    # Schedule: skip 2 steps, warm up 2 steps, capture 3 steps, repeat
    sched = schedule(wait=2, warmup=2, active=3, repeat=1)

    with profile(
        activities      = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule        = sched,
        on_trace_ready  = tensorboard_trace_handler("traces/tensorboard"),
        record_shapes   = True,
        profile_memory  = True,
        with_stack      = True,
        with_flops      = True,
    ) as prof:
        for step in range(8):  # wait(2) + warmup(2) + active(3) = 7 steps
            with record_function("train_step"):
                opt.zero_grad(set_to_none=True)
                with ctx:
                    _, loss = model(x, y)
                loss.backward()
                opt.step()
            prof.step()   # tell profiler this step is done

    print("TensorBoard trace saved: traces/tensorboard/")
    print("View: tensorboard --logdir traces/tensorboard --port 6006")

    # ── Method 3: Manual record_function labels ────────────────────────────
    print("\n\nMethod 3: Fine-grained labels")

    with profile(
        activities    = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes = True,
    ) as prof:
        with record_function("data_transfer"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            torch.cuda.synchronize()

        with record_function("embedding"):
            import torch.nn.functional as F
            tok_emb = model.token_emb(x)
            pos     = torch.arange(T, device=device)
            pos_emb = model.pos_emb(pos)
            hidden  = tok_emb + pos_emb

        with record_function("transformer_blocks"):
            for i, block in enumerate(model.blocks):
                with record_function(f"block_{i}"):
                    hidden = block(hidden)

        with record_function("lm_head"):
            logits = model.lm_head(model.ln_final(hidden))

    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=15
    ))

    trace_path = "traces/fine_grained_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nFine-grained trace: {trace_path}")


if __name__ == "__main__":
    main()
PY

python profile_basic.py
```

---

## 15.4 Lab 2 — Profiler Schedule for Training Loop

```bash
cat > profile_training.py <<'PY'
"""
Profile a real training loop with schedule.

The schedule pattern (wait → warmup → active) is critical:
  - wait:   let the loop reach steady state (skip early overhead)
  - warmup: CUDA caches warm up, allocator finds rhythm
  - active: the steps you actually profile (capture trace)

This gives you a trace of steady-state training, not startup noise.
"""
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
    record_function,
)

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def train_step(model, optimizer, x, y, ctx, scaler=None):
    """One training step with labeled sections."""
    optimizer.zero_grad(set_to_none=True)

    with record_function("forward_pass"):
        with ctx:
            _, loss = model(x, y)

    with record_function("backward_pass"):
        loss.backward()

    with record_function("optimizer_step"):
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item()


def profile_training_loop(
    n_profile_steps : int  = 10,
    wait_steps      : int  = 3,
    warmup_steps    : int  = 2,
    active_steps    : int  = 5,
    batch_size      : int  = 32,
    seq_len         : int  = 256,
    save_dir        : str  = "traces/training_profile",
):
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"

    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=seq_len,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0,
    )
    model = GPT(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ctx   = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    text = open("shakespeare.txt").read()
    tok  = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    def get_batch():
        ix = torch.randint(len(data) - seq_len, (batch_size,))
        x  = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
        y  = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
        return x, y

    # ── Pre-warmup (outside profiler) ────────────────────────────────────
    print("Pre-warming model (outside profiler)...")
    for _ in range(5):
        x, y = get_batch()
        train_step(model, opt, x, y, ctx)
    torch.cuda.synchronize()
    print("Pre-warm done.")

    # ── Profile with schedule ─────────────────────────────────────────────
    sched = schedule(
        wait   = wait_steps,
        warmup = warmup_steps,
        active = active_steps,
        repeat = 1,
    )

    total_steps    = wait_steps + warmup_steps + active_steps
    step_times     = []

    with profile(
        activities      = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule        = sched,
        on_trace_ready  = tensorboard_trace_handler(save_dir),
        record_shapes   = True,
        profile_memory  = True,
        with_stack      = False,
        with_flops      = True,
    ) as prof:
        for step in range(total_steps):
            x, y = get_batch()

            t0   = time.perf_counter()
            loss = train_step(model, opt, x, y, ctx)
            torch.cuda.synchronize()
            dt   = time.perf_counter() - t0

            step_times.append(dt * 1000)
            phase = ("wait" if step < wait_steps
                     else "warmup" if step < wait_steps + warmup_steps
                     else "ACTIVE")

            print(f"  step={step:2d} [{phase:6s}]  "
                  f"loss={loss:.4f}  dt={dt*1000:.1f}ms",
                  flush=True)

            prof.step()

    # ── Summary ───────────────────────────────────────────────────────────
    active_times = step_times[wait_steps + warmup_steps:]
    avg_ms       = sum(active_times) / len(active_times)
    tok_s        = batch_size * seq_len / (avg_ms / 1000)

    print(f"\n── Profile Summary ──────────────────────────────────────────")
    print(f"  Active steps:    {active_steps}")
    print(f"  Avg step time:   {avg_ms:.2f} ms")
    print(f"  Throughput:      {tok_s:.0f} tokens/s")
    print(f"  Saved to:        {save_dir}")
    print(f"\n── Top Ops (CUDA time) ──────────────────────────────────────")

    key_avgs = prof.key_averages()
    print(key_avgs.table(sort_by="cuda_time_total", row_limit=15))

    # ── Per-category breakdown ────────────────────────────────────────────
    print(f"\n── Time breakdown by category ───────────────────────────────")
    categories = {}
    for evt in key_avgs:
        cat = evt.key.split("::")[0] if "::" in evt.key else "other"
        categories[cat] = categories.get(cat, 0) + evt.cuda_time_total

    total_cuda = sum(categories.values())
    for cat, t in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * t / total_cuda if total_cuda > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cat:30s}: {t/1e6:8.2f}ms  {pct:5.1f}%  {bar}")

    # Save Chrome trace
    prof.export_chrome_trace("traces/training_trace.json")
    print(f"\nChrome trace: traces/training_trace.json")
    print("View: tensorboard --logdir traces/training_profile --port 6006")


if __name__ == "__main__":
    profile_training_loop()
PY

python profile_training.py
```

---

## 15.5 Lab 3 — Profile DDP Communication Overlap

```bash
cat > profile_ddp_comm.py <<'PY'
"""
Profile DDP training to see communication vs compute overlap.

Key things to look for in the trace:
  1. AllReduce kernels running DURING backward (good overlap)
  2. AllReduce kernels AFTER backward (poor overlap)
  3. CPU stalls waiting for GPU (Python overhead)
  4. DataLoader prefetch gap (CPU busy, GPU idle)

Run:
  torchrun --nproc_per_node=4 profile_ddp_comm.py
Then view trace for rank 0.
"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
    record_function,
)

from model import GPT, GPTConfig


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    torch.backends.cuda.matmul.allow_tf32 = True

    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=256,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0,
    )
    torch.manual_seed(42)
    model = GPT(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ctx   = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    B, T  = 32, 256
    x     = torch.randint(0, 65, (B, T), device=device)
    y     = torch.randint(0, 65, (B, T), device=device)

    # Pre-warmup
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()

    # Profile only rank 0 (or all ranks for comparison)
    trace_dir = f"traces/ddp_rank{rank}"

    sched = schedule(wait=2, warmup=2, active=5, repeat=1)

    with profile(
        activities      = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule        = sched,
        on_trace_ready  = tensorboard_trace_handler(trace_dir),
        record_shapes   = True,
        profile_memory  = True,
        with_stack      = False,
    ) as prof:
        for step in range(9):   # 2+2+5=9
            with record_function("zero_grad"):
                opt.zero_grad(set_to_none=True)

            with record_function("forward"):
                with ctx:
                    _, loss = model(x, y)

            with record_function("backward"):
                loss.backward()
                # AllReduce fires HERE (overlapped with backward via hooks)

            with record_function("optimizer"):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            dist.barrier()
            prof.step()

    if is_main:
        print(f"\n── DDP Profile (rank 0) ─────────────────────────────────")
        print(f"  world_size: {world_size}")
        print(f"  bucket_cap: 25 MB")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=15
        ))
        print(f"\nTraces saved to: traces/ddp_rank*/")
        print("Load ALL rank traces in TensorBoard to see comm overlap:")
        print("  tensorboard --logdir traces/ --port 6006")
        print("\nWhat to look for:")
        print("  ✓ ncclAllReduce running DURING aten::mm (good overlap)")
        print("  ✗ ncclAllReduce running AFTER all backward ops (no overlap)")
        print("  ✓ Multiple AllReduce calls (one per DDP bucket)")
        print("  ? Gap between backward end and AllReduce end = sync cost")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 profile_ddp_comm.py
```

---

## 15.6 Lab 4 — Nsight Systems (Timeline Profiling)

```bash
cat > nsys_profile_target.py <<'PY'
"""
Target script for Nsight Systems profiling.

Nsight Systems gives you a full GPU timeline:
  - Which kernels run and when
  - CPU/GPU overlap
  - NCCL collective timing
  - Memory transfers (H2D, D2H)
  - CUDA stream activity

We use NVTX ranges to annotate the trace.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    # Fallback: use torch.cuda.nvtx
    HAS_NVTX = False

from model import GPT, GPTConfig


def nvtx_range(name):
    """Context manager for NVTX range annotation."""
    class Range:
        def __enter__(self):
            if HAS_NVTX:
                nvtx.push_range(name)
            else:
                torch.cuda.nvtx.range_push(name)
        def __exit__(self, *args):
            if HAS_NVTX:
                nvtx.pop_range()
            else:
                torch.cuda.nvtx.range_pop()
    return Range()


def main():
    # Setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)

    use_dist = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE",1)) > 1
    if use_dist:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
    else:
        rank = 0

    torch.backends.cuda.matmul.allow_tf32 = True

    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=256,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0,
    )
    torch.manual_seed(42)
    model = GPT(cfg).to(device)
    if use_dist:
        model = DDP(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    B, T = 32, 256
    x    = torch.randint(0, 65, (B, T), device=device)
    y    = torch.randint(0, 65, (B, T), device=device)

    # Pre-warmup (not captured by nsys — run nsys with --capture-range)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    # ── Main profiled loop ─────────────────────────────────────────────
    # nsys will capture whatever happens here
    # We annotate with NVTX so the timeline has readable labels

    for step in range(5):
        with nvtx_range(f"step_{step}"):

            with nvtx_range("zero_grad"):
                opt.zero_grad(set_to_none=True)

            with nvtx_range("data_load"):
                # Simulate data loading
                x = torch.randint(0, 65, (B, T), device=device)
                y = torch.randint(0, 65, (B, T), device=device)

            with nvtx_range("forward"):
                with ctx:
                    _, loss = model(x, y)

            with nvtx_range("backward"):
                loss.backward()

            with nvtx_range("optimizer"):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0) \
                    if not use_dist else \
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            torch.cuda.synchronize()

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    import torch.nn as nn
    main()
PY

# Run with Nsight Systems (single GPU)
echo "=== Running Nsight Systems profile ==="

nsys profile \
    --output=profiles/gpt_single_gpu \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --stats=true \
    python nsys_profile_target.py

echo "Profile saved: profiles/gpt_single_gpu.nsys-rep"
echo "Open with: nsys-ui profiles/gpt_single_gpu.nsys-rep"
```

For DDP:
```bash
# Multi-GPU Nsight Systems profile
nsys profile \
    --output=profiles/gpt_ddp_%q{RANK} \
    --trace=cuda,nvtx,osrt,nccl,ucx \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    torchrun --nproc_per_node=4 nsys_profile_target.py
```

---

## 15.7 Lab 5 — Nsight Compute (Kernel Analysis)

```bash
cat > ncu_target.py <<'PY'
"""
Target script for Nsight Compute kernel analysis.

Nsight Compute profiles individual CUDA kernels:
  - Achieved occupancy
  - Memory throughput (% of peak HBM BW)
  - Compute throughput (% of peak TFLOPS)
  - Tensor Core utilization
  - Which metric is the bottleneck (roofline)

We profile specific ops to understand their efficiency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig


def profile_matmul():
    """Profile matrix multiplications of different sizes."""
    device = "cuda"
    torch.cuda.synchronize()

    sizes = [
        (256,  256,  256,  "tiny"),
        (1024, 1024, 1024, "small"),
        (4096, 4096, 4096, "large"),
        (8192, 8192, 8192, "huge"),
    ]

    for M, N, K, label in sizes:
        a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        b = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            c = a @ b
        torch.cuda.synchronize()

        # Profile target
        c = a @ b
        torch.cuda.synchronize()

        print(f"  GEMM {M}×{K}×{N} ({label}): done")


def profile_attention():
    """Profile attention kernel."""
    device = "cuda"
    B, H, T, D = 8, 6, 256, 64   # batch, heads, seq, head_dim

    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)

    # Standard (unfused) attention
    import math
    scale = math.sqrt(D)
    attn  = (q @ k.transpose(-2,-1) / scale).softmax(dim=-1)
    out   = attn @ v
    torch.cuda.synchronize()
    print(f"  Standard attention {B}×{H}×{T}×{D}: done")

    # Flash Attention (if available via PyTorch SDPA)
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        print(f"  Flash attention {B}×{H}×{T}×{D}: done")
    except Exception as e:
        print(f"  Flash attention unavailable: {e}")


def profile_layernorm():
    """Profile LayerNorm (memory-bandwidth-bound op)."""
    device = "cuda"
    shapes = [
        (32, 256, 384),
        (64, 256, 768),
        (16, 1024, 1024),
    ]
    ln = nn.LayerNorm(shapes[0][-1]).to(device)
    for shape in shapes:
        ln = nn.LayerNorm(shape[-1]).to(device).to(torch.bfloat16)
        x  = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        y  = ln(x)
        torch.cuda.synchronize()
        print(f"  LayerNorm {shape}: done")


def main():
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    print("GPU:", torch.cuda.get_device_name(0))

    # Warmup
    _ = torch.randn(1, device="cuda")
    torch.cuda.synchronize()

    print("\nProfiling matmul...")
    profile_matmul()

    print("\nProfiling attention...")
    profile_attention()

    print("\nProfiling LayerNorm...")
    profile_layernorm()

    print("\nProfile target complete.")


if __name__ == "__main__":
    main()
PY

# Run Nsight Compute on specific kernels
echo "=== Nsight Compute: profile all kernels in 1 step ==="

ncu \
    --target-processes all \
    --metrics \
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
        l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
        lts__throughput.avg.pct_of_peak_sustained_elapsed \
    --launch-count 1 \
    --output profiles/gpt_kernels \
    --force-overwrite \
    python ncu_target.py \
    2>&1 | head -100

echo ""
echo "Open report: ncu-ui profiles/gpt_kernels.ncu-rep"
```

Focused kernel analysis:
```bash
# Profile only matmul kernels
ncu \
    --kernel-name "ampere_bf16_s16816gemm.*" \
    --metrics \
        sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
        gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
    --launch-count 5 \
    --output profiles/matmul_kernels \
    python ncu_target.py

# Profile with full roofline analysis
ncu \
    --set roofline \
    --launch-count 3 \
    --output profiles/roofline \
    python ncu_target.py
```

---

## 15.8 Lab 6 — NCCL Debug and Communication Profiling

```bash
cat > nccl_profile.py <<'PY'
"""
Profile NCCL communication in detail.

Measures:
  1. AllReduce latency vs message size (bandwidth curve)
  2. Bucket size impact on DDP communication timing
  3. Comm vs compute overlap ratio
  4. Per-bucket AllReduce timing using hooks
"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict

from model import GPT, GPTConfig


class CommTimer:
    """
    Hook-based timer for DDP communication.
    Attaches to DDP's gradient reduction hooks to time AllReduce calls.
    """
    def __init__(self, model: DDP):
        self.model       = model
        self.comm_times  = defaultdict(list)
        self.bucket_sizes= []
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on DDP reducer buckets."""
        for i, bucket in enumerate(self.model.reducer.buckets):
            bucket_size_mb = sum(
                p.numel() * p.element_size()
                for p in bucket.parameters()
            ) / 1024**2
            self.bucket_sizes.append(bucket_size_mb)

    def reset(self):
        self.comm_times.clear()

    def report(self):
        if not self.comm_times:
            return

        print(f"\n── Bucket Communication Report ─────────────────────────────")
        total_comm = 0.0
        for bucket_id in sorted(self.comm_times):
            times    = self.comm_times[bucket_id]
            avg_ms   = sum(times) / len(times)
            total_comm += avg_ms
            size_mb  = (self.bucket_sizes[bucket_id]
                        if bucket_id < len(self.bucket_sizes) else 0)
            print(f"  Bucket {bucket_id}: "
                  f"size={size_mb:.1f}MB  "
                  f"avg={avg_ms:.2f}ms  "
                  f"n={len(times)}")
        print(f"  Total comm time per step: {total_comm:.2f}ms")


def benchmark_allreduce_bandwidth(world_size: int, device: torch.device):
    """Precise AllReduce bandwidth measurement."""
    sizes_mb = [0.1, 0.5, 1, 5, 10, 25, 50, 100, 256, 512]
    iters    = 30
    warmup   = 10

    results = []

    for smb in sizes_mb:
        numel = int(smb * 1024**2 / 4)   # float32 elements
        x     = torch.ones(numel, device=device, dtype=torch.float32)

        for _ in range(warmup):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        ms  = (time.perf_counter() - t0) * 1000 / iters
        bw  = 2 * smb * (world_size-1) / world_size / (ms/1000) / 1000
        results.append((smb, ms, bw))

    return results


def measure_overlap(model: DDP, opt, x, y, ctx, device, iters=20):
    """
    Measure how well comm overlaps with compute.
    Compare:
      A) step with DDP (comm overlapped with backward)
      B) step with no_sync + manual allreduce (comm sequential)
    """
    def step_with_overlap():
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()   # AllReduce fires here, overlapped with backward
        opt.step()
        return loss.item()

    def step_no_overlap():
        opt.zero_grad(set_to_none=True)
        with model.no_sync():   # suppress AllReduce during backward
            with ctx: _, loss = model(x, y)
            loss.backward()
        # Manual AllReduce AFTER backward (sequential, no overlap)
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        opt.step()
        return loss.item()

    # Warmup both
    for _ in range(3):
        step_with_overlap()
        step_no_overlap()
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark overlap
    t0 = time.perf_counter()
    for _ in range(iters):
        step_with_overlap()
    torch.cuda.synchronize()
    dist.barrier()
    ms_overlap = (time.perf_counter() - t0) * 1000 / iters

    # Benchmark sequential
    t0 = time.perf_counter()
    for _ in range(iters):
        step_no_overlap()
    torch.cuda.synchronize()
    dist.barrier()
    ms_sequential = (time.perf_counter() - t0) * 1000 / iters

    overlap_benefit = (ms_sequential - ms_overlap) / ms_sequential * 100

    return ms_overlap, ms_sequential, overlap_benefit


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    torch.backends.cuda.matmul.allow_tf32 = True

    # ── 1. AllReduce bandwidth curve ──────────────────────────────────────
    if is_main:
        print(f"\n{'='*65}")
        print(f"NCCL AllReduce Bandwidth  (world_size={world_size})")
        print(f"{'='*65}")
        print(f"  {'Size(MB)':>10}  {'avg_ms':>9}  "
              f"{'BW(GB/s)':>10}  {'regime':>15}")
        print(f"  {'─'*55}")

    results = benchmark_allreduce_bandwidth(world_size, device)

    if is_main:
        for smb, ms, bw in results:
            regime = ("latency-bound" if ms < 0.5
                      else "bandwidth-bound" if ms > 2 else "transition")
            print(f"  {smb:>10.1f}  {ms:>9.3f}  {bw:>10.1f}  {regime:>15}")

    # ── 2. Overlap analysis ───────────────────────────────────────────────
    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=256,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0,
    )
    torch.manual_seed(42)
    model = GPT(cfg).to(device)

    B, T  = 32, 256
    x     = torch.randint(0, 65, (B, T), device=device)
    y     = torch.randint(0, 65, (B, T), device=device)
    ctx   = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    for bucket_mb in [1, 5, 25, 100]:
        ddp_model = DDP(
            model, device_ids=[local_rank],
            bucket_cap_mb=bucket_mb
        )
        opt = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4, fused=True)

        ms_ov, ms_seq, benefit = measure_overlap(
            ddp_model, opt, x, y, ctx, device
        )

        if is_main:
            print(f"\n── bucket_cap={bucket_mb}MB ──────────────────────────────")
            print(f"  With overlap:         {ms_ov:.2f}ms")
            print(f"  Sequential (no_sync): {ms_seq:.2f}ms")
            print(f"  Overlap benefit:      {benefit:.1f}% speedup")

        del ddp_model, opt
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 nccl_profile.py
```

---

## 15.9 Lab 7 — Memory Profiler and Snapshot

```bash
cat > memory_profiler.py <<'PY'
"""
Detailed memory profiling:
1. Peak memory per operation
2. Memory timeline (allocated vs reserved over time)
3. PyTorch memory snapshot (full allocation trace)
4. Find memory leaks

All visualizable without external tools.
"""
import gc
import time
import torch
import torch.nn as nn
from typing import List, Tuple

from model import GPT, GPTConfig


# ─── Memory Timeline Tracker ──────────────────────────────────────────────────

class MemoryTracker:
    """
    Track memory allocation over time using forward hooks.
    Logs (op_name, allocated_MB, reserved_MB) after each operation.
    """
    def __init__(self):
        self.log     : List[Tuple[str, float, float]] = []
        self._handles = []

    def start(self, model: nn.Module):
        """Register hooks on all named modules."""
        for name, module in model.named_modules():
            h = module.register_forward_hook(self._make_hook(name))
            self._handles.append(h)

    def _make_hook(self, name):
        def hook(module, inp, out):
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024**2
            resv  = torch.cuda.memory_reserved()  / 1024**2
            self.log.append((name or "root", alloc, resv))
        return hook

    def stop(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def report(self, top_n=15):
        if not self.log:
            return

        print(f"\n── Memory after each module (top {top_n} by delta) ─────────")
        # Compute deltas
        prev_alloc = 0
        deltas     = []
        for name, alloc, resv in self.log:
            delta = alloc - prev_alloc
            deltas.append((name, alloc, resv, delta))
            prev_alloc = alloc

        # Sort by absolute delta
        sorted_d = sorted(deltas, key=lambda x: abs(x[3]), reverse=True)
        print(f"  {'Module':50s}  {'alloc_MB':>10}  {'delta_MB':>10}")
        print(f"  {'─'*75}")
        for name, alloc, resv, delta in sorted_d[:top_n]:
            sign = "+" if delta >= 0 else ""
            print(f"  {name:50s}  {alloc:>10.2f}  {sign}{delta:>9.2f}")

    def peak(self):
        if not self.log:
            return 0
        return max(a for _, a, _ in self.log)


# ─── Per-layer Backward Memory ────────────────────────────────────────────────

class BackwardMemoryTracker:
    """Track memory during backward pass."""
    def __init__(self):
        self.log      = []
        self._handles = []

    def start(self, model: nn.Module):
        for name, module in model.named_modules():
            h = module.register_full_backward_hook(
                self._make_hook(name)
            )
            self._handles.append(h)

    def _make_hook(self, name):
        def hook(module, grad_in, grad_out):
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024**2
            self.log.append((name or "root", alloc))
        return hook

    def stop(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ─── Memory Snapshot ──────────────────────────────────────────────────────────

def take_memory_snapshot(model, B, T, device, save_path="profiles/mem_snapshot.pkl"):
    """
    Full memory allocation trace using PyTorch's memory recorder.
    """
    import torch.cuda.memory as mem

    cfg = model.cfg if hasattr(model, 'cfg') else None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    x   = torch.randint(0, cfg.vocab_size if cfg else 65, (B, T), device=device)
    y   = torch.randint(0, cfg.vocab_size if cfg else 65, (B, T), device=device)

    # Start recording
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # One full training step
    opt.zero_grad(set_to_none=True)
    with ctx:
        _, loss = model(x, y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

    # Dump snapshot
    torch.cuda.memory._dump_snapshot(save_path)
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"\nMemory snapshot saved: {save_path}")
    print("To visualize:")
    print("  python -c \"")
    print("    from torch.cuda._memory_viz import profile_plot")
    print("    import pickle")
    print(f"    snap = pickle.load(open('{save_path}','rb'))")
    print("    print(type(snap))\"")
    print("  Or upload to: https://pytorch.org/memory_viz")


# ─── Memory Leak Detector ─────────────────────────────────────────────────────

class MemoryLeakDetector:
    """
    Detect memory leaks in training loops.
    Runs N steps and checks if allocated memory grows unboundedly.
    """
    def __init__(self, tolerance_mb=5.0):
        self.tolerance_mb = tolerance_mb
        self.baseline_mb  = None

    def check_step(self, step: int):
        torch.cuda.synchronize()
        alloc_mb = torch.cuda.memory_allocated() / 1024**2

        if self.baseline_mb is None:
            self.baseline_mb = alloc_mb
            return

        growth = alloc_mb - self.baseline_mb
        if abs(growth) > self.tolerance_mb:
            print(f"  [LeakDetector] step={step}: "
                  f"alloc grew by {growth:+.1f}MB "
                  f"(baseline={self.baseline_mb:.1f}MB → "
                  f"current={alloc_mb:.1f}MB)")
        else:
            print(f"  [LeakDetector] step={step}: "
                  f"alloc={alloc_mb:.1f}MB  growth={growth:+.1f}MB  OK")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    assert torch.cuda.is_available()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = GPTConfig(
        vocab_size=65, max_seq_len=256,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0,
    )
    model = GPT(cfg).to(device)
    B, T  = 16, 256

    # ── 1. Forward memory tracker ─────────────────────────────────────────
    print("="*60)
    print("1. Forward Pass Memory Tracker")
    print("="*60)

    tracker = MemoryTracker()
    tracker.start(model)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randint(0, 65, (B, T), device=device)
    y = torch.randint(0, 65, (B, T), device=device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)

    tracker.stop()
    tracker.report(top_n=15)
    print(f"\n  Peak allocated: {tracker.peak():.2f} MB")

    # ── 2. Memory summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("2. torch.cuda.memory_summary()")
    print("="*60)
    print(torch.cuda.memory_summary(device=device, abbreviated=True))

    # ── 3. Memory snapshot ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("3. Memory Snapshot")
    print("="*60)
    import os
    os.makedirs("profiles", exist_ok=True)
    take_memory_snapshot(model, B, T, device)

    # ── 4. Leak detection ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("4. Memory Leak Detection")
    print("="*60)

    opt     = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ctx     = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    leak_d  = MemoryLeakDetector(tolerance_mb=2.0)
    losses  = []   # good: use .item()

    # Simulate potential leak — accumulate tensor instead of .item()
    print("\n  BAD: accumulating loss tensors (leak)")
    losses_bad = []
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses_bad.append(loss)   # BAD: keeps autograd graph
        leak_d.check_step(step)

    del losses_bad
    gc.collect()
    torch.cuda.empty_cache()
    leak_d.baseline_mb = None

    print("\n  GOOD: using loss.item()")
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())   # GOOD: Python float, no graph kept
        leak_d.check_step(step)


if __name__ == "__main__":
    main()
PY

python memory_profiler.py
```

---

## 15.10 Lab 8 — FlashAttention vs Standard Attention Profile

```bash
cat > profile_attention_comparison.py <<'PY'
"""
Compare standard attention vs FlashAttention performance.

Standard attention:
  - Materializes full [B, H, T, T] attention matrix
  - Memory: O(T²) — bottleneck for long sequences
  - Memory-bandwidth-bound for most sizes

FlashAttention (SDPA in PyTorch 2.0+):
  - Tiles computation to stay in SRAM
  - Memory: O(T) — no materialized attention matrix
  - Compute-bound
  - 2-4× faster on A100 for typical transformer sizes
"""
import time
import math
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function


def standard_attention(q, k, v, causal=True):
    """Standard scaled dot-product attention (materializes full matrix)."""
    scale  = math.sqrt(q.size(-1))
    scores = q @ k.transpose(-2, -1) / scale   # [B,H,T,T]

    if causal:
        T    = q.size(-2)
        mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    w   = F.softmax(scores, dim=-1)
    return w @ v


def flash_attention(q, k, v, causal=True):
    """PyTorch's optimized SDPA (uses FlashAttention on supported hardware)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def bench(fn, *args, iters=50, warmup=10, label=""):
    """Benchmark a function."""
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0  = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize()
    ms   = (time.perf_counter() - t0) * 1000 / iters
    peak = torch.cuda.max_memory_allocated() / 1024**2

    return ms, peak


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"

    B = 8     # batch size
    H = 12    # attention heads
    D = 64    # head dimension

    print(f"\n{'='*70}")
    print(f"Standard vs FlashAttention (SDPA) Comparison")
    print(f"B={B}  H={H}  D={D}")
    print(f"{'='*70}")
    print(f"{'T':>6}  {'Std_ms':>9}  {'Flash_ms':>9}  "
          f"{'Speedup':>9}  {'Std_MB':>8}  {'Flash_MB':>9}")
    print(f"{'─'*70}")

    for T in [128, 256, 512, 1024, 2048, 4096]:
        q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)

        # Standard attention
        try:
            ms_std, mb_std = bench(
                standard_attention, q, k, v, label="standard"
            )
        except torch.cuda.OutOfMemoryError:
            ms_std, mb_std = float('inf'), float('inf')

        torch.cuda.empty_cache()

        # Flash attention
        try:
            ms_flash, mb_flash = bench(
                flash_attention, q, k, v, label="flash"
            )
        except Exception as e:
            ms_flash, mb_flash = float('inf'), float('inf')

        speedup = ms_std / ms_flash if ms_flash > 0 else float('inf')

        std_str   = f"{ms_std:.2f}" if ms_std < 1e6 else "OOM"
        flash_str = f"{ms_flash:.2f}" if ms_flash < 1e6 else "OOM"
        sp_str    = f"{speedup:.2f}×" if speedup < 1e6 else "N/A"
        std_mb    = f"{mb_std:.0f}" if mb_std < 1e6 else "OOM"
        flash_mb  = f"{mb_flash:.0f}" if mb_flash < 1e6 else "OOM"

        print(f"{T:>6}  {std_str:>9}  {flash_str:>9}  "
              f"{sp_str:>9}  {std_mb:>8}  {flash_mb:>9}")

    # Profile with PyTorch profiler
    print(f"\n── Profiling T=1024 with PyTorch profiler ──────────────────────")
    T  = 1024
    q  = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    k  = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    v  = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)

    with profile(
        activities    = [ProfilerActivity.CUDA],
        record_shapes = True,
    ) as prof:
        with record_function("standard_attention"):
            for _ in range(10):
                out = standard_attention(q, k, v)
        with record_function("flash_attention"):
            for _ in range(10):
                out = flash_attention(q, k, v)

    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=10
    ))

    prof.export_chrome_trace("traces/attention_comparison.json")
    print("Trace: traces/attention_comparison.json")

    # Enable SDPA backend info
    print(f"\n── SDPA backends available ──────────────────────────────────────")
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=True, enable_mem_efficient=True
    ):
        q2 = q.clone()
        try:
            out = F.scaled_dot_product_attention(q2, k, v, is_causal=True)
            print("  Flash + Math + MemEfficient: available")
        except:
            print("  Flash + Math + MemEfficient: ERROR")


if __name__ == "__main__":
    main()
PY

python profile_attention_comparison.py
```

---

## 15.11 Lab 9 — Find and Fix Top Bottlenecks

```bash
cat > find_bottlenecks.py <<'PY'
"""
Systematic bottleneck identification and fix.

Methodology:
  1. Profile baseline
  2. Identify top 3 bottlenecks by time
  3. Apply fix
  4. Re-profile to verify improvement

Bottlenecks we look for:
  A. DataLoader stall (GPU idle waiting for CPU)
  B. Memory-bandwidth-bound ops (LayerNorm, embeddings)
  C. Attention materialization (O(T²) memory)
  D. Optimizer overhead (unfused vs fused)
  E. Gradient sync not overlapped with backward
"""
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.data import DataLoader, TensorDataset

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def make_model_and_data(d_model=384, n_layers=6, seq_len=256, batch_size=32):
    device = "cuda"
    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=seq_len,
        d_model=d_model, n_heads=6, n_layers=n_layers,
        dropout=0.0,
    )
    model = GPT(cfg).to(device)

    text  = open("shakespeare.txt").read()
    tok   = CharTokenizer(text)
    data  = torch.tensor(tok.encode(text), dtype=torch.long)
    x_all = torch.stack([data[i:i+seq_len]
                         for i in range(0, min(1000, len(data)-seq_len))])
    y_all = torch.stack([data[i+1:i+seq_len+1]
                         for i in range(0, min(1000, len(data)-seq_len))])

    return model, x_all, y_all, cfg


def bench_step(model, get_batch, opt, ctx, iters=30, label=""):
    """Benchmark a training step."""
    # Warmup
    for _ in range(5):
        x, y = get_batch()
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        x, y = get_batch()
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    ms      = (time.perf_counter() - t0) * 1000 / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return ms, peak_gb


def main():
    assert torch.cuda.is_available()
    device    = "cuda"
    B, T      = 32, 256
    torch.backends.cuda.matmul.allow_tf32 = True

    model, x_all, y_all, cfg = make_model_and_data(batch_size=B, seq_len=T)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model.count_params()/1e6:.2f}M params")

    results = {}

    # ── Baseline ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BASELINE: fp32, unfused Adam, no compile")
    print(f"{'='*60}")

    opt  = torch.optim.Adam(model.parameters(), lr=1e-4)   # unfused
    ctx  = torch.amp.autocast(device_type="cuda", enabled=False)  # no AMP

    def get_batch_slow():
        """Simulate slow DataLoader (no pin_memory, num_workers=0)."""
        ix = torch.randint(len(x_all), (B,))
        x  = x_all[ix].to(device)   # slow: not pinned, no prefetch
        y  = y_all[ix].to(device)
        return x, y

    ms, peak = bench_step(model, get_batch_slow, opt, ctx, label="baseline")
    results["0_baseline"] = (ms, peak)
    print(f"  step_ms={ms:.2f}  peak_GB={peak:.3f}")

    # ── Fix 1: bf16 AMP ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FIX 1: Add bf16 AMP")
    print(f"{'='*60}")

    ctx  = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    ms, peak = bench_step(model, get_batch_slow, opt, ctx, label="amp")
    results["1_amp"] = (ms, peak)
    print(f"  step_ms={ms:.2f}  peak_GB={peak:.3f}")

    # ── Fix 2: Fused AdamW ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FIX 2: + Fused AdamW")
    print(f"{'='*60}")

    del opt
    opt  = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ms, peak = bench_step(model, get_batch_slow, opt, ctx, label="fused_adam")
    results["2_fused_adam"] = (ms, peak)
    print(f"  step_ms={ms:.2f}  peak_GB={peak:.3f}")

    # ── Fix 3: Fast DataLoader ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FIX 3: + Pin memory + preloaded batches")
    print(f"{'='*60}")

    x_pinned = x_all.pin_memory()
    y_pinned = y_all.pin_memory()

    def get_batch_fast():
        """Fast batch: pinned memory, async transfer."""
        ix = torch.randint(len(x_pinned), (B,))
        x  = x_pinned[ix].to(device, non_blocking=True)
        y  = y_pinned[ix].to(device, non_blocking=True)
        return x, y

    ms, peak = bench_step(model, get_batch_fast, opt, ctx, label="fast_loader")
    results["3_fast_loader"] = (ms, peak)
    print(f"  step_ms={ms:.2f}  peak_GB={peak:.3f}")

    # ── Fix 4: torch.compile ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FIX 4: + torch.compile (takes ~1min to compile)")
    print(f"{'='*60}")

    compiled_model = torch.compile(model, mode="reduce-overhead")
    opt_compiled   = torch.optim.AdamW(
        compiled_model.parameters(), lr=1e-4, fused=True
    )

    # First call triggers compilation (slow, don't time it)
    print("  Compiling (first call)...", flush=True)
    x, y = get_batch_fast()
    opt_compiled.zero_grad(set_to_none=True)
    with ctx: _, loss = compiled_model(x, y)
    loss.backward()
    opt_compiled.step()
    torch.cuda.synchronize()
    print("  Compilation done.")

    ms, peak = bench_step(
        compiled_model, get_batch_fast, opt_compiled, ctx,
        label="compiled"
    )
    results["4_compiled"] = (ms, peak)
    print(f"  step_ms={ms:.2f}  peak_GB={peak:.3f}")

    # ── Summary Table ─────────────────────────────────────────────────────
    baseline_ms   = results["0_baseline"][0]
    baseline_peak = results["0_baseline"][1]

    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':30s}  {'ms':>8}  {'speedup':>9}  "
          f"{'peak_GB':>9}  {'mem%':>7}")
    print(f"{'─'*70}")

    labels = {
        "0_baseline":    "Baseline (fp32, unfused Adam)",
        "1_amp":         "+ bf16 AMP",
        "2_fused_adam":  "+ Fused AdamW",
        "3_fast_loader": "+ Pin memory",
        "4_compiled":    "+ torch.compile",
    }

    for key in sorted(results):
        ms, peak = results[key]
        speedup  = baseline_ms / ms
        mem_pct  = peak / baseline_peak * 100
        print(f"  {labels[key]:28s}  {ms:>8.2f}  "
              f"{speedup:>8.2f}×  {peak:>9.3f}  {mem_pct:>6.1f}%")

    print(f"{'='*70}")
    total_speedup = baseline_ms / results[sorted(results)[-1]][0]
    print(f"\nTotal speedup from all optimizations: {total_speedup:.2f}×")
    print("(Individual contributions are additive in some cases)")


if __name__ == "__main__":
    main()
PY

python find_bottlenecks.py
```

---

## 15.12 Lab 10 — NCCL Debug Environment Variables

```bash
cat > nccl_debug_lab.py <<'PY'
"""
NCCL debugging: use environment variables to diagnose issues.

Demonstrates:
  1. NCCL_DEBUG=INFO: see topology detection and algorithm choice
  2. NCCL_DEBUG_SUBSYS=COLL: see each collective with sizes
  3. NCCL_DEBUG_FILE: redirect NCCL output to file per rank
  4. Common NCCL error patterns and their causes
"""
import os
import torch
import torch.distributed as dist


def test_with_nccl_debug():
    """Run a simple collective with NCCL debug output."""
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    x = torch.ones(1024, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"\nAllReduce result[0]={x[0].item()} (expected {world_size})")
        print(f"NCCL_DEBUG={os.environ.get('NCCL_DEBUG', 'not set')}")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_with_nccl_debug()
PY

# Run with different debug levels
echo "=== NCCL_DEBUG=VERSION (minimal) ==="
NCCL_DEBUG=VERSION \
torchrun --nproc_per_node=2 nccl_debug_lab.py 2>&1 | grep -i "nccl\|NCCL" | head -10

echo ""
echo "=== NCCL_DEBUG=WARN (warnings only) ==="
NCCL_DEBUG=WARN \
torchrun --nproc_per_node=2 nccl_debug_lab.py 2>&1 | grep -i "nccl\|NCCL" | head -10

echo ""
echo "=== NCCL_DEBUG=INFO (topology + init) ==="
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH \
torchrun --nproc_per_node=2 nccl_debug_lab.py 2>&1 | grep -i "nccl\|NCCL" | head -30

echo ""
echo "=== NCCL_DEBUG=INFO (each collective) ==="
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=COLL \
torchrun --nproc_per_node=2 nccl_debug_lab.py 2>&1 | grep -i "nccl\|NCCL" | head -20

echo ""
echo "=== Save NCCL debug to file per rank ==="
NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE="logs/nccl_rank%d.log" \
torchrun --nproc_per_node=2 nccl_debug_lab.py
echo "NCCL logs:"
ls logs/nccl_rank*.log 2>/dev/null && head -20 logs/nccl_rank0.log
```

---

## 15.13 Lab 11 — Profiling Dashboard (Automated Report)

```bash
cat > profiling_report.py <<'PY'
"""
Automated profiling report generator.

Runs your training script, collects metrics, and generates
a readable report with:
  - Step time breakdown (fwd/bwd/opt)
  - Memory usage
  - Throughput
  - Top bottlenecks by CUDA time
  - Recommendations

Run: python profiling_report.py
"""
import time
import json
import torch
import torch.nn as nn
from torch.profiler import (
    profile, ProfilerActivity, schedule, record_function
)

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def run_profiling_session(
    model, get_batch, optimizer, ctx,
    n_steps=20, warmup=5
) -> dict:
    """
    Run profiling session and collect structured metrics.
    Returns a dict of measurements.
    """
    timings = {"forward": [], "backward": [], "optimizer": [], "total": []}
    memories = []

    # Warmup
    for _ in range(warmup):
        x, y = get_batch()
        optimizer.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed steps with manual section timing
    events = {}
    for section in ["forward", "backward", "optimizer"]:
        events[section] = [
            (torch.cuda.Event(enable_timing=True),
             torch.cuda.Event(enable_timing=True))
            for _ in range(n_steps)
        ]

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch()

        # Forward
        events["forward"][step][0].record()
        with ctx: _, loss = model(x, y)
        events["forward"][step][1].record()

        # Backward
        events["backward"][step][0].record()
        loss.backward()
        events["backward"][step][1].record()

        # Optimizer
        events["optimizer"][step][0].record()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        events["optimizer"][step][1].record()

    torch.cuda.synchronize()

    # Extract timings
    for section in ["forward", "backward", "optimizer"]:
        for start, end in events[section]:
            timings[section].append(start.elapsed_time(end))

    # Total
    timings["total"] = [
        timings["forward"][i] + timings["backward"][i] + timings["optimizer"][i]
        for i in range(n_steps)
    ]

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    alloc_gb= torch.cuda.memory_allocated() / 1024**3

    return {
        "timings":      timings,
        "peak_gb":      peak_gb,
        "alloc_gb":     alloc_gb,
    }


def generate_report(model, get_batch, optimizer, ctx, batch_size, seq_len):
    """Generate a complete profiling report."""

    print("\n" + "="*70)
    print("GPT TRAINING PROFILING REPORT")
    print("="*70)
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")
    print(f"Model:  {model.count_params()/1e6:.2f}M params")
    print(f"Batch:  {batch_size}×{seq_len}")

    # Run profiling
    data = run_profiling_session(
        model, get_batch, optimizer, ctx, n_steps=30, warmup=10
    )

    t = data["timings"]

    def stats(lst):
        avg = sum(lst) / len(lst)
        mn  = min(lst)
        mx  = max(lst)
        return avg, mn, mx

    fwd_avg, fwd_min, fwd_max = stats(t["forward"])
    bwd_avg, bwd_min, bwd_max = stats(t["backward"])
    opt_avg, opt_min, opt_max = stats(t["optimizer"])
    tot_avg, tot_min, tot_max = stats(t["total"])

    print(f"\n── Step Time Breakdown ────────────────────────────────────────")
    print(f"  {'Section':20s}  {'avg_ms':>9}  {'min_ms':>9}  {'max_ms':>9}  {'%':>6}")
    print(f"  {'─'*60}")
    for name, avg, mn, mx in [
        ("Forward",   fwd_avg, fwd_min, fwd_max),
        ("Backward",  bwd_avg, bwd_min, bwd_max),
        ("Optimizer", opt_avg, opt_min, opt_max),
        ("Total",     tot_avg, tot_min, tot_max),
    ]:
        pct  = 100 * avg / tot_avg if tot_avg > 0 else 0
        print(f"  {name:20s}  {avg:>9.2f}  {mn:>9.2f}  {mx:>9.2f}  "
              f"{pct:>5.1f}%")

    tok_s = batch_size * seq_len / (tot_avg / 1000)
    print(f"\n── Throughput ─────────────────────────────────────────────────")
    print(f"  Tokens/second:       {tok_s:,.0f}")
    print(f"  Samples/second:      {batch_size / (tot_avg/1000):.1f}")
    print(f"  Step time (ms):      {tot_avg:.2f}")

    print(f"\n── Memory ─────────────────────────────────────────────────────")
    print(f"  Allocated:           {data['alloc_gb']:.3f} GB")
    print(f"  Peak:                {data['peak_gb']:.3f} GB")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Utilization:         {100*data['peak_gb']/vram:.1f}%")

    print(f"\n── Op-Level Profile (top 15) ──────────────────────────────────")

    def get_batch_fresh(): return get_batch()

    with profile(
        activities    = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes = True,
        with_flops    = True,
    ) as prof:
        for _ in range(5):
            x, y = get_batch_fresh()
            optimizer.zero_grad(set_to_none=True)
            with ctx: _, loss = model(x, y)
            loss.backward()
            optimizer.step()

    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=15
    ))

    # Recommendations
    print(f"\n── Recommendations ────────────────────────────────────────────")
    recs = []

    bwd_ratio = bwd_avg / fwd_avg
    if bwd_ratio > 3.0:
        recs.append(f"⚠  Backward/Forward ratio = {bwd_ratio:.1f}× (expected ~2×). "
                     "Check for gradient checkpointing overhead or extra ops in backward.")

    if data["peak_gb"] / vram > 0.95:
        recs.append(f"⚠  GPU memory nearly full ({data['peak_gb']:.2f}/{vram:.0f}GB). "
                     "Enable activation checkpointing or reduce batch size.")

    if data["peak_gb"] / vram < 0.5:
        recs.append(f"✓  Only {data['peak_gb']/vram*100:.0f}% GPU memory used. "
                     "Consider increasing batch size for better throughput.")

    if tot_avg > 200:
        recs.append(f"⚠  Step time {tot_avg:.0f}ms is high. "
                     "Consider torch.compile or larger batch size.")

    if not recs:
        recs.append("✓  No obvious bottlenecks detected. Looks healthy!")

    for r in recs:
        print(f"  {r}")

    # Save report
    report = {
        "gpu":        torch.cuda.get_device_name(0),
        "params_m":   model.count_params() / 1e6,
        "batch":      f"{batch_size}×{seq_len}",
        "fwd_ms":     round(fwd_avg, 3),
        "bwd_ms":     round(bwd_avg, 3),
        "opt_ms":     round(opt_avg, 3),
        "total_ms":   round(tot_avg, 3),
        "tok_s":      round(tok_s),
        "peak_gb":    round(data["peak_gb"], 3),
    }
    with open("profiles/training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report: profiles/training_report.json")
    print("="*70)


def main():
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    B, T   = 32, 256

    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=T,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0,
    )
    model = GPT(cfg).to(device)

    text  = open("shakespeare.txt").read()
    tok   = CharTokenizer(text)
    data  = torch.tensor(tok.encode(text), dtype=torch.long)

    def get_batch():
        ix = torch.randint(len(data) - T, (B,))
        x  = torch.stack([data[i:i+T] for i in ix]).to(device)
        y  = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
        return x, y

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    generate_report(model, get_batch, opt, ctx, B, T)


if __name__ == "__main__":
    import os
    os.makedirs("profiles", exist_ok=True)
    main()
PY

python profiling_report.py
```

---

## 15.14 Lab 12 — End-to-End: Profile GPT DDP and Fix Bottlenecks

```bash
cat > profile_gpt_ddp_full.py <<'PY'
"""
End-to-end profiling of GPT DDP training.

Produces:
  1. TensorBoard trace (per-rank)
  2. NCCL bandwidth curve
  3. Comm overhead measurement
  4. Memory profile
  5. Automated bottleneck report

This is the "full picture" profiling session.
"""
import os, time, gc
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import (
    profile, ProfilerActivity, schedule, tensorboard_trace_handler,
    record_function,
)

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42 + rank)

    B, T  = 32, 256

    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=T,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0,
    )
    torch.manual_seed(42)
    model = GPT(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ctx   = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    text = open("shakespeare.txt").read()
    tok  = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    def get_batch():
        ix = torch.randint(len(data)-T, (B,))
        x  = torch.stack([data[i:i+T] for i in ix]).to(device, non_blocking=True)
        y  = torch.stack([data[i+1:i+T+1] for i in ix]).to(device, non_blocking=True)
        return x, y

    # ── Pre-warmup ────────────────────────────────────────────────────────
    for _ in range(5):
        x, y = get_batch()
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()

    # ── Profile: schedule-based ───────────────────────────────────────────
    trace_dir = f"traces/ddp_full/rank{rank}"
    sched     = schedule(wait=2, warmup=2, active=5, repeat=1)

    with profile(
        activities      = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule        = sched,
        on_trace_ready  = tensorboard_trace_handler(trace_dir),
        record_shapes   = True,
        profile_memory  = True,
        with_flops      = True,
    ) as prof:
        for step in range(9):
            with record_function("zero_grad"):
                opt.zero_grad(set_to_none=True)
            with record_function("data"):
                x, y = get_batch()
            with record_function("forward"):
                with ctx: _, loss = model(x, y)
            with record_function("backward"):
                loss.backward()
            with record_function("clip_and_step"):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            dist.barrier()
            prof.step()

    # ── Comm overhead measurement ─────────────────────────────────────────
    iters = 20

    # With DDP (comm overlapped)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        x, y = get_batch()
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()
    ms_with_comm = (time.perf_counter() - t0) * 1000 / iters
    peak_gb      = torch.cuda.max_memory_allocated() / 1024**3

    # Without comm (no_sync)
    t0 = time.perf_counter()
    for _ in range(iters):
        x, y = get_batch()
        opt.zero_grad(set_to_none=True)
        with model.no_sync():
            with ctx: _, loss = model(x, y)
            loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()
    ms_no_comm = (time.perf_counter() - t0) * 1000 / iters

    comm_ms  = max(0, ms_with_comm - ms_no_comm)
    comm_pct = 100 * comm_ms / ms_with_comm
    tok_s    = B * T * world_size / (ms_with_comm / 1000)

    if is_main:
        print(f"\n{'='*65}")
        print(f"GPT DDP Profiling Report — world_size={world_size}")
        print(f"{'='*65}")
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        print(f"  params:          {model.module.count_params()/1e6:.2f}M")
        print(f"  batch:           {B}×{T}")
        print(f"  global_batch:    {B*world_size}×{T}")
        print(f"")
        print(f"  step_ms (full):  {ms_with_comm:.2f}ms")
        print(f"  step_ms (no AR): {ms_no_comm:.2f}ms")
        print(f"  comm_overhead:   {comm_ms:.2f}ms  ({comm_pct:.1f}%)")
        print(f"  throughput:      {tok_s:,.0f} tokens/s")
        print(f"  peak_memory:     {peak_gb:.3f}GB")
        print(f"")
        print(f"  Traces saved to: traces/ddp_full/rank*/")
        print(f"  View: tensorboard --logdir traces/ddp_full --port 6006")
        print(f"{'='*65}")

        # Automated assessment
        print(f"\n── Automated Assessment ───────────────────────────────────────")
        if comm_pct < 10:
            print(f"  ✓ Communication overhead is low ({comm_pct:.1f}%)")
        elif comm_pct < 25:
            print(f"  ~ Communication overhead is moderate ({comm_pct:.1f}%)")
            print(f"    Consider: larger bucket_cap_mb, gradient accumulation")
        else:
            print(f"  ✗ Communication overhead is HIGH ({comm_pct:.1f}%)")
            print(f"    Actions: reduce model size, use FSDP/ZeRO, gradient compression")

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if peak_gb / vram_gb < 0.5:
            print(f"  ~ Memory underutilized ({peak_gb:.1f}/{vram_gb:.0f}GB). "
                  f"Increase batch size.")
        elif peak_gb / vram_gb < 0.85:
            print(f"  ✓ Memory well utilized ({peak_gb:.1f}/{vram_gb:.0f}GB)")
        else:
            print(f"  ✗ Near OOM ({peak_gb:.1f}/{vram_gb:.0f}GB). "
                  f"Enable gradient checkpointing or reduce batch.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=8 profile_gpt_ddp_full.py
```

---

## 15.15 Complete Profiling Cheat Sheet

```bash
cat > profiling_cheatsheet.py <<'PY'
"""Prints the complete profiling reference."""

cheatsheet = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    PROFILING CHEAT SHEET                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  TOOL SELECTION                                                          ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Question                          → Tool                                ║
║  Where is time spent? (ops)        → PyTorch Profiler                   ║
║  GPU/CPU overlap? (timeline)       → Nsight Systems (nsys)              ║
║  Kernel efficiency? (occupancy)    → Nsight Compute (ncu)               ║
║  NCCL issue?                       → NCCL_DEBUG=INFO + nsys             ║
║  Memory allocation?                → PyTorch memory snapshot            ║
║  Memory leak?                      → MemoryLeakDetector + torch.cuda.   ║
║                                      memory_summary()                   ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  PYTORCH PROFILER (quick start)                                          ║
║  ─────────────────────────────────────────────────────────────────────  ║
║                                                                          ║
║  from torch.profiler import profile, ProfilerActivity, schedule         ║
║                                                                          ║
║  # One-shot                                                              ║
║  with profile(activities=[ProfilerActivity.CUDA],                       ║
║               profile_memory=True) as p:                                ║
║      train_step(...)                                                     ║
║  print(p.key_averages().table(sort_by="cuda_time_total", row_limit=15)) ║
║  p.export_chrome_trace("trace.json")                                    ║
║                                                                          ║
║  # Schedule (production)                                                 ║
║  sched = schedule(wait=2, warmup=2, active=5)                           ║
║  with profile(..., schedule=sched,                                       ║
║               on_trace_ready=tensorboard_trace_handler("dir")) as p:    ║
║      for step in range(9):                                               ║
║          train_step(...)                                                 ║
║          p.step()                                                        ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  NSIGHT SYSTEMS (timeline)                                               ║
║  ─────────────────────────────────────────────────────────────────────  ║
║                                                                          ║
║  nsys profile \\                                                         ║
║      --output=profiles/run \\                                            ║
║      --trace=cuda,nvtx,nccl \\                                           ║
║      --cuda-memory-usage=true \\                                         ║
║      python train.py                                                     ║
║                                                                          ║
║  # DDP (profile all ranks)                                               ║
║  nsys profile --output=profiles/rank_%q{RANK} ... torchrun ...          ║
║                                                                          ║
║  # Open: nsys-ui profiles/run.nsys-rep                                  ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  NSIGHT COMPUTE (kernel analysis)                                        ║
║  ─────────────────────────────────────────────────────────────────────  ║
║                                                                          ║
║  ncu \\                                                                  ║
║      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\      ║
║               gpu__compute_memory_throughput.avg.pct_of_peak... \\      ║
║      --set roofline \\                                                   ║
║      --launch-count 5 \\                                                 ║
║      --output profiles/kernels \\                                        ║
║      python train.py                                                     ║
║                                                                          ║
║  # Open: ncu-ui profiles/kernels.ncu-rep                                ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  NCCL DEBUG                                                              ║
║  ─────────────────────────────────────────────────────────────────────  ║
║                                                                          ║
║  NCCL_DEBUG=VERSION          minimal version info                       ║
║  NCCL_DEBUG=WARN             only warnings                              ║
║  NCCL_DEBUG=INFO             topology + algorithm selection             ║
║  NCCL_DEBUG_SUBSYS=INIT      initialization details                    ║
║  NCCL_DEBUG_SUBSYS=COLL      each collective call                       ║
║  NCCL_DEBUG_SUBSYS=GRAPH     communication graph                        ║
║  NCCL_DEBUG_FILE=logs/r%d.log  save per-rank                           ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  MEMORY PROFILING                                                        ║
║  ─────────────────────────────────────────────────────────────────────  ║
║                                                                          ║
║  torch.cuda.memory_allocated()    current usage                         ║
║  torch.cuda.memory_reserved()     allocator cache                       ║
║  torch.cuda.max_memory_allocated() peak since last reset                ║
║  torch.cuda.reset_peak_memory_stats()  reset peak counter               ║
║  torch.cuda.empty_cache()         release allocator cache               ║
║  torch.cuda.memory_summary()      full allocator stats                  ║
║                                                                          ║
║  # Full snapshot                                                         ║
║  torch.cuda.memory._record_memory_history(max_entries=100000)           ║
║  # ... run code ...                                                      ║
║  torch.cuda.memory._dump_snapshot("snap.pkl")                           ║
║  # upload snap.pkl to https://pytorch.org/memory_viz                    ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  BOTTLENECK DIAGNOSIS FLOWCHART                                          ║
║  ─────────────────────────────────────────────────────────────────────  ║
║                                                                          ║
║  GPU util < 50%? → CPU/DataLoader bound                                 ║
║    Fix: more workers, pin_memory, prefetch, faster storage              ║
║                                                                          ║
║  GPU util ~100% but slow? → Memory-BW or compute bound                  ║
║    → Run ncu roofline to distinguish                                    ║
║    Memory-BW: fuse kernels (torch.compile), use bf16                    ║
║    Compute:   already good, scale batch or GPUs                         ║
║                                                                          ║
║  comm_overhead > 20%? → Communication bound                             ║
║    Fix: overlap comm with backward, reduce model params,                ║
║          try FSDP, gradient compression                                 ║
║                                                                          ║
║  Memory near full? → Memory capacity bound                              ║
║    Fix: activation checkpointing, gradient accumulation,                ║
║          FSDP, reduce batch                                              ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
print(cheatsheet)
PY

python profiling_cheatsheet.py
```

---

## 15.16 Module 15 Pass Criteria

You're done when you can:

1) **Explain the roofline model**:
   - What "GPU utilization" really means (and why 99% isn't enough)
   - Compute-bound vs memory-bandwidth-bound
   - A100's ridge point (FLOPs/byte)

2) **Use PyTorch Profiler**:
   - One-shot profiling (3 lines of code)
   - Schedule-based profiling (wait/warmup/active)
   - Export Chrome trace and TensorBoard trace
   - Read op table: what `self_cuda_time_total` means

3) **Use Nsight Systems**:
   - `nsys profile` command for single GPU and DDP
   - Read timeline: identify comm/compute overlap
   - Use NVTX ranges to label your code sections

4) **Use Nsight Compute**:
   - `ncu` command with key metrics
   - Read occupancy, memory BW%, compute BW%
   - Identify if a kernel is memory or compute bound

5) **Diagnose NCCL issues**:
   - Set `NCCL_DEBUG=INFO` and read topology output
   - Measure AllReduce bandwidth at different message sizes
   - Quantify comm overhead with `no_sync()` experiment

6) **Run and interpret all labs**:
   - `python profile_basic.py` → op table
   - `python profile_training.py` → training trace
   - `torchrun --nproc_per_node=4 profile_ddp_comm.py` → comm trace
   - `torchrun --nproc_per_node=4 nccl_profile.py` → bandwidth + overlap
   - `python find_bottlenecks.py` → optimization cascade
   - `python profiling_report.py` → automated report

7) **Fill your bottleneck table**:

| Optimization | step_ms | speedup | peak_GB | mem% |
|---|---|---|---|---|
| Baseline (fp32) | | 1.0× | | 100% |
| + bf16 AMP | | ? | | |
| + fused AdamW | | ? | | |
| + pin_memory | | ? | | |
| + torch.compile | | ? | | |

---
