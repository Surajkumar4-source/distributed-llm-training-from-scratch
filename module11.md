# Module 11 — Memory Optimization

## Summary
Memory is the #1 constraint in LLM training. A model that doesn't fit in GPU memory simply cannot train. This module teaches you exactly where every byte goes, and gives you precise tools to reduce memory usage without sacrificing training quality.

- **Theory**: the 4 memory components (parameters, gradients, optimizer states, activations), mixed precision internals, activation recomputation
- **From scratch**: measure each component separately, implement gradient checkpointing manually
- **Tools**: `torch.cuda.memory_*`, memory snapshots, profiler traces
- **Experiments**: measure memory reduction from each technique individually
- **Production**: combine all techniques, fit 4× bigger model on same GPU

---

## 11.1 Theory — The 4 Components of Training Memory

### The complete memory budget

For a model with **N parameters** during training:

```
Component              Size (bytes)           Notes
─────────────────────────────────────────────────────────────
Parameters             N × 4                  fp32 master copy
Gradients              N × 4                  same dtype as params
Optimizer states       N × 8                  Adam: m + v (2 × fp32)
Activations            variable               depends on batch, seq_len, layers
─────────────────────────────────────────────────────────────
Minimum (fp32+Adam):   N × 16  +  activations
```

For our GPT (~10M params):
```
Parameters:      10M × 4 = 40 MB
Gradients:       10M × 4 = 40 MB
Adam states:     10M × 8 = 80 MB
─────────────────────────────────
Model overhead:          160 MB   (before any batch)
```

For GPT-2 (124M params):
```
Parameters:      124M × 4 = 496 MB
Gradients:                = 496 MB
Adam states:               = 992 MB
─────────────────────────────────────
Model overhead:           1.984 GB
```

For LLaMA-7B (7B params):
```
Parameters:      7B × 4 = 28 GB      ← already > 1 A100 40GB!
Gradients:               = 28 GB
Adam states:              = 56 GB
─────────────────────────────────────
Model overhead:          112 GB       ← needs 3+ A100 80GB just for states
```

This is why LLMs need FSDP, ZeRO, and model parallelism.

### Component 1: Parameters

Model weights stored as tensors on GPU.

```
For each nn.Linear(in, out):
  weight: [out, in]  → out × in × bytes_per_element
  bias:   [out]      → out × bytes_per_element
```

For our GPT (d_model=384, n_layers=6):
```
token_emb:     65 × 384         =   24,960  params
pos_emb:      256 × 384         =   98,304  params
per block:
  c_attn:     384 × (3×384)     =  442,368  params
  c_proj:     384 × 384         =  147,456  params
  ffn W1:     384 × 1536        =  589,824  params
  ffn W2:    1536 × 384         =  589,824  params
  LN1,LN2:    384 × 2 × 2       =    1,536  params
per block total:               1,771,008   params
6 blocks:                     10,626,048   params
lm_head:   tied to token_emb   (0 extra)
─────────────────────────────────────────────────────
Total:                        ~10.75M params
fp32:  10.75M × 4 =  43 MB
bf16:  10.75M × 2 =  21.5 MB
```

### Component 2: Gradients

One gradient tensor per parameter tensor.
Same shape and dtype as the parameter (usually fp32 for master gradients).

```
set_to_none=True  → grad=None when zeroed (saves memory between steps)
set_to_none=False → grad=zeros (wastes memory: always allocated)
```

Always use `optimizer.zero_grad(set_to_none=True)`.

### Component 3: Optimizer States

**SGD** (momentum):
```
1 extra tensor per param (momentum buffer)
extra memory = N × 4 bytes = 1× params
```

**Adam / AdamW**:
```
2 extra tensors per param (m = 1st moment, v = 2nd moment)
extra memory = N × 4 × 2 = 2× params = 8 bytes/param
```

**Adafactor** (memory-efficient alternative):
```
Uses factored representation of 2nd moment
extra memory ≈ 0.1–0.5× params
Useful for very large models
```

### Component 4: Activations (the hardest to control)

Activations = intermediate tensors stored during forward pass for use in backward.

For a single transformer block, the main tensors saved are:
```
Input to block:        [B, T, d_model]
After LN1:             [B, T, d_model]
Q, K, V:               [B, T, d_model] × 3
Attention scores:      [B, n_heads, T, T]
Attention weights:     [B, n_heads, T, T]
Attention output:      [B, T, d_model]
After LN2:             [B, T, d_model]
FFN hidden:            [B, T, 4×d_model]
FFN output:            [B, T, d_model]
```

Activation memory for one block (fp32):
```
d_model=384, T=256, B=64:

[B,T,d_model] tensors:  64 × 256 × 384 × 4 = 25.2 MB each
[B,nh,T,T] tensors:     64 × 6 × 256 × 256 × 4 = 100.7 MB each
FFN hidden:             64 × 256 × 1536 × 4 = 100.7 MB
```

With 6 layers, activations can easily dwarf parameter memory.

**Key insight**: activations scale with `B × T`, parameters don't.
Double batch → double activation memory. Double sequence → double activation memory.

---

## 11.2 Theory — Mixed Precision

### Float32 (fp32)
- 1 sign, 8 exponent, 23 mantissa bits
- Range: ~1.2e-38 to 3.4e38
- Standard training dtype
- 4 bytes per element

### Float16 (fp16)
- 1 sign, 5 exponent, 10 mantissa bits
- Range: ~6e-5 to 65504 (very narrow!)
- **Problem**: loss scaling required (gradients easily underflow/overflow)
- 2 bytes per element
- Requires GradScaler

### BFloat16 (bf16)
- 1 sign, 8 exponent, 7 mantissa bits
- Same range as fp32 (same exponent width)
- Less precision than fp16 but much more numerically stable
- **No loss scaling needed**
- 2 bytes per element
- Available on A100, H100 (not on V100)
- **Preferred dtype on A100**

### TF32 (TensorFloat-32)
- Only exists inside Tensor Cores (not a storage format)
- 10-bit mantissa (like fp16) but 8-bit exponent (like fp32)
- Tensor Core matmuls use TF32 internally with fp32 I/O
- Enabled with `torch.backends.cuda.matmul.allow_tf32 = True`
- ~8× faster than fp32 on A100, same numerical range
- No code changes needed (transparent)

### Mixed Precision Training flow (AMP)

```
Forward pass:
  weights (fp32 master) → cast to bf16/fp16
  compute in bf16/fp16  → fast Tensor Core ops
  activations stored in bf16/fp16 (2× memory savings)

Loss scaling (fp16 only):
  loss × scale_factor   → prevents gradient underflow
  gradients ÷ scale_factor after backward

Backward pass:
  gradients in bf16/fp16
  cast to fp32 for optimizer

Optimizer step:
  fp32 master weights updated
  fp32 momentum / variance updated
```

Memory with AMP (bf16):
```
fp32 master params:   N × 4 = 4 bytes
bf16 compute params:  N × 2 = 2 bytes  (created on-the-fly)
fp32 gradients:       N × 4 = 4 bytes
fp32 Adam m + v:      N × 8 = 8 bytes
─────────────────────────────────────
Total: 18 bytes/param
vs fp32: 16 bytes/param  (not huge for params!)

But activations:
fp32:  B×T×d_model×4 = 4 bytes/element
bf16:  B×T×d_model×2 = 2 bytes/element   ← 2× savings!
```

The biggest savings from mixed precision is **activation memory**, not parameter memory.

### GradScaler (fp16 specific)

fp16 gradients underflow to 0 below ~6e-5.
GradScaler multiplies loss by a large scale factor (e.g., 65536), preventing underflow.
After backward, gradients are unscaled before optimizer step.

```python
scaler = torch.cuda.amp.GradScaler()

with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(x, y)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)                    # unscale grads
torch.nn.utils.clip_grad_norm_(params, 1.0)  # clip after unscale
scaler.step(optimizer)
scaler.update()                               # adjust scale factor
```

For bf16: no GradScaler needed. Use `contextlib.nullcontext()` or just `enabled=False`.

---

## 11.3 Theory — Activation Checkpointing (Gradient Checkpointing)

### The problem
During backward, PyTorch needs the activations from forward to compute gradients.
By default, ALL activations are kept in memory from forward until backward completes.

For deep transformers, this is enormous:
```
n_layers=24 (GPT-2 large)
B=32, T=1024, d_model=1280

Activations per layer: ~300 MB
Total activations:     24 × 300 = 7.2 GB  ← just activations!
```

### The solution: recomputation
Throw away activations after forward. Recompute them during backward when needed.

```
Standard:
  forward:  save ALL activations → use in backward
  memory:   O(n_layers)  (grows with depth)

Checkpointing:
  forward:  save ONLY checkpoint inputs (one per block)
  backward: recompute block's activations from checkpoint input on the fly
  memory:   O(1) per layer group  (constant for full model!)
  cost:     ~33% extra compute (one extra forward per checkpoint)
```

### Granularity matters

```
Checkpoint every layer:
  memory: O(1)  (minimum)
  extra compute: 33% more
  
Checkpoint every K layers:
  memory: O(K)
  extra compute: 33%/K more (less overhead, more memory)
  
No checkpointing:
  memory: O(n_layers)
  extra compute: 0%
```

In practice: checkpoint every transformer block = best tradeoff.

### PyTorch API

```python
from torch.utils.checkpoint import checkpoint

# Without checkpointing
x = transformer_block(x)

# With checkpointing
x = checkpoint(transformer_block, x, use_reentrant=False)
```

`use_reentrant=False` is recommended (doesn't interfere with torch.compile).

---

## 11.4 Setup

```bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course
mkdir -p memory_module
cp gpt_scratch/tokenizer.py              memory_module/
cp gpt_scratch/attention.py              memory_module/
cp gpt_scratch/multihead_attention.py    memory_module/
cp gpt_scratch/feedforward.py            memory_module/
cp gpt_scratch/transformer_block.py      memory_module/
cp gpt_scratch/gpt.py                    memory_module/
cp gpt_scratch/shakespeare.txt           memory_module/
cd memory_module
```

---

## 11.5 Lab 1 — Measure Each Memory Component in Isolation

```bash
cat > measure_memory_components.py <<'PY'
"""
Measure exactly how much memory each training component uses.
Isolate: params, gradients, optimizer states, activations.
"""
import torch
import torch.nn as nn
import gc

from gpt import GPT, GPTConfig


def bytes_to_mb(b):  return b / 1024**2
def bytes_to_gb(b):  return b / 1024**3


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def snapshot(label, device="cuda"):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    resv  = torch.cuda.memory_reserved()
    peak  = torch.cuda.max_memory_allocated()
    print(f"  [{label}]")
    print(f"    allocated: {bytes_to_mb(alloc):8.2f} MB")
    print(f"    reserved:  {bytes_to_mb(resv):8.2f} MB")
    print(f"    peak:      {bytes_to_mb(peak):8.2f} MB")
    return alloc


def measure_all(
    batch_size:    int,
    seq_len:       int,
    d_model:       int   = 384,
    n_heads:       int   = 6,
    n_layers:      int   = 6,
    dtype:         str   = "float32",
    use_amp:       bool  = False,
    use_ckpt:      bool  = False,
    optimizer_type:str   = "adamw",
):
    device = "cuda"
    reset()

    print(f"\n{'='*65}")
    print(f"Config: B={batch_size} T={seq_len} d={d_model} "
          f"L={n_layers} dtype={dtype} amp={use_amp} ckpt={use_ckpt}")
    print(f"{'='*65}")

    # ── Baseline (empty GPU) ──────────────────────────────────────────────
    base = snapshot("baseline (empty)")

    # ── Parameters only ──────────────────────────────────────────────────
    cfg   = GPTConfig(vocab_size=65, max_seq_len=seq_len,
                      d_model=d_model, n_heads=n_heads,
                      n_layers=n_layers, dropout=0.0)
    torch.manual_seed(42)
    model = GPT(cfg)

    if dtype == "float16":
        model = model.half()
    elif dtype == "bfloat16":
        model = model.to(torch.bfloat16)

    model = model.to(device)
    after_params = snapshot("after model.to(device)")
    param_mem = after_params - base

    n_params = model.count_params()
    print(f"    n_params: {n_params/1e6:.2f}M")
    print(f"    theory:   {n_params * (2 if dtype != 'float32' else 4) / 1024**2:.2f} MB")
    print(f"    actual:   {bytes_to_mb(param_mem):.2f} MB")

    # ── Optimizer states ──────────────────────────────────────────────────
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif optimizer_type == "adafactor":
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(model.parameters(), scale_parameter=True,
                                  relative_step=True)
        except ImportError:
            print("    [Adafactor requires transformers, using AdamW]")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Optimizer states only exist after first step
    x = torch.randint(0, 65, (batch_size, seq_len), device=device)
    y = torch.randint(0, 65, (batch_size, seq_len), device=device)

    reset()
    torch.cuda.reset_peak_memory_stats()
    before_fwd = torch.cuda.memory_allocated()

    # ── Forward ───────────────────────────────────────────────────────────
    amp_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else torch.cuda.amp.autocast(enabled=False))

    with amp_ctx:
        _, loss = model(x, y)

    after_fwd = torch.cuda.memory_allocated()
    act_mem   = after_fwd - before_fwd
    snapshot("after forward")
    print(f"    activation memory:  {bytes_to_mb(act_mem):.2f} MB")

    # ── Backward ─────────────────────────────────────────────────────────
    loss.backward()
    after_bwd = torch.cuda.memory_allocated()
    grad_mem  = after_bwd - after_fwd
    snapshot("after backward")
    print(f"    gradient memory:    {bytes_to_mb(after_bwd - before_fwd - act_mem):.2f} MB")

    # ── Optimizer step ────────────────────────────────────────────────────
    optimizer.step()
    after_step = torch.cuda.memory_allocated()
    opt_mem    = after_step - after_bwd
    snapshot("after optimizer.step()")
    print(f"    optimizer state:    {bytes_to_mb(opt_mem):.2f} MB")

    # ── Summary ───────────────────────────────────────────────────────────
    total = torch.cuda.max_memory_allocated()
    print(f"\n  ── SUMMARY ─────────────────────────────────────────────")
    print(f"    parameters:    {bytes_to_mb(param_mem):8.2f} MB")
    print(f"    activations:   {bytes_to_mb(act_mem):8.2f} MB")
    print(f"    gradients:     {bytes_to_mb(grad_mem):8.2f} MB")
    print(f"    optimizer:     {bytes_to_mb(opt_mem):8.2f} MB")
    print(f"    PEAK TOTAL:    {bytes_to_gb(total):8.3f} GB")
    print(f"  ────────────────────────────────────────────────────────")

    # Cleanup
    del model, optimizer, x, y, loss
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.is_available()

    # ── Experiment 1: baseline fp32 ──────────────────────────────────────
    measure_all(batch_size=32, seq_len=256,
                dtype="float32", use_amp=False, use_ckpt=False)

    # ── Experiment 2: bfloat16 (AMP) ─────────────────────────────────────
    measure_all(batch_size=32, seq_len=256,
                dtype="float32", use_amp=True, use_ckpt=False)

    # ── Experiment 3: larger batch to stress activations ──────────────────
    measure_all(batch_size=64, seq_len=256,
                dtype="float32", use_amp=False, use_ckpt=False)

    # ── Experiment 4: longer sequence to stress attention activations ──────
    measure_all(batch_size=32, seq_len=512,
                dtype="float32", use_amp=False, use_ckpt=False)

    # ── Experiment 5: SGD vs AdamW optimizer state ────────────────────────
    print("\n\n── Optimizer comparison: SGD vs AdamW (same model+batch) ──")
    measure_all(batch_size=32, seq_len=256,
                dtype="float32", use_amp=False, optimizer_type="sgd")
    measure_all(batch_size=32, seq_len=256,
                dtype="float32", use_amp=False, optimizer_type="adamw")
PY

python measure_memory_components.py
```

---

## 11.6 Lab 2 — Mixed Precision Deep Dive

```bash
cat > mixed_precision_lab.py <<'PY'
"""
Deep dive into mixed precision:
1. Compare fp32 vs fp16 vs bf16 memory and speed
2. Demonstrate GradScaler for fp16
3. Show numerical differences between dtypes
4. Measure Tensor Core utilization
"""
import time, gc
import torch
import torch.nn as nn

from gpt import GPT, GPTConfig


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench_dtype(dtype_str, batch_size=32, seq_len=256, iters=50):
    """
    Benchmark training step for a given dtype.
    Returns: (step_ms, peak_gb, final_loss)
    """
    reset()
    device = "cuda"

    cfg   = GPTConfig(vocab_size=65, max_seq_len=seq_len,
                      d_model=384, n_heads=6, n_layers=6, dropout=0.0)
    torch.manual_seed(42)
    model = GPT(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    x = torch.randint(0, 65, (batch_size, seq_len), device=device)
    y = torch.randint(0, 65, (batch_size, seq_len), device=device)

    # ── Select context and scaler ─────────────────────────────────────────
    if dtype_str == "float32":
        ctx    = torch.amp.autocast(device_type="cuda", enabled=False)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    elif dtype_str == "float16":
        ctx    = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    elif dtype_str == "bfloat16":
        ctx    = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    # ── Warmup ───────────────────────────────────────────────────────────
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    reset()

    # ── Benchmark ─────────────────────────────────────────────────────────
    losses = []
    t0     = time.perf_counter()

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000 / iters
    peak_gb    = torch.cuda.max_memory_allocated() / 1024**3

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    return elapsed_ms, peak_gb, losses[-1]


def demonstrate_gradscaler():
    """Show GradScaler protecting against fp16 gradient underflow."""
    print("\n── GradScaler demonstration ────────────────────────────────────")
    device = "cuda"

    cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                      d_model=128, n_heads=4, n_layers=2, dropout=0.0)
    model = GPT(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x     = torch.randint(0, 65, (4, 64), device=device)
    y     = torch.randint(0, 65, (4, 64), device=device)

    # ── Without GradScaler (fp16, may underflow) ──────────────────────────
    ctx_fp16 = torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    opt.zero_grad(set_to_none=True)
    with ctx_fp16:
        _, loss = model(x, y)
    loss.backward()

    # Check for zero / nan gradients
    grad_norms    = [p.grad.norm().item()
                     for p in model.parameters() if p.grad is not None]
    n_zero_grads  = sum(1 for g in grad_norms if g == 0.0)
    n_nan_grads   = sum(1 for g in grad_norms if g != g)   # nan check

    print(f"  Without scaler fp16: "
          f"zero_grads={n_zero_grads}/{len(grad_norms)} "
          f"nan_grads={n_nan_grads}")

    # ── With GradScaler ───────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler()
    opt.zero_grad(set_to_none=True)
    with ctx_fp16:
        _, loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)

    grad_norms2   = [p.grad.norm().item()
                     for p in model.parameters() if p.grad is not None]
    n_zero_grads2 = sum(1 for g in grad_norms2 if g == 0.0)
    n_nan_grads2  = sum(1 for g in grad_norms2 if g != g)

    print(f"  With scaler    fp16: "
          f"zero_grads={n_zero_grads2}/{len(grad_norms2)} "
          f"nan_grads={n_nan_grads2}")
    print(f"  Scaler scale factor: {scaler.get_scale()}")
    del model, opt


def show_dtype_numerical_range():
    """
    Show the numerical range of each dtype.
    Demonstrates why fp16 overflows but bf16 doesn't.
    """
    print("\n── Dtype numerical properties ───────────────────────────────────")

    dtypes = [
        (torch.float32,  "fp32"),
        (torch.float16,  "fp16"),
        (torch.bfloat16, "bf16"),
    ]

    for dt, name in dtypes:
        info = torch.finfo(dt)
        print(f"  {name}:  "
              f"max={info.max:.2e}  "
              f"min={info.tiny:.2e}  "
              f"eps={info.eps:.2e}  "
              f"bits={info.bits}")

    print()
    # Show overflow for fp16 but not bf16
    big_value = torch.tensor(70000.0)  # above fp16 max (65504)
    print(f"  Value 70000.0:")
    print(f"    fp32: {big_value.to(torch.float32).item()}")
    print(f"    fp16: {big_value.to(torch.float16).item()}  "
          f"(overflow → inf!)")
    print(f"    bf16: {big_value.to(torch.bfloat16).item()}  "
          f"(ok, same range as fp32)")


def show_tf32():
    """Demonstrate TF32 speedup on A100."""
    print("\n── TF32 experiment ──────────────────────────────────────────────")

    device = "cuda"
    n      = 4096

    a = torch.randn(n, n, device=device, dtype=torch.float32)
    b = torch.randn(n, n, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()

    def bench_matmul(tf32_enabled, iters=100):
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / iters

    ms_no_tf32 = bench_matmul(False)
    ms_tf32    = bench_matmul(True)

    print(f"  fp32 matmul ({n}×{n}):")
    print(f"    TF32 disabled: {ms_no_tf32:.3f} ms")
    print(f"    TF32 enabled:  {ms_tf32:.3f} ms")
    print(f"    Speedup:       {ms_no_tf32/ms_tf32:.2f}×")

    # Re-enable for rest of script
    torch.backends.cuda.matmul.allow_tf32 = True


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Numerical properties ──────────────────────────────────────────────
    show_dtype_numerical_range()

    # ── TF32 speedup ──────────────────────────────────────────────────────
    show_tf32()

    # ── GradScaler demo ───────────────────────────────────────────────────
    demonstrate_gradscaler()

    # ── Speed + memory comparison ─────────────────────────────────────────
    print("\n── Training step benchmark: fp32 vs fp16 vs bf16 ────────────────")
    print(f"  {'dtype':>8}  {'step_ms':>10}  {'peak_GB':>10}  "
          f"{'speedup':>10}  {'mem_saving':>12}")
    print("-" * 60)

    results = {}
    for dtype in ["float32", "bfloat16", "float16"]:
        ms, gb, loss = bench_dtype(dtype, batch_size=32, seq_len=256)
        results[dtype] = (ms, gb, loss)

    baseline_ms = results["float32"][0]
    baseline_gb = results["float32"][1]

    for dtype, (ms, gb, loss) in results.items():
        speedup  = baseline_ms / ms
        mem_save = (1 - gb / baseline_gb) * 100
        print(f"  {dtype:>8}  {ms:>10.2f}  {gb:>10.3f}  "
              f"{speedup:>9.2f}×  {mem_save:>10.1f}%")

    print()
    print("Recommendation for A100: use bfloat16")
    print("  - Same numerical range as fp32")
    print("  - No GradScaler needed")
    print("  - 1.5-3× faster matmuls via Tensor Cores")
    print("  - ~40% less activation memory")


if __name__ == "__main__":
    main()
PY

python mixed_precision_lab.py
```

---

## 11.7 Lab 3 — Implement Activation Checkpointing From Scratch

### First: understand what gets saved

```bash
cat > understand_autograd_saves.py <<'PY'
"""
See exactly what tensors autograd saves during forward.
This shows WHY activation memory grows with depth.
"""
import torch
import torch.nn as nn
from gpt import GPT, GPTConfig


def count_saved_tensors(model, x, y):
    """Count tensors saved by autograd for backward."""
    saved = []

    # Hook to intercept pack_hook (what gets saved)
    def pack_hook(tensor):
        saved.append(tensor.shape)
        return tensor

    def unpack_hook(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        _, loss = model(x, y)
        loss.backward()

    total_elements = sum(s.numel() for s in saved)
    total_mb       = total_elements * 4 / 1024**2   # fp32

    return len(saved), total_elements, total_mb


def main():
    device = "cuda"

    print("Autograd saved tensor analysis")
    print("(how many tensors are kept in memory for backward)\n")

    for n_layers in [1, 2, 4, 6, 12]:
        cfg   = GPTConfig(vocab_size=65, max_seq_len=128,
                          d_model=256, n_heads=4, n_layers=n_layers,
                          dropout=0.0)
        model = GPT(cfg).to(device)
        x     = torch.randint(0, 65, (8, 128), device=device)
        y     = torch.randint(0, 65, (8, 128), device=device)

        n_tensors, n_elements, mb = count_saved_tensors(model, x, y)

        print(f"  n_layers={n_layers:2d}  "
              f"saved_tensors={n_tensors:4d}  "
              f"elements={n_elements/1e6:7.2f}M  "
              f"memory={mb:7.2f}MB")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
PY

python understand_autograd_saves.py
```

### Now implement checkpointing

```bash
cat > gpt_with_checkpointing.py <<'PY'
"""
GPT with activation checkpointing (gradient checkpointing).

Two implementations:
1. Manual: recompute activations during backward
2. PyTorch: torch.utils.checkpoint.checkpoint (production)

Also implements a memory-efficient GPT variant with:
  - activation checkpointing
  - bf16 mixed precision
  - fused AdamW
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from dataclasses import dataclass

from multihead_attention import MultiHeadAttention
from feedforward import FeedForward


# ─── Checkpointed Transformer Block ────────────────────────────────────────────

class CheckpointedTransformerBlock(nn.Module):
    """
    Transformer block with optional activation checkpointing.

    When use_checkpoint=True:
      - forward activations are NOT stored during the first pass
      - they are recomputed from the block input during backward
      - memory: O(d_model) per block instead of O(T × d_model + ...)
      - compute: ~33% more (one extra forward per block)
    """
    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int, dropout: float = 0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.ln1            = nn.LayerNorm(d_model)
        self.attn           = MultiHeadAttention(d_model, n_heads,
                                                 max_seq_len, dropout)
        self.ln2            = nn.LayerNorm(d_model)
        self.ffn            = FeedForward(d_model, dropout)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """The actual forward logic (called directly or via checkpoint)."""
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # use_reentrant=False: modern, compatible with torch.compile
            return torch_checkpoint(self._forward, x,
                                    use_reentrant=False)
        return self._forward(x)


# ─── Memory-Efficient GPT ──────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size:      int   = 65
    max_seq_len:     int   = 256
    d_model:         int   = 384
    n_heads:         int   = 6
    n_layers:        int   = 6
    dropout:         float = 0.1
    use_checkpoint:  bool  = False   # activation checkpointing


class MemoryEfficientGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config    = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop_emb  = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            CheckpointedTransformerBlock(
                d_model        = config.d_model,
                n_heads        = config.n_heads,
                max_seq_len    = config.max_seq_len,
                dropout        = config.dropout,
                use_checkpoint = config.use_checkpoint,
            )
            for _ in range(config.n_layers)
        ])

        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size,
                                   bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        n = self.count_params()
        ckpt = "✓" if config.use_checkpoint else "✗"
        print(f"MemoryEfficientGPT: {n/1e6:.2f}M params "
              f"[checkpoint={ckpt}]")

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters()) \
               - self.lm_head.weight.numel()

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor = None):
        import torch.nn.functional as F
        B, T    = idx.shape
        device  = idx.device
        pos     = torch.arange(T, device=device)
        x       = self.drop_emb(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x      = self.ln_final(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        import torch.nn.functional as F
        self.eval()
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx


if __name__ == "__main__":
    import gc

    device = "cuda"
    B, T   = 32, 256

    def measure_step(model, B, T, label):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x   = torch.randint(0, 65, (B, T), device=device)
        y   = torch.randint(0, 65, (B, T), device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()

        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  {label:40s}  peak={peak:.3f}GB")
        del opt
        torch.cuda.empty_cache()

    print("Activation checkpointing memory comparison:")
    print(f"  (B={B}, T={T}, d=384, L=6)")
    print()

    for use_ckpt in [False, True]:
        cfg   = GPTConfig(use_checkpoint=use_ckpt)
        model = MemoryEfficientGPT(cfg).to(device)
        label = "no checkpointing" if not use_ckpt else "with checkpointing"
        measure_step(model, B, T, label)
        del model
PY

python gpt_with_checkpointing.py
```

---

## 11.8 Lab 4 — Systematic Memory Reduction Experiments

```bash
cat > memory_reduction_experiments.py <<'PY'
"""
Systematically apply memory optimizations one at a time.
Measure exactly how much each technique saves.

Techniques applied in order:
  0. Baseline (fp32, no AMP, no checkpoint)
  1. + bf16 AMP
  2. + activation checkpointing
  3. + set_to_none=True
  4. + fused AdamW
  5. + torch.compile (optional)
"""
import time, gc
import torch
import torch.nn as nn

from gpt_with_checkpointing import MemoryEfficientGPT, GPTConfig


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_experiment(
    label:          str,
    batch_size:     int,
    seq_len:        int,
    use_amp:        bool,
    use_checkpoint: bool,
    set_to_none:    bool,
    fused_adam:     bool,
    iters:          int = 20,
):
    reset()
    device = "cuda"

    cfg   = GPTConfig(
        vocab_size      = 65,
        max_seq_len     = seq_len,
        d_model         = 384,
        n_heads         = 6,
        n_layers        = 6,
        dropout         = 0.0,
        use_checkpoint  = use_checkpoint,
    )
    torch.manual_seed(42)
    model = MemoryEfficientGPT(cfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=1e-4,
        fused=fused_adam
    )

    x = torch.randint(0, 65, (batch_size, seq_len), device=device)
    y = torch.randint(0, 65, (batch_size, seq_len), device=device)

    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_amp else torch.amp.autocast(device_type="cuda",
                                               enabled=False))

    # Warmup
    for _ in range(3):
        opt.zero_grad(set_to_none=set_to_none)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step()

    reset()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=set_to_none)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    alloc_gb= torch.cuda.memory_allocated() / 1024**3
    tok_s   = batch_size * seq_len / (ms / 1000)

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label":    label,
        "peak_gb":  peak_gb,
        "alloc_gb": alloc_gb,
        "step_ms":  ms,
        "tok_s":    tok_s,
    }


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB\n")

    B, T = 64, 256

    experiments = [
        dict(label="0. baseline (fp32, no tricks)",
             use_amp=False, use_checkpoint=False,
             set_to_none=False, fused_adam=False),

        dict(label="1. + bf16 AMP",
             use_amp=True,  use_checkpoint=False,
             set_to_none=False, fused_adam=False),

        dict(label="2. + activation checkpoint",
             use_amp=True,  use_checkpoint=True,
             set_to_none=False, fused_adam=False),

        dict(label="3. + set_to_none=True",
             use_amp=True,  use_checkpoint=True,
             set_to_none=True, fused_adam=False),

        dict(label="4. + fused AdamW",
             use_amp=True,  use_checkpoint=True,
             set_to_none=True, fused_adam=True),
    ]

    print(f"B={B} T={T} d=384 L=6")
    print(f"\n{'Label':45s}  {'Peak(GB)':>9}  "
          f"{'Save(GB)':>9}  {'step_ms':>9}  {'tok/s':>8}")
    print("─" * 90)

    results = []
    baseline_peak = None

    for exp in experiments:
        r = run_experiment(batch_size=B, seq_len=T, **exp)
        results.append(r)

        if baseline_peak is None:
            baseline_peak = r["peak_gb"]
            saved = 0.0
        else:
            saved = baseline_peak - r["peak_gb"]

        print(f"  {r['label']:45s}  "
              f"{r['peak_gb']:>9.3f}  "
              f"{saved:>+9.3f}  "
              f"{r['step_ms']:>9.1f}  "
              f"{r['tok_s']:>8.0f}")

    # ── Find max trainable batch size ─────────────────────────────────────
    print(f"\n── Max batch size experiment ─────────────────────────────────────")
    print(f"   (with all optimizations enabled, seq_len={T})")

    for bs in [32, 64, 128, 256, 512]:
        try:
            r = run_experiment(
                label=f"B={bs}",
                batch_size=bs, seq_len=T,
                use_amp=True, use_checkpoint=True,
                set_to_none=True, fused_adam=True,
                iters=5,
            )
            print(f"  batch_size={bs:4d}: peak={r['peak_gb']:.3f}GB  "
                  f"tok/s={r['tok_s']:.0f}  ✓")
        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={bs:4d}: OOM  ✗")
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
PY

python memory_reduction_experiments.py
```

---

## 11.9 Lab 5 — Memory Profiler (PyTorch Memory Snapshot)

```bash
cat > memory_profiler_lab.py <<'PY'
"""
Use PyTorch's memory profiler and snapshot API.
Produces a detailed breakdown of every tensor allocation.
"""
import torch
import torch.nn as nn
from gpt import GPT, GPTConfig


def profile_with_memory_stats():
    """Use torch.profiler to profile memory allocations."""
    from torch.profiler import profile, ProfilerActivity, record_function

    device = "cuda"
    cfg    = GPTConfig(vocab_size=65, max_seq_len=128,
                       d_model=256, n_heads=4, n_layers=4, dropout=0.0)
    model  = GPT(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x      = torch.randint(0, 65, (16, 128), device=device)
    y      = torch.randint(0, 65, (16, 128), device=device)

    with profile(
        activities   = [ProfilerActivity.CUDA],
        profile_memory = True,
        record_shapes  = True,
        with_stack     = False,
    ) as prof:
        for _ in range(3):
            with record_function("zero_grad"):
                opt.zero_grad(set_to_none=True)

            with record_function("forward"):
                with torch.amp.autocast(device_type="cuda",
                                        dtype=torch.bfloat16):
                    _, loss = model(x, y)

            with record_function("backward"):
                loss.backward()

            with record_function("optimizer_step"):
                opt.step()

    # Print top memory-consuming ops
    print("\n── Top 15 ops by CUDA memory usage ──────────────────────────────")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=15
    ))

    del model, opt
    torch.cuda.empty_cache()


def profile_per_layer_memory():
    """
    Hook-based profiling: measure memory before and after each layer.
    Shows exactly where memory is consumed in the forward pass.
    """
    device = "cuda"
    cfg    = GPTConfig(vocab_size=65, max_seq_len=256,
                       d_model=384, n_heads=6, n_layers=6, dropout=0.0)
    model  = GPT(cfg).to(device)

    memory_log = []

    def make_hook(name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024**2
            memory_log.append((name, alloc))
        return hook

    # Register hooks on each major sub-module
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.Linear,
                                nn.MultiheadAttention)):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 65, (32, 256), device=device)
    y = torch.randint(0, 65, (32, 256), device=device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)

    # Remove hooks
    for h in handles:
        h.remove()

    print("\n── Memory growth during forward pass ─────────────────────────────")
    prev = memory_log[0][1] if memory_log else 0
    for i, (name, alloc) in enumerate(memory_log[:30]):  # first 30 events
        delta = alloc - prev
        bar   = "+" * max(0, int(delta / 2)) if delta > 0 else ""
        print(f"  {name:55s}  {alloc:8.1f}MB  Δ{delta:+6.1f}MB  {bar}")
        prev  = alloc

    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n  Peak memory: {peak:.1f}MB")
    del model
    torch.cuda.empty_cache()


def memory_snapshot_demo():
    """
    Use torch.cuda.memory._snapshot() for detailed analysis.
    Creates a pickle file you can analyze with torch's memory visualizer.
    """
    device = "cuda"
    cfg    = GPTConfig(vocab_size=65, max_seq_len=128,
                       d_model=256, n_heads=4, n_layers=4, dropout=0.0)
    model  = GPT(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x      = torch.randint(0, 65, (16, 128), device=device)
    y      = torch.randint(0, 65, (16, 128), device=device)

    # Start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=100000)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    opt.step()

    # Save snapshot
    snap_path = "memory_snapshot.pkl"
    torch.cuda.memory._dump_snapshot(snap_path)
    torch.cuda.memory._record_memory_history(enabled=None)  # stop recording

    print(f"\n── Memory snapshot saved: {snap_path}")
    print("  To visualize: upload to https://pytorch.org/memory_viz")
    print("  Or: python -c \"import pickle; snap=pickle.load(open('memory_snapshot.pkl','rb')); print(type(snap))\"")

    del model, opt
    torch.cuda.empty_cache()


def print_memory_summary():
    """
    torch.cuda.memory_summary(): complete allocator statistics.
    """
    device = "cuda"
    cfg    = GPTConfig(vocab_size=65, max_seq_len=128,
                       d_model=256, n_heads=4, n_layers=4, dropout=0.0)
    model  = GPT(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x      = torch.randint(0, 65, (16, 128), device=device)
    y      = torch.randint(0, 65, (16, 128), device=device)

    torch.cuda.reset_peak_memory_stats()

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    opt.step()

    print("\n── torch.cuda.memory_summary() ───────────────────────────────────")
    print(torch.cuda.memory_summary(device=device, abbreviated=True))

    del model, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.is_available()

    profile_per_layer_memory()
    print_memory_summary()
    profile_with_memory_stats()
    memory_snapshot_demo()
PY

python memory_profiler_lab.py
```

---

## 11.10 Lab 6 — OOM Handling and Recovery

```bash
cat > oom_handling.py <<'PY'
"""
Graceful OOM handling in training:
1. Catch OOM error
2. Clear cache
3. Reduce batch size
4. Continue training

Also shows common OOM patterns and how to debug them.
"""
import gc, torch
import torch.nn as nn

from gpt import GPT, GPTConfig


def train_with_oom_recovery(
    initial_batch: int,
    seq_len:       int,
    min_batch:     int = 1,
):
    """
    Try to train with initial_batch.
    If OOM, halve batch size and retry.
    """
    device = "cuda"
    cfg    = GPTConfig(vocab_size=65, max_seq_len=seq_len,
                       d_model=384, n_heads=6, n_layers=6)
    model  = GPT(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = initial_batch

    while batch_size >= min_batch:
        try:
            gc.collect()
            torch.cuda.empty_cache()

            x = torch.randint(0, 65, (batch_size, seq_len), device=device)
            y = torch.randint(0, 65, (batch_size, seq_len), device=device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda",
                                    dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            opt.step()

            alloc = torch.cuda.memory_allocated() / 1024**3
            peak  = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  batch_size={batch_size:4d}: SUCCESS  "
                  f"alloc={alloc:.2f}GB peak={peak:.2f}GB")
            return batch_size

        except torch.cuda.OutOfMemoryError as e:
            print(f"  batch_size={batch_size:4d}: OOM → halving...")
            del x, y
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch_size //= 2

    print(f"  Could not fit even batch_size={min_batch}")
    return 0


def find_max_batch_size(
    seq_len:   int,
    d_model:   int   = 384,
    n_layers:  int   = 6,
    use_amp:   bool  = True,
    use_ckpt:  bool  = False,
):
    """
    Binary search for maximum batch size.
    More efficient than halving.
    """
    from gpt_with_checkpointing import MemoryEfficientGPT, GPTConfig as MConfig

    device = "cuda"
    cfg    = MConfig(vocab_size=65, max_seq_len=seq_len,
                     d_model=d_model, n_heads=6, n_layers=n_layers,
                     use_checkpoint=use_ckpt)
    model  = MemoryEfficientGPT(cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ctx    = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
              if use_amp else torch.amp.autocast(device_type="cuda",
                                                  enabled=False))

    lo, hi  = 1, 512
    best    = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            gc.collect()
            torch.cuda.empty_cache()

            x = torch.randint(0, 65, (mid, seq_len), device=device)
            y = torch.randint(0, 65, (mid, seq_len), device=device)

            opt.zero_grad(set_to_none=True)
            with ctx:
                _, loss = model(x, y)
            loss.backward()
            opt.step()

            best = mid
            lo   = mid + 1

        except torch.cuda.OutOfMemoryError:
            hi = mid - 1
        finally:
            if 'x' in dir(): del x, y
            gc.collect()
            torch.cuda.empty_cache()

    del model, opt
    return best


def demonstrate_memory_leak():
    """
    Common memory leak: storing loss tensors in a list without .item().
    Shows how to detect and fix it.
    """
    device = "cuda"
    cfg    = GPTConfig(vocab_size=65, max_seq_len=64,
                       d_model=128, n_heads=4, n_layers=2)
    model  = GPT(cfg).to(device)

    print("\n── Memory leak demonstration ─────────────────────────────────────")

    # BAD: keeps autograd graph alive
    losses_bad = []
    for step in range(10):
        x = torch.randint(0, 65, (4, 64), device=device)
        y = torch.randint(0, 65, (4, 64), device=device)
        _, loss = model(x, y)
        losses_bad.append(loss)   # BAD: tensor keeps graph alive!

    mem_bad = torch.cuda.memory_allocated() / 1024**2
    print(f"  BAD  (keeping loss tensors): {mem_bad:.1f}MB")

    del losses_bad, x, y
    gc.collect()
    torch.cuda.empty_cache()

    # GOOD: detach or .item()
    losses_good = []
    for step in range(10):
        x = torch.randint(0, 65, (4, 64), device=device)
        y = torch.randint(0, 65, (4, 64), device=device)
        _, loss = model(x, y)
        losses_good.append(loss.item())   # GOOD: just Python float

    mem_good = torch.cuda.memory_allocated() / 1024**2
    print(f"  GOOD (using loss.item()):    {mem_good:.1f}MB")
    print(f"  Memory saved: {mem_bad - mem_good:.1f}MB")

    del model
    torch.cuda.empty_cache()


def main():
    assert torch.cuda.is_available()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {total_gb:.1f}GB\n")

    # ── OOM recovery ──────────────────────────────────────────────────────
    print("── OOM recovery (start at B=256, auto-reduce) ───────────────────")
    final_batch = train_with_oom_recovery(
        initial_batch=256, seq_len=256, min_batch=4
    )
    print(f"  Final batch size used: {final_batch}")

    # ── Binary search for max batch size ──────────────────────────────────
    print("\n── Max batch size (binary search) ───────────────────────────────")
    configs = [
        dict(label="fp32, no ckpt",   use_amp=False, use_ckpt=False),
        dict(label="bf16 AMP, no ckpt", use_amp=True, use_ckpt=False),
        dict(label="bf16 AMP + ckpt", use_amp=True,  use_ckpt=True),
    ]

    for c in configs:
        max_b = find_max_batch_size(
            seq_len=256,
            use_amp=c["use_amp"],
            use_ckpt=c["use_ckpt"],
        )
        print(f"  {c['label']:30s}: max_batch={max_b}")

    # ── Memory leak demo ──────────────────────────────────────────────────
    demonstrate_memory_leak()


if __name__ == "__main__":
    main()
PY

python oom_handling.py
```

---

## 11.11 Lab 7 — Full Memory-Optimized Training Script

This is the Module 9 training script with all memory optimizations applied:

```bash
cat > train_gpt_memory_optimized.py <<'PY'
"""
Memory-optimized GPT training on single GPU.
Combines all techniques:
  - bf16 mixed precision
  - activation checkpointing
  - gradient accumulation (simulate large batch on small GPU)
  - set_to_none=True
  - fused AdamW
  - memory-efficient generate
  - OOM-safe evaluation

Usage:
  # Standard (no tricks)
  python train_gpt_memory_optimized.py --no_amp --no_checkpoint

  # All tricks
  python train_gpt_memory_optimized.py

  # Max memory savings
  python train_gpt_memory_optimized.py --batch_size=8 --accum_steps=8
"""
import os, gc, time, math, argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import CharTokenizer
from gpt_with_checkpointing import MemoryEfficientGPT, GPTConfig


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx     : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def get_lr(step, warmup, total, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > total:
        return min_lr
    p     = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * p))
    return min_lr + coeff * (lr - min_lr)


def estimate_memory_budget(cfg, batch_size, accum_steps, use_amp, use_ckpt):
    """
    Rough memory estimate before training.
    Helps you know if config will fit before spending time on setup.
    """
    bytes_per_param = 2 if use_amp else 4
    n_params        = (
        cfg.vocab_size * cfg.d_model +          # token_emb (shared)
        cfg.max_seq_len * cfg.d_model +          # pos_emb
        cfg.n_layers * (
            4 * cfg.d_model**2 +                 # attention projections
            8 * cfg.d_model**2 +                 # FFN
            4 * cfg.d_model                      # LN params
        )
    )

    param_gb   = n_params * bytes_per_param / 1024**3
    grad_gb    = n_params * 4 / 1024**3           # fp32 grads
    adam_gb    = n_params * 8 / 1024**3           # m + v

    # Activation estimate
    act_per_layer_gb = (
        batch_size * cfg.max_seq_len * cfg.d_model * bytes_per_param
        / 1024**3
    )
    if use_ckpt:
        act_gb = act_per_layer_gb * 2  # only checkpoint inputs
    else:
        act_gb = act_per_layer_gb * cfg.n_layers * 6  # all intermediates

    total_est = param_gb + grad_gb + adam_gb + act_gb
    print(f"\n── Memory estimate ──────────────────────────────────────────────")
    print(f"  params ({bytes_per_param}B/p): {param_gb:.3f} GB")
    print(f"  grads  (fp32):   {grad_gb:.3f} GB")
    print(f"  adam states:     {adam_gb:.3f} GB")
    print(f"  activations:     {act_gb:.3f} GB  (ckpt={use_ckpt})")
    print(f"  ESTIMATE TOTAL:  {total_est:.3f} GB")
    print(f"  GPU VRAM:        "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"────────────────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--accum_steps",   type=int,   default=4)
    parser.add_argument("--max_iters",     type=int,   default=5000)
    parser.add_argument("--seq_len",       type=int,   default=256)
    parser.add_argument("--d_model",       type=int,   default=384)
    parser.add_argument("--n_layers",      type=int,   default=6)
    parser.add_argument("--n_heads",       type=int,   default=6)
    parser.add_argument("--lr",            type=float, default=6e-4)
    parser.add_argument("--no_amp",        action="store_true")
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--log_interval",  type=int,   default=50)
    parser.add_argument("--eval_interval", type=int,   default=500)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)

    use_amp  = not args.no_amp
    use_ckpt = not args.no_checkpoint

    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"bf16 AMP:  {use_amp}")
    print(f"act_ckpt:  {use_ckpt}")
    print(f"batch:     {args.batch_size} × {args.accum_steps} accum "
          f"= {args.batch_size * args.accum_steps} effective")

    # ── Data ─────────────────────────────────────────────────────────────
    text = open("shakespeare.txt").read()
    tok  = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n    = int(len(data) * 0.9)
    train_ds = TextDataset(data[:n],  args.seq_len)
    val_ds   = TextDataset(data[n:],  args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4,
                              pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4,
                              pin_memory=True,
                              persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────
    cfg = GPTConfig(
        vocab_size      = tok.vocab_size,
        max_seq_len     = args.seq_len,
        d_model         = args.d_model,
        n_heads         = args.n_heads,
        n_layers        = args.n_layers,
        dropout         = 0.1,
        use_checkpoint  = use_ckpt,
    )
    estimate_memory_budget(cfg, args.batch_size, args.accum_steps,
                           use_amp, use_ckpt)

    model = MemoryEfficientGPT(cfg).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────
    decay_p    = [p for n,p in model.named_parameters()
                  if p.requires_grad and p.dim() >= 2]
    no_decay_p = [p for n,p in model.named_parameters()
                  if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": 0.1},
         {"params": no_decay_p, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True
    )

    ctx    = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
              if use_amp else torch.amp.autocast(device_type="cuda",
                                                  enabled=False))

    # ── Training ──────────────────────────────────────────────────────────
    import contextlib
    torch.cuda.reset_peak_memory_stats()
    best_val  = float("inf")
    os.makedirs("checkpoints_memopt", exist_ok=True)
    train_iter = iter(train_loader)
    t0         = time.perf_counter()

    for step in range(args.max_iters):
        lr = get_lr(step, 200, args.max_iters, args.lr, 6e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(args.accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                _, loss = model(x, y)
            (loss / args.accum_steps).backward()
            accum_loss += loss.item() / args.accum_steps

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0:
            torch.cuda.synchronize()
            t1      = time.perf_counter()
            ms      = (t1 - t0) * 1000 / args.log_interval
            tok_s   = (args.batch_size * args.accum_steps *
                       args.seq_len * args.log_interval) / (t1 - t0)
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            t0      = t1
            print(f"step={step:5d} loss={accum_loss:.4f} "
                  f"lr={lr:.2e} ms={ms:.1f} tok/s={tok_s:.0f} "
                  f"peak={peak_gb:.3f}GB",
                  flush=True)

        if step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    if i > 50: break
                    x, y = x.to(device), y.to(device)
                    with ctx:
                        _, loss = model(x, y)
                    val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            print(f"\n  eval val_loss={val_loss:.4f} "
                  f"peak={torch.cuda.max_memory_allocated()/1024**3:.3f}GB\n")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           "checkpoints_memopt/best.pt")
            model.train()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nTraining complete. Peak memory: {peak:.3f}GB")


if __name__ == "__main__":
    main()
PY
```

Run comparison experiments:
```bash
# Baseline (no tricks)
python train_gpt_memory_optimized.py \
  --batch_size=32 --accum_steps=1 \
  --no_amp --no_checkpoint \
  --max_iters=500 --log_interval=50

# All optimizations
python train_gpt_memory_optimized.py \
  --batch_size=64 --accum_steps=4 \
  --max_iters=500 --log_interval=50

# Maximum memory savings (tiny batch, max accum)
python train_gpt_memory_optimized.py \
  --batch_size=8 --accum_steps=16 \
  --max_iters=500 --log_interval=50
```

---

## 11.12 Debug Drills

### Drill 1 — Gradient checkpoint breaks `no_sync()` (DDP + checkpoint)

When using activation checkpointing inside DDP, you need `use_reentrant=False`:

```bash
cat > ckpt_ddp_compat.py <<'PY'
"""
Test activation checkpointing compatibility with DDP.
use_reentrant=True (old API) can cause issues with DDP.
use_reentrant=False (new API) is safe.
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from gpt_with_checkpointing import MemoryEfficientGPT, GPTConfig


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                      d_model=128, n_heads=4, n_layers=2,
                      use_checkpoint=True)
    model = MemoryEfficientGPT(cfg).to(device)
    ddp   = DDP(model, device_ids=[local_rank])
    opt   = torch.optim.AdamW(ddp.parameters(), lr=1e-4)

    x = torch.randint(0, 65, (4, 64), device=device)
    y = torch.randint(0, 65, (4, 64), device=device)

    # This should work cleanly with use_reentrant=False
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = ddp(x, y)
    loss.backward()
    opt.step()

    if rank == 0:
        print("DDP + activation checkpoint (use_reentrant=False): PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=2 ckpt_ddp_compat.py
```

---

### Drill 2 — GradScaler scale goes to 0 (fp16 instability)

```bash
python - <<'PY'
"""Show GradScaler scale dynamics during training."""
import torch
import torch.nn as nn
from gpt import GPT, GPTConfig

device = "cuda"
cfg    = GPTConfig(vocab_size=65, max_seq_len=64,
                   d_model=128, n_heads=4, n_layers=2, dropout=0.0)
model  = GPT(cfg).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=1e-4)

# GradScaler starts at 65536, adjusts automatically
scaler = torch.cuda.amp.GradScaler(
    init_scale        = 65536,
    growth_factor     = 2.0,
    backoff_factor    = 0.5,
    growth_interval   = 2000,
)

ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)

for step in range(20):
    x = torch.randint(0, 65, (4, 64), device=device)
    y = torch.randint(0, 65, (4, 64), device=device)

    opt.zero_grad(set_to_none=True)
    with ctx:
        _, loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

    scale   = scaler.get_scale()
    inf_skp = scaler._found_inf_per_device
    print(f"  step={step:3d}  loss={loss.item():.4f}  "
          f"scale={scale:.0f}  "
          f"skipped_step={'yes' if any(inf_skp.values()) else 'no'}")
PY
```

---

### Drill 3 — Memory fragmentation (why reserved > allocated)

```bash
python - <<'PY'
"""
Demonstrates PyTorch CUDA memory caching allocator:
- memory_allocated: actual tensor data
- memory_reserved:  memory held by PyTorch allocator cache
- They differ because PyTorch keeps freed memory in a cache pool
"""
import torch
import gc

device = "cuda"

def show(label):
    alloc = torch.cuda.memory_allocated() / 1024**2
    resv  = torch.cuda.memory_reserved()  / 1024**2
    print(f"  {label:40s}  alloc={alloc:8.1f}MB  reserved={resv:8.1f}MB")

torch.cuda.empty_cache()
show("start")

# Allocate 1GB tensor
big = torch.randn(256_000_000, device=device)  # ~1GB
show("after allocating 1GB tensor")

# Delete it
del big
gc.collect()
show("after del (cache NOT cleared)")

# Force cache clear
torch.cuda.empty_cache()
show("after empty_cache()")

# Allocate and free many small tensors
for i in range(100):
    t = torch.randn(1_000_000, device=device)
    del t
show("after 100× alloc/free cycles (no empty_cache)")

torch.cuda.empty_cache()
show("after empty_cache()")

print("\nNote: reserved stays high because allocator caches memory")
print("      for reuse. This is normal and expected behavior.")
PY
```

---

## 11.13 Memory Optimization Summary Table

```bash
cat > print_summary.py <<'PY'
"""
Print the complete memory optimization techniques and their tradeoffs.
"""

techniques = [
    {
        "name":     "bf16 Mixed Precision (AMP)",
        "memory":   "~40% less activation memory",
        "speed":    "1.5-3× faster (Tensor Cores)",
        "quality":  "Same as fp32 on A100",
        "code":     "torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)",
        "tradeoff": "Tiny precision reduction (rarely matters)",
    },
    {
        "name":     "Activation Checkpointing",
        "memory":   "O(n_layers) → O(1) per layer group",
        "speed":    "~33% slower (recompute forward)",
        "quality":  "Identical (mathematically equivalent)",
        "code":     "torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)",
        "tradeoff": "33% more compute",
    },
    {
        "name":     "set_to_none=True",
        "memory":   "Saves grad tensor allocation between steps",
        "speed":    "Slightly faster (skip zero fill)",
        "quality":  "Identical",
        "code":     "optimizer.zero_grad(set_to_none=True)",
        "tradeoff": "None",
    },
    {
        "name":     "Gradient Accumulation",
        "memory":   "Reduce effective batch in memory",
        "speed":    "Slower (serial micro-steps)",
        "quality":  "Same global batch = same training",
        "code":     "for micro in range(accum_steps): loss.backward()",
        "tradeoff": "Slower per effective batch",
    },
    {
        "name":     "Fused AdamW",
        "memory":   "Slightly less peak (kernel fusion)",
        "speed":    "~10-20% faster optimizer step",
        "quality":  "Identical",
        "code":     "torch.optim.AdamW(..., fused=True)",
        "tradeoff": "CUDA only",
    },
    {
        "name":     "TF32 (A100 only)",
        "memory":   "None (transparent)",
        "speed":    "~8× faster matmuls",
        "quality":  "Tiny precision reduction (rarely matters)",
        "code":     "torch.backends.cuda.matmul.allow_tf32 = True",
        "tradeoff": "10-bit mantissa (vs 23-bit fp32)",
    },
    {
        "name":     "torch.compile",
        "memory":   "Often reduces peak (kernel fusion)",
        "speed":    "10-50% faster (graph optimization)",
        "quality":  "Identical",
        "code":     "model = torch.compile(model)",
        "tradeoff": "Long compile time (~1-5 min first run)",
    },
]

print("\n" + "="*80)
print("MEMORY OPTIMIZATION TECHNIQUES — COMPLETE REFERENCE")
print("="*80)
for i, t in enumerate(techniques):
    print(f"\n{i+1}. {t['name']}")
    print(f"   Memory:    {t['memory']}")
    print(f"   Speed:     {t['speed']}")
    print(f"   Quality:   {t['quality']}")
    print(f"   Tradeoff:  {t['tradeoff']}")
    print(f"   Code:      {t['code']}")
print("\n" + "="*80)

print("\nRECOMMENDED STACK for A100:")
print("  1. bf16 AMP                         ← always on")
print("  2. TF32                             ← always on A100")
print("  3. fused AdamW + set_to_none=True   ← always on")
print("  4. activation checkpointing         ← when batch_size/seq_len is large")
print("  5. gradient accumulation            ← when batch doesn't fit")
print("  6. torch.compile                    ← when training > 1 hour")
print()
PY

python print_summary.py
```

---

## 11.14 Module 11 Pass Criteria

You're done when you can:

1) **Explain the 4 memory components** and their byte cost formulas:
   - parameters: `N × bytes_per_param`
   - gradients: `N × 4` (fp32)
   - Adam states: `N × 8` (m + v in fp32)
   - activations: `B × T × d_model × n_layers × constant`

2) **Explain mixed precision**:
   - why bf16 is preferred over fp16 on A100
   - what GradScaler does and when it's needed
   - what TF32 is (compute-only, not a storage format)

3) **Explain activation checkpointing**:
   - what is recomputed vs stored
   - the 33% compute overhead tradeoff
   - why `use_reentrant=False`

4) **Run and interpret all labs**:
   - `python measure_memory_components.py` → see all 4 components separately
   - `python mixed_precision_lab.py` → fp32 vs fp16 vs bf16 comparison
   - `python gpt_with_checkpointing.py` → checkpoint vs no checkpoint memory
   - `python memory_reduction_experiments.py` → progressive improvement table
   - `python oom_handling.py` → max batch size search

5) **Fill your memory reduction table**:

| Config | Peak GB | vs Baseline |
|--------|---------|-------------|
| fp32, no AMP, no checkpoint | | 100% |
| + bf16 AMP | | ?% |
| + activation checkpoint | | ?% |
| + set_to_none | | ?% |
| + fused AdamW | | ?% |

---

