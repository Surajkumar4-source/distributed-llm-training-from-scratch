# Module 13 — DeepSpeed

## Summary
DeepSpeed is Microsoft's distributed training library. It implements the same ZeRO ideas as FSDP but with a different philosophy: **configuration-driven** (JSON config controls everything) rather than code-driven. It also goes further than FSDP with CPU offloading, NVMe offloading, and ZeRO-Infinity.

- **Theory**: DeepSpeed architecture, ZeRO Stage 1/2/3 implementation details, CPU offload, NVMe offload, ZeRO-Infinity
- **Config**: every config field explained in depth
- **From scratch**: convert GPT to DeepSpeed engine step by step
- **Launcher**: `deepspeed` CLI and SLURM integration
- **Benchmark**: DeepSpeed vs FSDP vs DDP on same model
- **Advanced**: gradient accumulation, mixed precision, activation checkpointing with DeepSpeed
- **Debug drills**: config errors, dtype mismatches, NaN losses, checkpoint issues

---

## 13.1 Theory — DeepSpeed Architecture

### How DeepSpeed differs from FSDP

```
FSDP approach:
  - PyTorch-native (part of torch.distributed)
  - Code-driven: wrap modules explicitly in Python
  - Config via Python objects (ShardingStrategy, MixedPrecision, etc.)
  - Tight PyTorch integration

DeepSpeed approach:
  - Standalone library (pip install deepspeed)
  - Config-driven: JSON file controls everything
  - Wraps model + optimizer into a "DeepSpeed Engine"
  - More features: NVMe offload, ZeRO-Infinity, sparse attention,
    kernel fusion, communication compression
  - Used by: Megatron-LM, Bloom, OPT, many production LLMs
```

### DeepSpeed Engine

```python
import deepspeed

# Before DeepSpeed:
model     = GPT(cfg)
optimizer = Adam(model.parameters(), lr=1e-4)
# train manually

# After DeepSpeed:
engine, optimizer, _, _ = deepspeed.initialize(
    model           = model,
    model_parameters= model.parameters(),
    config          = ds_config,   # JSON dict or path
)
# engine replaces both model and optimizer
engine.backward(loss)   # instead of loss.backward()
engine.step()           # instead of optimizer.step()
```

The engine handles:
- ZeRO sharding of params/grads/optimizer states
- Mixed precision (FP16/BF16)
- Gradient clipping
- LR scheduling
- Gradient accumulation
- CPU/NVMe offloading
- Checkpointing

### ZeRO Stage 1, 2, 3 in DeepSpeed

Same math as Module 12, different implementation details:

```
Stage 1: Optimizer State Partitioning
  - Adam m,v split across ranks
  - Gradients: full AllReduce (same as DDP)
  - Parameters: replicated
  - Memory: N×4 + N×4 + N×8/P = N×(8 + 8/P) bytes

Stage 2: + Gradient Partitioning
  - Gradients also split
  - ReduceScatter instead of AllReduce
  - Memory: N×4 + N×4/P + N×8/P = N×(4 + 12/P) bytes

Stage 3: + Parameter Partitioning
  - Parameters also split
  - AllGather before each forward computation
  - Memory: N×4/P + N×4/P + N×8/P = N×16/P bytes
  - (same as FSDP FULL_SHARD)
```

For world_size=8 (P=8):

| Stage | Memory formula | Memory (10M params) | vs Baseline |
|-------|---------------|---------------------|-------------|
| DDP (no ZeRO) | N×16 | 160 MB | 1.0× |
| ZeRO-1 | N×(8+1) = N×9 | 90 MB | 1.8× |
| ZeRO-2 | N×(4+1.5) = N×5.5 | 55 MB | 2.9× |
| ZeRO-3 | N×16/P = N×2 | 20 MB | 8.0× |

### DeepSpeed-specific features beyond ZeRO

**CPU Offload (ZeRO-Offload)**:
```
optimizer states → CPU RAM  (during optimizer step)
gradients        → CPU RAM  (accumulate on CPU)
parameters       → GPU      (kept on GPU for compute)

Use case: GPU has 40GB, model optimizer states need 56GB
→ offload states to CPU RAM (typically 256-512GB)
→ trade speed for memory
```

**NVMe Offload (ZeRO-Infinity)**:
```
optimizer states → NVMe SSD  (via async I/O)
gradients        → NVMe SSD
parameters       → NVMe SSD (optional)

Use case: trillion-parameter models
→ offload to fast NVMe (3-7 GB/s read)
→ very slow but enables training otherwise impossible models
```

**Communication Compression**:
```
1-bit Adam: compress gradients to 1-bit before AllReduce
→ 32× less communication bandwidth
→ slight accuracy degradation (sometimes negligible)
```

**Kernel Fusion**:
```
DeepSpeed fuses operations for speed:
- Fused Adam optimizer
- Fused Layer Norm
- Fused softmax
- Transformer kernel (custom CUDA)
```

---

## 13.2 Theory — DeepSpeed Config (Every Field Explained)

The config JSON controls everything. Here is the complete reference:

```json
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 1,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 6e-5,
            "warmup_max_lr": 6e-4,
            "warmup_num_steps": 200,
            "total_num_steps": 5000
        }
    },

    "fp16": {
        "enabled": false
    },

    "bf16": {
        "enabled": true
    },

    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        }
    },

    "gradient_clipping": 1.0,

    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    },

    "wall_clock_breakdown": false,
    "steps_per_print": 50
}
```

### Field explanations

```
train_batch_size:
  Global batch size = micro_batch × grad_accum × world_size
  DeepSpeed computes and validates this automatically

train_micro_batch_size_per_gpu:
  Batch size on each GPU per forward/backward pass

gradient_accumulation_steps:
  Number of micro-steps before optimizer.step()
  Effective batch = micro_batch × grad_accum × world_size

optimizer.type:
  "AdamW", "Adam", "Lamb", "OneBitAdam"
  (DeepSpeed has its own fused implementations)

scheduler.type:
  "WarmupLR", "WarmupDecayLR", "CosineAnnealingLR"
  Or set to null and manage LR yourself

fp16.enabled / bf16.enabled:
  Mixed precision. bf16 preferred on A100.
  DO NOT enable both simultaneously.

fp16.loss_scale:
  Initial loss scale for GradScaler (default: 0 = dynamic)

zero_optimization.stage:
  0: DDP (no ZeRO)
  1: optimizer state partitioning
  2: + gradient partitioning
  3: + parameter partitioning

zero_optimization.overlap_comm:
  Overlap AllGather/ReduceScatter with backward compute
  Similar to DDP's gradient bucketing
  Strongly recommended: true

zero_optimization.contiguous_gradients:
  Allocate gradient buffer as contiguous memory
  Improves communication performance

zero_optimization.allgather_bucket_size:
  Size of parameter chunks for AllGather (bytes)
  Larger = fewer AllGather ops but larger temporary tensors
  Default 500MB is reasonable

zero_optimization.reduce_bucket_size:
  Size of gradient chunks for ReduceScatter
  Same tradeoff as allgather_bucket_size

zero_optimization.offload_optimizer.device:
  "none": keep optimizer on GPU
  "cpu":  move optimizer states to CPU RAM
  "nvme": move optimizer states to NVMe SSD

zero_optimization.offload_param.device:
  "none": keep params on GPU
  "cpu":  move unused param shards to CPU
  "nvme": move unused param shards to NVMe

gradient_clipping:
  Max gradient norm. DeepSpeed handles this correctly
  across ZeRO stages (similar to FSDP.clip_grad_norm_)

activation_checkpointing.partition_activations:
  Partition activations across ranks (extra memory savings)
  Only useful for Stage 3 with very large activations
```

---

## 13.3 Setup

```bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course
pip install deepspeed
mkdir -p deepspeed_module
cp gpt_scratch/tokenizer.py          deepspeed_module/
cp gpt_scratch/shakespeare.txt       deepspeed_module/
cp fsdp_module/model.py              deepspeed_module/
cd deepspeed_module

# Verify DeepSpeed installation
python -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)"

# Check what ops are available (CUDA extensions)
ds_report
```

---

## 13.4 Lab 1 — DeepSpeed Hello World

```bash
cat > ds_hello.py <<'PY'
"""
Minimal DeepSpeed example.
Shows: engine initialization, forward, backward, step.
Compare to DDP and FSDP patterns.
"""
import os
import torch
import torch.distributed as dist
import deepspeed

from model import GPT, GPTConfig


def main():
    # ── Deepspeed handles dist.init_process_group internally ─────────────
    # BUT we still need LOCAL_RANK for device assignment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # ── DeepSpeed config ──────────────────────────────────────────────────
    ds_config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 6e-4,
                "betas": [0.9, 0.95],
                "weight_decay": 0.1,
            }
        },
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }

    # ── Model ─────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=128,
        d_model=256, n_heads=4, n_layers=4, dropout=0.0
    )
    model = GPT(cfg)

    # ── Initialize DeepSpeed engine ───────────────────────────────────────
    # This replaces:
    #   model = DDP(model)         ← DDP
    #   model = FSDP(model)        ← FSDP
    #   optimizer = AdamW(...)     ← manual optimizer
    engine, optimizer, _, _ = deepspeed.initialize(
        model            = model,
        model_parameters = model.parameters(),
        config           = ds_config,
    )

    rank       = engine.global_rank
    world_size = engine.world_size
    device     = engine.device

    if rank == 0:
        print(f"\nDeepSpeed Engine initialized")
        print(f"  world_size:  {world_size}")
        print(f"  ZeRO stage:  {ds_config['zero_optimization']['stage']}")
        print(f"  bf16:        {ds_config['bf16']['enabled']}")
        print(f"  device:      {device}")
        print(f"  model type:  {type(engine.module).__name__}")
        print(f"  optimizer:   {type(optimizer).__name__}")

    # ── Training step ─────────────────────────────────────────────────────
    device = torch.device(f"cuda:{local_rank}")
    x      = torch.randint(0, 65, (4, 128), device=device)
    y      = torch.randint(0, 65, (4, 128), device=device)

    for step in range(5):
        # Forward (same as normal PyTorch)
        _, loss = engine(x, y)

        # Backward — use engine.backward() NOT loss.backward()
        engine.backward(loss)

        # Optimizer step — use engine.step() NOT optimizer.step()
        engine.step()

        # zero_grad is handled by engine automatically
        # (no need to call explicitly)

        if rank == 0:
            print(f"  step={step}  loss={loss.item():.4f}",
                  flush=True)

    if rank == 0:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak memory: {peak_gb:.3f}GB")
        print("DeepSpeed hello: PASSED")


if __name__ == "__main__":
    main()
PY

deepspeed --num_gpus=4 ds_hello.py
```

---

## 13.5 Lab 2 — ZeRO Stage Comparison

```bash
cat > ds_zero_stages.py <<'PY'
"""
Compare DeepSpeed ZeRO stages 0,1,2,3 on the same model.
Measures: memory per GPU, step time, throughput.

This is the DeepSpeed equivalent of Module 12's DDP vs FSDP comparison.
"""
import os, gc, time
import torch
import deepspeed

from model import GPT, GPTConfig


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def build_config(stage, use_bf16=True, overlap_comm=True):
    """Build DeepSpeed config for a given ZeRO stage."""
    cfg = {
        "train_micro_batch_size_per_gpu": 32,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 6e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            }
        },
        "bf16": {"enabled": use_bf16},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": overlap_comm,
            "contiguous_gradients": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
            "reduce_scatter": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 9999,
        "wall_clock_breakdown": False,
    }
    return cfg


def bench_stage(stage, B, T, iters=30, warmup=5):
    """Run training with a specific ZeRO stage and measure performance."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    reset()

    torch.manual_seed(42)
    cfg_model = GPTConfig(
        vocab_size=65, max_seq_len=T,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0
    )
    model     = GPT(cfg_model)
    ds_config = build_config(stage=stage)

    engine, optimizer, _, _ = deepspeed.initialize(
        model            = model,
        model_parameters = model.parameters(),
        config           = ds_config,
    )

    device = torch.device(f"cuda:{local_rank}")
    x      = torch.randint(0, 65, (B, T), device=device)
    y      = torch.randint(0, 65, (B, T), device=device)

    # Warmup
    for _ in range(warmup):
        _, loss = engine(x, y)
        engine.backward(loss)
        engine.step()

    torch.cuda.synchronize()
    reset()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(iters):
        _, loss = engine(x, y)
        engine.backward(loss)
        engine.step()

    torch.cuda.synchronize()
    ms      = (time.perf_counter() - t0) * 1000 / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    tok_s   = B * T * engine.world_size / (ms / 1000)
    loss_val= loss.item()

    engine.destroy()
    del engine, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    return ms, peak_gb, tok_s, loss_val


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main    = (local_rank == 0)

    B, T       = 32, 256

    stages     = [0, 1, 2, 3]
    results    = []

    for stage in stages:
        ms, peak, tok_s, loss = bench_stage(stage=stage, B=B, T=T)
        results.append((stage, ms, peak, tok_s, loss))
        if is_main:
            print(f"ZeRO-{stage}: done  "
                  f"ms={ms:.1f}  peak={peak:.3f}GB  "
                  f"tok/s={tok_s:.0f}", flush=True)

    if is_main:
        baseline_ms   = results[0][1]
        baseline_peak = results[0][2]

        print(f"\n{'='*75}")
        print(f"DeepSpeed ZeRO Stage Comparison")
        print(f"B={B}  T={T}  d=384  L=6")
        print(f"{'='*75}")
        print(f"{'Stage':12s}  {'ms':>8}  {'speedup':>8}  "
              f"{'peak_GB':>9}  {'mem_save':>10}  {'tok/s':>10}")
        print(f"{'─'*75}")

        for stage, ms, peak, tok_s, _ in results:
            speedup  = baseline_ms / ms
            mem_save = (1 - peak/baseline_peak) * 100
            print(f"  ZeRO-{stage}:      {ms:>8.2f}  "
                  f"{speedup:>7.2f}×  {peak:>9.3f}  "
                  f"{mem_save:>9.1f}%  {tok_s:>10.0f}")

        print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
PY

deepspeed --num_gpus=8 ds_zero_stages.py
```

---

## 13.6 Lab 3 — DeepSpeed Config Files (Production Pattern)

In production you save configs as JSON files and load them:

```bash
# Create config directory
mkdir -p ds_configs

# Config: ZeRO Stage 2 (good default for most models)
cat > ds_configs/zero2.json <<'JSON'
{
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 1,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 6e-5,
            "warmup_max_lr": 6e-4,
            "warmup_num_steps": 200,
            "total_num_steps": 5000
        }
    },

    "bf16": {"enabled": true},
    "fp16": {"enabled": false},

    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 50,
    "wall_clock_breakdown": false
}
JSON

# Config: ZeRO Stage 3 (maximum memory savings)
cat > ds_configs/zero3.json <<'JSON'
{
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 1,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "bf16": {"enabled": true},
    "fp16": {"enabled": false},

    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "stage3_prefetch_bucket_size": 500000000,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none",
            "pin_memory": true
        }
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 50,
    "wall_clock_breakdown": false
}
JSON

# Config: ZeRO Stage 3 + CPU Offload
cat > ds_configs/zero3_offload.json <<'JSON'
{
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 4,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "bf16": {"enabled": true},

    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 50
}
JSON

# Config: ZeRO Stage 1 (minimal, closest to DDP)
cat > ds_configs/zero1.json <<'JSON'
{
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 1,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "bf16": {"enabled": true},

    "zero_optimization": {
        "stage": 1
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 50
}
JSON

echo "Configs created:"
ls -la ds_configs/
```

---

## 13.7 Lab 4 — Full GPT DeepSpeed Training Script

```bash
cat > train_gpt_deepspeed.py <<'PY'
"""
Full GPT training with DeepSpeed.

Features:
  - ZeRO Stage 0/1/2/3 (via config)
  - bf16 mixed precision
  - Activation checkpointing
  - Gradient accumulation
  - DeepSpeed checkpointing
  - Distributed evaluation
  - Throughput measurement

Launch:
  deepspeed --num_gpus=8 train_gpt_deepspeed.py --ds_config=ds_configs/zero2.json
  deepspeed --num_gpus=8 train_gpt_deepspeed.py --ds_config=ds_configs/zero3.json

  With SLURM:
  srun deepspeed --num_gpus=8 train_gpt_deepspeed.py ...
"""
import os, gc, time, math, argparse, json
import torch
import torch.distributed as dist
import deepspeed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


# ─── Dataset ──────────────────────────────────────────────────────────────────

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


# ─── LR Schedule (manual, overrides DeepSpeed scheduler) ─────────────────────

def get_lr(step, warmup, total, lr, min_lr):
    """Cosine schedule with warmup."""
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > total:
        return min_lr
    p     = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * p))
    return min_lr + coeff * (lr - min_lr)


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(engine, loader, device, eval_iters):
    """
    Distributed evaluation using DeepSpeed engine.
    engine.module gives access to underlying GPT model.
    """
    engine.eval()
    total_loss  = torch.zeros(1, device=device)
    total_steps = torch.zeros(1, device=device)

    for i, (x, y) in enumerate(loader):
        if i >= eval_iters:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = engine(x, y)
        total_loss  += loss.detach()
        total_steps += 1

    dist.all_reduce(total_loss,  op=dist.ReduceOp.SUM)
    dist.all_reduce(total_steps, op=dist.ReduceOp.SUM)
    engine.train()
    return (total_loss / total_steps).item()


# ─── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(engine, step, val_loss, save_dir, rank):
    """
    DeepSpeed checkpointing is different from PyTorch:
    engine.save_checkpoint() handles ZeRO sharding correctly.

    Creates: save_dir/step_<N>/  with rank-specific files + metadata
    """
    tag = f"step_{step}"
    engine.save_checkpoint(save_dir, tag=tag)

    # Save metadata separately (rank 0 only)
    if rank == 0:
        meta = {"step": step, "val_loss": val_loss}
        meta_path = os.path.join(save_dir, tag, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"[rank 0] Checkpoint saved → {save_dir}/{tag} "
              f"(val_loss={val_loss:.4f})", flush=True)

    dist.barrier()
    return tag


def load_checkpoint(engine, save_dir, tag, rank):
    """
    Load DeepSpeed checkpoint.
    engine.load_checkpoint() handles ZeRO resharding if world_size changed.
    """
    _, client_state = engine.load_checkpoint(
        save_dir,
        tag                  = tag,
        load_optimizer_states= True,
        load_lr_scheduler_states = True,
    )

    # Read metadata
    meta_path = os.path.join(save_dir, tag, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if rank == 0:
            print(f"Resumed from step={meta['step']} "
                  f"val_loss={meta['val_loss']:.4f}")
        return meta["step"], meta["val_loss"]

    return 0, float("inf")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_config",     type=str,
                        default="ds_configs/zero2.json")
    parser.add_argument("--max_iters",     type=int,   default=5000)
    parser.add_argument("--seq_len",       type=int,   default=256)
    parser.add_argument("--d_model",       type=int,   default=384)
    parser.add_argument("--n_layers",      type=int,   default=6)
    parser.add_argument("--n_heads",       type=int,   default=6)
    parser.add_argument("--lr",            type=float, default=6e-4)
    parser.add_argument("--use_checkpoint",action="store_true")
    parser.add_argument("--log_interval",  type=int,   default=50)
    parser.add_argument("--eval_interval", type=int,   default=500)
    parser.add_argument("--ckpt_dir",      type=str,
                        default="checkpoints_ds/run")
    parser.add_argument("--resume_tag",    type=str,   default=None)

    # DeepSpeed adds its own args (--local_rank, --deepspeed, etc.)
    parser = deepspeed.add_config_arguments(parser)
    args   = parser.parse_args()

    # ── Init ─────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device  = torch.device(f"cuda:{local_rank}")

    # DeepSpeed initializes distributed internally
    # (no need to call dist.init_process_group manually)
    deepspeed.init_distributed()
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    is_main    = (rank == 0)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(1337 + rank)

    # ── Load and validate DS config ───────────────────────────────────────
    with open(args.ds_config) as f:
        ds_config = json.load(f)

    # Derive train_batch_size if not set
    micro_bs = ds_config.get("train_micro_batch_size_per_gpu", 64)
    grad_acc = ds_config.get("gradient_accumulation_steps", 1)
    if "train_batch_size" not in ds_config:
        ds_config["train_batch_size"] = micro_bs * grad_acc * world_size

    zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)

    if is_main:
        print(f"\n{'='*60}")
        print(f"DeepSpeed GPT Training")
        print(f"  config:      {args.ds_config}")
        print(f"  ZeRO stage:  {zero_stage}")
        print(f"  world_size:  {world_size}")
        print(f"  micro_bs:    {micro_bs}")
        print(f"  grad_accum:  {grad_acc}")
        print(f"  global_bs:   {ds_config['train_batch_size']}")
        print(f"  bf16:        {ds_config.get('bf16',{}).get('enabled',False)}")
        print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────
    if is_main:
        text = open("shakespeare.txt").read()
        tok  = CharTokenizer(text)
        data = torch.tensor(tok.encode(text), dtype=torch.long)
        n    = int(len(data) * 0.9)
        train_data, val_data = data[:n], data[n:]
        vocab_size = tok.vocab_size
    else:
        train_data = val_data = None
        vocab_size = None

    obj_list = [vocab_size, train_data, val_data]
    dist.broadcast_object_list(obj_list, src=0)
    vocab_size, train_data, val_data = obj_list

    train_ds = TextDataset(train_data.cpu(), args.seq_len)
    val_ds   = TextDataset(val_data.cpu(),   args.seq_len)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=micro_bs, sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_bs, sampler=val_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    gpt_cfg = GPTConfig(
        vocab_size      = vocab_size,
        max_seq_len     = args.seq_len,
        d_model         = args.d_model,
        n_heads         = args.n_heads,
        n_layers        = args.n_layers,
        dropout         = 0.1,
        use_checkpoint  = args.use_checkpoint,
    )

    torch.manual_seed(1337)
    model = GPT(gpt_cfg)

    if is_main:
        n_params = model.count_params()
        print(f"Model: {n_params/1e6:.2f}M params")

    # ── DeepSpeed Engine ───────────────────────────────────────────────────
    # Note: we do NOT create optimizer manually —
    # DeepSpeed creates it from config
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model            = model,
        model_parameters = model.parameters(),
        config           = ds_config,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    best_val   = float("inf")

    if args.resume_tag and os.path.exists(args.ckpt_dir):
        start_step, best_val = load_checkpoint(
            engine, args.ckpt_dir, args.resume_tag, rank
        )

    # ── Training loop ──────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_iter = iter(train_loader)
    t0         = time.perf_counter()

    if is_main:
        print("── Training starts ────────────────────────────────────────\n")

    for step in range(start_step, args.max_iters):

        # ── LR update (manual, overrides DS scheduler) ─────────────────
        # Comment this out to use DS scheduler instead
        lr = get_lr(step, 200, args.max_iters, args.lr, 6e-5)
        for pg in engine.optimizer.param_groups:
            pg["lr"] = lr

        # ── Sampler epoch ────────────────────────────────────────────────
        epoch = step * micro_bs // max(1, len(train_ds))
        train_sampler.set_epoch(epoch)

        # ── Batch ────────────────────────────────────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ── Forward ───────────────────────────────────────────────────────
        # engine() handles mixed precision automatically
        _, loss = engine(x, y)

        # ── Backward ─────────────────────────────────────────────────────
        # engine.backward() handles:
        #   - loss scaling (fp16)
        #   - gradient accumulation bookkeeping
        #   - ZeRO gradient handling
        engine.backward(loss)

        # ── Optimizer step ────────────────────────────────────────────────
        # engine.step() handles:
        #   - gradient clipping (configured in JSON)
        #   - optimizer.step()
        #   - lr_scheduler.step()
        #   - gradient zero (automatically)
        engine.step()

        # ── Logging ───────────────────────────────────────────────────────
        if step % args.log_interval == 0:
            torch.cuda.synchronize()
            t1    = time.perf_counter()
            ms    = (t1 - t0) * 1000 / args.log_interval
            tok_s = micro_bs * args.seq_len * world_size \
                    * args.log_interval / (t1 - t0)
            peak  = torch.cuda.max_memory_allocated(device) / 1024**3
            alloc = torch.cuda.memory_allocated(device) / 1024**3
            t0    = t1

            if is_main:
                print(
                    f"step={step:5d}  loss={loss.item():.4f}  "
                    f"lr={lr:.2e}  ms={ms:.1f}  tok/s={tok_s:.0f}  "
                    f"alloc={alloc:.3f}GB  peak={peak:.3f}GB",
                    flush=True
                )

        # ── Evaluation ────────────────────────────────────────────────────
        if step % args.eval_interval == 0 or step == args.max_iters - 1:

            val_loss = evaluate(engine, val_loader, device, eval_iters=50)

            if is_main:
                print(f"\n── eval step={step} val_loss={val_loss:.4f} ────")

                if val_loss < best_val:
                    best_val = val_loss
                    tag      = save_checkpoint(
                        engine, step, val_loss, args.ckpt_dir, rank
                    )

                print("────────────────────────────────────────────────────\n",
                      flush=True)

            dist.barrier()

    if is_main:
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"\n── Training complete ─────────────────────────────────────")
        print(f"  best_val_loss: {best_val:.4f}")
        print(f"  peak_memory:   {peak:.3f}GB")
        print("──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
PY
```

Run with different configs:
```bash
# ZeRO Stage 2 (good default)
deepspeed --num_gpus=8 train_gpt_deepspeed.py \
  --ds_config=ds_configs/zero2.json \
  --max_iters=2000

# ZeRO Stage 3 (maximum memory savings)
deepspeed --num_gpus=8 train_gpt_deepspeed.py \
  --ds_config=ds_configs/zero3.json \
  --max_iters=2000

# ZeRO Stage 3 + CPU offload
deepspeed --num_gpus=8 train_gpt_deepspeed.py \
  --ds_config=ds_configs/zero3_offload.json \
  --max_iters=2000

# With activation checkpointing
deepspeed --num_gpus=8 train_gpt_deepspeed.py \
  --ds_config=ds_configs/zero2.json \
  --use_checkpoint \
  --max_iters=2000
```

---

## 13.8 Lab 5 — DeepSpeed Checkpointing Deep Dive

```bash
cat > ds_checkpoint_lab.py <<'PY'
"""
DeepSpeed checkpoint internals.

DeepSpeed saves:
  checkpoints/<tag>/
    mp_rank_00_model_states.pt    ← model state dict (rank 0)
    zero_pp_rank_N_mp_rank_00_optim_states.pt  ← per-rank ZeRO states
    bf16_zero_pp_rank_N_mp_rank_00_optim_states.pt  (if bf16)
    latest                        ← file containing latest tag name

Loading works even if world_size changes (ZeRO resharding).
"""
import os, json
import torch
import torch.distributed as dist
import deepspeed

from model import GPT, GPTConfig


def print_checkpoint_structure(ckpt_dir, tag):
    """Show what files DeepSpeed saved."""
    tag_dir = os.path.join(ckpt_dir, tag)
    if not os.path.exists(tag_dir):
        print(f"No checkpoint found at {tag_dir}")
        return

    print(f"\n── Checkpoint structure: {tag_dir} ────────────────────────────")
    for f in sorted(os.listdir(tag_dir)):
        fpath = os.path.join(tag_dir, f)
        size  = os.path.getsize(fpath) / 1024**2
        print(f"  {f:60s}  {size:8.2f} MB")
    print()


def consolidate_zero3_checkpoint(ckpt_dir, tag, output_path):
    """
    For ZeRO Stage 3: params are sharded across ranks.
    To get a normal PyTorch state dict, you must consolidate shards.

    DeepSpeed provides a utility for this.
    """
    from deepspeed.utils.zero_to_fp32 import (
        get_fp32_state_dict_from_zero_checkpoint
    )

    print(f"Consolidating ZeRO-3 checkpoint → {output_path}")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        ckpt_dir, tag=tag
    )

    torch.save(state_dict, output_path)
    n_params = sum(v.numel() for v in state_dict.values())
    print(f"  Saved fp32 state dict: {n_params/1e6:.2f}M params")
    return state_dict


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    deepspeed.init_distributed()
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    is_main    = (rank == 0)

    cfg = GPTConfig(vocab_size=65, max_seq_len=128,
                    d_model=256, n_heads=4, n_layers=4, dropout=0.0)

    for stage in [2, 3]:
        if is_main:
            print(f"\n{'='*55}")
            print(f"Checkpoint test: ZeRO Stage {stage}")
            print(f"{'='*55}")

        ds_config = {
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps":    1,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-3, "weight_decay": 0.1}
            },
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "stage3_gather_16bit_weights_on_model_save": True
                    if stage == 3 else False,
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 999,
        }

        torch.manual_seed(42)
        model  = GPT(cfg)
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(),
            config=ds_config,
        )

        # ── Train 5 steps ─────────────────────────────────────────────────
        x = torch.randint(0, 65, (4, 128), device=device)
        y = torch.randint(0, 65, (4, 128), device=device)
        losses_before = []
        for _ in range(5):
            _, loss = engine(x, y)
            engine.backward(loss)
            engine.step()
            losses_before.append(loss.item())

        if is_main:
            print(f"Losses before save: {[f'{l:.4f}' for l in losses_before]}")

        # ── Save checkpoint ───────────────────────────────────────────────
        ckpt_dir = f"checkpoints_ds/stage{stage}_test"
        tag      = "step_5"
        engine.save_checkpoint(ckpt_dir, tag=tag)
        dist.barrier()

        # ── Show checkpoint files ─────────────────────────────────────────
        if is_main:
            print_checkpoint_structure(ckpt_dir, tag)

        # ── Load checkpoint ───────────────────────────────────────────────
        _, client_state = engine.load_checkpoint(
            ckpt_dir, tag=tag,
            load_optimizer_states=True,
        )
        dist.barrier()

        # ── Verify loss hasn't changed ────────────────────────────────────
        losses_after = []
        with torch.no_grad():
            for _ in range(3):
                _, loss = engine(x, y)
                losses_after.append(loss.item())

        if is_main:
            print(f"Losses after reload: {[f'{l:.4f}' for l in losses_after]}")
            match = abs(losses_after[0] - losses_before[-1]) < 0.01
            print(f"Loss matches: {match}")
            print(f"ZeRO-{stage} checkpoint: {'PASSED' if match else 'FAILED'}")

        engine.destroy()
        del engine, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # ── Consolidate ZeRO-3 checkpoint to fp32 ─────────────────────────
        if stage == 3 and is_main:
            output_path = f"{ckpt_dir}/consolidated_fp32.pt"
            consolidate_zero3_checkpoint(ckpt_dir, tag, output_path)
            print(f"Consolidated model saved: {output_path}")

    if rank == 0:
        print("\nAll checkpoint tests complete.")


if __name__ == "__main__":
    main()
PY

mkdir -p checkpoints_ds
deepspeed --num_gpus=4 ds_checkpoint_lab.py
```

---

## 13.9 Lab 6 — CPU and NVMe Offloading

```bash
cat > ds_offload_lab.py <<'PY'
"""
CPU and NVMe offloading experiments.

CPU offload: moves optimizer states (and optionally params) to CPU RAM
  → GPU memory used ≈ params only
  → 3-5× slower than GPU optimizer (PCIe bandwidth bottleneck)

NVMe offload: moves to SSD (even more extreme)
  → Only for models that don't fit in CPU RAM
  → Very slow (disk I/O bottleneck)

This lab measures:
  - GPU memory with and without offload
  - Training speed penalty
"""
import os, gc, time
import torch
import torch.distributed as dist
import deepspeed

from model import GPT, GPTConfig


def build_offload_config(offload_device="none", stage=3,
                         micro_bs=16, nvme_path=None):
    """Build DS config with different offload options."""
    offload_opt = {"device": offload_device, "pin_memory": True}
    offload_par = {"device": offload_device, "pin_memory": True}

    if offload_device == "nvme" and nvme_path:
        offload_opt["nvme_path"] = nvme_path
        offload_par["nvme_path"] = nvme_path
        offload_opt["buffer_count"]       = 4
        offload_opt["buffer_size"]        = 1e8
        offload_par["buffer_count"]       = 5
        offload_par["buffer_size"]        = 1e8

    return {
        "train_micro_batch_size_per_gpu": micro_bs,
        "gradient_accumulation_steps":    1,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 6e-4, "weight_decay": 0.1}
        },
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "offload_optimizer": offload_opt,
            "offload_param": offload_par,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 999,
    }


def bench_offload(cfg_model, ds_config, device, B, T, iters=10):
    """Measure training step with given offload config."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    model   = GPT(cfg_model)
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config=ds_config,
    )

    x = torch.randint(0, 65, (B, T), device=device)
    y = torch.randint(0, 65, (B, T), device=device)

    # Warmup
    for _ in range(3):
        _, loss = engine(x, y)
        engine.backward(loss)
        engine.step()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        _, loss = engine(x, y)
        engine.backward(loss)
        engine.step()
    torch.cuda.synchronize()

    ms      = (time.perf_counter() - t0) * 1000 / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    engine.destroy()
    del engine, model
    gc.collect()
    torch.cuda.empty_cache()

    return ms, peak_gb


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    deepspeed.init_distributed()
    rank    = dist.get_rank()
    is_main = (rank == 0)

    B, T = 16, 256
    cfg  = GPTConfig(vocab_size=65, max_seq_len=T,
                     d_model=384, n_heads=6, n_layers=6, dropout=0.0)

    # Check available NVMe path
    nvme_path = "/tmp/nvme_offload" if os.path.exists("/tmp") else None
    if nvme_path:
        os.makedirs(nvme_path, exist_ok=True)

    experiments = [
        ("ZeRO-3 no offload",
         build_offload_config("none",  stage=3, micro_bs=B)),
        ("ZeRO-3 CPU opt offload",
         build_offload_config("cpu",   stage=3, micro_bs=B)),
    ]

    if nvme_path:
        experiments.append((
            "ZeRO-3 NVMe offload",
            build_offload_config("nvme", stage=3, micro_bs=B,
                                 nvme_path=nvme_path)
        ))

    results = []
    for name, ds_cfg in experiments:
        if is_main:
            print(f"Benchmarking: {name}...", flush=True)
        ms, peak = bench_offload(cfg, ds_cfg, device, B, T)
        results.append((name, ms, peak))
        if is_main:
            print(f"  done: ms={ms:.1f}  peak={peak:.3f}GB")

    if is_main:
        baseline_ms   = results[0][1]
        baseline_peak = results[0][2]

        print(f"\n{'='*65}")
        print(f"CPU/NVMe Offload Comparison  B={B}  T={T}")
        print(f"{'='*65}")
        print(f"{'Config':35s}  {'ms':>8}  "
              f"{'slowdown':>10}  {'peak_GB':>9}  {'mem%':>7}")
        print(f"{'─'*65}")

        for name, ms, peak in results:
            slowdown = ms / baseline_ms
            mem_pct  = peak / baseline_peak * 100
            print(f"  {name:33s}  {ms:>8.1f}  "
                  f"{slowdown:>9.2f}×  {peak:>9.3f}  "
                  f"{mem_pct:>6.1f}%")

        print(f"{'='*65}")
        print("\nCPU offload: slower but much less GPU memory.")
        print("NVMe offload: slowest but enables largest models.")
        print("Rule: use offload only when model doesn't fit otherwise.\n")


if __name__ == "__main__":
    main()
PY

deepspeed --num_gpus=4 ds_offload_lab.py
```

---

## 13.10 Lab 7 — DeepSpeed vs FSDP vs DDP Benchmark

```bash
cat > final_benchmark.py <<'PY'
"""
Head-to-head benchmark: DDP vs FSDP vs DeepSpeed.

Same model, same batch size, same hardware.
Measures: memory, speed, throughput.

This is the key experiment of Module 13.
"""
import os, gc, time, functools, json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import deepspeed

from model import GPT, GPTConfig, TransformerBlock


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ─── DDP benchmark ────────────────────────────────────────────────────────────

def bench_ddp(cfg, device, local_rank, B, T, iters=30):
    reset()
    torch.manual_seed(42)
    model = GPT(cfg).to(device)
    model = DDP(model, device_ids=[local_rank])
    opt   = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
    ctx   = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    x = torch.randint(0, 65, (B, T), device=device)
    y = torch.randint(0, 65, (B, T), device=device)

    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    torch.cuda.synchronize()
    dist.barrier()
    reset()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()

    ms   = (time.perf_counter() - t0) * 1000 / iters
    peak = torch.cuda.max_memory_allocated() / 1024**3

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return ms, peak


# ─── FSDP benchmark ───────────────────────────────────────────────────────────

def bench_fsdp(cfg, device, strategy_str, B, T, iters=30):
    reset()
    strategies = {
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD":    ShardingStrategy.FULL_SHARD,
    }
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    torch.manual_seed(42)
    model = FSDP(
        GPT(cfg),
        auto_wrap_policy  = wrap_policy,
        mixed_precision   = mp_policy,
        sharding_strategy = strategies[strategy_str],
        device_id         = torch.cuda.current_device(),
        use_orig_params   = True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    x = torch.randint(0, 65, (B, T), device=device)
    y = torch.randint(0, 65, (B, T), device=device)

    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        model.clip_grad_norm_(1.0)
        opt.step()

    torch.cuda.synchronize()
    dist.barrier()
    reset()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        model.clip_grad_norm_(1.0)
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()

    ms   = (time.perf_counter() - t0) * 1000 / iters
    peak = torch.cuda.max_memory_allocated() / 1024**3

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return ms, peak


# ─── DeepSpeed benchmark ──────────────────────────────────────────────────────

def bench_deepspeed(cfg, device, stage, B, T, iters=30):
    reset()
    ds_config = {
        "train_micro_batch_size_per_gpu": B,
        "gradient_accumulation_steps":    1,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 6e-4, "weight_decay": 0.1}
        },
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 9999,
    }

    torch.manual_seed(42)
    model  = GPT(cfg)
    engine, opt, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config=ds_config,
    )

    x = torch.randint(0, 65, (B, T), device=device)
    y = torch.randint(0, 65, (B, T), device=device)

    for _ in range(5):
        _, loss = engine(x, y)
        engine.backward(loss)
        engine.step()

    torch.cuda.synchronize()
    dist.barrier()
    reset()

    t0 = time.perf_counter()
    for _ in range(iters):
        _, loss = engine(x, y)
        engine.backward(loss)
        engine.step()
    torch.cuda.synchronize()
    dist.barrier()

    ms   = (time.perf_counter() - t0) * 1000 / iters
    peak = torch.cuda.max_memory_allocated() / 1024**3

    engine.destroy()
    del engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return ms, peak


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    deepspeed.init_distributed()
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    is_main    = (rank == 0)

    torch.backends.cuda.matmul.allow_tf32 = True

    B, T = 32, 256
    cfg  = GPTConfig(vocab_size=65, max_seq_len=T,
                     d_model=384, n_heads=6, n_layers=6, dropout=0.0)

    configs = [
        ("DDP",                      "ddp",        None),
        ("FSDP ZeRO-2",              "fsdp",       "SHARD_GRAD_OP"),
        ("FSDP ZeRO-3",              "fsdp",       "FULL_SHARD"),
        ("DeepSpeed ZeRO-1",         "deepspeed",  1),
        ("DeepSpeed ZeRO-2",         "deepspeed",  2),
        ("DeepSpeed ZeRO-3",         "deepspeed",  3),
    ]

    results = []

    for name, backend, param in configs:
        if is_main:
            print(f"Benchmarking: {name}...", flush=True)

        if backend == "ddp":
            ms, peak = bench_ddp(cfg, device, local_rank, B, T)
        elif backend == "fsdp":
            ms, peak = bench_fsdp(cfg, device, param, B, T)
        elif backend == "deepspeed":
            ms, peak = bench_deepspeed(cfg, device, param, B, T)

        tok_s = B * T * world_size / (ms / 1000)
        results.append((name, ms, peak, tok_s))

        if is_main:
            print(f"  {name}: ms={ms:.1f}  peak={peak:.3f}GB  "
                  f"tok/s={tok_s:.0f}")

    if is_main:
        baseline_ms   = results[0][1]
        baseline_peak = results[0][2]

        print(f"\n{'='*80}")
        print(f"DDP vs FSDP vs DeepSpeed — Final Benchmark")
        print(f"world_size={world_size}  GPU={torch.cuda.get_device_name(0)}")
        print(f"B={B}  T={T}  d=384  L=6")
        print(f"{'='*80}")
        print(f"{'Backend':25s}  {'ms':>7}  {'speedup':>8}  "
              f"{'peak_GB':>9}  {'mem%':>7}  {'tok/s':>10}")
        print(f"{'─'*80}")

        for name, ms, peak, tok_s in results:
            speedup  = baseline_ms / ms
            mem_pct  = peak / baseline_peak * 100
            print(f"  {name:23s}  {ms:>7.2f}  {speedup:>7.2f}×  "
                  f"{peak:>9.3f}  {mem_pct:>6.1f}%  {tok_s:>10.0f}")

        print(f"{'='*80}")
        print("\nGuide:")
        print("  DDP:              use when model fits easily (fastest)")
        print("  FSDP ZeRO-2/3:   use when model is tight/doesn't fit")
        print("  DeepSpeed ZeRO-2: similar to FSDP ZeRO-2, more configs")
        print("  DeepSpeed ZeRO-3: use for very large models + offload\n")


if __name__ == "__main__":
    main()
PY

deepspeed --num_gpus=8 final_benchmark.py
```

---

## 13.11 DeepSpeed Launcher Reference

### Method 1: `deepspeed` CLI (simplest)
```bash
# Single node, 8 GPUs
deepspeed --num_gpus=8 train.py --ds_config=zero2.json

# Single node, specific GPUs
deepspeed --include=localhost:0,1,2,3 train.py

# Multi-node (2 nodes, 8 GPUs each)
# Create hostfile first:
echo "node0 slots=8" > hostfile
echo "node1 slots=8" >> hostfile

deepspeed --num_nodes=2 --num_gpus=8 \
  --hostfile=hostfile train.py
```

### Method 2: `torchrun` + DeepSpeed (when using SLURM)
```bash
# DeepSpeed works with torchrun too
torchrun --nproc_per_node=8 \
  train_gpt_deepspeed.py \
  --ds_config=ds_configs/zero2.json
```

### Method 3: SLURM + DeepSpeed

```bash
cat > jobs/job_gpt_deepspeed_16gpu.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=gpt_deepspeed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/deepspeed_%j.out
#SBATCH --error=logs/deepspeed_%j.err

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/deepspeed_module

mkdir -p logs checkpoints_ds

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_ADDR MASTER_PORT

# DeepSpeed hostfile for multi-node
scontrol show hostnames "$SLURM_JOB_NODELIST" | \
  awk '{print $1 " slots=8"}' > /tmp/ds_hostfile_${SLURM_JOB_ID}

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

echo "MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "Hostfile:"
cat /tmp/ds_hostfile_${SLURM_JOB_ID}

# Method: torchrun + DeepSpeed
srun bash -c "
  torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_gpt_deepspeed.py \
    --ds_config=ds_configs/zero3.json \
    --max_iters=5000 \
    --ckpt_dir=checkpoints_ds/run_16gpu
"
SLURM

mkdir -p jobs logs
sbatch jobs/job_gpt_deepspeed_16gpu.sh
```

---

## 13.12 Debug Drills

### Drill 1 — Config validation error

```bash
cat > drill_config_error.py <<'PY'
"""
Common DeepSpeed config errors and how to debug them.
"""
import os
import torch
import deepspeed

from model import GPT, GPTConfig


def test_config(config, name):
    """Try to initialize with a config, catch errors."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                      d_model=128, n_heads=4, n_layers=2)
    model = GPT(cfg)

    try:
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(),
            config=config,
        )
        engine.destroy()
        print(f"  {name}: OK")
    except Exception as e:
        print(f"  {name}: FAILED - {type(e).__name__}: {str(e)[:80]}")
    finally:
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()


def main():
    deepspeed.init_distributed()

    print("\n── DeepSpeed Config Error Demo ─────────────────────────────────")

    # ── Error 1: fp16 and bf16 both enabled ─────────────────────────────
    test_config({
        "train_micro_batch_size_per_gpu": 4,
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
        "fp16": {"enabled": True},
        "bf16": {"enabled": True},   # ERROR: both enabled
        "zero_optimization": {"stage": 1},
    }, "both fp16+bf16 enabled")

    # ── Error 2: batch size mismatch ─────────────────────────────────────
    # train_batch_size must == micro_bs * grad_accum * world_size
    import torch.distributed as dist
    world_size = dist.get_world_size()
    test_config({
        "train_batch_size": 100,                          # wrong!
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        # 4 * 1 * world_size != 100 → error
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 1},
    }, f"batch size mismatch (100 != 4*1*{world_size}={4*world_size})")

    # ── Correct config ────────────────────────────────────────────────────
    test_config({
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2},
    }, "correct config")

    if dist.get_rank() == 0:
        print("\nFixes:")
        print("  1. Never enable both fp16 and bf16")
        print("  2. Ensure: train_batch = micro_bs × grad_accum × world_size")
        print("     OR: omit train_batch_size (DS computes it)")


if __name__ == "__main__":
    main()
PY

deepspeed --num_gpus=2 drill_config_error.py
```

---

### Drill 2 — NaN loss with fp16 (need loss scaling)

```bash
cat > drill_nan_loss.py <<'PY'
"""
fp16 with DeepSpeed: gradient overflow → NaN loss without scaler.
DeepSpeed handles this automatically when fp16.enabled=True.

This drill shows:
  1. NaN loss can happen with fp16 and large learning rates
  2. DeepSpeed's dynamic loss scaling prevents it
  3. bf16 doesn't have this problem
"""
import os
import torch
import torch.distributed as dist
import deepspeed

from model import GPT, GPTConfig


def run_training(use_fp16, use_bf16, lr, steps=20):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cfg = GPTConfig(vocab_size=65, max_seq_len=64,
                    d_model=128, n_heads=4, n_layers=2)
    model = GPT(cfg)

    ds_config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "weight_decay": 0.1}
        },
        "fp16": {
            "enabled": use_fp16,
            "loss_scale": 0,          # 0 = dynamic scaling
            "loss_scale_window": 100,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {"enabled": use_bf16},
        "zero_optimization": {"stage": 1},
        "gradient_clipping": 1.0,
        "steps_per_print": 999,
    }

    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config=ds_config,
    )

    x = torch.randint(0, 65, (4, 64), device=device)
    y = torch.randint(0, 65, (4, 64), device=device)

    nan_count = 0
    for step in range(steps):
        _, loss = engine(x, y)
        engine.backward(loss)
        engine.step()
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1

    final_loss = loss.item()
    engine.destroy()
    del engine, model
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return final_loss, nan_count


def main():
    deepspeed.init_distributed()
    rank    = dist.get_rank()
    is_main = (rank == 0)

    if is_main:
        print("\n── fp16 vs bf16 stability test ────────────────────────────────")
        print(f"{'Config':40s}  {'final_loss':>12}  {'nan_steps':>10}")
        print("─" * 70)

    tests = [
        ("fp16, lr=6e-4 (normal)",    True,  False, 6e-4),
        ("fp16, lr=1e-2 (high lr)",   True,  False, 1e-2),
        ("bf16, lr=1e-2 (high lr)",   False, True,  1e-2),
        ("bf16, lr=6e-4 (normal)",    False, True,  6e-4),
    ]

    for name, fp16, bf16, lr in tests:
        final_loss, nan_count = run_training(fp16, bf16, lr)
        if is_main:
            status = "⚠ unstable" if nan_count > 0 else "✓ stable"
            print(f"  {name:38s}  {final_loss:>12.4f}  "
                  f"{nan_count:>10}  {status}")

    if is_main:
        print("\nConclusion: bf16 is more stable than fp16 for large LRs.")
        print("DeepSpeed's dynamic loss scaling helps fp16 but can't always prevent NaN.")


if __name__ == "__main__":
    main()
PY

deepspeed --num_gpus=2 drill_nan_loss.py
```

---

### Drill 3 — engine.module vs engine direct access

```bash
python - <<'PY'
"""
DeepSpeed engine wraps the model.
Access the underlying model via engine.module.
"""
import os
import torch
import deepspeed
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29599")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

from model import GPT, GPTConfig

deepspeed.init_distributed()
torch.cuda.set_device(0)

cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                  d_model=128, n_heads=4, n_layers=2)
model = GPT(cfg)

engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config={
        "train_micro_batch_size_per_gpu": 2,
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 1},
        "steps_per_print": 999,
    }
)

print(f"\nengine type:        {type(engine)}")
print(f"engine.module type: {type(engine.module)}")

# WRONG: call generate on engine directly
try:
    out = engine.generate(
        torch.zeros(1, 1, dtype=torch.long, device="cuda"), 5
    )
except AttributeError as e:
    print(f"\nWRONG (engine.generate): {e}")

# CORRECT: call generate on engine.module
with torch.no_grad():
    out = engine.module.generate(
        torch.zeros(1, 1, dtype=torch.long, device="cuda"),
        max_new_tokens=5
    )
print(f"\nCORRECT (engine.module.generate): shape={out.shape}")

# WRONG: save engine state dict directly
try:
    sd = engine.state_dict()
    print(f"\nWRONG: engine.state_dict() returned {len(sd)} keys")
    # This may work but gives ZeRO shards, not full model
except Exception as e:
    print(f"\nWRONG (engine.state_dict): {e}")

# CORRECT: use engine.save_checkpoint() for DeepSpeed
print("CORRECT: use engine.save_checkpoint(path, tag) instead")

engine.destroy()
PY
```

---

### Drill 4 — ZeRO-3 parameter access outside forward

```bash
cat > drill_zero3_params.py <<'PY'
"""
With ZeRO Stage 3, parameters are sharded.
Accessing them outside forward/backward gives you only the local shard.

To inspect full parameters, use:
  deepspeed.zero.GatheredParameters() context manager
"""
import os
import torch
import torch.distributed as dist
import deepspeed

from model import GPT, GPTConfig


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    deepspeed.init_distributed()
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    is_main    = (rank == 0)

    cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                      d_model=128, n_heads=4, n_layers=2)
    model = GPT(cfg)

    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": 2,
            "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 3},
            "steps_per_print": 999,
        }
    )

    # Get first parameter
    first_param = next(engine.module.parameters())

    # Outside forward: see shard
    shard_numel = first_param.ds_numel  # ds_numel = shard size
    full_numel  = first_param.ds_shape  # ds_shape = original shape

    if is_main:
        print(f"\nZeRO-3 Parameter Access")
        print(f"  world_size:  {world_size}")
        print(f"  shard_numel: {shard_numel}  (local shard)")
        print(f"  full_shape:  {full_numel}  (original)")

    # Gather full parameters using context manager
    with deepspeed.zero.GatheredParameters(
        [first_param], modifier_rank=0
    ):
        if rank == 0:
            full_numel_gathered = first_param.numel()
            print(f"  full_numel (gathered): {full_numel_gathered}")
            print(f"  ratio: {full_numel_gathered/shard_numel:.1f}×")

    if is_main:
        print("\nRule: use GatheredParameters() to inspect ZeRO-3 params.")
        print("Outside GatheredParameters, you only see the local shard.")

    engine.destroy()


if __name__ == "__main__":
    main()
PY

deepspeed --num_gpus=4 drill_zero3_params.py
```

---

## 13.13 ZeRO-Infinity Config (NVMe, for reference)

```bash
cat > ds_configs/zero_infinity.json <<'JSON'
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    "bf16": {"enabled": true},

    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true,

        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        }
    },

    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": false,
        "overlap_events": true
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 50
}
JSON

echo "ZeRO-Infinity config created (requires NVMe at /local_nvme)"
echo "Use case: trillion-parameter models that don't fit in CPU RAM"
```

---

## 13.14 When to Use What — Decision Guide

```bash
cat > decision_guide.py <<'PY'
"""
Prints a decision guide for choosing between DDP, FSDP, DeepSpeed.
"""

guide = """
╔══════════════════════════════════════════════════════════════════════════╗
║          DISTRIBUTED TRAINING STRATEGY DECISION GUIDE                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Question 1: Does the model fit on a single GPU?                        ║
║                                                                          ║
║    YES → Use DDP                                                         ║
║      - Simplest, fastest, most debuggable                               ║
║      - Module 4 code                                                     ║
║                                                                          ║
║    NO → Go to Question 2                                                 ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Question 2: Prefer PyTorch-native or external library?                 ║
║                                                                          ║
║    PyTorch-native → FSDP                                                 ║
║      - Built into PyTorch (no extra dependencies)                       ║
║      - Code-driven (Pythonic)                                            ║
║      - ZeRO-2: SHARD_GRAD_OP                                            ║
║      - ZeRO-3: FULL_SHARD                                               ║
║      - With CPU offload: CPUOffload(offload_params=True)                ║
║                                                                          ║
║    External library → DeepSpeed                                          ║
║      - More config options                                               ║
║      - NVMe offload (ZeRO-Infinity)                                     ║
║      - Better support for very large models (>100B)                     ║
║      - Used by: Bloom, OPT, Megatron-LM                                 ║
║      - JSON config (easier to share / reproduce)                        ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Memory per GPU reference (N params, P GPUs):                           ║
║                                                                          ║
║  DDP:                    N × 16 bytes                                   ║
║  FSDP/DS ZeRO-1:         N × 10 bytes (approx)                         ║
║  FSDP/DS ZeRO-2:         N ×  6 bytes (approx)                         ║
║  FSDP/DS ZeRO-3:         N × 16/P bytes                                ║
║  ZeRO-3 + CPU opt:       N ×  4/P bytes (params only on GPU)           ║
║  ZeRO-Infinity:          N ×  2/P bytes (params sharded + NVMe)        ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Speed vs Memory tradeoff:                                               ║
║                                                                          ║
║  Fastest     DDP > FSDP ZeRO-2 > FSDP ZeRO-3 ≈ DS ZeRO-2 ≈ DS ZeRO-3 ║
║  Most memory savings:                                                   ║
║              ZeRO-Infinity > ZeRO-3+CPU > ZeRO-3 > ZeRO-2 > ZeRO-1    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
print(guide)
PY

python decision_guide.py
```

---

## 13.15 Module 13 Pass Criteria

You're done when you can:

1) **Explain DeepSpeed core concepts**:
   - ZeRO Stage 1/2/3 memory formulas
   - What the DeepSpeed Engine replaces (model + optimizer + scaler + grad clip)
   - Why `engine.backward()` not `loss.backward()`
   - Why `engine.save_checkpoint()` not `torch.save()`
   - CPU offload mechanism and when to use it

2) **Read and write DeepSpeed JSON configs**:
   - Set train_micro_batch_size_per_gpu, gradient_accumulation_steps
   - Configure ZeRO stage
   - Enable bf16 correctly (not both fp16 and bf16)
   - Set up CPU/NVMe offload
   - Understand that `train_batch_size = micro_bs × grad_acc × world_size`

3) **Run successfully**:
   - `deepspeed --num_gpus=8 ds_zero_stages.py` → compare all stages
   - `deepspeed --num_gpus=8 train_gpt_deepspeed.py` → training converges
   - `deepspeed --num_gpus=4 ds_checkpoint_lab.py` → checkpoint PASSED
   - `deepspeed --num_gpus=8 final_benchmark.py` → complete comparison table

4) **Fill your comparison table**:

| Backend | Stage | step_ms | peak_GB | tok/s |
|---------|-------|---------|---------|-------|
| DDP | - | | | |
| FSDP | ZeRO-2 | | | |
| FSDP | ZeRO-3 | | | |
| DeepSpeed | ZeRO-1 | | | |
| DeepSpeed | ZeRO-2 | | | |
| DeepSpeed | ZeRO-3 | | | |

5) **Know the key differences** between FSDP and DeepSpeed:
   - PyTorch-native vs external library
   - Code-driven vs config-driven
   - FSDP: `model.clip_grad_norm_()` vs DS: handled in `engine.step()`
   - FSDP: `FSDP.state_dict_type()` vs DS: `engine.save_checkpoint()`

---

