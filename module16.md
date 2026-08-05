# Module 16 — Production Training

## Summary
Everything so far you built from scratch. Now you read, annotate, and modify **real production code** used to train GPT-4, LLaMA, Bloom, OPT, and every major LLM. The goal: after this module, you can open any production training repo and understand every line.

- **NanoGPT**: Karpathy's minimal complete system — read every line, understand every design choice
- **HuggingFace Transformers**: `Trainer`, how it auto-selects DDP/FSDP/DeepSpeed
- **Megatron-LM**: `pretrain_gpt.py`, model sharding, data pipeline
- **DeepSpeed examples**: ZeRO config patterns used in production
- **PyTorch FSDP examples**: wrapping policies, mixed precision, checkpointing
- **Deep dives**: `launch.py`, `trainer.py`, `model.py`, `config.py`, `distributed.py` patterns
- **Hands-on**: clone each repo, instrument the code, run and measure

---

## 16.1 Theory — What Production Code Adds Over Our Code

### The gap between your Module 9-15 code and production

```
Your code (Module 9-15):
  ✓ Correct training loop
  ✓ DDP / FSDP / DeepSpeed
  ✓ Mixed precision
  ✓ Checkpointing
  ✓ Profiling

Production code adds:
  + Fault tolerance     (resume from ANY failure, not just clean exit)
  + Config management   (hydra, argparse, yaml — every hyperparameter CLI-settable)
  + Experiment tracking (wandb, tensorboard, MLflow — every run logged)
  + Evaluation harness  (standardized benchmarks, not just val loss)
  + Data pipeline       (streaming datasets, multiple datasets, sampling ratios)
  + LR scheduling       (warmup + decay + restart + cooldown)
  + Gradient clipping   (with monitoring — detect instability early)
  + Throughput tracking (tokens/sec, MFU — model FLOP utilization)
  + Logging discipline  (rank-0-only, structured, parseable)
  + Reproducibility     (exact seeds, dataset fingerprints, git commits logged)
  + Distributed testing (unit tests that run across 2/4/8 GPUs)
  + Model export        (save in multiple formats: PT, safetensors, ONNX)
```

### Model FLOP Utilization (MFU) — the production metric

Production teams don't report "GPU utilization %". They report MFU:

```
MFU = actual_throughput_flops / peak_hardware_flops

actual_throughput_flops = tokens_per_second × flops_per_token

flops_per_token (GPT, forward only):
  ≈ 6 × N   (where N = number of parameters)
  (2× matmul forward, 2× matmul backward, 2× optimizer)

A100 peak BF16 FLOPS: 312 × 10^12 FLOPS/s

Example: GPT-3 (175B params) at 142 tokens/s on 1024 A100s:
  actual = 142 × 6 × 175e9 = 149e12 FLOPS/s per A100
  MFU    = 149e12 / 312e12 = 47.8%

Typical MFU targets:
  Good single-node:   40-60%
  Multi-node DDP:     35-55%
  With tensor parallel: 30-50%
  Below 30%: something is wrong
```

---

## 16.2 Setup — Clone All Repos

```bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course
mkdir -p production_module
cd production_module

# Clone all repos
git clone https://github.com/karpathy/nanoGPT.git
git clone https://github.com/huggingface/transformers.git --depth=1
git clone https://github.com/NVIDIA/Megatron-LM.git      --depth=1
git clone https://github.com/microsoft/DeepSpeed.git     --depth=1
git clone https://github.com/pytorch/examples.git        --depth=1

# Install
pip install transformers datasets accelerate
pip install wandb
pip install omegaconf hydra-core
pip install safetensors

echo "All repos cloned:"
ls -d */
```

---

## 16.3 Study 1 — NanoGPT (Start Here)

NanoGPT is the best starting point: small, complete, production-quality, well-commented.

```bash
cat > study_nanogpt.py <<'PY'
"""
Annotated study of NanoGPT.

We read nanoGPT/train.py line by line, understand every decision,
then compare to our Module 9-15 implementations.

Key files:
  nanoGPT/model.py   — GPT model
  nanoGPT/train.py   — training loop
  nanoGPT/config/    — configurations
  nanoGPT/data/      — dataset preparation
"""

# ─── Read and annotate each section ────────────────────────────────────────

ANNOTATIONS = {
    "model.py": {
        "CausalSelfAttention": """
Key design decisions vs our Module 9 code:
  1. Uses F.scaled_dot_product_attention (FlashAttention when available)
     → we used manual attention matrix (less memory efficient)
  2. Single c_attn projection (Q+K+V combined)
     → same as ours (efficient: one big matmul vs 3 small ones)
  3. dropout on attn weights and residuals
     → same as ours
  4. register_buffer for bias (causal mask)
     → same as ours

Production difference:
  → Uses SDPA which auto-selects FlashAttention, memory-efficient, or math
     Based on: dtype, head_dim, sequence length, available CUDA version
        """,

        "MLP": """
Key design decisions:
  1. Uses nn.GELU(approximate='tanh')
     → tanh approximation is ~10% faster than exact GELU
     → minimal accuracy difference
  2. Simple nn.Linear (not our Column/Row parallel)
     → NanoGPT is single-GPU; no TP needed
        """,

        "Block": """
Same as ours:
  - Pre-LN (LayerNorm before attention and FFN)
  - Residual connections
  - ln_1, ln_2, attn, mlp
        """,

        "GPT.__init__": """
Key production additions:
  1. Weight initialization scaling:
       std = 0.02 / sqrt(2 * n_layer)
       Applied to c_proj (output projection) residual path
     → Prevents variance explosion in deep networks
     → GPT-2 paper technique

  2. Weight tying:
       self.lm_head.weight = self.transformer.wte.weight
     → Same as ours

  3. Parameter count logging
     → Always log param count (sanity check)
        """,

        "GPT.configure_optimizers": """
THIS IS A KEY PRODUCTION PATTERN:

  decay_params = [params with dim >= 2]  (weight matrices)
  no_decay_params = [params with dim < 2]  (biases, layernorm)

  AdamW only decays weight matrices, NOT biases or LayerNorm params.
  This is because weight decay on LayerNorm is harmful.

  Also checks for fused AdamW availability (CUDA extension):
    if fused available and device == cuda: use fused
    → Fused AdamW is significantly faster (fewer kernel launches)

  Returns optimizer with correct param groups.
  → Our code does the same (we learned from NanoGPT!)
        """,

        "GPT.estimate_mfu": """
Production metric: Model FLOP Utilization

  flops_per_token = 6 * n_params
  (approximate: each param touched ~6 times per token in fwd+bwd)

  actual_flops = flops_per_token * tokens_per_second
  mfu = actual_flops / peak_gpu_flops

  A100 bf16: peak = 312e12 FLOPS/s

  This is the REAL metric production teams optimize.
  Not "GPU utilization %" (which is meaningless).
        """,
    },

    "train.py": {
        "config_keys": """
NanoGPT uses a simple but effective config system:
  1. All hyperparameters as module-level globals
  2. Override from config file (exec(open(config_file).read()))
  3. Override from CLI (for k, v in overrides: exec(f'{k}={v}'))

More sophisticated than our argparse but simpler than Hydra.
Production teams use Hydra for larger projects.
        """,

        "ddp_setup": """
DDP setup pattern (same as our Module 4):
  ddp = int(os.environ.get('RANK', -1)) != -1
  if ddp:
      init_process_group(backend='nccl')
      ddp_rank = int(os.environ['RANK'])
      ddp_local_rank = int(os.environ['LOCAL_RANK'])
      ddp_world_size = int(os.environ['WORLD_SIZE'])
      device = f'cuda:{ddp_local_rank}'
      master_process = ddp_rank == 0
  else:
      # single GPU / CPU fallback
      ddp_rank = 0
      master_process = True

KEY: supports BOTH single-GPU and DDP with same code.
Use: if ddp: ... else: single-gpu-path
        """,

        "data_loading": """
NanoGPT uses memory-mapped numpy arrays:
  train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')

  def get_batch(split):
      data = train_data if split == 'train' else val_data
      ix = torch.randint(len(data) - block_size, (batch_size,))
      x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
      y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
      return x.to(device), y.to(device)

WHY memmap:
  - Dataset doesn't fit in RAM
  - OS handles paging transparently
  - Much faster than reading from disk per batch
  - Standard pattern for pretraining (dataset = single huge binary file)

COMPARISON TO OURS:
  Our TextDataset: loads full text into memory (fine for Shakespeare ~1MB)
  Production: memmap over TB-scale datasets
        """,

        "gradient_accumulation": """
NanoGPT gradient accumulation pattern:

  for micro_step in range(gradient_accumulation_steps):
      if ddp:
          model.require_backward_grad_sync = (
              micro_step == gradient_accumulation_steps - 1
          )
      with ctx:
          logits, loss = model(X, Y)
          loss = loss / gradient_accumulation_steps
      X, Y = get_batch('train')
      scaler.scale(loss).backward()

KEY: require_backward_grad_sync
  This is the NO_SYNC equivalent for older PyTorch versions.
  We use model.no_sync() context manager (cleaner API).
  Both achieve the same: suppress AllReduce until last micro-step.

ALSO NOTE: loss = loss / gradient_accumulation_steps
  Must normalize by accum steps or effective LR is too high.
        """,

        "lr_schedule": """
NanoGPT LR schedule:

  def get_lr(it):
      if it < warmup_iters:
          return learning_rate * it / warmup_iters
      if it > lr_decay_iters:
          return min_lr
      decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
      coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
      return min_lr + coeff * (learning_rate - min_lr)

IDENTICAL to our implementation in Module 10.
This is the standard GPT LR schedule (cosine decay after warmup).
        """,

        "checkpoint_save_load": """
NanoGPT checkpointing:

SAVE:
  if master_process:
      checkpoint = {
          'model': raw_model.state_dict(),  # raw_model = model.module if ddp
          'optimizer': optimizer.state_dict(),
          'model_args': model_args,
          'iter_num': iter_num,
          'best_val_loss': best_val_loss,
          'config': config,
      }
      torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

LOAD:
  checkpoint = torch.load(ckpt_path, map_location=device)
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  iter_num = checkpoint['iter_num']
  best_val_loss = checkpoint['best_val_loss']

KEY DIFFERENCES FROM OURS:
  1. Saves model_args (can reconstruct model from scratch)
  2. Saves config dict (full hyperparameter record)
  3. Saves iter_num (exact resume position)
  4. Only master_process saves (same as our rank==0 pattern)
        """,

        "mfu_tracking": """
Production throughput tracking in NanoGPT:

  t1 = time.time()
  dt = t1 - t0
  t0 = t1
  tokens_per_sec = gradient_accumulation_steps * ddp_world_size
                   * batch_size * block_size / dt
  mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)

  if master_process:
      print(f'iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, '
            f'mfu {mfu*100:.2f}%')

KEY: every log line shows MFU, not just loss.
This is how production teams know if training is efficient.
        """,
    }
}


def print_study(repo, file, section, text):
    print(f"\n{'='*70}")
    print(f"REPO: {repo}  FILE: {file}  SECTION: {section}")
    print(f"{'='*70}")
    print(text.strip())


def main():
    for file, sections in ANNOTATIONS.items():
        for section, annotation in sections.items():
            print_study("nanoGPT", file, section, annotation)

    print(f"\n{'='*70}")
    print("SUMMARY: What NanoGPT Does Better Than Our Code")
    print(f"{'='*70}")
    improvements = [
        "1. FlashAttention (SDPA) — we used manual attention matrix",
        "2. GELU approximate='tanh' — 10% faster, same accuracy",
        "3. memmap datasets — handles TB-scale, not just small files",
        "4. MFU tracking — meaningful metric, not 'GPU util %'",
        "5. model_args in checkpoint — can reconstruct without code changes",
        "6. Single-GPU + DDP same code path — cleaner fallback",
        "7. Weight decay separation — decay only 2D+ params",
    ]
    for imp in improvements:
        print(f"  {imp}")

    print("\nWhat WE have that NanoGPT doesn't:")
    ours = [
        "1. FSDP support (Module 12)",
        "2. DeepSpeed support (Module 13)",
        "3. Tensor/Pipeline parallelism (Module 14)",
        "4. Profiling infrastructure (Module 15)",
        "5. SLURM integration (Module 8)",
        "6. Multi-node support (Module 7)",
    ]
    for o in ours:
        print(f"  {o}")


if __name__ == "__main__":
    main()
PY

python study_nanogpt.py
```

---

## 16.4 NanoGPT — Run and Instrument

```bash
cat > run_nanogpt.sh <<'BASH'
#!/bin/bash
# Run NanoGPT on Shakespeare (our benchmark dataset)
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/production_module

# Prepare NanoGPT dataset
cd nanoGPT
python data/shakespeare_char/prepare.py

# Single GPU run with our usual config
python train.py \
    config/train_shakespeare_char.py \
    --device=cuda \
    --compile=False \
    --eval_iters=50 \
    --log_interval=10 \
    --block_size=256 \
    --batch_size=64 \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0 \
    --out_dir=out-shakespeare-nanogpt

cd ..
BASH

bash run_nanogpt.sh

# Multi-GPU NanoGPT
cat > run_nanogpt_ddp.sh <<'BASH'
#!/bin/bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/production_module/nanoGPT

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    config/train_shakespeare_char.py \
    --device=cuda \
    --compile=False \
    --eval_iters=50 \
    --log_interval=10 \
    --block_size=256 \
    --batch_size=64 \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --max_iters=2000 \
    --out_dir=out-shakespeare-nanogpt-ddp
BASH

bash run_nanogpt_ddp.sh
```

---

## 16.5 NanoGPT — Add MFU to Our Code

Extract the MFU calculation and add it to our training scripts:

```bash
cat > mfu_tracker.py <<'PY'
"""
Model FLOP Utilization (MFU) tracker.
Extracted from NanoGPT, adapted for our GPT + DDP/FSDP/DeepSpeed.

Usage:
  tracker = MFUTracker(n_params=model.count_params(), device="cuda")
  ...
  mfu = tracker.update(tokens_this_step)
  print(f"MFU: {mfu*100:.2f}%")
"""
import time
import torch
from dataclasses import dataclass, field
from typing import Optional


# A100 peak FLOPS by dtype (TFLOPS)
GPU_PEAK_FLOPS = {
    "A100": {
        "float32": 19.5e12,
        "float16": 77.97e12,
        "bfloat16": 77.97e12,
        "tf32": 156e12,
    },
    "H100": {
        "float32": 67e12,
        "float16": 1979e12,  # with sparsity
        "bfloat16": 1979e12,
        "tf32": 989e12,
    },
    "V100": {
        "float32": 14e12,
        "float16": 112e12,
        "bfloat16": 14e12,
        "tf32": 14e12,  # V100 doesn't support TF32
    },
}


def get_gpu_peak_flops(dtype="bfloat16") -> float:
    """Auto-detect GPU and return peak FLOPS."""
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" in gpu_name:
        gpu = "A100"
    elif "H100" in gpu_name:
        gpu = "H100"
    elif "V100" in gpu_name:
        gpu = "V100"
    else:
        # Unknown GPU: use conservative estimate
        return 100e12

    dtype_key = str(dtype).replace("torch.", "")
    return GPU_PEAK_FLOPS.get(gpu, {}).get(dtype_key, 100e12)


def estimate_flops_per_token(
    n_params:    int,
    n_layers:    int,
    d_model:     int,
    seq_len:     int,
    n_heads:     int,
    fwd_only:    bool = False,
) -> float:
    """
    Estimate FLOPs per token for GPT-style model.

    Follows PaLM paper convention:
      Per token, per forward:
        Attention: 4 * seq_len * d_model² / n_heads * n_heads = 4 * seq_len * d_model²
          (QKV + output projections)
        Actually simpler: 6N where N = params (for fwd+bwd)

    The 6N rule:
      fwd: 2N (matmul convention: 2 FLOPs per multiply-add)
      bwd: 4N (bwd is ~2× fwd due to two gradient computations)
      Total: 6N

    More precise for transformers:
      Attention per layer:
        QKV: 2 * seq * 3d²  = 6 * seq * d²
        Scores: 2 * seq² * d (Q@K^T)
        Values: 2 * seq² * d (Attn@V)
        Output: 2 * seq * d² = 2 * seq * d²
      FFN per layer:
        W1: 2 * seq * d * 4d = 8 * seq * d²
        W2: 2 * seq * 4d * d = 8 * seq * d²
      Total per layer: (6+2+2+2+8+8) * seq * d² ≈ 24 * seq * d²

    Simple approximation: 6 * n_params
    """
    # Simple approximation (from nanoGPT/PaLM papers)
    fwd_flops = 6 * n_params   # per token, for fwd+bwd combined
    if fwd_only:
        fwd_flops = 2 * n_params

    # Add attention FLOPs (quadratic in sequence length)
    # For each layer: 2 × 2 × seq × d = 4 × seq × d per head × n_heads
    attn_flops = n_layers * 2 * seq_len * d_model * 2
    # Usually small compared to 6N for short sequences

    return fwd_flops + attn_flops


@dataclass
class MFUTracker:
    """
    Tracks Model FLOP Utilization over training.

    Usage:
        tracker = MFUTracker(
            n_params=model.count_params(),
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            seq_len=cfg.max_seq_len,
        )

        # In training loop:
        t0 = time.perf_counter()
        # ... step ...
        dt = time.perf_counter() - t0
        mfu = tracker.update(tokens_processed=batch*seq, dt=dt)
    """
    n_params:    int
    n_layers:    int      = 6
    d_model:     int      = 384
    n_heads:     int      = 6
    seq_len:     int      = 256
    dtype:       str      = "bfloat16"
    world_size:  int      = 1

    # Running stats
    _mfu_history:  list   = field(default_factory=list)
    _step:         int    = 0

    def __post_init__(self):
        self.peak_flops     = get_gpu_peak_flops(self.dtype)
        self.flops_per_tok  = estimate_flops_per_token(
            self.n_params, self.n_layers, self.d_model,
            self.seq_len, self.n_heads,
        )
        if torch.cuda.is_available():
            print(f"MFUTracker: GPU={torch.cuda.get_device_name(0)}")
        print(f"  peak_flops:    {self.peak_flops/1e12:.1f} TFLOPS ({self.dtype})")
        print(f"  flops_per_tok: {self.flops_per_tok/1e6:.2f}M")
        print(f"  world_size:    {self.world_size}")

    def update(self, tokens_processed: int, dt: float) -> float:
        """
        Update MFU estimate.

        Args:
            tokens_processed: total tokens this step (batch × seq × world_size)
            dt:               elapsed wall time in seconds

        Returns:
            MFU (fraction 0-1)
        """
        tok_per_sec    = tokens_processed / dt
        actual_flops   = tok_per_sec * self.flops_per_tok
        mfu            = actual_flops / self.peak_flops

        self._mfu_history.append(mfu)
        self._step += 1
        return mfu

    def running_avg_mfu(self, last_n: int = 100) -> float:
        """Average MFU over last N steps."""
        hist = self._mfu_history[-last_n:]
        return sum(hist) / len(hist) if hist else 0.0

    def report(self):
        if not self._mfu_history:
            return
        avg  = self.running_avg_mfu(len(self._mfu_history))
        peak = max(self._mfu_history)
        print(f"\nMFU Report:")
        print(f"  Steps tracked: {self._step}")
        print(f"  Avg MFU:       {avg*100:.2f}%")
        print(f"  Peak MFU:      {peak*100:.2f}%")
        if avg < 0.30:
            print(f"  ⚠ MFU < 30%: significant inefficiency")
            print(f"    Check: DataLoader speed, kernel fusion, batch size")
        elif avg < 0.45:
            print(f"  ~ MFU 30-45%: room for improvement")
            print(f"    Try: torch.compile, FlashAttention, larger batches")
        else:
            print(f"  ✓ MFU > 45%: good utilization")


# ─── Test ──────────────────────────────────────────────────────────────────

def test_mfu_tracker():
    import time, math
    from model import GPT, GPTConfig

    device = "cuda"
    cfg    = GPTConfig(
        vocab_size=65, max_seq_len=256,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0,
    )
    model = GPT(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ctx   = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    B, T  = 32, 256

    tracker = MFUTracker(
        n_params   = model.count_params(),
        n_layers   = cfg.n_layers,
        d_model    = cfg.d_model,
        n_heads    = cfg.n_heads,
        seq_len    = cfg.max_seq_len,
        dtype      = "bfloat16",
        world_size = 1,
    )

    x = torch.randint(0, 65, (B, T), device=device)
    y = torch.randint(0, 65, (B, T), device=device)

    # Warmup
    for _ in range(3):
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    print("\nTraining with MFU tracking:")
    for step in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        opt.step()

        torch.cuda.synchronize()
        dt  = time.perf_counter() - t0
        mfu = tracker.update(tokens_processed=B*T, dt=dt)

        if step % 5 == 0:
            tok_s = B * T / dt
            print(f"  step={step:3d}  loss={loss.item():.4f}  "
                  f"tok/s={tok_s:.0f}  MFU={mfu*100:.2f}%")

    tracker.report()


if __name__ == "__main__":
    test_mfu_tracker()
PY

python mfu_tracker.py
```

---

## 16.6 Study 2 — HuggingFace Trainer

```bash
cat > study_hf_trainer.py <<'PY'
"""
Study HuggingFace Transformers Trainer.

Key files:
  transformers/trainer.py           — main Trainer class
  transformers/training_args.py     — TrainingArguments
  transformers/trainer_utils.py     — utilities
  transformers/integrations/        — wandb, deepspeed, etc.

Key design: ONE Trainer class handles:
  - Single GPU
  - DDP (auto-detected from torchrun env vars)
  - FSDP (via TrainingArguments.fsdp)
  - DeepSpeed (via TrainingArguments.deepspeed)

It auto-selects based on what's available and configured.
"""

HF_TRAINER_ANNOTATIONS = {
    "TrainingArguments": {
        "distributed_detection": """
HuggingFace auto-detects distributed context:

  # From trainer.py:
  if is_torch_tpu_available():
      self.place_model_on_device = False  # TPU handles placement
  elif is_sagemaker_mp_enabled():
      ...
  elif is_sagemaker_dp_enabled():
      ...
  elif self.args.local_rank != -1:   # torchrun sets this
      torch.distributed.init_process_group(backend="nccl")
      self.place_model_on_device = True

  HuggingFace checks env vars set by torchrun:
    LOCAL_RANK → args.local_rank
    RANK       → dist.get_rank()
    WORLD_SIZE → dist.get_world_size()

  Same pattern as our Module 4/7 code.
        """,

        "fsdp_setup": """
FSDP setup in HuggingFace:

  # From trainer.py:
  if self.fsdp is not None:
      if self.args.fsdp_min_num_params > 0:
          self.fsdp_config["min_num_params"] = self.args.fsdp_min_num_params

      fsdp_kwargs = self.args.xla_fsdp_config
      self.model = FSDP(
          model,
          auto_wrap_policy=...,      # from fsdp_config
          sharding_strategy=...,     # FULL_SHARD etc
          mixed_precision=...,       # bf16/fp16
          device_id=local_rank,
          **fsdp_kwargs,
      )

  Triggered by: --fsdp "full_shard auto_wrap"
  Much simpler than our Module 12 code because HF wraps it.
        """,

        "deepspeed_setup": """
DeepSpeed setup in HuggingFace:

  # From trainer.py:
  if self.args.deepspeed:
      from .integrations import HfDeepSpeedConfig
      hf_deepspeed_config = HfDeepSpeedConfig(self.args.deepspeed)
      # Adjusts DeepSpeed config based on HF model config
      # (batch size, grad accum, etc.)

      model, optimizer, _, lr_scheduler = deepspeed.initialize(
          model=model,
          model_parameters=model.parameters(),
          config=hf_deepspeed_config.config,
      )

  Triggered by: --deepspeed ds_config.json

  HF's DeepSpeed integration automatically:
    - Derives train_batch_size from HF training args
    - Handles fp16/bf16 config
    - Sets up optimizer from DeepSpeed config OR HF config
        """,

        "gradient_accumulation": """
HuggingFace gradient accumulation:

  # Simplified from trainer.py training_step:
  with self.compute_loss_context_manager():
      loss = self.compute_loss(model, inputs)

  if self.args.gradient_accumulation_steps > 1:
      loss = loss / self.args.gradient_accumulation_steps

  if self.do_grad_scaling:
      self.scaler.scale(loss).backward()
  elif self.use_apex:
      with amp.scale_loss(loss, self.optimizer) as scaled_loss:
          scaled_loss.backward()
  else:
      self.accelerator.backward(loss)

  Note: HF uses accelerate.backward() which routes to:
    - loss.backward() for DDP
    - engine.backward() for DeepSpeed
    - loss.backward() for FSDP
        """,

        "evaluation": """
HuggingFace evaluation:

  def evaluate(self, eval_dataset=None, ...):
      # Handles distributed eval automatically
      # Creates eval DataLoader with eval_sampler
      # If DDP: each rank evaluates its shard
      # Aggregates results via dist.all_reduce

      for inputs in eval_dataloader:
          with torch.no_grad():
              outputs = model(**inputs)

      # Gather predictions across all ranks
      all_preds = distributed_concat(preds, num_total_examples)
      all_labels = distributed_concat(labels, num_total_examples)

      # Compute metrics
      metrics = compute_metrics(EvalPrediction(
          predictions=all_preds, label_ids=all_labels
      ))

  Same pattern as our distributed evaluation in Module 10.
        """,
    },

    "model.py (GPT2)": {
        "flash_attention": """
HuggingFace GPT-2 uses:
  attn_output = torch.nn.functional.scaled_dot_product_attention(
      query, key, value,
      attn_mask=attention_mask,
      dropout_p=self.attn_dropout.p if self.training else 0.0,
      is_causal=is_causal,
  )

Exactly like NanoGPT — both moved to SDPA for FlashAttention support.
Our Module 9 uses manual attention matrix (less efficient).
        """,
    }
}


def print_section(section, content):
    print(f"\n{'─'*65}")
    print(f"  {section}")
    print(f"{'─'*65}")
    print(content.strip())


def main():
    print("="*65)
    print("HuggingFace Trainer — Architecture Study")
    print("="*65)

    for component, sections in HF_TRAINER_ANNOTATIONS.items():
        print(f"\n\n{'='*65}")
        print(f"COMPONENT: {component}")
        print(f"{'='*65}")
        for section, annotation in sections.items():
            print_section(section, annotation)

    print(f"\n\n{'='*65}")
    print("HuggingFace Trainer — Key Takeaways")
    print(f"{'='*65}")
    takeaways = [
        "1. One Trainer class, all backends: DDP / FSDP / DeepSpeed",
        "2. Auto-detection of distributed context (torchrun env vars)",
        "3. accelerate.backward() abstracts backend differences",
        "4. distributed_concat() for proper multi-rank evaluation",
        "5. DeepSpeed config JSON fully integrated (batch size auto-adjusted)",
        "6. FSDP via TrainingArguments.fsdp string ('full_shard auto_wrap')",
        "7. Uses SDPA (FlashAttention) in all modern HF models",
    ]
    for t in takeaways:
        print(f"  {t}")


if __name__ == "__main__":
    main()
PY

python study_hf_trainer.py
```

---

## 16.7 Run HuggingFace Training

```bash
cat > train_with_hf.py <<'PY'
"""
Train using HuggingFace Trainer with our GPT model.

Demonstrates:
  - HF Trainer with DDP (via torchrun)
  - HF Trainer with FSDP (via --fsdp flag)
  - HF Trainer with DeepSpeed (via --deepspeed flag)

All with the SAME training script — Trainer handles the backend.
"""
import os
import math
import torch
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    GPT2LMHeadModel,
    GPT2Config,
    default_data_collator,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


# ─── Build a simple char tokenizer in HF format ────────────────────────────────
class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int, vocab_size: int = 65):
        chars           = sorted(set(text))
        self.ch2id      = {c: i for i, c in enumerate(chars)}
        self.id2ch      = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_len    = seq_len
        self.tokens     = [self.ch2id[c] for c in text]

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx    : idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1  : idx+self.seq_len+1], dtype=torch.long)
        return {"input_ids": x, "labels": y}


# ─── Build HF-compatible GPT2 model ────────────────────────────────────────────
def build_gpt2_model(vocab_size: int = 65, n_ctx: int = 256,
                     n_embd: int = 384, n_layer: int = 6,
                     n_head: int = 6) -> GPT2LMHeadModel:
    cfg   = GPT2Config(
        vocab_size = vocab_size,
        n_ctx      = n_ctx,
        n_embd     = n_embd,
        n_layer    = n_layer,
        n_head     = n_head,
        bos_token_id = 0,
        eos_token_id = 0,
    )
    model = GPT2LMHeadModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HF GPT2 model: {n_params/1e6:.2f}M params")
    return model


# ─── Custom Trainer with MFU logging ───────────────────────────────────────────
class MFUTrainer(Trainer):
    """
    Extends HF Trainer with MFU calculation.
    Shows how to hook into Trainer's training loop.
    """
    def __init__(self, *args, n_params=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_params    = n_params
        self._mfu_log    = []

    def training_step(self, model, inputs):
        """Override to add MFU tracking."""
        import time
        t0   = time.perf_counter()
        loss = super().training_step(model, inputs)
        torch.cuda.synchronize()
        dt   = time.perf_counter() - t0

        # Compute MFU
        seq_len   = inputs["input_ids"].shape[1]
        bs        = inputs["input_ids"].shape[0]
        world     = self.args.world_size
        tok_s     = bs * seq_len * world / dt
        flops_tok = 6 * self.n_params
        peak      = 312e12   # A100 bf16
        mfu       = tok_s * flops_tok / peak

        self._mfu_log.append(mfu)
        return loss

    def log(self, logs):
        """Add MFU to logs."""
        if self._mfu_log:
            avg_mfu = sum(self._mfu_log[-50:]) / len(self._mfu_log[-50:])
            logs["mfu"] = round(avg_mfu * 100, 2)
        super().log(logs)


def main():
    text  = open("shakespeare.txt").read()
    SEQ   = 256
    train_size = int(len(text) * 0.9)

    train_ds = CharDataset(text[:train_size], seq_len=SEQ)
    val_ds   = CharDataset(text[train_size:], seq_len=SEQ)

    vocab_size = train_ds.vocab_size
    model      = build_gpt2_model(vocab_size=vocab_size)

    # ── Training Arguments ────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir              = "hf_output",
        num_train_epochs        = 5,
        per_device_train_batch_size = 64,
        per_device_eval_batch_size  = 64,
        gradient_accumulation_steps = 1,
        warmup_steps            = 200,
        learning_rate           = 6e-4,
        weight_decay            = 0.1,
        adam_beta1              = 0.9,
        adam_beta2              = 0.95,
        bf16                    = True,
        fp16                    = False,
        logging_steps           = 50,
        eval_steps              = 500,
        save_steps              = 500,
        evaluation_strategy     = "steps",
        save_strategy           = "steps",
        load_best_model_at_end  = True,
        dataloader_num_workers  = 4,
        dataloader_pin_memory   = True,
        report_to               = [],   # disable wandb for now
        run_name                = "shakespeare_gpt2",
        # For FSDP: add --fsdp "full_shard auto_wrap" to CLI
        # For DeepSpeed: add --deepspeed ds_configs/zero2.json to CLI
    )

    trainer = MFUTrainer(
        model          = model,
        args           = args,
        train_dataset  = train_ds,
        eval_dataset   = val_ds,
        data_collator  = default_data_collator,
        n_params       = sum(p.numel() for p in model.parameters()),
    )

    print(f"\nTrainer backend: {args.parallel_mode}")
    print(f"World size: {args.world_size}")
    print(f"Local rank: {args.local_rank}")

    trainer.train()

    print(f"\n── Training complete ─────────────────────────────────────────")
    if trainer._mfu_log:
        avg_mfu = sum(trainer._mfu_log) / len(trainer._mfu_log)
        print(f"  Average MFU: {avg_mfu*100:.2f}%")
    print(f"  Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
PY

# Single GPU (DDP = off)
python train_with_hf.py

# 8 GPU DDP
torchrun --nproc_per_node=8 train_with_hf.py

# 8 GPU FSDP
torchrun --nproc_per_node=8 train_with_hf.py \
    --fsdp "full_shard auto_wrap"

# 8 GPU DeepSpeed ZeRO-2
torchrun --nproc_per_node=8 train_with_hf.py \
    --deepspeed ../deepspeed_module/ds_configs/zero2.json
```

---

## 16.8 Study 3 — Megatron-LM Architecture

```bash
cat > study_megatron.py <<'PY'
"""
Study Megatron-LM code architecture.

Key files:
  pretrain_gpt.py           — entry point
  megatron/training.py      — training loop
  megatron/model/gpt_model.py — model definition
  megatron/arguments.py     — all CLI arguments
  megatron/initialize.py    — distributed init
  megatron/data/            — data pipeline

Megatron is what trains GPT-3 scale models at NVIDIA/Microsoft.
"""

MEGATRON_ANNOTATIONS = {
    "pretrain_gpt.py": """
ENTRY POINT PATTERN:

  def train_valid_test_datasets_provider(train_val_test_num_samples):
      # Returns dataset objects
      # Called AFTER distributed init
      ...

  def model_provider(pre_process=True, post_process=True):
      # Returns model for this pipeline stage
      # pre_process=True: include embedding layer (first PP stage)
      # post_process=True: include LM head (last PP stage)
      args = get_args()
      model = GPTModel(
          config=config,
          num_tokentypes=0,
          parallel_output=True,   # keep output distributed across TP ranks
          pre_process=pre_process,
          post_process=post_process,
      )
      return model

  def forward_step(data_iterator, model):
      # One forward step (called by Megatron's training loop)
      args = get_args()
      data = next(data_iterator)
      tokens, labels, mask = data['tokens'], data['labels'], data['mask']
      output_tensor = model(tokens, mask)
      return output_tensor, partial(loss_func, labels)

  if __name__ == '__main__':
      pretrain(
          train_valid_test_datasets_provider,
          model_provider,
          ModelType.encoder_or_decoder,
          forward_step,
      )

  KEY DESIGN: user provides 3 functions, Megatron handles:
    - distributed init (TP/PP/DP groups)
    - model wrapping for all parallelism types
    - training loop with 1F1B pipeline schedule
    - checkpointing
    - logging
    """,

    "megatron/initialize.py": """
DISTRIBUTED INIT (TP + PP + DP):

  def initialize_megatron(args):
      # 1. Init torch distributed
      torch.distributed.init_process_group(backend='nccl')

      # 2. Set up parallel process groups
      from megatron.core import parallel_state
      parallel_state.initialize_model_parallel(
          tensor_model_parallel_size  = args.tensor_model_parallel_size,
          pipeline_model_parallel_size= args.pipeline_model_parallel_size,
          # + sequence_parallel, expert_parallel, etc.
      )

      # 3. Set random seeds per rank
      # (different seeds ensure different data per DP rank)

  # parallel_state provides:
  parallel_state.get_tensor_model_parallel_group()
  parallel_state.get_pipeline_model_parallel_group()
  parallel_state.get_data_parallel_group()
  parallel_state.get_tensor_model_parallel_rank()
  parallel_state.get_pipeline_model_parallel_rank()
  parallel_state.is_pipeline_first_stage()
  parallel_state.is_pipeline_last_stage()

  This is exactly what we built in Module 14's process_groups_3d.py
  but production-grade and handles edge cases.
    """,

    "megatron/model/transformer.py": """
MEGATRON TRANSFORMER LAYER:

  CoreAttention:
    Uses flash attention (FA2) if available
    Falls back to memory-efficient or manual

  SelfAttention.forward():
    # TP: QKV via Column Parallel
    mixed_x_layer = self.query_key_value(hidden_states)
    # Split Q, K, V (local to TP rank's heads)
    (query, key, value) = split_tensor_along_last_dim(mixed_x_layer, 3)

    # Core attention (flash if available)
    context_layer = self.core_attention(query, key, value, mask)

    # Output via Row Parallel (AllReduce inside)
    output = self.dense(context_layer)

  MLP.forward():
    # TP: Column Parallel (GELU then Row Parallel)
    intermediate = self.dense_h_to_4h(hidden_states)  # ColParallel
    intermediate = self.activation_func(intermediate)  # local GELU
    output       = self.dense_4h_to_h(intermediate)    # RowParallel (AllReduce)

  This is exactly our Module 14 TensorParallelMLP and TensorParallelSelfAttention.
  Megatron calls them differently but the math is identical.
    """,

    "megatron/training.py": """
TRAINING LOOP:

  def train_step(forward_step_func, data_iterator, model, optimizer, lr_scheduler):
      if config.pipeline_model_parallel_size > 1:
          losses_reduced = forward_backward_pipelining_without_interleaving(
              forward_step_func, data_iterator, model,
              num_microbatches=get_num_microbatches(),
          )
      else:
          losses_reduced = forward_backward_no_pipelining(
              forward_step_func, data_iterator, model,
          )

      # Gradient clip
      clip_grad_norm(model, optimizer, args.clip_grad)

      # Optimizer step (handles ZeRO if using DeepSpeed)
      optimizer.step()

      return losses_reduced

  PIPELINE SCHEDULE (our Module 14 1F1B):
  forward_backward_pipelining_without_interleaving() implements:
    - Startup (warmup) phase: forward only
    - Steady state: 1F1B pairs
    - Cooldown: backward only
    - P2P send/recv between pipeline stages

  LOSSES REDUCED:
    After pipeline backward, losses are AllReduced across DP ranks.
    Only the last PP stage has losses (others have 0 contribution).
    AllReduce collects all contributions.
    """,

    "megatron/data/": """
DATA PIPELINE (production-grade):

  megatron/data/gpt_dataset.py:
    - Uses numpy memmap over tokenized binary files
    - Supports multiple datasets with sampling ratios
    - Example: 60% C4 + 30% Wikipedia + 10% Books

  DataLoader pattern:
    dataset = BlendedMegatronDataset(
        datasets = [c4_ds, wiki_ds, books_ds],
        weights  = [0.6, 0.3, 0.1],
    )

    Each DP rank gets different samples (via DistributedSampler or
    direct index partitioning based on DP rank + DP world size).

  KEY: data is tokenized OFFLINE (before training):
    python tools/preprocess_data.py
        --input raw_text.txt
        --output-prefix tokenized_data
        --vocab gpt2-vocab.json
        --tokenizer-type GPT2BPETokenizer

  Training reads from .bin (tokens) + .idx (document boundaries).
  This is 100× faster than tokenizing during training.
    """,
}


def main():
    print("="*65)
    print("Megatron-LM Architecture Study")
    print("="*65)
    for section, annotation in MEGATRON_ANNOTATIONS.items():
        print(f"\n{'='*65}")
        print(f"FILE: {section}")
        print(f"{'='*65}")
        print(annotation.strip())

    print(f"\n\n{'='*65}")
    print("Megatron-LM — What's Different From Our Code")
    print(f"{'='*65}")
    diffs = [
        "1. Production TP+PP+SP+EP in one unified framework",
        "2. Megatron Core (megatron.core): clean TP/PP API",
        "3. Sequence Parallelism (SP): reduces activation memory by TP factor",
        "4. Interleaved pipeline schedule: better pipeline efficiency",
        "5. Multi-dataset blending with sampling ratios",
        "6. Offline tokenization (binary format, much faster than online)",
        "7. FusedLayerNorm (CUDA extension): faster than PyTorch native",
        "8. Fused softmax and attention kernels",
        "9. Activation recomputation built into every block",
    ]
    for d in diffs:
        print(f"  {d}")


if __name__ == "__main__":
    main()
PY

python study_megatron.py
```

---

## 16.9 Build a Production Trainer (Putting It All Together)

```bash
cat > production_trainer.py <<'PY'
"""
Production-grade GPT trainer.

Synthesizes best practices from:
  - NanoGPT: config, checkpointing, MFU, memmap data
  - HuggingFace: one-class all-backends
  - Megatron: offline data, structured logging
  - Module 4-15: DDP, FSDP, profiling, memory

Features:
  ✓ Single GPU / DDP / FSDP (auto-detected)
  ✓ bf16 mixed precision
  ✓ Gradient accumulation
  ✓ MFU tracking
  ✓ Structured logging
  ✓ Checkpoint save + resume
  ✓ Fault-tolerant (signal handler saves checkpoint)
  ✓ Config from YAML / CLI
  ✓ Reproducibility (seeds + git hash)
  ✓ Memory profiling (optional)
  ✓ TensorBoard logging
"""
import os
import sys
import time
import math
import signal
import hashlib
import argparse
import subprocess
import contextlib
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from model import GPT, GPTConfig
from tokenizer import CharTokenizer
from mfu_tracker import MFUTracker


# ─── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Paths
    data_path:      str   = "shakespeare.txt"
    out_dir:        str   = "production_output"
    resume:         Optional[str] = None

    # Model
    vocab_size:     int   = 65
    max_seq_len:    int   = 256
    d_model:        int   = 384
    n_heads:        int   = 6
    n_layers:       int   = 6
    dropout:        float = 0.1

    # Training
    batch_size:     int   = 64     # per GPU
    accum_steps:    int   = 1
    max_iters:      int   = 5000
    lr:             float = 6e-4
    min_lr:         float = 6e-5
    warmup_iters:   int   = 200
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0

    # System
    dtype:          str   = "bfloat16"
    compile_model:  bool  = False
    seed:           int   = 1337

    # Logging
    log_interval:   int   = 50
    eval_interval:  int   = 500
    eval_iters:     int   = 50
    save_interval:  int   = 1000

    # Misc
    wandb_project:  Optional[str] = None
    profile_steps:  int   = 0     # 0 = no profiling

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


# ─── Reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed: int, rank: int):
    """Set seeds for reproducibility. Different per rank for data."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed + rank)


def get_git_hash() -> str:
    """Get current git commit hash for experiment tracking."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"


def get_data_fingerprint(path: str) -> str:
    """MD5 fingerprint of dataset file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:8]


# ─── Dataset ───────────────────────────────────────────────────────────────────

class StreamingTextDataset(Dataset):
    """
    Memory-mapped text dataset.
    Production pattern: data as flat binary file (like NanoGPT).
    """
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx     : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ─── Scheduler ─────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: Config) -> float:
    """Cosine LR with warmup (NanoGPT pattern)."""
    if step < cfg.warmup_iters:
        return cfg.lr * (step + 1) / cfg.warmup_iters
    if step > cfg.max_iters:
        return cfg.min_lr
    ratio = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ─── Production Trainer ────────────────────────────────────────────────────────

class ProductionTrainer:
    """
    Production-grade trainer.
    Handles: DDP, bf16, grad accum, checkpointing, MFU, logging.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._setup_distributed()
        self._setup_dirs()
        self._setup_logging()

    def _setup_distributed(self):
        """Auto-detect and init distributed."""
        self.is_ddp = (int(os.environ.get("RANK", -1)) != -1)

        if self.is_ddp:
            dist.init_process_group("nccl")
            self.rank       = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
        else:
            self.rank       = 0
            self.world_size = 1
            self.local_rank = 0

        self.device  = torch.device("cuda", self.local_rank)
        self.is_main = (self.rank == 0)

        torch.backends.cuda.matmul.allow_tf32 = True
        set_seeds(self.cfg.seed, self.rank)

    def _setup_dirs(self):
        if self.is_main:
            os.makedirs(self.cfg.out_dir, exist_ok=True)
            self.cfg.save(os.path.join(self.cfg.out_dir, "config.json"))

    def _setup_logging(self):
        self.writer  = None
        self.use_wandb = False

        if self.is_main:
            log_dir = os.path.join(self.cfg.out_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir)

            if self.cfg.wandb_project:
                try:
                    import wandb
                    wandb.init(
                        project = self.cfg.wandb_project,
                        config  = asdict(self.cfg),
                        name    = f"run_{get_git_hash()}",
                    )
                    self.use_wandb = True
                except ImportError:
                    self.log("wandb not installed, skipping")

    def log(self, msg: str):
        if self.is_main:
            print(msg, flush=True)

    def log_metrics(self, step: int, metrics: dict):
        if not self.is_main:
            return
        for k, v in metrics.items():
            if self.writer:
                self.writer.add_scalar(k, v, step)
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def _setup_data(self):
        """Load and distribute data."""
        if self.is_main:
            text = open(self.cfg.data_path).read()
            tok  = CharTokenizer(text)
            data = torch.tensor(tok.encode(text), dtype=torch.long)
            n    = int(len(data) * 0.9)
            train_data = data[:n]
            val_data   = data[n:]
            vocab_size = tok.vocab_size
            fingerprint = get_data_fingerprint(self.cfg.data_path)
            self.log(f"Data: {len(data):,} chars | "
                     f"vocab={vocab_size} | fingerprint={fingerprint}")
        else:
            train_data = val_data = None
            vocab_size = None

        if self.is_ddp:
            obj = [vocab_size, train_data, val_data]
            dist.broadcast_object_list(obj, src=0)
            vocab_size, train_data, val_data = obj

        self.cfg.vocab_size = vocab_size

        train_ds = StreamingTextDataset(train_data.cpu(), self.cfg.max_seq_len)
        val_ds   = StreamingTextDataset(val_data.cpu(),   self.cfg.max_seq_len)

        if self.is_ddp:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size,
                rank=self.rank, shuffle=True, drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_ds, num_replicas=self.world_size,
                rank=self.rank, shuffle=False, drop_last=False,
            )
        else:
            train_sampler = None
            val_sampler   = None

        self.train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size,
            sampler=train_sampler, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, persistent_workers=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size,
            sampler=val_sampler, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )
        self.train_sampler = train_sampler

    def _setup_model(self):
        """Build and wrap model."""
        gpt_cfg = GPTConfig(
            vocab_size  = self.cfg.vocab_size,
            max_seq_len = self.cfg.max_seq_len,
            d_model     = self.cfg.d_model,
            n_heads     = self.cfg.n_heads,
            n_layers    = self.cfg.n_layers,
            dropout     = self.cfg.dropout,
        )
        torch.manual_seed(self.cfg.seed)   # same init across all ranks
        model = GPT(gpt_cfg).to(self.device)

        if self.cfg.compile_model:
            self.log("Compiling model...")
            model = torch.compile(model)

        if self.is_ddp:
            model = DDP(model, device_ids=[self.local_rank])

        self.model    = model
        self.raw_model = model.module if self.is_ddp else model

        n_params = self.raw_model.count_params()
        self.log(f"Model: {n_params/1e6:.2f}M params")

        # MFU tracker
        self.mfu_tracker = MFUTracker(
            n_params   = n_params,
            n_layers   = self.cfg.n_layers,
            d_model    = self.cfg.d_model,
            n_heads    = self.cfg.n_heads,
            seq_len    = self.cfg.max_seq_len,
            dtype      = self.cfg.dtype,
            world_size = self.world_size,
        )

    def _setup_optimizer(self):
        """AdamW with weight decay separation (NanoGPT pattern)."""
        decay_p    = [p for n, p in self.raw_model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
        no_decay_p = [p for n, p in self.raw_model.named_parameters()
                      if p.requires_grad and p.dim() < 2]

        self.optimizer = torch.optim.AdamW(
            [{"params": decay_p,    "weight_decay": self.cfg.weight_decay},
             {"params": no_decay_p, "weight_decay": 0.0}],
            lr     = self.cfg.lr,
            betas  = (self.cfg.beta1, self.cfg.beta2),
            fused  = True,
        )

    def _amp_context(self):
        return torch.amp.autocast(
            device_type="cuda",
            dtype=getattr(torch, self.cfg.dtype),
        )

    def _save_checkpoint(self, step: int, val_loss: float, tag: str = "best"):
        """Save checkpoint (rank 0 only)."""
        if not self.is_main:
            return
        ckpt = {
            "step":       step,
            "model":      self.raw_model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "val_loss":   val_loss,
            "config":     asdict(self.cfg),
            "git_hash":   get_git_hash(),
        }
        path = os.path.join(self.cfg.out_dir, f"ckpt_{tag}.pt")
        torch.save(ckpt, path)
        self.log(f"[rank 0] Checkpoint → {path} (val_loss={val_loss:.4f})")

    def _load_checkpoint(self) -> int:
        """Load checkpoint if resume path set."""
        if not self.cfg.resume:
            return 0
        self.log(f"Resuming from {self.cfg.resume}")
        ckpt = torch.load(self.cfg.resume, map_location=self.device)
        self.raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt["step"]

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Distributed evaluation."""
        self.model.eval()
        total = torch.zeros(2, device=self.device)  # [loss_sum, count]

        ctx = self._amp_context()
        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.cfg.eval_iters:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with ctx:
                _, loss = self.model(x, y)
            total[0] += loss.item()
            total[1] += 1

        if self.is_ddp:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        self.model.train()
        return (total[0] / total[1]).item()

    def _install_signal_handler(self):
        """Save checkpoint on SIGTERM (SLURM sends this before kill)."""
        def handler(signum, frame):
            self.log(f"Signal {signum} received — saving emergency checkpoint")
            self._save_checkpoint(
                self._current_step, float("inf"), tag="emergency"
            )
            sys.exit(0)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT,  handler)

    def train(self):
        """Main training loop."""
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._install_signal_handler()

        start_step  = self._load_checkpoint()
        best_val    = float("inf")
        ctx         = self._amp_context()

        self.log(f"\n{'='*60}")
        self.log(f"Production GPT Training")
        self.log(f"  world_size:    {self.world_size}")
        self.log(f"  batch/GPU:     {self.cfg.batch_size}")
        self.log(f"  accum_steps:   {self.cfg.accum_steps}")
        self.log(f"  global_batch:  "
                 f"{self.cfg.batch_size*self.cfg.accum_steps*self.world_size}")
        self.log(f"  dtype:         {self.cfg.dtype}")
        self.log(f"  git_hash:      {get_git_hash()}")
        self.log(f"  data_fp:       {get_data_fingerprint(self.cfg.data_path)}")
        self.log(f"{'='*60}\n")

        torch.cuda.reset_peak_memory_stats(self.device)
        train_iter         = iter(self.train_loader)
        t0                 = time.perf_counter()
        self._current_step = start_step

        for step in range(start_step, self.cfg.max_iters):
            self._current_step = step

            # LR
            lr = get_lr(step, self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Sampler epoch
            if self.train_sampler is not None:
                epoch = step * self.cfg.batch_size // max(1, len(self.train_loader.dataset))
                self.train_sampler.set_epoch(epoch)

            # Gradient accumulation
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro in range(self.cfg.accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                is_last = (micro == self.cfg.accum_steps - 1)
                sync_ctx = contextlib.nullcontext() if is_last \
                           else (self.model.no_sync() if self.is_ddp
                                 else contextlib.nullcontext())

                with sync_ctx:
                    with ctx:
                        _, loss = self.model(x, y)
                    (loss / self.cfg.accum_steps).backward()

                accum_loss += loss.item() / self.cfg.accum_steps

            # Gradient clip + step
            if self.is_ddp:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            else:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.optimizer.step()

            # Logging
            if step % self.cfg.log_interval == 0:
                torch.cuda.synchronize()
                t1    = time.perf_counter()
                dt    = (t1 - t0) / self.cfg.log_interval
                t0    = t1

                tokens = (self.cfg.batch_size * self.cfg.accum_steps *
                          self.cfg.max_seq_len * self.world_size)
                mfu   = self.mfu_tracker.update(tokens_processed=tokens, dt=dt)
                tok_s = tokens / dt
                peak  = torch.cuda.max_memory_allocated(self.device) / 1024**3

                self.log(
                    f"step={step:5d}  loss={accum_loss:.4f}  "
                    f"lr={lr:.2e}  tok/s={tok_s:.0f}  "
                    f"mfu={mfu*100:.1f}%  peak={peak:.2f}GB"
                )

                self.log_metrics(step, {
                    "train/loss":   accum_loss,
                    "train/lr":     lr,
                    "train/mfu":    mfu * 100,
                    "perf/tok_s":   tok_s,
                    "mem/peak_gb":  peak,
                })

            # Evaluation
            if step % self.cfg.eval_interval == 0 or \
               step == self.cfg.max_iters - 1:
                val_loss = self._evaluate()
                self.log(f"\n── eval step={step} val_loss={val_loss:.4f} ──")
                self.log_metrics(step, {"eval/val_loss": val_loss})

                if val_loss < best_val:
                    best_val = val_loss
                    self._save_checkpoint(step, val_loss, tag="best")

                self.log("────────────────────────────────────────────\n")
                if self.is_ddp:
                    dist.barrier()

            # Periodic save
            if step > 0 and step % self.cfg.save_interval == 0:
                self._save_checkpoint(step, accum_loss, tag=f"step{step}")

        # Final summary
        self.mfu_tracker.report()
        if self.writer:
            self.writer.close()
        if self.is_ddp:
            dist.destroy_process_group()
        self.log(f"\nTraining complete. Best val_loss={best_val:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str,   default=None)
    parser.add_argument("--data_path",     type=str,   default="shakespeare.txt")
    parser.add_argument("--out_dir",       type=str,   default="production_output")
    parser.add_argument("--resume",        type=str,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--accum_steps",   type=int,   default=1)
    parser.add_argument("--max_iters",     type=int,   default=5000)
    parser.add_argument("--lr",            type=float, default=6e-4)
    parser.add_argument("--d_model",       type=int,   default=384)
    parser.add_argument("--n_layers",      type=int,   default=6)
    parser.add_argument("--n_heads",       type=int,   default=6)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--log_interval",  type=int,   default=50)
    parser.add_argument("--eval_interval", type=int,   default=500)
    parser.add_argument("--wandb_project", type=str,   default=None)
    args = parser.parse_args()

    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg = Config(**{k: v for k, v in vars(args).items()
                        if v is not None and k != "config"})

    trainer = ProductionTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
PY
```

Run it:
```bash
# Single GPU
python production_trainer.py \
    --batch_size=64 \
    --max_iters=2000 \
    --out_dir=production_output/single_gpu

# 8 GPUs DDP
torchrun --nproc_per_node=8 production_trainer.py \
    --batch_size=64 \
    --max_iters=2000 \
    --out_dir=production_output/8gpu

# Resume from checkpoint
torchrun --nproc_per_node=8 production_trainer.py \
    --batch_size=64 \
    --max_iters=2000 \
    --resume=production_output/8gpu/ckpt_best.pt \
    --out_dir=production_output/8gpu_resume
```

---

## 16.10 Study 4 — DeepSpeed Production Patterns

```bash
cat > study_deepspeed_prod.py <<'PY'
"""
Study production DeepSpeed patterns from the DeepSpeed repo.

Key files:
  deepspeed/runtime/engine.py          — DeepSpeedEngine
  deepspeed/runtime/zero/             — ZeRO implementation
  deepspeed/runtime/pipe/             — Pipeline Parallelism
  deepspeed/ops/                      — Custom CUDA ops
"""

DS_ANNOTATIONS = {
    "DeepSpeedEngine": """
The engine wraps your model and handles:
  1. ZeRO partitioning of params/grads/optim states
  2. Mixed precision (fp16/bf16) with loss scaling
  3. Gradient accumulation tracking
  4. Checkpoint save/load (per-rank ZeRO files)
  5. Wall clock timing breakdown

Key methods:
  engine.backward(loss)        — handles loss scaling, gradient accumulation
  engine.step()                — gradient clip + optimizer step + LR step
  engine.save_checkpoint(dir)  — ZeRO-aware checkpoint
  engine.load_checkpoint(dir)  — ZeRO-aware load (handles world_size change)

engine.global_rank    → dist.get_rank()
engine.world_size     → dist.get_world_size()
engine.local_rank     → LOCAL_RANK
engine.device         → torch.device("cuda", local_rank)
engine.module         → original model (unwrapped)
    """,

    "Zero3 communication": """
ZeRO Stage 3 communication flow (DeepSpeed implementation):

Forward pass:
  for each FWD_LINEAR in model:
      _allgather_params(linear.weight)    ← gather shard from all ranks
      output = linear(input)
      _free_params(linear.weight)         ← free gathered params immediately

Backward pass:
  for each BWD_LINEAR in model (reverse):
      _allgather_params(linear.weight)    ← re-gather for gradient computation
      grad = compute_gradient(output_grad, linear.weight)
      _free_params(linear.weight)         ← free again
      _reducescatter_grad(grad)           ← each rank gets its gradient shard

This is the same as FSDP FULL_SHARD but implemented in DeepSpeed.
The key difference: DeepSpeed hooks into PyTorch's autograd engine
via register_forward_pre_hook() and register_backward_hook().

Our Module 14 tensor parallel uses the same hook pattern
(CopyToModelParallelRegion, ReduceFromModelParallelRegion).
    """,

    "ZeRO-Infinity (NVMe)": """
ZeRO-Infinity extends ZeRO-3 to NVMe:

  GPU → [compute buffer]
  CPU → [optimizer states, some params]
  NVMe → [most params, gradients, optimizer states overflow]

Implementation uses async I/O (aio):
  - POSIX AIO or libaio for async NVMe reads
  - Prefetch next layer's params while current layer computes
  - Overlap: NVMe→CPU→GPU transfer with computation

This enables training models larger than GPU+CPU RAM combined.
Practical for: 1T+ parameter models on commodity hardware.

Config:
  "offload_param": {"device": "nvme", "nvme_path": "/local/nvme"}
  "offload_optimizer": {"device": "nvme", "nvme_path": "/local/nvme"}
  "aio": {"block_size": 1048576, "queue_depth": 8}
    """,

    "Pipeline Parallelism in DS": """
DeepSpeed pipeline (deepspeed/runtime/pipe/engine.py):

  class PipelineEngine(DeepSpeedEngine):
      def train_batch(self, data_iter):
          # 1F1B schedule
          # Exactly like our Module 14 pipeline_parallel.py
          # but integrated with ZeRO and mixed precision

  Key addition over Megatron:
    - Works with ZeRO (can shard params within each PP stage)
    - Integrated with DeepSpeed's communication engine
    - Supports gradient checkpointing per micro-batch

  Usage:
    model = deepspeed.pipe.PipelineModule(
        layers=[layer0, layer1, ..., layerN],
        num_stages=4,
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model, config=ds_config,
    )
    engine.train_batch(data_iter)
    """,
}


def main():
    print("="*65)
    print("DeepSpeed Production Patterns Study")
    print("="*65)
    for section, annotation in DS_ANNOTATIONS.items():
        print(f"\n{'='*65}")
        print(f"SECTION: {section}")
        print(f"{'='*65}")
        print(annotation.strip())

    print(f"\n\n{'='*65}")
    print("Production DeepSpeed Config — Complete Reference")
    print(f"{'='*65}")

    # Show annotated production config
    prod_config = """
{
    // Batch sizes: MUST satisfy:
    // train_batch_size = micro_bs × grad_accum × world_size
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 4,

    // AdamW with HF-compatible settings
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },

    // Cosine LR with warmup
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 6e-5,
            "warmup_max_lr": 6e-4,
            "warmup_num_steps": 200,
            "total_num_steps": 5000
        }
    },

    // bf16 on A100 (no GradScaler needed)
    "bf16": {"enabled": true},
    "fp16": {"enabled": false},

    // ZeRO-3 with reasonable buckets
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,    // 500MB AllGather bucket
        "overlap_comm": true,             // overlap comm with backward
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,           // optimizer sub-group size
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,  // params < 1M stay on GPU
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true,

        // CPU offload (comment out if memory allows without it)
        // "offload_optimizer": {"device": "cpu", "pin_memory": true},
        // "offload_param":     {"device": "cpu", "pin_memory": true}
    },

    // Gradient clipping (handles ZeRO correctly unlike nn.clip_grad)
    "gradient_clipping": 1.0,

    // Performance
    "steps_per_print": 50,
    "wall_clock_breakdown": false,   // true to see comm/compute/IO breakdown

    // Activation checkpointing (optional, integrated with ZeRO)
    "activation_checkpointing": {
        "partition_activations": false,  // true for very large models
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false
    }
}
"""
    print(prod_config)


if __name__ == "__main__":
    main()
PY

python study_deepspeed_prod.py
```

---

## 16.11 Study 5 — FSDP Examples (PyTorch Official)

```bash
cat > study_fsdp_examples.py <<'PY'
"""
Study PyTorch's official FSDP examples.

Location: examples/distributed/FSDP/

Key files:
  T5_training.py       — complete FSDP training example
  multinode_fsdp.py    — multi-node FSDP
  utils/               — wrapping policies, checkpointing utilities
"""

FSDP_EXAMPLE_ANNOTATIONS = {
    "Wrapping Policy (transformer)": """
Production FSDP wrapping for transformer:

  from torch.distributed.fsdp.wrap import (
      transformer_auto_wrap_policy,
      enable_wrap,
      wrap,
  )
  from functools import partial

  # Define which module class to wrap
  my_auto_wrap_policy = partial(
      transformer_auto_wrap_policy,
      transformer_layer_cls={TransformerDecoderLayer},  # your block class
  )

  model = FSDP(
      model,
      auto_wrap_policy     = my_auto_wrap_policy,
      sharding_strategy    = ShardingStrategy.FULL_SHARD,
      mixed_precision      = MixedPrecision(
          param_dtype  = torch.bfloat16,
          reduce_dtype = torch.float32,
          buffer_dtype = torch.bfloat16,
      ),
      device_id            = torch.cuda.current_device(),
      use_orig_params      = True,   # required for AdamW param groups
      limit_all_gathers    = True,   # prefetch control (memory vs speed)
  )

  KEY PARAMETER: limit_all_gathers
    False (default): prefetch next layer's AllGather eagerly
    True: limit prefetching (less memory, potentially slower)
    Use True when close to OOM with FSDP.
    """,

    "Mixed Precision in FSDP": """
FSDP MixedPrecision has 3 dtypes:

  param_dtype:  dtype for parameters DURING computation
    - bf16: compute in bf16 (fast Tensor Cores)
    - float32: compute in fp32 (slow but precise)

  reduce_dtype: dtype for AllGather/ReduceScatter
    - float32: more stable gradient accumulation (recommended)
    - bf16: faster communication, slight numerical difference

  buffer_dtype: dtype for buffers (attention mask, position bias, etc.)
    - Usually same as param_dtype

  Recommended A100 config:
    MixedPrecision(
        param_dtype  = torch.bfloat16,
        reduce_dtype = torch.float32,  # stable gradients
        buffer_dtype = torch.bfloat16,
    )

  fp16 config:
    MixedPrecision(
        param_dtype  = torch.float16,
        reduce_dtype = torch.float32,  # must be fp32 for stability
        buffer_dtype = torch.float16,
    )
    # Also need GradScaler with fp16!
    """,

    "FSDP Checkpoint (production)": """
Production FSDP checkpoint pattern:

  SAVE:
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(
            offload_to_cpu = True,   # gather to CPU (not GPU)
            rank0_only     = True,   # only rank 0 gets full dict
        ),
    ):
        state_dict = model.state_dict()

    if dist.get_rank() == 0:
        torch.save({"model": state_dict, "step": step}, path)

  LOAD:
    state_dict = torch.load(path, map_location="cpu")["model"]
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        model.load_state_dict(state_dict)

    KEY: load with rank0_only=False so ALL ranks load from same dict.
    FSDP then shards the loaded state appropriately.

  ALTERNATIVE: ShardedStateDictConfig
    Each rank saves its own shard (faster, but world_size dependent).
    Use when model is too large to fit full state on CPU.
    """,

    "Hybrid Sharding (intra-node)": """
HYBRID_SHARD: shard within node, replicate across nodes.

  model = FSDP(
      model,
      sharding_strategy = ShardingStrategy.HYBRID_SHARD,
      # Within-node: full param sharding (NVLink)
      # Across-nodes: replicated (like DDP, AllReduce over IB)
  )

  Memory: same as FULL_SHARD within node (params split across 8 GPUs)
  Communication: AllGather (NVLink, fast) + AllReduce (IB, slow)
  vs FULL_SHARD: AllGather (IB, slow across all GPUs)

  WHEN TO USE:
    - Model fits across one node's GPUs (8 × 40GB = 320GB)
    - Cross-node IB bandwidth is the bottleneck
    - Hybrid minimizes slow IB traffic (only gradients, not params)

  Real-world example:
    LLaMA-7B on 2 nodes × 8 A100s:
      FULL_SHARD: AllGather params across all 16 GPUs (all over IB)
      HYBRID_SHARD: AllGather within node (NVLink) + AllReduce across (IB)
      → HYBRID_SHARD is typically 15-30% faster for this scenario
    """,
}


def main():
    print("="*65)
    print("PyTorch FSDP Production Patterns")
    print("="*65)
    for section, annotation in FSDP_EXAMPLE_ANNOTATIONS.items():
        print(f"\n{'='*65}")
        print(f"SECTION: {section}")
        print(f"{'='*65}")
        print(annotation.strip())

    print(f"\n\n{'='*65}")
    print("FSDP Configuration Decision Table")
    print(f"{'='*65}")
    table = """
  Scenario                              Recommended Config
  ──────────────────────────────────────────────────────────────────────
  Model fits on 1 GPU                   DDP (skip FSDP)
  Model barely fits (tight)             FULL_SHARD (ZeRO-3)
  Multi-node, IB bottleneck             HYBRID_SHARD
  Single-node only                      FULL_SHARD
  Near OOM even with FULL_SHARD         FULL_SHARD + CPUOffload
  Very long sequence (activation OOM)   FULL_SHARD + activation_ckpt
  Fast inference required               FULL_SHARD + use_orig_params=True
                                        (then compile)
    """
    print(table)


if __name__ == "__main__":
    main()
PY

python study_fsdp_examples.py
```

---

## 16.12 Key Patterns Reference Card

```bash
cat > patterns_reference.py <<'PY'
"""
Production training patterns reference card.

Every pattern you'll encounter in real repos.
"""

PATTERNS = {
    "Config Management": {
        "simple (NanoGPT)": """
# All hyperparams as globals, override via exec()
# config/train.py:
out_dir = 'out'
max_iters = 600000
...

# Override from CLI:
# python train.py config/train.py --max_iters=10000 --lr=1e-3
""",
        "argparse (ours)": """
parser = argparse.ArgumentParser()
parser.add_argument('--max_iters', type=int, default=5000)
...
args = parser.parse_args()
""",
        "Hydra (large teams)": """
@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.train()

# Launch: python train.py training=gpt3_175b
# Override: python train.py training.lr=1e-4 model.n_layers=96
""",
    },

    "DDP Setup": {
        "minimal": """
dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = DDP(model, device_ids=[local_rank])
""",
        "auto-detect (NanoGPT pattern)": """
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    master_process = ddp_rank == 0
else:
    ddp_rank = 0; master_process = True; device = 'cuda'
""",
    },

    "Checkpoint Save": {
        "DDP": """
if rank == 0:
    torch.save({
        'model':      model.module.state_dict(),
        'optimizer':  opt.state_dict(),
        'step':       step,
        'val_loss':   val_loss,
        'config':     asdict(cfg),
        'git_hash':   git_hash,
    }, 'ckpt.pt')
dist.barrier()
""",
        "FSDP": """
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
    state = model.state_dict()
if rank == 0:
    torch.save({'model': state, 'step': step}, 'ckpt.pt')
dist.barrier()
""",
        "DeepSpeed": """
engine.save_checkpoint(save_dir, tag=f'step_{step}')
# Consolidate ZeRO-3: python -m deepspeed.utils.zero_to_fp32 save_dir step_N
""",
    },

    "Gradient Accumulation (DDP)": {
        "no_sync pattern": """
for micro_step in range(accum_steps):
    is_last = (micro_step == accum_steps - 1)
    ctx = contextlib.nullcontext() if is_last else model.no_sync()
    with ctx:
        with amp_ctx: _, loss = model(x, y)
        (loss / accum_steps).backward()
opt.step()
""",
    },

    "Distributed Evaluation": {
        "standard": """
model.eval()
total_loss = torch.zeros(1, device=device)
count      = torch.zeros(1, device=device)
with torch.no_grad():
    for x, y in val_loader:
        _, loss = model(x.to(device), y.to(device))
        total_loss += loss; count += 1
dist.all_reduce(total_loss); dist.all_reduce(count)
val_loss = (total_loss / count).item()
model.train()
""",
    },

    "MFU Calculation": {
        "from NanoGPT": """
# tokens_per_sec * flops_per_token / gpu_peak_flops
flops_per_token = 6 * n_params  # approx (fwd+bwd)
actual_flops    = tokens_per_sec * flops_per_token
mfu             = actual_flops / a100_bf16_peak_flops
# a100_bf16_peak_flops = 312e12
""",
    },

    "Fault Tolerance": {
        "SIGTERM handler (SLURM)": """
def save_and_exit(signum, frame):
    if rank == 0:
        torch.save(checkpoint, 'emergency_ckpt.pt')
    sys.exit(0)
signal.signal(signal.SIGTERM, save_and_exit)
signal.signal(signal.SIGUSR1, save_and_exit)
""",
        "SLURM auto-resubmit": """
# In sbatch script:
#SBATCH --signal=SIGUSR1@90  # send 90s before timeout

# In Python, catch SIGUSR1, save checkpoint, resubmit:
def handler(sig, frame):
    save_checkpoint(...)
    if rank == 0:
        os.system(f'sbatch {sys.argv[0]}')
    sys.exit(0)
signal.signal(signal.SIGUSR1, handler)
""",
    },

    "Logging discipline": {
        "rank 0 only": """
def log(msg):
    if rank == 0:
        print(msg, flush=True)
        if writer: writer.add_scalar(...)
        if wandb_run: wandb.log(...)
""",
        "structured (JSON lines)": """
import json
if rank == 0:
    log_entry = {
        'step': step, 'loss': loss, 'mfu': mfu,
        'lr': lr, 'tok_s': tok_s, 'ts': time.time(),
    }
    with open('logs/train.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\\n')
""",
    },
}


def main():
    print("="*70)
    print("PRODUCTION TRAINING PATTERNS REFERENCE")
    print("="*70)

    for category, patterns in PATTERNS.items():
        print(f"\n\n{'─'*70}")
        print(f"  {category}")
        print(f"{'─'*70}")
        for name, code in patterns.items():
            print(f"\n  [{name}]")
            for line in code.strip().split('\n'):
                print(f"    {line}")


if __name__ == "__main__":
    main()
PY

python patterns_reference.py
```

---

## 16.13 Lab — Read and Modify Real Repo Code

```bash
cat > lab_read_repos.sh <<'BASH'
#!/bin/bash
# Guided exploration of real repos

echo "════════════════════════════════════════════"
echo "LAB: Reading Real Production Code"
echo "════════════════════════════════════════════"

echo ""
echo "1. NanoGPT — key functions"
echo "─────────────────────────────────────────"
echo "Files to read:"
echo "  nanoGPT/model.py"
echo "  nanoGPT/train.py"
echo ""
echo "grep: how NanoGPT handles DDP:"
grep -n "ddp\|DDP\|distributed" nanoGPT/train.py | head -20

echo ""
echo "grep: how NanoGPT does gradient accumulation:"
grep -n "gradient_accumulation\|micro_step\|no_sync\|require_backward" \
    nanoGPT/train.py | head -15

echo ""
echo "grep: MFU in NanoGPT:"
grep -n "mfu\|MFU\|flops" nanoGPT/model.py | head -10

echo ""
echo "════════════════════════════════════════════"
echo "2. HuggingFace Trainer"
echo "─────────────────────────────────────────"
echo "Files to read:"
echo "  transformers/src/transformers/trainer.py"
echo "  transformers/src/transformers/training_args.py"
echo ""
echo "grep: FSDP setup in HF:"
grep -rn "FullyShardedDataParallel\|FSDP\|fsdp" \
    transformers/src/transformers/trainer.py 2>/dev/null | head -10

echo ""
echo "grep: DeepSpeed setup in HF:"
grep -rn "deepspeed.initialize\|deepspeed_init" \
    transformers/src/transformers/trainer.py 2>/dev/null | head -10

echo ""
echo "════════════════════════════════════════════"
echo "3. Megatron-LM"
echo "─────────────────────────────────────────"
echo "Files to read:"
echo "  Megatron-LM/pretrain_gpt.py"
echo "  Megatron-LM/megatron/training.py"
echo "  Megatron-LM/megatron/model/transformer.py"
echo ""
echo "grep: pipeline schedule in Megatron:"
grep -n "forward_backward_pipelining\|schedule\|1F1B" \
    Megatron-LM/megatron/training.py 2>/dev/null | head -10

echo ""
echo "grep: tensor parallel in Megatron:"
grep -n "ColumnParallelLinear\|RowParallelLinear" \
    Megatron-LM/megatron/model/transformer.py 2>/dev/null | head -10

echo ""
echo "════════════════════════════════════════════"
echo "4. DeepSpeed ZeRO"
echo "─────────────────────────────────────────"
echo "grep: AllGather in ZeRO stage 3:"
grep -rn "allgather\|all_gather" \
    DeepSpeed/deepspeed/runtime/zero/stage3.py 2>/dev/null | head -10

echo ""
echo "grep: ReduceScatter in ZeRO:"
grep -rn "reduce_scatter\|reducescatter" \
    DeepSpeed/deepspeed/runtime/zero/stage3.py 2>/dev/null | head -10
BASH

bash lab_read_repos.sh
```

---

## 16.14 Module 16 Pass Criteria

You're done when you can:

1) **Read NanoGPT** (`train.py` + `model.py`) line by line and explain:
   - How DDP is auto-detected (no crash if not distributed)
   - Why `require_backward_grad_sync` vs `no_sync()`
   - What `estimate_mfu()` computes and why
   - The memmap data pattern and why it matters at scale

2) **Navigate HuggingFace Trainer**:
   - How it detects DDP/FSDP/DeepSpeed from TrainingArguments
   - Where `accelerator.backward(loss)` routes for each backend
   - How to add a custom metric (like MFU) to `Trainer.log()`

3) **Understand Megatron-LM**:
   - What `model_provider(pre_process, post_process)` returns per PP stage
   - How `parallel_state` manages TP/PP/DP groups
   - Where AllReduce for TP happens (RowParallel forward)

4) **Run production_trainer.py**:
   - `python production_trainer.py` (single GPU)
   - `torchrun --nproc_per_node=8 production_trainer.py` (DDP)
   - Verify MFU is reported every `log_interval` steps
   - Verify checkpoint resumes correctly: `--resume=output/ckpt_best.pt`

5) **Know all production patterns**:
   - Config: argparse vs Hydra vs exec-based
   - Checkpoint: save on rank 0, load barrier pattern
   - Logging: rank 0 only, structured JSON lines
   - Fault tolerance: SIGTERM handler, SLURM auto-resubmit
   - Evaluation: all_reduce correct/total across ranks
   - Grad accum: `no_sync()` + loss normalization

6) **Implement MFU tracker**:
   - `python mfu_tracker.py` runs cleanly
   - Gives ≥ 30% MFU on your A100 for the GPT config

---

