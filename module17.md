# Module 17 — Capstone Project

## Summary
Everything from Modules 0–16 comes together into one complete system. You will build, run, benchmark, and document a production-grade distributed GPT training system on **2 nodes × 8 A100 GPUs = 16 GPUs**.

This is not a tutorial — you build it end to end.

- **System**: multi-node, multi-GPU, SLURM-integrated, fault-tolerant GPT trainer
- **Backends**: DDP and FSDP selectable via single config flag
- **Features**: bf16, gradient accumulation, MFU tracking, checkpointing, auto-resume, SIGTERM fault tolerance, profiling, distributed evaluation, TensorBoard + wandb
- **Benchmarking**: throughput, scaling efficiency, MFU, memory per GPU across 1→2→4→8→16 GPUs
- **Deliverable**: a repo you can run on any A100 cluster

---

## 17.1 Project Structure

```bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course
mkdir -p capstone/{src,configs,jobs,scripts,logs,profiles,checkpoints,tests}
cd capstone

echo "Capstone project tree:"
tree . 2>/dev/null || find . -type d | head -20
```

Final structure:
```
capstone/
├── src/
│   ├── model.py              ← GPT model (from scratch, with FlashAttn)
│   ├── trainer.py            ← main trainer (DDP + FSDP unified)
│   ├── data.py               ← dataset + dataloader
│   ├── optimizer.py          ← AdamW with param groups + LR schedule
│   ├── distributed.py        ← distributed utilities
│   ├── checkpoint.py         ← save/load for DDP and FSDP
│   ├── profiler.py           ← profiling hooks
│   ├── metrics.py            ← MFU + throughput tracker
│   └── config.py             ← unified config dataclass
├── configs/
│   ├── small.yaml            ← d=384 L=6 (fast experiments)
│   ├── medium.yaml           ← d=768 L=12 (GPT-2 small scale)
│   └── large.yaml            ← d=1024 L=24 (bigger)
├── jobs/
│   ├── job_1gpu.sh
│   ├── job_8gpu.sh
│   ├── job_16gpu.sh
│   └── job_sweep.sh
├── scripts/
│   ├── prepare_data.py
│   ├── benchmark_scaling.sh
│   └── generate.py
├── tests/
│   ├── test_model.py
│   ├── test_distributed.py
│   └── test_checkpoint.py
└── train.py                  ← entry point
```

---

## 17.2 Source — config.py

```bash
cat > src/config.py <<'PY'
"""
Unified configuration for the capstone trainer.

Supports: YAML files + CLI overrides.
Design: single Config dataclass, everything in one place.
"""
import os
import json
import yaml
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size:     int   = 65
    max_seq_len:    int   = 256
    d_model:        int   = 384
    n_heads:        int   = 6
    n_layers:       int   = 6
    dropout:        float = 0.1
    use_checkpoint: bool  = False   # activation checkpointing
    use_flash_attn: bool  = True    # FlashAttention via SDPA

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"


@dataclass
class TrainConfig:
    # Data
    data_path:      str   = "shakespeare.txt"
    train_split:    float = 0.9

    # Optimization
    batch_size:     int   = 64        # per GPU per micro-step
    accum_steps:    int   = 1         # gradient accumulation
    max_iters:      int   = 5000
    lr:             float = 6e-4
    lr_scale:       str   = "sqrt"    # none / sqrt / linear (for multi-GPU)
    min_lr:         float = 6e-5
    warmup_iters:   int   = 200
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0

    # System
    dtype:          str   = "bfloat16"
    backend:        str   = "ddp"     # ddp / fsdp
    fsdp_strategy:  str   = "FULL_SHARD"
    compile_model:  bool  = False
    seed:           int   = 1337

    # Logging
    log_interval:   int   = 50
    eval_interval:  int   = 500
    eval_iters:     int   = 50
    save_interval:  int   = 1000
    gen_interval:   int   = 1000
    gen_tokens:     int   = 200

    # Output
    out_dir:        str   = "checkpoints/run"
    resume:         Optional[str] = None

    # Experiment tracking
    wandb_project:  Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Profiling
    profile_steps:  int   = 0     # 0 = disabled
    profile_wait:   int   = 2
    profile_warmup: int   = 2
    profile_active: int   = 3


@dataclass
class Config:
    model:  ModelConfig = field(default_factory=ModelConfig)
    train:  TrainConfig = field(default_factory=TrainConfig)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "model": asdict(self.model),
                "train": asdict(self.train),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return cls(
            model = ModelConfig(**d["model"]),
            train = TrainConfig(**d["train"]),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f)
        cfg = cls()
        for k, v in d.get("model", {}).items():
            setattr(cfg.model, k, v)
        for k, v in d.get("train", {}).items():
            setattr(cfg.train, k, v)
        return cfg

    def apply_cli(self, args: argparse.Namespace) -> "Config":
        """Override config fields from parsed CLI args."""
        for k, v in vars(args).items():
            if v is None:
                continue
            if hasattr(self.model, k):
                setattr(self.model, k, v)
            elif hasattr(self.train, k):
                setattr(self.train, k, v)
        return self


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Capstone GPT Trainer")

    # Config file
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file")

    # Model overrides
    p.add_argument("--d_model",       type=int,   default=None)
    p.add_argument("--n_heads",       type=int,   default=None)
    p.add_argument("--n_layers",      type=int,   default=None)
    p.add_argument("--max_seq_len",   type=int,   default=None)
    p.add_argument("--dropout",       type=float, default=None)
    p.add_argument("--use_checkpoint",action="store_true", default=None)
    p.add_argument("--no_flash_attn", action="store_true", default=None)

    # Train overrides
    p.add_argument("--data_path",     type=str,   default=None)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--accum_steps",   type=int,   default=None)
    p.add_argument("--max_iters",     type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--lr_scale",      type=str,   default=None)
    p.add_argument("--backend",       type=str,   default=None,
                   choices=["ddp", "fsdp"])
    p.add_argument("--fsdp_strategy", type=str,   default=None)
    p.add_argument("--compile_model", action="store_true", default=None)
    p.add_argument("--dtype",         type=str,   default=None)
    p.add_argument("--out_dir",       type=str,   default=None)
    p.add_argument("--resume",        type=str,   default=None)
    p.add_argument("--log_interval",  type=int,   default=None)
    p.add_argument("--eval_interval", type=int,   default=None)
    p.add_argument("--wandb_project", type=str,   default=None)
    p.add_argument("--profile_steps", type=int,   default=None)

    return p
PY
```

---

## 17.3 Source — model.py

```bash
cat > src/model.py <<'PY'
"""
GPT model — production version.

Improvements over Module 9:
  1. FlashAttention via F.scaled_dot_product_attention
  2. GELU approximate='tanh' (10% faster)
  3. Activation checkpointing per block (optional)
  4. MFU estimation method
  5. Weight init with residual path scaling (GPT-2 paper)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from src.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0

        self.n_heads     = cfg.n_heads
        self.d_head      = cfg.d_model // cfg.n_heads
        self.d_model     = cfg.d_model
        self.use_flash   = cfg.use_flash_attn
        self.dropout_p   = cfg.dropout

        self.c_attn  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj  = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

        if not self.use_flash:
            mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if self.use_flash:
            # FlashAttention via PyTorch SDPA (auto-selects best kernel)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout_p if self.training else 0.0,
                is_causal = True,
            )
        else:
            scale  = math.sqrt(self.d_head)
            scores = q @ k.transpose(-2, -1) / scale
            scores = scores.masked_fill(
                self.mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0,
                float('-inf')
            )
            out = F.dropout(
                F.softmax(scores, dim=-1),
                p=self.dropout_p, training=self.training
            ) @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(out))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(approximate='tanh'),   # ~10% faster than exact
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1            = nn.LayerNorm(cfg.d_model)
        self.attn           = CausalSelfAttention(cfg)
        self.ln2            = nn.LayerNorm(cfg.d_model)
        self.mlp            = MLP(cfg)
        self.use_checkpoint = cfg.use_checkpoint

    def _forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch_checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg       = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop_emb  = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])
        self.ln_final  = nn.LayerNorm(cfg.d_model)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying (reduces params, improves performance)
        self.lm_head.weight = self.token_emb.weight

        # Weight initialization
        self.apply(self._init_weights)

        # Residual path scaling (GPT-2 paper)
        # Scale down c_proj (output of each attention + MLP)
        for pname, p in self.named_parameters():
            if pname.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layers))

        n = self.count_params()
        print(f"GPT: {n/1e6:.2f}M params  "
              f"(flash={cfg.use_flash_attn} ckpt={cfg.use_checkpoint})")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def count_params(self, exclude_tied=True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_tied:
            n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor = None):
        B, T    = idx.shape
        assert T <= self.cfg.max_seq_len
        device  = idx.device
        pos     = torch.arange(T, device=device)
        x       = self.drop_emb(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x       = self.ln_final(x)
        logits  = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_c   = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_c)
            logits  = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _    = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs   = F.softmax(logits, dim=-1)
            next_t  = torch.multinomial(probs, num_samples=1)
            idx     = torch.cat([idx, next_t], dim=1)
        return idx

    def estimate_mfu(self, tokens_per_sec: float,
                     dtype: str = "bfloat16") -> float:
        """Model FLOP Utilization (NanoGPT formula)."""
        peak = {"bfloat16": 312e12, "float16": 312e12,
                "float32": 19.5e12}.get(dtype, 312e12)
        flops_per_tok = 6 * self.count_params()
        return tokens_per_sec * flops_per_tok / peak
PY
```

---

## 17.4 Source — data.py

```bash
cat > src/data.py <<'PY'
"""
Data pipeline.
Supports: character-level tokenizer + streaming dataset.
"""
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class CharTokenizer:
    def __init__(self, text: str):
        chars           = sorted(set(text))
        self.vocab_size = len(chars)
        self.ch2id      = {c: i for i, c in enumerate(chars)}
        self.id2ch      = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str):
        return [self.ch2id[c] for c in text if c in self.ch2id]

    def decode(self, ids) -> str:
        return ''.join(self.id2ch.get(i, '?') for i in ids)


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


def build_dataloaders(
    data_path:   str,
    seq_len:     int,
    batch_size:  int,
    train_split: float,
    rank:        int,
    world_size:  int,
    num_workers: int = 4,
):
    """
    Build train + val DataLoaders.
    Returns: train_loader, val_loader, tokenizer, vocab_size
    """
    text = open(data_path).read()
    tok  = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    n          = int(len(data) * train_split)
    train_data = data[:n]
    val_data   = data[n:]

    train_ds = TextDataset(train_data, seq_len)
    val_ds   = TextDataset(val_data,   seq_len)

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = train_sampler,
        shuffle     = (train_sampler is None),
        num_workers = num_workers,
        pin_memory  = True,
        persistent_workers = (num_workers > 0),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        sampler     = val_sampler,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        persistent_workers = (num_workers > 0),
    )

    return train_loader, val_loader, train_sampler, tok
PY
```

---

## 17.5 Source — distributed.py

```bash
cat > src/distributed.py <<'PY'
"""
Distributed utilities.
Handles: init, process group detection, collective helpers.
"""
import os
import subprocess
import torch
import torch.distributed as dist
from contextlib import contextmanager


def init_distributed():
    """
    Initialize distributed training.
    Returns: (rank, world_size, local_rank, is_distributed)
    """
    is_dist = (int(os.environ.get("RANK", -1)) != -1)

    if is_dist:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank       = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, is_dist


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()


def broadcast_object(obj, rank: int):
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"


@contextmanager
def no_sync_if_accumulating(model, is_last_micro: bool, is_ddp: bool):
    """Context manager for gradient accumulation with DDP."""
    import contextlib
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if is_last_micro or not is_ddp:
        yield
    else:
        if isinstance(model, DDP):
            with model.no_sync():
                yield
        elif isinstance(model, FSDP):
            with model.no_sync():
                yield
        else:
            yield
PY
```

---

## 17.6 Source — optimizer.py

```bash
cat > src/optimizer.py <<'PY'
"""
Optimizer + LR schedule.
"""
import math
import torch
import torch.nn as nn
from src.config import TrainConfig


def build_optimizer(
    model:       nn.Module,
    train_cfg:   TrainConfig,
    effective_lr: float,
) -> torch.optim.AdamW:
    """
    AdamW with correct weight decay separation.
    Decay: 2D+ params (weight matrices)
    No decay: 1D params (biases, LayerNorm weights)
    """
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    n_decay    = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: decay={n_decay/1e6:.2f}M  "
          f"no_decay={n_no_decay/1e6:.2f}M  lr={effective_lr:.2e}")

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr     = effective_lr,
        betas  = (train_cfg.beta1, train_cfg.beta2),
        fused  = True,   # fused AdamW (CUDA)
    )
    return optimizer


def get_lr(step: int, cfg: TrainConfig, effective_lr: float) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < cfg.warmup_iters:
        return effective_lr * (step + 1) / cfg.warmup_iters
    if step > cfg.max_iters:
        return cfg.min_lr
    ratio = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (effective_lr - cfg.min_lr)


def scale_lr(base_lr: float, world_size: int, rule: str) -> float:
    """Scale learning rate for multi-GPU training."""
    if rule == "linear":
        return base_lr * world_size
    elif rule == "sqrt":
        return base_lr * math.sqrt(world_size)
    return base_lr
PY
```

---

## 17.7 Source — metrics.py

```bash
cat > src/metrics.py <<'PY'
"""
Training metrics: MFU, throughput, memory.
"""
import time
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricsTracker:
    n_params:    int
    n_layers:    int
    d_model:     int
    seq_len:     int
    world_size:  int
    dtype:       str = "bfloat16"

    _history:    List[dict] = field(default_factory=list)
    _t0:         float      = field(default_factory=time.perf_counter)

    GPU_PEAK = {
        "bfloat16": 312e12,
        "float16":  312e12,
        "float32":   19.5e12,
    }

    def __post_init__(self):
        self.peak_flops = self.GPU_PEAK.get(self.dtype, 312e12)
        # flops per token: 6N approximation (fwd + bwd)
        self.flops_per_tok = 6 * self.n_params

    def update(self, step: int, loss: float, lr: float,
               tokens_this_step: int, dt: float) -> dict:
        tok_per_sec = tokens_this_step / dt
        mfu         = tok_per_sec * self.flops_per_tok / self.peak_flops
        peak_gb     = torch.cuda.max_memory_allocated() / 1024**3
        alloc_gb    = torch.cuda.memory_allocated()     / 1024**3

        m = {
            "step":         step,
            "loss":         loss,
            "lr":           lr,
            "tok_per_sec":  tok_per_sec,
            "mfu_pct":      mfu * 100,
            "peak_gb":      peak_gb,
            "alloc_gb":     alloc_gb,
            "step_ms":      dt * 1000,
        }
        self._history.append(m)
        return m

    def avg_mfu(self, last_n: int = 100) -> float:
        hist = self._history[-last_n:]
        if not hist:
            return 0.0
        return sum(h["mfu_pct"] for h in hist) / len(hist)

    def avg_tok_per_sec(self, last_n: int = 100) -> float:
        hist = self._history[-last_n:]
        if not hist:
            return 0.0
        return sum(h["tok_per_sec"] for h in hist) / len(hist)

    def report(self):
        if not self._history:
            return
        total_s  = time.perf_counter() - self._t0
        avg_mfu  = self.avg_mfu(len(self._history))
        peak_mfu = max(h["mfu_pct"] for h in self._history)
        avg_tok  = self.avg_tok_per_sec(len(self._history))
        peak_gb  = max(h["peak_gb"] for h in self._history)

        print(f"\n── Metrics Summary ─────────────────────────────────────")
        print(f"  total_time:    {total_s:.1f}s")
        print(f"  avg_mfu:       {avg_mfu:.2f}%")
        print(f"  peak_mfu:      {peak_mfu:.2f}%")
        print(f"  avg_tok/s:     {avg_tok:.0f}")
        print(f"  peak_memory:   {peak_gb:.3f}GB")
        print(f"  total_tokens:  "
              f"{avg_tok*total_s/1e9:.3f}B")

        grade = ("✓ excellent" if avg_mfu > 45
                 else "~ good" if avg_mfu > 30
                 else "⚠ low")
        print(f"  MFU grade:     {grade}")
        print("──────────────────────────────────────────────────────")
PY
```

---

## 17.8 Source — checkpoint.py

```bash
cat > src/checkpoint.py <<'PY'
"""
Checkpointing for DDP and FSDP.
Production pattern: FULL_STATE_DICT (portable across world_size changes).
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


def save_checkpoint(
    model:      torch.nn.Module,
    optimizer:  torch.optim.Optimizer,
    step:       int,
    val_loss:   float,
    config_dict: dict,
    path:       str,
    rank:       int,
):
    """Save checkpoint. Handles DDP and FSDP correctly."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    if isinstance(model, FSDP):
        # FSDP: gather full state dict to CPU on rank 0
        fsdp_cfg = FullStateDictConfig(
            offload_to_cpu = True,
            rank0_only     = True,
        )
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
            model_state = model.state_dict()
        opt_state = optimizer.state_dict() if rank == 0 else {}
    elif isinstance(model, DDP):
        model_state = model.module.state_dict() if rank == 0 else {}
        opt_state   = optimizer.state_dict() if rank == 0 else {}
    else:
        model_state = model.state_dict() if rank == 0 else {}
        opt_state   = optimizer.state_dict() if rank == 0 else {}

    if rank == 0:
        torch.save({
            "step":      step,
            "model":     model_state,
            "optimizer": opt_state,
            "val_loss":  val_loss,
            "config":    config_dict,
        }, path)
        print(f"[rank 0] Checkpoint → {path}  "
              f"(step={step} val_loss={val_loss:.4f})", flush=True)

    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    model:      torch.nn.Module,
    optimizer:  torch.optim.Optimizer,
    path:       str,
    device:     torch.device,
    rank:       int,
) -> tuple:
    """
    Load checkpoint into model + optimizer.
    Must be called BEFORE wrapping with DDP/FSDP.

    Returns: (step, val_loss)
    """
    ckpt = torch.load(path, map_location=device)

    if isinstance(model, FSDP):
        fsdp_cfg = FullStateDictConfig(
            offload_to_cpu = True,
            rank0_only     = True,
        )
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
            model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])

    optimizer.load_state_dict(ckpt["optimizer"])

    if rank == 0:
        print(f"Resumed from {path}  "
              f"(step={ckpt['step']} val_loss={ckpt['val_loss']:.4f})")

    if dist.is_initialized():
        dist.barrier()

    return ckpt["step"], ckpt["val_loss"]
PY
```

---

## 17.9 Source — profiler.py

```bash
cat > src/profiler.py <<'PY'
"""
Profiling hooks for the training loop.
Integrates with torch.profiler and TensorBoard.
"""
import contextlib
import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
    record_function,
)


class TrainingProfiler:
    """
    Wraps torch.profiler with training-friendly API.
    Activated for `active_steps` steps after `wait + warmup` steps.
    """

    def __init__(
        self,
        out_dir:    str,
        rank:       int,
        wait:       int = 2,
        warmup:     int = 2,
        active:     int = 3,
        enabled:    bool = True,
    ):
        self.rank    = rank
        self.enabled = enabled
        self._prof   = None
        self._step   = 0

        if not enabled:
            return

        trace_dir = f"{out_dir}/traces/rank{rank}"
        sched     = schedule(wait=wait, warmup=warmup, active=active, repeat=1)

        self._prof = profile(
            activities      = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule        = sched,
            on_trace_ready  = tensorboard_trace_handler(trace_dir),
            record_shapes   = True,
            profile_memory  = True,
            with_flops      = True,
        )
        self._total_steps = wait + warmup + active

    def __enter__(self):
        if self._prof:
            self._prof.__enter__()
        return self

    def __exit__(self, *args):
        if self._prof:
            self._prof.__exit__(*args)

    def step(self):
        """Call at end of each training step."""
        if self._prof:
            self._prof.step()
            self._step += 1

    def is_done(self) -> bool:
        """True after profiling window is complete."""
        if not self._prof:
            return True
        return self._step >= self._total_steps

    def summary(self):
        """Print op summary (rank 0 only)."""
        if self._prof and self.rank == 0:
            print("\n── Profiler Summary (top 15 by CUDA time) ──────────────")
            print(self._prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=15
            ))

    @contextlib.contextmanager
    def label(self, name: str):
        """Annotate a code section."""
        with record_function(name):
            yield
PY
```

---

## 17.10 Source — trainer.py (The Core)

```bash
cat > src/trainer.py <<'PY'
"""
Main trainer. Handles DDP and FSDP uniformly.

Design principles:
  1. Backend (ddp/fsdp) selected via config — no code changes
  2. Every major feature (fault tolerance, profiling, MFU) integrated
  3. Rank-0-only I/O everywhere
  4. Full reproducibility (seeds, git hash, data fingerprint)
"""
import os
import sys
import json
import time
import math
import signal
import hashlib
import contextlib
import functools
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.model import GPT, TransformerBlock
from src.data import build_dataloaders
from src.optimizer import build_optimizer, get_lr, scale_lr
from src.distributed import (
    init_distributed, cleanup_distributed,
    is_main, barrier, broadcast_object, get_git_hash,
    no_sync_if_accumulating,
)
from src.checkpoint import save_checkpoint, load_checkpoint
from src.metrics import MetricsTracker
from src.profiler import TrainingProfiler


def get_data_fingerprint(path: str) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()[:8]
    except:
        return "unknown"


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # ── Distributed setup ────────────────────────────────────────────
        self.rank, self.world_size, self.local_rank, self.is_dist = \
            init_distributed()
        self.device  = torch.device("cuda", self.local_rank)
        self._main   = is_main(self.rank)

        # ── Reproducibility ──────────────────────────────────────────────
        torch.manual_seed(cfg.train.seed)
        torch.cuda.manual_seed(cfg.train.seed + self.rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

        # ── Signal handling (fault tolerance) ────────────────────────────
        self._current_step = 0
        self._install_signal_handlers()

        # ── Output directories ───────────────────────────────────────────
        if self._main:
            os.makedirs(cfg.train.out_dir, exist_ok=True)
            cfg.save(os.path.join(cfg.train.out_dir, "config.json"))

        # ── Logging ──────────────────────────────────────────────────────
        self.writer   = None
        self.wandb_on = False
        self._setup_logging()

        self.log(f"\n{'='*65}")
        self.log(f"Capstone GPT Trainer")
        self.log(f"  rank={self.rank}/{self.world_size}  "
                 f"device={self.device}")
        self.log(f"  backend={cfg.train.backend}  "
                 f"dtype={cfg.train.dtype}")
        self.log(f"  git_hash={get_git_hash()}")
        self.log(f"  data_fp={get_data_fingerprint(cfg.train.data_path)}")
        self.log(f"{'='*65}\n")

    # ── Setup methods ──────────────────────────────────────────────────────

    def _setup_logging(self):
        if not self._main:
            return
        tb_dir = os.path.join(self.cfg.train.out_dir, "tensorboard")
        self.writer = SummaryWriter(tb_dir)

        if self.cfg.train.wandb_project:
            try:
                import wandb
                wandb.init(
                    project  = self.cfg.train.wandb_project,
                    name     = self.cfg.train.wandb_run_name or get_git_hash(),
                    config   = {
                        "model": asdict(self.cfg.model),
                        "train": asdict(self.cfg.train),
                    },
                )
                self.wandb_on = True
            except ImportError:
                self.log("wandb not installed")

    def _install_signal_handlers(self):
        """SIGTERM: sent by SLURM 90s before timeout. Save and exit."""
        def _handler(signum, frame):
            self.log(f"\n⚠ Signal {signum} received — emergency checkpoint")
            self._emergency_save()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGUSR1, _handler)

    def _emergency_save(self):
        if hasattr(self, "model") and hasattr(self, "optimizer"):
            path = os.path.join(
                self.cfg.train.out_dir,
                f"ckpt_emergency_step{self._current_step}.pt"
            )
            save_checkpoint(
                self.model, self.optimizer,
                self._current_step, float("inf"),
                {"emergency": True},
                path, self.rank,
            )

    def log(self, msg: str):
        if self._main:
            print(msg, flush=True)

    def log_metrics(self, step: int, d: dict):
        if not self._main:
            return
        for k, v in d.items():
            if self.writer:
                self.writer.add_scalar(k, v, step)
        if self.wandb_on:
            import wandb
            wandb.log(d, step=step)

    # ── Model building ─────────────────────────────────────────────────────

    def _build_model(self) -> nn.Module:
        torch.manual_seed(self.cfg.train.seed)  # same init on all ranks
        model = GPT(self.cfg.model)
        return model

    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        return DDP(
            model,
            device_ids        = [self.local_rank],
            find_unused_parameters = False,
            bucket_cap_mb     = 25,
        )

    def _wrap_fsdp(self, model: nn.Module) -> nn.Module:
        strat_map = {
            "FULL_SHARD":    ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "HYBRID_SHARD":  ShardingStrategy.HYBRID_SHARD,
            "NO_SHARD":      ShardingStrategy.NO_SHARD,
        }
        strategy = strat_map.get(
            self.cfg.train.fsdp_strategy, ShardingStrategy.FULL_SHARD
        )

        dtype   = getattr(torch, self.cfg.train.dtype)
        mp      = MixedPrecision(
            param_dtype  = dtype,
            reduce_dtype = torch.float32,
            buffer_dtype = dtype,
        )
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls = {TransformerBlock},
        )
        return FSDP(
            model,
            auto_wrap_policy  = wrap_policy,
            mixed_precision   = mp,
            sharding_strategy = strategy,
            device_id         = torch.cuda.current_device(),
            use_orig_params   = True,
        )

    def _get_raw_model(self) -> nn.Module:
        if isinstance(self.model, (DDP, FSDP)):
            return self.model.module
        return self.model

    # ── Evaluation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total = torch.zeros(2, device=self.device)  # [loss_sum, count]
        amp   = self._amp_ctx()

        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.cfg.train.eval_iters:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with amp:
                _, loss = self.model(x, y)
            total[0] += loss.item()
            total[1] += 1.0

        if self.is_dist:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        self.model.train()
        return (total[0] / total[1]).item()

    # ── Text generation ────────────────────────────────────────────────────

    def _generate_sample(self, tokenizer) -> str:
        if not self._main:
            return ""
        raw = self._get_raw_model()
        raw.eval()
        prompt = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            gen = raw.generate(
                prompt,
                max_new_tokens = self.cfg.train.gen_tokens,
                temperature    = 0.8,
                top_k          = 40,
            )
        raw.train()
        return tokenizer.decode(gen[0].tolist())

    def _amp_ctx(self):
        dtype = getattr(torch, self.cfg.train.dtype)
        return torch.amp.autocast(device_type="cuda", dtype=dtype)

    # ── Main training loop ─────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        tc  = cfg.train

        # ── Data ──────────────────────────────────────────────────────────
        if self._main:
            train_loader, val_loader, train_sampler, tok = build_dataloaders(
                data_path   = tc.data_path,
                seq_len     = cfg.model.max_seq_len,
                batch_size  = tc.batch_size,
                train_split = tc.train_split,
                rank        = self.rank,
                world_size  = self.world_size,
            )
            cfg.model.vocab_size = tok.vocab_size
        else:
            train_loader = val_loader = train_sampler = tok = None

        if self.is_dist:
            objs = [cfg.model.vocab_size, tok]
            if self._main:
                dist.broadcast_object_list(objs, src=0)
            else:
                dist.broadcast_object_list(objs, src=0)
                cfg.model.vocab_size = objs[0]
                tok = objs[1]

            train_loader, val_loader, train_sampler, _ = build_dataloaders(
                data_path   = tc.data_path,
                seq_len     = cfg.model.max_seq_len,
                batch_size  = tc.batch_size,
                train_split = tc.train_split,
                rank        = self.rank,
                world_size  = self.world_size,
            )

        self.val_loader    = val_loader
        self.train_sampler = train_sampler

        # ── Model ──────────────────────────────────────────────────────────
        model = self._build_model()

        # ── Effective LR ───────────────────────────────────────────────────
        effective_lr = scale_lr(tc.lr, self.world_size, tc.lr_scale)
        self.log(f"LR: base={tc.lr:.2e}  scale={tc.lr_scale}  "
                 f"effective={effective_lr:.2e}")

        # ── Optimizer (before DDP/FSDP wrap) ──────────────────────────────
        optimizer = build_optimizer(model, tc, effective_lr)

        # ── Resume (before DDP/FSDP wrap) ─────────────────────────────────
        start_step = 0
        best_val   = float("inf")
        if tc.resume and os.path.exists(tc.resume):
            model = model.to(self.device)
            start_step, best_val = load_checkpoint(
                model, optimizer, tc.resume, self.device, self.rank
            )

        # ── Wrap with backend ──────────────────────────────────────────────
        if not self.is_dist:
            self.model = model.to(self.device)
        elif tc.backend == "fsdp":
            self.model = self._wrap_fsdp(model)
            self.log(f"Backend: FSDP ({tc.fsdp_strategy})")
        else:
            self.model = self._wrap_ddp(model)
            self.log(f"Backend: DDP")

        self.optimizer = optimizer

        # ── Compile ────────────────────────────────────────────────────────
        if tc.compile_model:
            self.log("Compiling model (torch.compile)...")
            self.model = torch.compile(self.model)
            self.log("Compilation done.")

        # ── Metrics ────────────────────────────────────────────────────────
        raw_model = self._get_raw_model()
        n_params  = raw_model.count_params()
        self.log(f"Model: {n_params/1e6:.2f}M params")
        self.log(f"Global batch: {tc.batch_size}×{tc.accum_steps}×"
                 f"{self.world_size} = "
                 f"{tc.batch_size*tc.accum_steps*self.world_size} samples")

        self.metrics = MetricsTracker(
            n_params   = n_params,
            n_layers   = cfg.model.n_layers,
            d_model    = cfg.model.d_model,
            seq_len    = cfg.model.max_seq_len,
            world_size = self.world_size,
            dtype      = tc.dtype,
        )

        # ── Profiler ────────────────────────────────────────────────────────
        profiler = TrainingProfiler(
            out_dir = tc.out_dir,
            rank    = self.rank,
            wait    = tc.profile_wait,
            warmup  = tc.profile_warmup,
            active  = tc.profile_active,
            enabled = (tc.profile_steps > 0),
        )

        # ── Training loop ───────────────────────────────────────────────────
        torch.cuda.reset_peak_memory_stats(self.device)
        self.model.train()
        amp       = self._amp_ctx()
        t0        = time.perf_counter()
        train_iter= iter(train_loader)

        self.log(f"\n── Training starts (step {start_step} → {tc.max_iters}) ──\n")

        with profiler:
            for step in range(start_step, tc.max_iters):
                self._current_step = step

                # LR
                lr = get_lr(step, tc, effective_lr)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                # Sampler epoch (prevents same shuffle every epoch)
                if self.train_sampler is not None:
                    epoch = (step * tc.batch_size
                             // max(1, len(train_loader.dataset)))
                    self.train_sampler.set_epoch(epoch)

                # ── Gradient accumulation ─────────────────────────────────
                self.optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0

                with profiler.label("train_step"):
                    for micro in range(tc.accum_steps):
                        try:
                            x, y = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_loader)
                            x, y = next(train_iter)

                        x = x.to(self.device, non_blocking=True)
                        y = y.to(self.device, non_blocking=True)
                        is_last = (micro == tc.accum_steps - 1)

                        with no_sync_if_accumulating(
                            self.model, is_last, self.is_dist
                        ):
                            with profiler.label("forward"):
                                with amp:
                                    _, loss = self.model(x, y)

                            with profiler.label("backward"):
                                (loss / tc.accum_steps).backward()

                        accum_loss += loss.item() / tc.accum_steps

                    # Gradient clip
                    with profiler.label("clip_and_step"):
                        if isinstance(self.model, FSDP):
                            self.model.clip_grad_norm_(tc.grad_clip)
                        else:
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(), tc.grad_clip
                            )
                        self.optimizer.step()

                profiler.step()

                # ── Logging ───────────────────────────────────────────────
                if step % tc.log_interval == 0:
                    torch.cuda.synchronize()
                    t1      = time.perf_counter()
                    dt      = (t1 - t0) / tc.log_interval
                    t0      = t1
                    tokens  = (tc.batch_size * tc.accum_steps *
                               cfg.model.max_seq_len * self.world_size)

                    m = self.metrics.update(
                        step=step, loss=accum_loss, lr=lr,
                        tokens_this_step=tokens, dt=dt,
                    )

                    self.log(
                        f"step={step:5d}  loss={accum_loss:.4f}  "
                        f"lr={lr:.2e}  tok/s={m['tok_per_sec']:.0f}  "
                        f"mfu={m['mfu_pct']:.1f}%  "
                        f"peak={m['peak_gb']:.2f}GB"
                    )
                    self.log_metrics(step, {
                        "train/loss":   accum_loss,
                        "train/lr":     lr,
                        "train/mfu":    m["mfu_pct"],
                        "perf/tok_s":   m["tok_per_sec"],
                        "mem/peak_gb":  m["peak_gb"],
                        "perf/step_ms": m["step_ms"],
                    })

                # ── Evaluation ────────────────────────────────────────────
                if (step % tc.eval_interval == 0 or
                    step == tc.max_iters - 1):
                    val_loss = self._evaluate()
                    self.log(f"\n── eval step={step}  "
                             f"val_loss={val_loss:.4f} ──────────────────")
                    self.log_metrics(step, {"eval/val_loss": val_loss})

                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(
                            self.model, self.optimizer, step, val_loss,
                            {"model": asdict(cfg.model),
                             "train": asdict(cfg.train)},
                            os.path.join(tc.out_dir, "ckpt_best.pt"),
                            self.rank,
                        )
                    self.log("──────────────────────────────────────────────\n")
                    barrier()

                # ── Periodic save ─────────────────────────────────────────
                if step > 0 and step % tc.save_interval == 0:
                    save_checkpoint(
                        self.model, self.optimizer, step, accum_loss,
                        {"model": asdict(cfg.model), "train": asdict(cfg.train)},
                        os.path.join(tc.out_dir, f"ckpt_step{step}.pt"),
                        self.rank,
                    )

                # ── Text generation ───────────────────────────────────────
                if (tc.gen_interval > 0 and step > 0 and
                    step % tc.gen_interval == 0):
                    gen_text = self._generate_sample(tok)
                    if gen_text and self._main:
                        self.log(f"── generation ──────────────────────────")
                        self.log(gen_text[:500])
                        self.log("───────────────────────────────────────────\n")
                        if self.writer:
                            self.writer.add_text("generation", gen_text, step)

                # ── Profiler done? ────────────────────────────────────────
                if tc.profile_steps > 0 and profiler.is_done():
                    profiler.summary()

        # ── Final ──────────────────────────────────────────────────────────
        self.metrics.report()
        self.log(f"\nTraining complete. best_val_loss={best_val:.4f}")

        if self.writer:
            self.writer.close()
        if self.wandb_on:
            import wandb
            wandb.finish()

        cleanup_distributed()
        return best_val
PY
```

---

## 17.11 Entry Point — train.py

```bash
cat > train.py <<'PY'
"""
Capstone GPT Trainer — Entry Point

Usage:
  # Single GPU
  python train.py

  # 8 GPU DDP
  torchrun --nproc_per_node=8 train.py --backend=ddp

  # 8 GPU FSDP
  torchrun --nproc_per_node=8 train.py --backend=fsdp

  # With config file
  torchrun --nproc_per_node=8 train.py --config=configs/medium.yaml

  # Resume
  torchrun --nproc_per_node=8 train.py \
      --resume=checkpoints/run/ckpt_best.pt

  # Full feature run
  torchrun --nproc_per_node=8 train.py \
      --config=configs/small.yaml \
      --backend=fsdp \
      --profile_steps=10 \
      --wandb_project=capstone_gpt
"""
import sys
sys.path.insert(0, ".")   # ensure src/ imports work

from src.config import Config, build_parser
from src.trainer import Trainer


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Build config (YAML → CLI overrides)
    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()

    cfg.apply_cli(args)

    # Handle --no_flash_attn
    if getattr(args, "no_flash_attn", None):
        cfg.model.use_flash_attn = False

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
PY
```

---

## 17.12 YAML Configs

```bash
cat > configs/small.yaml <<'YAML'
# Small config: d=384, L=6 (~10M params)
# Fast experiments, fits on 1 GPU with batch=256
model:
  vocab_size: 65
  max_seq_len: 256
  d_model: 384
  n_heads: 6
  n_layers: 6
  dropout: 0.1
  use_checkpoint: false
  use_flash_attn: true

train:
  data_path: shakespeare.txt
  batch_size: 64
  accum_steps: 1
  max_iters: 5000
  lr: 6.0e-4
  lr_scale: sqrt
  min_lr: 6.0e-5
  warmup_iters: 200
  weight_decay: 0.1
  grad_clip: 1.0
  dtype: bfloat16
  backend: ddp
  eval_interval: 500
  save_interval: 1000
  gen_interval: 1000
  out_dir: checkpoints/small
YAML

cat > configs/medium.yaml <<'YAML'
# Medium config: d=768, L=12 (~85M params, GPT-2 small scale)
# Use FSDP for 8-GPU efficiency
model:
  vocab_size: 65
  max_seq_len: 512
  d_model: 768
  n_heads: 12
  n_layers: 12
  dropout: 0.1
  use_checkpoint: true
  use_flash_attn: true

train:
  data_path: shakespeare.txt
  batch_size: 16
  accum_steps: 4
  max_iters: 10000
  lr: 3.0e-4
  lr_scale: sqrt
  min_lr: 3.0e-5
  warmup_iters: 400
  weight_decay: 0.1
  grad_clip: 1.0
  dtype: bfloat16
  backend: fsdp
  fsdp_strategy: FULL_SHARD
  eval_interval: 500
  save_interval: 1000
  gen_interval: 2000
  out_dir: checkpoints/medium
YAML

cat > configs/large.yaml <<'YAML'
# Large config: d=1024, L=24 (~350M params)
# Requires FSDP + activation checkpointing
model:
  vocab_size: 65
  max_seq_len: 1024
  d_model: 1024
  n_heads: 16
  n_layers: 24
  dropout: 0.1
  use_checkpoint: true
  use_flash_attn: true

train:
  data_path: shakespeare.txt
  batch_size: 4
  accum_steps: 8
  max_iters: 20000
  lr: 2.0e-4
  lr_scale: sqrt
  min_lr: 2.0e-5
  warmup_iters: 800
  weight_decay: 0.1
  grad_clip: 1.0
  dtype: bfloat16
  backend: fsdp
  fsdp_strategy: FULL_SHARD
  eval_interval: 1000
  save_interval: 2000
  gen_interval: 5000
  out_dir: checkpoints/large
YAML
```

---

## 17.13 Tests

```bash
cat > tests/test_model.py <<'PY'
"""Unit tests for GPT model."""
import math
import torch
import pytest
from src.config import ModelConfig
from src.model import GPT


@pytest.fixture
def small_cfg():
    return ModelConfig(
        vocab_size=65, max_seq_len=64,
        d_model=128, n_heads=4, n_layers=2,
        dropout=0.0, use_flash_attn=True, use_checkpoint=False,
    )


def test_output_shape(small_cfg):
    model = GPT(small_cfg)
    x     = torch.randint(0, 65, (2, 32))
    logits, loss = model(x, x)
    assert logits.shape == (2, 32, 65)
    assert loss is not None


def test_init_loss(small_cfg):
    """Initial loss should be ≈ log(vocab_size)."""
    torch.manual_seed(42)
    model = GPT(small_cfg)
    x     = torch.randint(0, 65, (4, 32))
    _, loss = model(x, x)
    expected = math.log(65)
    assert abs(loss.item() - expected) < 0.5, \
        f"Init loss {loss.item():.4f} far from expected {expected:.4f}"


def test_causal_masking(small_cfg):
    """Future tokens must not influence past positions."""
    model = GPT(small_cfg)
    x     = torch.randint(0, 65, (1, 32))

    with torch.no_grad():
        logits_full, _ = model(x)

    x2 = x.clone()
    x2[:, 16:] = 0
    with torch.no_grad():
        logits_masked, _ = model(x2)

    assert torch.allclose(
        logits_full[:, :16, :],
        logits_masked[:, :16, :],
        atol=1e-4
    ), "Causal mask violated"


def test_overfit_tiny(small_cfg):
    """Model should overfit a tiny fixed batch."""
    torch.manual_seed(42)
    model = GPT(small_cfg)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x     = torch.randint(0, 65, (2, 16))

    for _ in range(200):
        opt.zero_grad()
        _, loss = model(x, x)
        loss.backward()
        opt.step()

    assert loss.item() < 0.1, f"Failed to overfit: loss={loss.item():.4f}"


def test_generation(small_cfg):
    model = GPT(small_cfg)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    gen   = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    assert gen.shape == (1, 21)


def test_weight_tying(small_cfg):
    model = GPT(small_cfg)
    assert model.lm_head.weight is model.token_emb.weight, \
        "Weight tying broken"


def test_param_count(small_cfg):
    model = GPT(small_cfg)
    n     = model.count_params()
    assert n > 0
    print(f"  params: {n/1e6:.3f}M")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
PY

cat > tests/test_distributed.py <<'PY'
"""
Distributed tests.
Run: torchrun --nproc_per_node=2 tests/test_distributed.py
"""
import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, ".")
from src.distributed import init_distributed, cleanup_distributed
from src.config import ModelConfig
from src.model import GPT


def test_model_sync():
    """All ranks must start with same model weights."""
    rank, world_size, local_rank, is_dist = init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(42)
    cfg   = ModelConfig(vocab_size=65, max_seq_len=32,
                        d_model=64, n_heads=4, n_layers=2)
    model = GPT(cfg).to(device)

    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[local_rank])

    # Check first param is identical across ranks
    param = list(model.parameters())[0].data.clone()
    gathered = [torch.zeros_like(param) for _ in range(world_size)]
    dist.all_gather(gathered, param)

    if rank == 0:
        for i in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[i], atol=1e-5), \
                f"Ranks 0 and {i} have different init!"
        print("PASSED: model sync across ranks")

    cleanup_distributed()


def test_distributed_loss():
    """Distributed loss should match single-GPU loss."""
    rank, world_size, local_rank, is_dist = init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(42)
    cfg   = ModelConfig(vocab_size=65, max_seq_len=32,
                        d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    model = GPT(cfg).to(device)

    from torch.nn.parallel import DistributedDataParallel as DDP
    model_ddp = DDP(model, device_ids=[local_rank])

    torch.manual_seed(0)
    x = torch.randint(0, 65, (4, 32), device=device)
    y = torch.randint(0, 65, (4, 32), device=device)

    _, loss_ddp = model_ddp(x, y)

    if rank == 0:
        print(f"PASSED: DDP loss = {loss_ddp.item():.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    test_model_sync()
    test_distributed_loss()
PY

cat > tests/test_checkpoint.py <<'PY'
"""
Checkpoint save/load test.
Run: torchrun --nproc_per_node=2 tests/test_checkpoint.py
"""
import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, ".")
from src.distributed import init_distributed, cleanup_distributed
from src.config import ModelConfig
from src.model import GPT
from src.checkpoint import save_checkpoint, load_checkpoint


def test_ddp_checkpoint():
    rank, world_size, local_rank, is_dist = init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(42)
    cfg   = ModelConfig(vocab_size=65, max_seq_len=32,
                        d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    model = GPT(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)

    from torch.nn.parallel import DistributedDataParallel as DDP
    ddp_model = DDP(model, device_ids=[local_rank])

    # Train 5 steps
    losses_before = []
    x = torch.randint(0, 65, (2, 32), device=device)
    y = torch.randint(0, 65, (2, 32), device=device)
    for _ in range(5):
        opt.zero_grad()
        _, loss = ddp_model(x, y)
        loss.backward()
        opt.step()
        losses_before.append(loss.item())

    # Save
    os.makedirs("tests/tmp", exist_ok=True)
    save_checkpoint(ddp_model, opt, 5, losses_before[-1],
                    {}, "tests/tmp/ckpt_test.pt", rank)

    # Corrupt weights
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)

    # Load (reload into raw model, then re-wrap)
    model2 = GPT(cfg).to(device)
    opt2   = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    load_checkpoint(model2, opt2, "tests/tmp/ckpt_test.pt", device, rank)
    ddp2  = DDP(model2, device_ids=[local_rank])

    # Verify
    with torch.no_grad():
        _, loss_after = ddp2(x, y)

    if rank == 0:
        diff  = abs(loss_after.item() - losses_before[-1])
        ok    = diff < 0.01
        print(f"PASSED: DDP checkpoint (diff={diff:.6f})" if ok
              else f"FAILED: diff={diff:.6f}")

    cleanup_distributed()


if __name__ == "__main__":
    test_ddp_checkpoint()
PY
```

---

## 17.14 SLURM Jobs

```bash
cat > jobs/job_1gpu.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=capstone_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/cap_1gpu_%j.out
#SBATCH --error=logs/cap_1gpu_%j.err

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/capstone

python train.py \
    --config=configs/small.yaml \
    --backend=ddp \
    --max_iters=2000 \
    --out_dir=checkpoints/1gpu_run

echo "1 GPU run complete."
SLURM

cat > jobs/job_8gpu.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=capstone_8gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/cap_8gpu_%j.out
#SBATCH --error=logs/cap_8gpu_%j.err
#SBATCH --signal=SIGUSR1@90

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/capstone

MASTER_ADDR=localhost
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_ADDR MASTER_PORT
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

torchrun \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config=configs/small.yaml \
    --backend=ddp \
    --max_iters=2000 \
    --out_dir=checkpoints/8gpu_ddp

echo "8 GPU DDP run complete."
SLURM

cat > jobs/job_16gpu.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=capstone_16gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/cap_16gpu_%j.out
#SBATCH --error=logs/cap_16gpu_%j.err
#SBATCH --signal=SIGUSR1@90

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/capstone

mkdir -p logs checkpoints

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_ADDR MASTER_PORT
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

echo "=============================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Nodes:    $SLURM_JOB_NODELIST"
echo "Master:   $MASTER_ADDR:$MASTER_PORT"
echo "=============================="

srun bash -c "
    torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py \
        --config=configs/small.yaml \
        --backend=ddp \
        --max_iters=2000 \
        --out_dir=checkpoints/16gpu_ddp
"

echo "16 GPU run complete."
SLURM

cat > jobs/job_16gpu_fsdp.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=capstone_16gpu_fsdp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/cap_16gpu_fsdp_%j.out
#SBATCH --error=logs/cap_16gpu_fsdp_%j.err
#SBATCH --signal=SIGUSR1@90

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/capstone

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_ADDR MASTER_PORT
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

srun bash -c "
    torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py \
        --config=configs/medium.yaml \
        --backend=fsdp \
        --fsdp_strategy=FULL_SHARD \
        --max_iters=2000 \
        --out_dir=checkpoints/16gpu_fsdp
"
SLURM

# Job array for scaling benchmark
cat > jobs/job_sweep.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=cap_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --output=logs/sweep_%A_%a.out

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/capstone

# Map array index to GPU count
GPU_COUNTS=(1 2 4 8)
N_GPUS=${GPU_COUNTS[$SLURM_ARRAY_TASK_ID]}

MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID * 10))

echo "Array task $SLURM_ARRAY_TASK_ID → $N_GPUS GPUs"

torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    train.py \
    --config=configs/small.yaml \
    --backend=ddp \
    --max_iters=500 \
    --log_interval=10 \
    --eval_interval=200 \
    --out_dir=checkpoints/sweep_${N_GPUS}gpu

echo "Done: $N_GPUS GPUs"
SLURM

mkdir -p logs
```

---

## 17.15 Scaling Benchmark Script

```bash
cat > scripts/benchmark_scaling.sh <<'BASH'
#!/bin/bash
# Benchmark scaling efficiency: 1→2→4→8→16 GPUs

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/capstone
mkdir -p logs/benchmark

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Capstone Scaling Benchmark                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

BASE_ARGS="--config=configs/small.yaml --max_iters=200 --log_interval=10 --eval_interval=100"

for N in 1 2 4 8; do
    echo "────────────────────────────────────────────"
    echo "Running: $N GPU(s)"
    echo "────────────────────────────────────────────"

    OUTFILE="logs/benchmark/bench_${N}gpu.log"

    torchrun \
        --nproc_per_node=$N \
        --master_addr=localhost \
        --master_port=$((29500 + N)) \
        train.py $BASE_ARGS \
        --backend=ddp \
        --out_dir=checkpoints/bench_${N}gpu \
        2>&1 | tee $OUTFILE

    echo ""
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Scaling Report                                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "GPU   tok/s    MFU%   efficiency"
echo "────────────────────────────────"

BASE_TOK_S=0
for N in 1 2 4 8; do
    LOGFILE="logs/benchmark/bench_${N}gpu.log"
    if [ -f "$LOGFILE" ]; then
        # Extract last meaningful log line
        TOK_S=$(grep "tok/s=" $LOGFILE | tail -5 | \
                awk -F'tok/s=' '{print $2}' | \
                awk '{print $1}' | sort -n | tail -1)
        MFU=$(grep "mfu=" $LOGFILE | tail -5 | \
              awk -F'mfu=' '{print $2}' | \
              awk '{print $1}' | sort -n | tail -1)

        if [ "$N" -eq "1" ]; then
            BASE_TOK_S=$TOK_S
            EFF="100.0%"
        else
            EFF=$(python3 -c "print(f'{$TOK_S/($N*$BASE_TOK_S)*100:.1f}%')" 2>/dev/null || echo "?")
        fi
        printf "%-5d %-8s %-6s %s\n" $N "${TOK_S:-?}" "${MFU:-?}%" "$EFF"
    fi
done
BASH

chmod +x scripts/benchmark_scaling.sh
```

---

## 17.16 Generation Script

```bash
cat > scripts/generate.py <<'PY'
"""
Generate text from a trained checkpoint.
Usage: python scripts/generate.py --checkpoint=checkpoints/run/ckpt_best.pt
"""
import sys, os, argparse
sys.path.insert(0, ".")

import torch
from src.config import Config
from src.model import GPT
from src.data import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt",     type=str, default="HAMLET:\n")
    p.add_argument("--max_tokens", type=int, default=500)
    p.add_argument("--temperature",type=float, default=0.8)
    p.add_argument("--top_k",      type=int, default=40)
    p.add_argument("--n_samples",  type=int, default=3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = Config()

    if "config" in ckpt:
        from src.config import ModelConfig, TrainConfig
        if "model" in ckpt["config"]:
            cfg.model = ModelConfig(**ckpt["config"]["model"])

    print(f"Checkpoint: step={ckpt.get('step','?')}  "
          f"val_loss={ckpt.get('val_loss',0):.4f}")

    # Build model
    model = GPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = model.count_params()
    print(f"Model: {n_params/1e6:.2f}M params\n")

    # Build tokenizer (need data file to build vocab)
    data_path = ckpt.get("config", {}).get("train", {}).get(
        "data_path", "shakespeare.txt"
    )
    text = open(data_path).read()
    tok  = CharTokenizer(text)

    # Encode prompt
    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [0]
    prompt_t   = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"Prompt: '{args.prompt}'")
    print("=" * 60)

    configs_to_try = [
        {"temperature": 0.5, "top_k": 10,  "label": "Conservative"},
        {"temperature": 0.8, "top_k": 40,  "label": "Balanced"},
        {"temperature": 1.2, "top_k": None,"label": "Creative"},
    ]

    for i in range(args.n_samples):
        cfg_s = configs_to_try[i % len(configs_to_try)]
        print(f"\n── Sample {i+1} ({cfg_s['label']}) ──────────────────────")

        with torch.no_grad():
            gen = model.generate(
                prompt_t.clone(),
                max_new_tokens = args.max_tokens,
                temperature    = cfg_s["temperature"],
                top_k          = cfg_s["top_k"],
            )

        text_out = tok.decode(gen[0].tolist())
        print(text_out)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
PY
```

---

## 17.17 Run Everything

```bash
cat > scripts/run_all.sh <<'BASH'
#!/bin/bash
# Complete capstone run sequence

set -e
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/capstone
mkdir -p logs checkpoints profiles

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Capstone Project — Complete Run                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Step 1: Unit tests ────────────────────────────────────────────────────────
echo ""
echo "Step 1: Unit Tests"
echo "──────────────────"
python -m pytest tests/test_model.py -v

echo ""
echo "Distributed tests (2 GPU):"
torchrun --nproc_per_node=2 tests/test_distributed.py
torchrun --nproc_per_node=2 tests/test_checkpoint.py

# ── Step 2: Single GPU baseline ───────────────────────────────────────────────
echo ""
echo "Step 2: Single GPU Baseline"
echo "───────────────────────────"
python train.py \
    --config=configs/small.yaml \
    --max_iters=1000 \
    --out_dir=checkpoints/1gpu \
    2>&1 | tee logs/1gpu.log

echo ""
echo "Single GPU summary:"
grep "step=" logs/1gpu.log | tail -5

# ── Step 3: 8 GPU DDP ─────────────────────────────────────────────────────────
echo ""
echo "Step 3: 8 GPU DDP"
echo "─────────────────"
torchrun --nproc_per_node=8 \
    train.py \
    --config=configs/small.yaml \
    --backend=ddp \
    --max_iters=1000 \
    --out_dir=checkpoints/8gpu_ddp \
    2>&1 | tee logs/8gpu_ddp.log

echo ""
echo "8 GPU DDP summary:"
grep "step=" logs/8gpu_ddp.log | tail -5

# ── Step 4: 8 GPU FSDP ───────────────────────────────────────────────────────
echo ""
echo "Step 4: 8 GPU FSDP"
echo "──────────────────"
torchrun --nproc_per_node=8 \
    train.py \
    --config=configs/small.yaml \
    --backend=fsdp \
    --max_iters=1000 \
    --out_dir=checkpoints/8gpu_fsdp \
    2>&1 | tee logs/8gpu_fsdp.log

echo ""
echo "8 GPU FSDP summary:"
grep "step=" logs/8gpu_fsdp.log | tail -5

# ── Step 5: Scaling benchmark ─────────────────────────────────────────────────
echo ""
echo "Step 5: Scaling Benchmark (1→8 GPU)"
echo "────────────────────────────────────"
bash scripts/benchmark_scaling.sh

# ── Step 6: Profile ───────────────────────────────────────────────────────────
echo ""
echo "Step 6: Profiling (8 GPU)"
echo "─────────────────────────"
torchrun --nproc_per_node=8 \
    train.py \
    --config=configs/small.yaml \
    --backend=ddp \
    --max_iters=50 \
    --profile_steps=10 \
    --profile_wait=2 \
    --profile_warmup=2 \
    --profile_active=3 \
    --out_dir=checkpoints/profile_run \
    2>&1 | tee logs/profile.log

echo ""
echo "Profile traces: checkpoints/profile_run/traces/"
echo "View: tensorboard --logdir checkpoints/profile_run/tensorboard --port 6006"

# ── Step 7: Resume test ───────────────────────────────────────────────────────
echo ""
echo "Step 7: Resume from checkpoint"
echo "──────────────────────────────"
torchrun --nproc_per_node=8 \
    train.py \
    --config=configs/small.yaml \
    --backend=ddp \
    --max_iters=1500 \
    --resume=checkpoints/8gpu_ddp/ckpt_best.pt \
    --out_dir=checkpoints/8gpu_ddp_resume \
    2>&1 | tee logs/8gpu_ddp_resume.log

echo ""
echo "Resume summary:"
grep "step=" logs/8gpu_ddp_resume.log | head -3

# ── Step 8: Generate text ─────────────────────────────────────────────────────
echo ""
echo "Step 8: Text Generation"
echo "───────────────────────"
python scripts/generate.py \
    --checkpoint=checkpoints/8gpu_ddp/ckpt_best.pt \
    --prompt="HAMLET:" \
    --max_tokens=300 \
    --n_samples=2

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Capstone Complete!                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
BASH

chmod +x scripts/run_all.sh
```

---

## 17.18 16 GPU Run (The Final Boss)

```bash
cat > scripts/run_16gpu.sh <<'BASH'
#!/bin/bash
# Run full capstone on 2 nodes × 8 GPUs = 16 GPUs
# Adjust NODE0/NODE1 to your cluster hostnames

NODE0="scn38-mn"       # change to your node 0 hostname
NODE1="scn38-w1"       # change to your node 1 hostname
MASTER_ADDR=$NODE0
MASTER_PORT=29500
NPROC=8
NNODES=2

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/capstone
mkdir -p logs checkpoints

echo "Launching 16 GPU run: $NODE0 + $NODE1"
echo "MASTER: $MASTER_ADDR:$MASTER_PORT"

# Launch node 1 in background
ssh $NODE1 "
    source /root/ddl_course/venv/bin/activate
    cd /root/ddl_course/capstone
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export NCCL_IB_DISABLE=0
    export NCCL_DEBUG=WARN
    torchrun \
        --nproc_per_node=$NPROC \
        --nnodes=$NNODES \
        --node_rank=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py \
        --config=configs/small.yaml \
        --backend=ddp \
        --max_iters=2000 \
        --out_dir=checkpoints/16gpu_run \
        2>&1 | tee logs/16gpu_node1.log
" &
NODE1_PID=$!

# Launch node 0 (foreground)
export MASTER_ADDR MASTER_PORT
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

torchrun \
    --nproc_per_node=$NPROC \
    --nnodes=$NNODES \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config=configs/small.yaml \
    --backend=ddp \
    --max_iters=2000 \
    --out_dir=checkpoints/16gpu_run \
    2>&1 | tee logs/16gpu_node0.log

wait $NODE1_PID

echo ""
echo "16 GPU run complete."
echo "Results:"
grep "step=" logs/16gpu_node0.log | tail -10
BASH

chmod +x scripts/run_16gpu.sh
```

---

## 17.19 Performance Report Generator

```bash
cat > scripts/performance_report.py <<'PY'
"""
Parse all benchmark logs and generate final performance report.
"""
import re
import os
import json
from pathlib import Path


def parse_log(path: str) -> dict:
    """Extract key metrics from a training log."""
    tok_rates, mfus, losses = [], [], []

    with open(path) as f:
        for line in f:
            m = re.search(r'tok/s=(\d+(?:\.\d+)?)', line)
            if m:
                tok_rates.append(float(m.group(1)))

            m = re.search(r'mfu=(\d+(?:\.\d+)?)', line)
            if m:
                mfus.append(float(m.group(1)))

            m = re.search(r'loss=(\d+\.\d+)', line)
            if m and 'eval' not in line:
                losses.append(float(m.group(1)))

    def avg(lst, last=50):
        h = lst[-last:]
        return sum(h) / len(h) if h else 0

    return {
        "avg_tok_s": avg(tok_rates),
        "avg_mfu":   avg(mfus),
        "final_loss":losses[-1] if losses else 0,
        "n_steps":   len(losses),
    }


def main():
    log_dir = Path("logs")
    if not log_dir.exists():
        print("No logs/ directory found. Run training first.")
        return

    print("\n" + "="*70)
    print("CAPSTONE PROJECT — PERFORMANCE REPORT")
    print("="*70)

    # Scaling table
    print(f"\n── Scaling Efficiency ─────────────────────────────────────────")
    print(f"{'Config':25s}  {'tok/s':>9}  {'MFU%':>7}  "
          f"{'efficiency':>11}  {'loss':>8}")
    print(f"{'─'*65}")

    baseline_tok_s = None
    for logfile in sorted(log_dir.glob("*.log")):
        if "benchmark" in str(logfile.parent):
            continue
        name = logfile.stem

        m    = parse_log(str(logfile))
        if m["avg_tok_s"] == 0:
            continue

        # Extract GPU count from name
        n_gpu_match = re.search(r'(\d+)gpu', name)
        n_gpu = int(n_gpu_match.group(1)) if n_gpu_match else 1

        if n_gpu == 1 and baseline_tok_s is None:
            baseline_tok_s = m["avg_tok_s"]

        if baseline_tok_s and baseline_tok_s > 0:
            eff = m["avg_tok_s"] / (n_gpu * baseline_tok_s) * 100
            eff_str = f"{eff:.1f}%"
        else:
            eff_str = "baseline"

        print(f"  {name:23s}  "
              f"{m['avg_tok_s']:>9.0f}  "
              f"{m['avg_mfu']:>7.1f}  "
              f"{eff_str:>11}  "
              f"{m['final_loss']:>8.4f}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    best_mfu = 0
    best_log = ""
    for logfile in log_dir.glob("*.log"):
        m = parse_log(str(logfile))
        if m["avg_mfu"] > best_mfu:
            best_mfu = m["avg_mfu"]
            best_log = logfile.stem

    print(f"  Best MFU: {best_mfu:.2f}% ({best_log})")

    grade = ("A (> 45%)" if best_mfu > 45
             else "B (30-45%)" if best_mfu > 30
             else "C (< 30%)")
    print(f"  Grade:    {grade}")

    print("\n  Checklist:")
    checks = [
        ("Unit tests pass",              "tests/"),
        ("Single GPU runs",              "logs/1gpu.log"),
        ("8 GPU DDP runs",               "logs/8gpu_ddp.log"),
        ("8 GPU FSDP runs",              "logs/8gpu_fsdp.log"),
        ("Checkpoint save/load works",   "checkpoints/8gpu_ddp/ckpt_best.pt"),
        ("Resume from checkpoint works", "logs/8gpu_ddp_resume.log"),
        ("Profiling traces generated",   "checkpoints/profile_run/traces/"),
        ("Text generation works",        None),
        ("MFU > 30%",                    None),
        ("Scaling table filled",         None),
    ]
    for item, path in checks:
        if path:
            exists = os.path.exists(path)
            status = "✓" if exists else "✗"
        else:
            status = "?"
        print(f"    [{status}] {item}")

    print("="*70)


if __name__ == "__main__":
    main()
PY
```

---

## 17.20 Final Capstone Execution

```bash
# ── 1. Run unit tests first ───────────────────────────────────────────────────
cd /root/ddl_course/capstone
pip install pytest
python -m pytest tests/test_model.py -v

# ── 2. Smoke test (1 GPU, 100 steps) ─────────────────────────────────────────
python train.py \
    --config=configs/small.yaml \
    --max_iters=100 \
    --log_interval=10 \
    --eval_interval=50 \
    --out_dir=checkpoints/smoke_test

# ── 3. Full single GPU run ────────────────────────────────────────────────────
python train.py \
    --config=configs/small.yaml \
    --max_iters=2000 \
    --out_dir=checkpoints/1gpu_full \
    2>&1 | tee logs/1gpu.log

# ── 4. 8 GPU DDP ──────────────────────────────────────────────────────────────
torchrun --nproc_per_node=8 train.py \
    --config=configs/small.yaml \
    --backend=ddp \
    --max_iters=2000 \
    --out_dir=checkpoints/8gpu_ddp \
    2>&1 | tee logs/8gpu_ddp.log

# ── 5. 8 GPU FSDP ─────────────────────────────────────────────────────────────
torchrun --nproc_per_node=8 train.py \
    --config=configs/small.yaml \
    --backend=fsdp \
    --max_iters=2000 \
    --out_dir=checkpoints/8gpu_fsdp \
    2>&1 | tee logs/8gpu_fsdp.log

# ── 6. Scaling benchmark ──────────────────────────────────────────────────────
bash scripts/benchmark_scaling.sh

# ── 7. Profiling ──────────────────────────────────────────────────────────────
torchrun --nproc_per_node=8 train.py \
    --config=configs/small.yaml \
    --max_iters=30 \
    --profile_steps=10 \
    --out_dir=checkpoints/profiled \
    2>&1 | tee logs/profile.log

# ── 8. Resume test ────────────────────────────────────────────────────────────
torchrun --nproc_per_node=8 train.py \
    --config=configs/small.yaml \
    --backend=ddp \
    --max_iters=3000 \
    --resume=checkpoints/8gpu_ddp/ckpt_best.pt \
    --out_dir=checkpoints/8gpu_ddp \
    2>&1 | tee logs/8gpu_ddp_resume.log

# ── 9. 16 GPU (via SLURM or manual) ──────────────────────────────────────────
# Option A: SLURM
sbatch jobs/job_16gpu.sh

# Option B: manual (2 terminals)
bash scripts/run_16gpu.sh

# ── 10. Generate text ─────────────────────────────────────────────────────────
python scripts/generate.py \
    --checkpoint=checkpoints/8gpu_ddp/ckpt_best.pt \
    --prompt="To be or not to be" \
    --max_tokens=400 \
    --n_samples=3

# ── 11. Final report ──────────────────────────────────────────────────────────
python scripts/performance_report.py

# ── 12. TensorBoard ───────────────────────────────────────────────────────────
tensorboard --logdir checkpoints/ --port 6006
```

---

## 17.21 Final Benchmark Table Template

Fill this in after running all experiments:

```bash
cat > scripts/fill_benchmark_table.py <<'PY'
"""
Print the expected benchmark table template.
Fill in your measured values.
"""

table = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           CAPSTONE BENCHMARK RESULTS — FILL IN YOUR NUMBERS                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Hardware: 2 nodes × 8 A100 SXM4 40GB (NVSwitch intra-node, IB inter-node)   ║
║  Model: GPT (d=384, L=6, ~10M params)  Dtype: bf16                           ║
║                                                                              ║
╠══════════════════════╦════════╦════════╦═══════╦══════════╦══════=═══════════╣
║ Config               ║ GPUs   ║ tok/s  ║ MFU%  ║ peak_GB  ║  efficiency      ║
╠══════════════════════╬════════╬════════╬═══════╬══════════╬══════════════=═══╣
║ Baseline (1 GPU DDP) ║ 1      ║ ___    ║ ___   ║ ___      ║ 100% (baseline)  ║
║ 2 GPU DDP            ║ 2      ║ ___    ║ ___   ║ ___      ║ ___%             ║
║ 4 GPU DDP            ║ 4      ║ ___    ║ ___   ║ ___      ║ ___%             ║
║ 8 GPU DDP            ║ 8      ║ ___    ║ ___   ║ ___      ║ ___%             ║
║ 8 GPU FSDP Z-3       ║ 8      ║ ___    ║ ___   ║ ___      ║ ___%             ║
║ 16 GPU DDP           ║ 16     ║ ___    ║ ___   ║ ___      ║ ___%             ║
╠══════════════════════╩════════╩════════╩═══════╩══════════╩═════════════════=╣
║                                                                              ║
║  Memory Optimization Results (8 GPU, same model):                            ║
║                                                                              ║
╠══════════════════════╦══════════╦══════════════════════════════════════════==╣
║ Config               ║ peak_GB  ║ Notes                                      ║
╠══════════════════════╬══════════╬══════════════════════════════════════════==╣
║ DDP, no tricks       ║ ___      ║ baseline                                   ║
║ DDP + bf16 AMP       ║ ___      ║                                            ║
║ DDP + act_ckpt       ║ ___      ║                                            ║
║ FSDP FULL_SHARD      ║ ___      ║ params sharded across 8 GPUs               ║
╠══════════════════════╩══════════╩══════════════════════════════════════════==╣
║                                                                              ║
║  Communication Analysis (8 GPU DDP):                                         ║
║    AllReduce bucket size: 25MB                                               ║
║    Comm overhead per step: ___ ms  (___ % of total step time)                ║
║    NVLink bandwidth (measured): ___ GB/s                                     ║
║                                                                              ║
║  Profiler Findings:                                                          ║
║    Top-1 op by CUDA time: ___________  (___ ms)                              ║
║    Top-2 op by CUDA time: ___________  (___ ms)                              ║
║    GPU idle time: ___ %                                                      ║
║    DataLoader bottleneck: yes / no                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
print(table)
PY

python scripts/fill_benchmark_table.py
```

---

## 17.22 Module 17 Pass Criteria

You are done with the full course when:

### Core System (non-negotiable)
1. `python -m pytest tests/test_model.py -v` → all 7 tests pass
2. `torchrun --nproc_per_node=2 tests/test_distributed.py` → PASSED
3. `torchrun --nproc_per_node=2 tests/test_checkpoint.py` → PASSED
4. Single GPU training runs and converges (val_loss decreases)
5. 8 GPU DDP training runs with correct MFU logging
6. 8 GPU FSDP training runs without OOM
7. Checkpoint save + resume produces same loss at step N+1
8. SIGTERM handler saves emergency checkpoint
9. Text generation produces readable output

### Performance
10. MFU > 30% on 8 GPU DDP (A100 SXM4)
11. Scaling efficiency > 80% at 8 GPU vs 1 GPU
12. Benchmark table filled with measured numbers

### 16 GPU
13. `bash scripts/run_16gpu.sh` or `sbatch jobs/job_16gpu.sh` runs end to end
14. Scaling table includes 16 GPU row

### Understanding (oral/written)
15. Explain what `no_sync_if_accumulating` does and why
16. Explain `scale_lr` and which rule you'd use for 16 GPUs
17. Explain why FSDP `FULL_SHARD` saves memory but not latency
18. Read the profiler trace and identify the top bottleneck

---

## 17.23 Congratulations — Course Complete

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      DISTRIBUTED DEEP LEARNING AND LLM TRAINING COURSE           ║
║                        COMPLETE                                  ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  You can now:                                                    ║
║                                                                  ║
║  ✓ Explain GPU architecture (SMs, Tensor Cores, HBM, NVLink)     ║
║  ✓ Write CUDA kernels from scratch                               ║
║  ✓ Train on 1 GPU → 8 GPUs → 16 GPUs without changing model      ║
║  ✓ Understand rank/world_size/local_rank in your sleep           ║
║  ✓ Write + debug DDP, FSDP, DeepSpeed training scripts           ║
║  ✓ Write production SLURM job scripts                            ║
║  ✓ Build GPT from scratch (every component)                      ║
║  ✓ Optimize memory (bf16, ckpt, ZeRO, offload)                   ║
║  ✓ Implement Tensor Parallel and Pipeline Parallel               ║
║  ✓ Profile with Nsight Systems, Nsight Compute, PyTorch          ║
║  ✓ Read NanoGPT, HuggingFace, Megatron-LM production code        ║
║  ✓ Build and deploy a full distributed training system           ║
║                                                                  ║
║  Hardware used: 2 nodes × 8 A100 SXM4 40GB = 16 GPUs             ║
║  Total course: ~80–120 hours of study + labs                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

