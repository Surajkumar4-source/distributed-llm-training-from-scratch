# Module 12 — Fully Sharded Data Parallel (FSDP)

## Summary
DDP keeps a **full copy** of the model on every GPU. If your model has 70B parameters, you need 70B × 16 bytes = 1.12 TB just for parameters + optimizer states — impossible on any single GPU. FSDP solves this by sharding everything across GPUs.

- **Theory**: why DDP fails for large models, ZeRO stages 1/2/3, FSDP internals (AllGather + ReduceScatter), sharding strategies
- **From scratch**: implement a toy ZeRO-1 optimizer to understand sharding math
- **Hands-on**: wrap GPT with FSDP, run 1→8 GPUs
- **Compare**: DDP vs FSDP memory and speed side by side
- **Advanced**: mixed precision, activation checkpointing, FSDP checkpointing (different from DDP)
- **Debug drills**: common FSDP failures, wrong wrapping policy, checkpoint mismatches

---

## 12.1 Theory — Why DDP Fails for Large Models

### DDP memory model (the problem)

```
DDP with world_size=8:

GPU0: full_model_params + full_gradients + full_optimizer_states
GPU1: full_model_params + full_gradients + full_optimizer_states
GPU2: full_model_params + full_gradients + full_optimizer_states
...
GPU7: full_model_params + full_gradients + full_optimizer_states

Memory per GPU = N×4 + N×4 + N×8 = N×16 bytes
Total cluster memory used = 8 × N×16 bytes = 128×N bytes
Memory actually needed = N×16 bytes (only 1/8 is unique data!)

Waste factor = 8× (all redundancy)
```

For GPT-3 (175B params):
```
Memory per GPU (DDP):
  params:     175B × 4  = 700 GB    ← impossible on single GPU
  grads:      175B × 4  = 700 GB
  Adam:       175B × 8  = 1.4 TB
  total:                  2.8 TB per GPU
```

### ZeRO — Zero Redundancy Optimizer

ZeRO (from DeepSpeed, Microsoft) eliminates the redundancy in DDP.

Three stages, each removing more redundancy:

```
                     Params    Grads    Optimizer States    Memory/GPU
─────────────────────────────────────────────────────────────────────
DDP (no ZeRO)          Full     Full          Full         N×16 bytes
ZeRO Stage 1           Full     Full         Sharded        N×10 bytes
ZeRO Stage 2           Full    Sharded       Sharded        N× 6 bytes
ZeRO Stage 3          Sharded  Sharded       Sharded        N× 2 bytes
─────────────────────────────────────────────────────────────────────
```

### ZeRO Stage 1 — Shard Optimizer States

Each GPU owns 1/N of the optimizer states (m and v for Adam).
After gradient AllReduce, each GPU updates only its own 1/N of parameters.

```
world_size=4, N params:

GPU0 owns optimizer state for params [0    .. N/4  ]
GPU1 owns optimizer state for params [N/4  .. N/2  ]
GPU2 owns optimizer state for params [N/2  .. 3N/4 ]
GPU3 owns optimizer state for params [3N/4 .. N    ]

Optimizer memory: N×8 / 4 = N×2 per GPU (vs N×8 in DDP)
```

### ZeRO Stage 2 — Shard Optimizer States + Gradients

Each GPU also only keeps 1/N of the gradient tensor.

After backward, instead of AllReduce (everyone gets all grads), we do ReduceScatter (each GPU gets its own gradient shard).

```
ReduceScatter:
  Input:  each GPU has a full gradient tensor
  Output: each GPU has 1/N of the fully reduced gradient

Then each GPU:
  updates its 1/N of parameters using its 1/N gradient shard
```

### ZeRO Stage 3 — Shard Everything (what FSDP implements)

Even parameters are sharded. Each GPU only stores 1/N of parameters.

But we need full parameters for the forward pass!

Solution: **AllGather** before each layer's forward, **free** after.

```
For each FlatParameter group (FSDP unit):

Before forward:
  AllGather: reconstruct full params from shards
  Each GPU has full params for this layer

During forward:
  Compute forward normally

After forward:
  Free the gathered (full) parameters
  Only keep the shard

During backward:
  AllGather: reconstruct full params again (for gradient computation)
  Compute gradients
  ReduceScatter: scatter reduced gradients (each GPU gets its shard)
  Free gathered params again
  Update own parameter shard with own gradient shard
```

Memory during forward (FSDP):
```
At any moment, only ONE FSDP unit's full params are in memory
All other units: only their 1/N shard is stored
```

### Communication pattern comparison

```
DDP:
  backward: AllReduce(full_gradients)
  Cost: 2 × N × bytes (ring allreduce sends 2× data)

FSDP:
  forward:  AllGather(param_shards)  ← extra vs DDP
  backward: AllGather(param_shards)  ← extra vs DDP
            ReduceScatter(gradients) ← equivalent to AllReduce
  Cost: similar total bytes, but different timing

Why FSDP comm cost ≈ DDP comm cost:
  AllGather: N bytes
  ReduceScatter: N bytes
  Total: 2N bytes ≈ DDP AllReduce 2N bytes
```

FSDP doesn't cost more communication — it trades memory for the same comm.

### The key FSDP parameters

```
sharding_strategy:
  FULL_SHARD       → ZeRO-3 (params+grads+optim sharded) ← most memory savings
  SHARD_GRAD_OP    → ZeRO-2 (grads+optim sharded, params gathered after forward)
  NO_SHARD         → equivalent to DDP (no sharding)
  HYBRID_SHARD     → shard within node, replicate across nodes

auto_wrap_policy:
  Controls which modules become FSDP units (wrapping granularity)
  Each FSDP unit = one AllGather/ReduceScatter group
  Too coarse: large temporary memory spikes
  Too fine:   too many small AllGather calls (latency dominated)

mixed_precision:
  MixedPrecision(param_dtype=torch.bfloat16,
                 reduce_dtype=torch.float32,
                 buffer_dtype=torch.bfloat16)

cpu_offload:
  CPUOffload(offload_params=True)
  → move unused shards to CPU RAM (saves GPU VRAM, slower)
```

---

## 12.2 Theory — FSDP Wrapping Policy (Critical Concept)

FSDP needs to know which sub-modules to wrap as independent FSDP units.

### Option 1: Wrap everything as one unit (bad)
```python
model = FSDP(model)  # one giant unit
```
Problem: AllGather reconstructs the ENTIRE model at once → peak memory = full model.

### Option 2: Wrap each transformer block (correct)
```python
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock}
)
model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

Now each block is its own FSDP unit:
- Only one block's params are fully reconstructed at a time
- All other blocks: only shards stored
- Peak temporary memory = one block's full params (much smaller)

### Option 3: Size-based wrapping
```python
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1_000_000
)
```
Wraps any sub-module with >= 1M params as an FSDP unit.

### What wrapping looks like internally

```
Before FSDP wrapping:
  GPT
  ├── token_emb  [65, 384]
  ├── pos_emb    [256, 384]
  ├── blocks[0]  (TransformerBlock)
  │   ├── attn
  │   └── ffn
  ├── blocks[1]  (TransformerBlock)
  ...
  ├── blocks[5]  (TransformerBlock)
  └── ln_final

After FSDP wrapping (wrap each TransformerBlock):
  FSDP(GPT)                       ← outer wrapper
  ├── token_emb  [65, 384]
  ├── pos_emb    [256, 384]
  ├── FSDP(blocks[0])             ← unit 0: AllGather before fwd, free after
  ├── FSDP(blocks[1])             ← unit 1
  ...
  ├── FSDP(blocks[5])             ← unit 5
  └── ln_final

FlatParameter:
  Within each FSDP unit, all params are flattened into one 1D tensor.
  Then that tensor is sharded across GPUs.
```

---

## 12.3 Theory — FlatParameter and Sharding Math

FSDP flattens all parameters within a unit into a single 1D tensor:

```
TransformerBlock params:
  attn.c_attn.weight: [384, 1152]  → flatten → [442368]
  attn.c_proj.weight: [384, 384]   → flatten → [147456]
  ffn.net.0.weight:   [384, 1536]  → flatten → [589824]
  ...

FlatParameter: concatenate all → [1,771,008]

With world_size=8:
  Each GPU shard: [1,771,008 / 8] = [221,376]

Padding: if not divisible, FSDP pads to nearest multiple
```

During AllGather:
```
GPU0 shard: flat[0:221376]
GPU1 shard: flat[221376:442752]
...
GPU7 shard: flat[1549632:1771008]

AllGather → full flat[0:1771008]
Reshape back to original param shapes for use in forward
```

---

## 12.4 Setup

```bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course
mkdir -p fsdp_module
cp gpt_scratch/tokenizer.py              fsdp_module/
cp gpt_scratch/shakespeare.txt           fsdp_module/
cp memory_module/gpt_with_checkpointing.py fsdp_module/
cp gpt_scratch/multihead_attention.py    fsdp_module/
cp gpt_scratch/feedforward.py            fsdp_module/
cd fsdp_module
```

We need a clean GPT that FSDP can wrap. Let's create it:

```bash
cat > model.py <<'PY'
"""
GPT model ready for FSDP wrapping.
Key difference from Module 9: TransformerBlock is importable
separately so FSDP can identify it as a wrapping target.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint as torch_checkpoint


@dataclass
class GPTConfig:
    vocab_size:     int   = 65
    max_seq_len:    int   = 256
    d_model:        int   = 384
    n_heads:        int   = 6
    n_layers:       int   = 6
    dropout:        float = 0.1
    use_checkpoint: bool  = False

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.c_attn  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj  = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop    = nn.Dropout(cfg.dropout)

        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.c_attn(x).split(self.d_model, dim=2)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        scale  = math.sqrt(self.d_head)
        scores = Q @ K.transpose(-2, -1) / scale
        scores = scores.masked_fill(
            self.mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0, float('-inf')
        )
        w   = self.drop(F.softmax(scores, dim=-1))
        out = w @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


# TransformerBlock is the FSDP wrapping unit
# FSDP will wrap each instance of this class separately
class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1            = nn.LayerNorm(cfg.d_model)
        self.attn           = CausalSelfAttention(cfg)
        self.ln2            = nn.LayerNorm(cfg.d_model)
        self.ffn            = FeedForward(cfg)
        self.use_checkpoint = cfg.use_checkpoint

    def _forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch_checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg       = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop_emb  = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([TransformerBlock(cfg)
                                        for _ in range(cfg.n_layers)])
        self.ln_final  = nn.LayerNorm(cfg.d_model)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters()) \
               - self.lm_head.weight.numel()

    def forward(self, idx, targets=None):
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
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx
PY
```

---

## 12.5 Lab 1 — Toy ZeRO-1 Optimizer (understand sharding math)

```bash
cat > toy_zero1.py <<'PY'
"""
Implement ZeRO Stage 1 from scratch (no NCCL, simulated).

Goal: understand the sharding logic before using FSDP.

ZeRO-1:
  - each rank owns 1/N of optimizer states
  - after AllReduce of gradients (same as DDP),
    each rank only updates its 1/N of parameters
  - then AllGather to sync updated parameters
"""
import math
import torch
import torch.distributed as dist
import os


class ZeroOneOptimizer:
    """
    Simulates ZeRO Stage 1 sharding of optimizer states.
    Wraps any base optimizer.
    """
    def __init__(self, params, base_optimizer_cls, rank, world_size, **kwargs):
        self.rank       = rank
        self.world_size = world_size

        # Flatten all params into a list
        self.all_params = list(params)
        n               = len(self.all_params)

        # ── Partition params across ranks ─────────────────────────────────
        # Each rank owns a contiguous slice of parameters
        params_per_rank = math.ceil(n / world_size)
        start           = rank * params_per_rank
        end             = min(start + params_per_rank, n)
        self.my_params  = self.all_params[start:end]

        # ── Only create optimizer state for this rank's params ────────────
        self.optimizer  = base_optimizer_cls(self.my_params, **kwargs)

        print(f"[rank {rank}] ZeRO-1: owns {len(self.my_params)}/{n} params")

    def step(self):
        """
        Step 1: update this rank's parameters (optimizer states already local)
        Step 2: AllGather updated parameters so all ranks have full model
        """
        # Update our subset of parameters
        self.optimizer.step()

        # AllGather: sync updated params to all ranks
        # In real ZeRO-1, this is a parameter broadcast from each rank
        for i, param in enumerate(self.all_params):
            # Determine which rank owns this param
            params_per_rank = math.ceil(len(self.all_params) / self.world_size)
            owner_rank      = i // params_per_rank

            # Broadcast from owner to everyone
            dist.broadcast(param.data, src=owner_rank)

    def zero_grad(self, set_to_none=True):
        if set_to_none:
            for p in self.all_params:
                p.grad = None
        else:
            for p in self.all_params:
                if p.grad is not None:
                    p.grad.zero_()

    def memory_stats(self):
        """Report optimizer state memory."""
        state_bytes = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state.get(p, {})
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        state_bytes += v.numel() * v.element_size()
        return state_bytes / 1024**2   # MB


def train_with_zero1():
    """Train a small model using ZeRO-1 optimizer."""
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from model import GPT, GPTConfig
    torch.manual_seed(42)
    cfg   = GPTConfig(vocab_size=65, max_seq_len=128,
                      d_model=256, n_heads=4, n_layers=4, dropout=0.0)
    model = GPT(cfg).to(device)

    # Sync model params across ranks (same as DDP init)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # ZeRO-1: each rank owns 1/world_size of optimizer states
    zero_opt = ZeroOneOptimizer(
        params            = model.parameters(),
        base_optimizer_cls= torch.optim.AdamW,
        rank              = rank,
        world_size        = world_size,
        lr                = 1e-3,
    )

    n_params   = cfg.d_model  # rough
    ddp_optim_mb  = sum(p.numel() for p in model.parameters()) * 8 / 1024**2
    zero1_optim_mb = zero_opt.memory_stats()

    if rank == 0:
        print(f"\nOptimizer memory comparison:")
        print(f"  DDP (full Adam states): {ddp_optim_mb:.1f} MB")
        print(f"  ZeRO-1 (sharded):       ~{ddp_optim_mb/world_size:.1f} MB/rank")

    # ── Training loop ──────────────────────────────────────────────────────
    for step in range(10):
        x = torch.randint(0, 65, (4, 128), device=device)
        y = torch.randint(0, 65, (4, 128), device=device)

        zero_opt.zero_grad(set_to_none=True)

        # Forward (same on all ranks — each rank has full model)
        _, loss = model(x, y)

        # Backward
        loss.backward()

        # AllReduce gradients (same as DDP)
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # ZeRO-1 step: each rank updates its own params + AllGather
        zero_opt.step()

        if rank == 0 and step % 3 == 0:
            after_step_mb = zero_opt.memory_stats()
            print(f"  step={step} loss={loss.item():.4f} "
                  f"optim_state_mb={after_step_mb:.1f}",
                  flush=True)

    if rank == 0:
        print("ZeRO-1 training: PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    train_with_zero1()
PY

torchrun --nproc_per_node=4 toy_zero1.py
```

---

## 12.6 Lab 2 — FSDP Hello World

```bash
cat > fsdp_hello.py <<'PY'
"""
Minimal FSDP example.
Shows: wrapping, parameter inspection, shard size.
"""
import os, functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

from model import GPT, GPTConfig, TransformerBlock


def inspect_fsdp_model(model, rank, world_size):
    """
    Inspect how FSDP sharded the parameters.
    Shows shard sizes vs full sizes.
    """
    if rank != 0:
        return

    print(f"\n── FSDP Parameter Inspection (rank=0) ─────────────────────────")
    total_local  = 0
    total_full   = 0

    for name, param in model.named_parameters():
        local_numel = param.numel()   # shard size on this rank
        # Full size would be local × world_size (roughly, ignoring padding)
        full_numel  = local_numel * world_size
        total_local += local_numel
        total_full  += full_numel
        local_mb = local_numel * param.element_size() / 1024**2
        print(f"  {name:50s}  "
              f"shard={local_numel:8,}  "
              f"full≈{full_numel:8,}  "
              f"local_MB={local_mb:.3f}")

    print(f"\n  total local (this rank):    {total_local:,} params  "
          f"({total_local*4/1024**2:.2f}MB)")
    print(f"  total full (whole model):   {total_full:,} params  "
          f"({total_full*4/1024**2:.2f}MB)")
    print(f"  sharding factor:            {world_size}×")
    print("────────────────────────────────────────────────────────────────\n")


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    torch.manual_seed(42)
    cfg   = GPTConfig(
        vocab_size=65, max_seq_len=256,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0
    )

    # ── Create model on CPU first (important for large models!) ───────────
    # For huge models: init on CPU, FSDP shards + moves to GPU
    model = GPT(cfg)

    # ── Define wrap policy ────────────────────────────────────────────────
    # Wrap each TransformerBlock as an independent FSDP unit
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # ── Mixed precision config ────────────────────────────────────────────
    mp_policy = MixedPrecision(
        param_dtype  = torch.bfloat16,  # store params in bf16
        reduce_dtype = torch.float32,   # AllGather/ReduceScatter in fp32
        buffer_dtype = torch.bfloat16,  # buffers (like mask) in bf16
    )

    # ── Wrap with FSDP ────────────────────────────────────────────────────
    model = FSDP(
        model,
        auto_wrap_policy  = wrap_policy,
        mixed_precision   = mp_policy,
        sharding_strategy = ShardingStrategy.FULL_SHARD,  # ZeRO-3
        device_id         = torch.cuda.current_device(),   # shard to this GPU
        use_orig_params   = True,   # important for optimizers + compile
    )

    if is_main:
        print(f"\nworld_size={world_size}")
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"\nFSDP model structure:")
        print(model)

    # Inspect sharded parameters
    inspect_fsdp_model(model, rank, world_size)

    # ── Test forward pass ─────────────────────────────────────────────────
    x = torch.randint(0, 65, (4, 64), device=device)
    y = torch.randint(0, 65, (4, 64), device=device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)

    if is_main:
        print(f"Forward pass: logits={logits.shape} loss={loss.item():.4f}")

    # ── Backward pass ──────────────────────────────────────────────────────
    loss.backward()

    if is_main:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        alloc_gb= torch.cuda.memory_allocated() / 1024**3
        print(f"Backward done: alloc={alloc_gb:.3f}GB  peak={peak_gb:.3f}GB")
        print("FSDP hello: PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 fsdp_hello.py
```

---

## 12.7 Lab 3 — DDP vs FSDP Memory Comparison

```bash
cat > ddp_vs_fsdp_memory.py <<'PY'
"""
Side-by-side memory comparison: DDP vs FSDP.

Runs the same training step with:
  - DDP (full model on each GPU)
  - FSDP FULL_SHARD (ZeRO-3)
  - FSDP SHARD_GRAD_OP (ZeRO-2)

Reports peak memory, throughput, communication overhead.
"""
import os, gc, time, functools
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

from model import GPT, GPTConfig, TransformerBlock


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench_step(model, x, y, optimizer, ctx, iters=30, warmup=5):
    """Benchmark a training step. Returns (avg_ms, peak_gb)."""
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()
    reset_memory()

    t0 = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()
    ms      = (time.perf_counter() - t0) * 1000 / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return ms, peak_gb


def build_ddp(cfg, device, local_rank):
    torch.manual_seed(42)
    model = GPT(cfg).to(device)
    model = DDP(model, device_ids=[local_rank])
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    return model, opt


def build_fsdp(cfg, device, strategy, use_mp):
    torch.manual_seed(42)
    model = GPT(cfg)

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    mp_policy = MixedPrecision(
        param_dtype  = torch.bfloat16,
        reduce_dtype = torch.float32,
        buffer_dtype = torch.bfloat16,
    ) if use_mp else None

    model = FSDP(
        model,
        auto_wrap_policy  = wrap_policy,
        mixed_precision   = mp_policy,
        sharding_strategy = strategy,
        device_id         = torch.cuda.current_device(),
        use_orig_params   = True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    return model, opt


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

    B, T = 32, 256
    cfg  = GPTConfig(
        vocab_size=65, max_seq_len=T,
        d_model=384, n_heads=6, n_layers=6, dropout=0.0
    )

    x = torch.randint(0, 65, (B, T), device=device)
    y = torch.randint(0, 65, (B, T), device=device)

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    results = []

    # ── DDP ───────────────────────────────────────────────────────────────
    reset_memory()
    model, opt = build_ddp(cfg, device, local_rank)
    ms, peak   = bench_step(model, x, y, opt, amp_ctx)
    tok_s      = B * T * world_size / (ms / 1000)
    results.append(("DDP (ZeRO-0)", ms, peak, tok_s))
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    # ── FSDP SHARD_GRAD_OP (ZeRO-2) ──────────────────────────────────────
    reset_memory()
    model, opt = build_fsdp(cfg, device,
                             ShardingStrategy.SHARD_GRAD_OP, use_mp=True)
    ms, peak   = bench_step(model, x, y, opt, amp_ctx)
    tok_s      = B * T * world_size / (ms / 1000)
    results.append(("FSDP SHARD_GRAD_OP (ZeRO-2)", ms, peak, tok_s))
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    # ── FSDP FULL_SHARD (ZeRO-3) ─────────────────────────────────────────
    reset_memory()
    model, opt = build_fsdp(cfg, device,
                             ShardingStrategy.FULL_SHARD, use_mp=True)
    ms, peak   = bench_step(model, x, y, opt, amp_ctx)
    tok_s      = B * T * world_size / (ms / 1000)
    results.append(("FSDP FULL_SHARD (ZeRO-3)", ms, peak, tok_s))
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    # ── FSDP FULL_SHARD + activation checkpoint ───────────────────────────
    reset_memory()
    cfg_ckpt = GPTConfig(vocab_size=65, max_seq_len=T,
                          d_model=384, n_heads=6, n_layers=6,
                          dropout=0.0, use_checkpoint=True)
    model, opt = build_fsdp(cfg_ckpt, device,
                             ShardingStrategy.FULL_SHARD, use_mp=True)
    ms, peak   = bench_step(model, x, y, opt, amp_ctx)
    tok_s      = B * T * world_size / (ms / 1000)
    results.append(("FSDP FULL_SHARD + act_ckpt", ms, peak, tok_s))
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    # ── Print results ─────────────────────────────────────────────────────
    if is_main:
        baseline_peak = results[0][2]
        baseline_ms   = results[0][1]

        print(f"\n{'='*80}")
        print(f"DDP vs FSDP Comparison  |  world_size={world_size}  "
              f"B={B}  T={T}  d={cfg.d_model}  L={cfg.n_layers}")
        print(f"{'='*80}")
        print(f"{'Strategy':35s}  {'step_ms':>9}  {'speedup':>9}  "
              f"{'peak_GB':>9}  {'mem_save':>9}  {'tok/s':>10}")
        print(f"{'─'*80}")

        for name, ms, peak, tok_s in results:
            speedup  = baseline_ms / ms
            mem_save = (1 - peak / baseline_peak) * 100
            print(f"  {name:33s}  {ms:>9.2f}  {speedup:>9.2f}×  "
                  f"{peak:>9.3f}  {mem_save:>8.1f}%  {tok_s:>10.0f}")

        print(f"{'='*80}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

# Run on 2, 4, 8 GPUs and observe memory reduction
torchrun --nproc_per_node=2 ddp_vs_fsdp_memory.py
torchrun --nproc_per_node=4 ddp_vs_fsdp_memory.py
torchrun --nproc_per_node=8 ddp_vs_fsdp_memory.py
```

---

## 12.8 Lab 4 — FSDP Checkpointing (Different from DDP!)

FSDP checkpointing requires special APIs because parameters are sharded.

```bash
cat > fsdp_checkpoint.py <<'PY'
"""
FSDP Checkpointing — 3 methods explained:

Method 1: FULL_STATE_DICT
  - Gather full model to rank 0, save as normal state dict
  - Simple, compatible with non-FSDP models
  - Expensive: requires full model in CPU RAM on rank 0

Method 2: SHARDED_STATE_DICT (recommended for large models)
  - Each rank saves its own shard
  - Shards saved as separate files
  - Load must use same world_size (or use resharding)
  - Most memory efficient

Method 3: LOCAL_STATE_DICT
  - Raw local shard (implementation detail, rarely used directly)

We implement and test Methods 1 and 2.
"""
import os, functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model import GPT, GPTConfig, TransformerBlock


def build_fsdp_model(cfg, device):
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    mp_policy = MixedPrecision(
        param_dtype  = torch.bfloat16,
        reduce_dtype = torch.float32,
        buffer_dtype = torch.bfloat16,
    )
    model = FSDP(
        GPT(cfg),
        auto_wrap_policy  = wrap_policy,
        mixed_precision   = mp_policy,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        device_id         = torch.cuda.current_device(),
        use_orig_params   = True,
    )
    return model


def save_full_state_dict(model, optimizer, step, path, rank):
    """
    Method 1: FULL_STATE_DICT
    Gathers full model on CPU (rank 0 only saves).
    """
    # Context manager: FSDP gathers all shards to rank 0
    save_policy = FullStateDictConfig(
        offload_to_cpu = True,   # gather to CPU (not GPU) to save GPU memory
        rank0_only     = True,   # only rank 0 has the full dict
    )

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        save_policy,
    ):
        cpu_state = model.state_dict()

    # Only rank 0 saves (others have empty dict with rank0_only=True)
    if rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "step":      step,
            "model":     cpu_state,
            "optimizer": optimizer.state_dict(),
        }, path)
        print(f"[rank 0] Saved FULL_STATE_DICT → {path}")

    dist.barrier()   # wait for rank 0 to finish saving


def load_full_state_dict(model, optimizer, path, device, rank):
    """Load FULL_STATE_DICT checkpoint into FSDP model."""
    ckpt = torch.load(path, map_location="cpu")

    load_policy = FullStateDictConfig(
        offload_to_cpu = True,
        rank0_only     = True,
    )

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        load_policy,
    ):
        model.load_state_dict(ckpt["model"])

    optimizer.load_state_dict(ckpt["optimizer"])

    if rank == 0:
        print(f"[rank 0] Loaded FULL_STATE_DICT ← {path} "
              f"(step={ckpt['step']})")
    dist.barrier()
    return ckpt["step"]


def save_sharded_state_dict(model, optimizer, step, save_dir, rank):
    """
    Method 2: SHARDED_STATE_DICT
    Each rank saves its own shard.
    More scalable for large models.
    """
    os.makedirs(save_dir, exist_ok=True)

    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    ):
        sharded_state = model.state_dict()

    # Each rank saves its own file
    rank_path = os.path.join(save_dir, f"rank_{rank:04d}.pt")
    torch.save({
        "step":  step,
        "model": sharded_state,
        "optim": optimizer.state_dict(),
        "rank":  rank,
    }, rank_path)

    dist.barrier()   # all ranks done saving

    if rank == 0:
        print(f"[all ranks] Saved SHARDED_STATE_DICT → {save_dir}/")


def load_sharded_state_dict(model, optimizer, save_dir, rank):
    """Load SHARDED_STATE_DICT. Each rank loads its own file."""
    rank_path = os.path.join(save_dir, f"rank_{rank:04d}.pt")
    ckpt      = torch.load(rank_path, map_location="cpu")

    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    ):
        model.load_state_dict(ckpt["model"])

    optimizer.load_state_dict(ckpt["optim"])
    dist.barrier()

    if rank == 0:
        print(f"[all ranks] Loaded SHARDED_STATE_DICT ← {save_dir}/")
    return ckpt["step"]


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    cfg   = GPTConfig(vocab_size=65, max_seq_len=128,
                      d_model=256, n_heads=4, n_layers=4, dropout=0.0)

    # ── Train 5 steps ─────────────────────────────────────────────────────
    torch.manual_seed(42)
    model = build_fsdp_model(cfg, device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, 65, (4, 128), device=device)
    y = torch.randint(0, 65, (4, 128), device=device)

    losses_before = []
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses_before.append(loss.item())

    if is_main:
        print(f"\nLosses before save: {[f'{l:.4f}' for l in losses_before]}")

    # ── Method 1: FULL_STATE_DICT ─────────────────────────────────────────
    save_full_state_dict(model, opt, step=5,
                          path="checkpoints_fsdp/full/checkpoint.pt",
                          rank=rank)

    # ── Method 2: SHARDED_STATE_DICT ─────────────────────────────────────
    save_sharded_state_dict(model, opt, step=5,
                             save_dir="checkpoints_fsdp/sharded",
                             rank=rank)

    # ── Corrupt model weights and reload ─────────────────────────────────
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)

    # Test Method 1 reload
    load_full_state_dict(model, opt,
                          path="checkpoints_fsdp/full/checkpoint.pt",
                          device=device, rank=rank)

    # Verify loss matches before save
    losses_after = []
    with torch.no_grad():
        for _ in range(3):
            _, loss = model(x, y)
            losses_after.append(loss.item())

    if is_main:
        print(f"Losses after reload (FULL): {[f'{l:.4f}' for l in losses_after]}")
        match = abs(losses_after[0] - losses_before[-1]) < 0.01
        print(f"Loss matches: {match}")
        print(f"FULL_STATE_DICT checkpoint: {'PASSED' if match else 'FAILED'}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

mkdir -p checkpoints_fsdp
torchrun --nproc_per_node=4 fsdp_checkpoint.py
```

---

## 12.9 Lab 5 — Full FSDP Training Script

```bash
cat > train_gpt_fsdp.py <<'PY'
"""
Full GPT training with FSDP.

Features:
  - FULL_SHARD (ZeRO-3) or SHARD_GRAD_OP (ZeRO-2)
  - bf16 mixed precision
  - Activation checkpointing
  - Gradient clipping (with FSDP-specific API)
  - FULL_STATE_DICT checkpointing
  - Distributed evaluation
  - Throughput measurement
  - Scaling efficiency vs DDP

Launch:
  torchrun --nproc_per_node=8 train_gpt_fsdp.py
  torchrun --nproc_per_node=8 train_gpt_fsdp.py --strategy=SHARD_GRAD_OP
  torchrun --nproc_per_node=8 train_gpt_fsdp.py --no_mp  # fp32
"""
import os, gc, time, math, argparse, functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model import GPT, GPTConfig, TransformerBlock
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


# ─── LR Schedule ──────────────────────────────────────────────────────────────

def get_lr(step, warmup, total, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > total:
        return min_lr
    p     = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * p))
    return min_lr + coeff * (lr - min_lr)


# ─── FSDP Utilities ───────────────────────────────────────────────────────────

def build_fsdp_model(cfg, device, strategy, use_mp, use_ckpt):
    """Build and wrap GPT model with FSDP."""

    if use_ckpt:
        cfg = GPTConfig(
            vocab_size=cfg.vocab_size, max_seq_len=cfg.max_seq_len,
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers=cfg.n_layers, dropout=cfg.dropout,
            use_checkpoint=True,
        )

    # Initialize on CPU for large models
    torch.manual_seed(42)
    model = GPT(cfg)

    # Wrap policy: each TransformerBlock = one FSDP unit
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Mixed precision
    mp_policy = MixedPrecision(
        param_dtype  = torch.bfloat16,
        reduce_dtype = torch.float32,
        buffer_dtype = torch.bfloat16,
    ) if use_mp else None

    # Sharding strategy
    strat_map = {
        "FULL_SHARD":    ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD":      ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD":  ShardingStrategy.HYBRID_SHARD,
    }

    model = FSDP(
        model,
        auto_wrap_policy  = wrap_policy,
        mixed_precision   = mp_policy,
        sharding_strategy = strat_map[strategy],
        device_id         = torch.cuda.current_device(),
        use_orig_params   = True,
    )

    return model


def fsdp_save(model, optimizer, step, val_loss, path, rank):
    """Save FSDP model using FULL_STATE_DICT (portable)."""
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()

    if rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "step":      step,
            "model":     state,
            "optimizer": optimizer.state_dict(),
            "val_loss":  val_loss,
        }, path)
        print(f"[rank 0] checkpoint saved → {path} "
              f"(step={step} val_loss={val_loss:.4f})", flush=True)
    dist.barrier()


def fsdp_load(model, optimizer, path, rank):
    """Load FULL_STATE_DICT into FSDP model."""
    ckpt = torch.load(path, map_location="cpu")
    cfg  = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    dist.barrier()
    return ckpt["step"], ckpt.get("val_loss", float("inf"))


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, eval_iters, ctx, rank, world_size):
    model.eval()
    total_loss  = torch.zeros(1, device=device)
    total_steps = torch.zeros(1, device=device)

    for i, (x, y) in enumerate(loader):
        if i >= eval_iters:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with ctx:
            _, loss = model(x, y)
        total_loss  += loss.detach()
        total_steps += 1

    dist.all_reduce(total_loss,  op=dist.ReduceOp.SUM)
    dist.all_reduce(total_steps, op=dist.ReduceOp.SUM)
    model.train()
    return (total_loss / total_steps).item()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",      type=str, default="FULL_SHARD",
                        choices=["FULL_SHARD","SHARD_GRAD_OP",
                                 "NO_SHARD","HYBRID_SHARD"])
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--max_iters",     type=int,   default=5000)
    parser.add_argument("--seq_len",       type=int,   default=256)
    parser.add_argument("--d_model",       type=int,   default=384)
    parser.add_argument("--n_layers",      type=int,   default=6)
    parser.add_argument("--n_heads",       type=int,   default=6)
    parser.add_argument("--lr",            type=float, default=6e-4)
    parser.add_argument("--no_mp",         action="store_true")
    parser.add_argument("--no_ckpt",       action="store_true")
    parser.add_argument("--resume",        type=str,   default=None)
    parser.add_argument("--log_interval",  type=int,   default=50)
    parser.add_argument("--eval_interval", type=int,   default=500)
    parser.add_argument("--ckpt_dir",      type=str,
                        default="checkpoints_fsdp/training")
    args = parser.parse_args()

    # ── Init distributed ──────────────────────────────────────────────────
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(1337 + rank)

    use_mp   = not args.no_mp
    use_ckpt = not args.no_ckpt

    if is_main:
        print(f"\n{'='*60}")
        print(f"FSDP Training")
        print(f"  strategy:  {args.strategy}")
        print(f"  world_size:{world_size}")
        print(f"  bf16 AMP:  {use_mp}")
        print(f"  act_ckpt:  {use_ckpt}")
        print(f"  batch/GPU: {args.batch_size}")
        print(f"  global_b:  {args.batch_size * world_size}")
        print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
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
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    cfg = GPTConfig(
        vocab_size  = vocab_size,
        max_seq_len = args.seq_len,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        dropout     = 0.1,
    )

    model = build_fsdp_model(cfg, device,
                              strategy = args.strategy,
                              use_mp   = use_mp,
                              use_ckpt = use_ckpt)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model params (local shard): {n_params/1e6:.2f}M")
        print(f"Model params (full model):  "
              f"{n_params*world_size/1e6:.2f}M\n")

    # ── Optimizer ─────────────────────────────────────────────────────────
    # FSDP + use_orig_params=True → can use standard optimizer API
    decay_p    = [p for n,p in model.named_parameters()
                  if p.requires_grad and p.dim() >= 2]
    no_decay_p = [p for n,p in model.named_parameters()
                  if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [{"params": decay_p,    "weight_decay": 0.1},
         {"params": no_decay_p, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    best_val   = float("inf")
    if args.resume and os.path.exists(args.resume):
        start_step, best_val = fsdp_load(model, optimizer,
                                          args.resume, rank)
        if is_main:
            print(f"Resumed from step={start_step} val_loss={best_val:.4f}")

    # ── Mixed precision context ────────────────────────────────────────────
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_mp else torch.amp.autocast(device_type="cuda",
                                              enabled=False))

    # ── Training loop ──────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    train_iter = iter(train_loader)
    t0         = time.perf_counter()

    if is_main:
        print("── Training starts ─────────────────────────────────────────\n")

    for step in range(start_step, args.max_iters):
        # LR schedule
        lr = get_lr(step, 200, args.max_iters, args.lr, 6e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sampler epoch
        epoch = step * args.batch_size // max(1, len(train_ds))
        train_sampler.set_epoch(epoch)

        # Batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Forward + backward
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()

        # Gradient clipping — FSDP-specific API
        # (must call on FSDP model, not model.parameters())
        model.clip_grad_norm_(1.0)

        optimizer.step()

        # Logging
        if step % args.log_interval == 0:
            torch.cuda.synchronize()
            t1     = time.perf_counter()
            ms     = (t1 - t0) * 1000 / args.log_interval
            tok_s  = args.batch_size * args.seq_len * world_size \
                     * args.log_interval / (t1 - t0)
            peak   = torch.cuda.max_memory_allocated(device) / 1024**3
            alloc  = torch.cuda.memory_allocated(device) / 1024**3
            t0     = t1

            if is_main:
                print(
                    f"step={step:5d}  loss={loss.item():.4f}  "
                    f"lr={lr:.2e}  ms={ms:.1f}  tok/s={tok_s:.0f}  "
                    f"alloc={alloc:.3f}GB  peak={peak:.3f}GB",
                    flush=True
                )

        # Evaluation
        if step % args.eval_interval == 0 or \
           step == args.max_iters - 1:
            val_loss = evaluate(model, val_loader, device,
                                50, ctx, rank, world_size)

            if is_main:
                print(f"\n── eval step={step} val_loss={val_loss:.4f} ────")
                if val_loss < best_val:
                    best_val = val_loss
                    fsdp_save(
                        model, optimizer, step, val_loss,
                        path=os.path.join(args.ckpt_dir, "best.pt"),
                        rank=rank,
                    )
                print("────────────────────────────────────────────────────\n",
                      flush=True)

            dist.barrier()

    if is_main:
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"\n── Training complete ───────────────────────────────────")
        print(f"  best_val_loss: {best_val:.4f}")
        print(f"  peak memory:   {peak:.3f}GB")
        print("────────────────────────────────────────────────────────\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run experiments:
```bash
# FULL_SHARD (ZeRO-3) — maximum memory savings
torchrun --nproc_per_node=8 train_gpt_fsdp.py \
  --strategy=FULL_SHARD \
  --batch_size=64 \
  --max_iters=2000

# SHARD_GRAD_OP (ZeRO-2) — less memory savings, potentially faster
torchrun --nproc_per_node=8 train_gpt_fsdp.py \
  --strategy=SHARD_GRAD_OP \
  --batch_size=64 \
  --max_iters=2000

# No sharding (equivalent to DDP, for comparison)
torchrun --nproc_per_node=8 train_gpt_fsdp.py \
  --strategy=NO_SHARD \
  --batch_size=64 \
  --max_iters=2000
```

---

## 12.10 Lab 6 — FSDP Wrapping Policy Experiment

```bash
cat > fsdp_wrap_policy_experiment.py <<'PY'
"""
Compare different FSDP wrapping policies.

Too coarse (whole model = 1 unit):
  - Huge memory spike during AllGather (whole model in memory at once)
  - Defeats the purpose of FSDP

Too fine (every Linear = 1 unit):
  - Many small AllGather calls → latency dominated
  - High communication overhead

Just right (each TransformerBlock = 1 unit):
  - Only one block in memory at a time during forward
  - Reasonable AllGather call frequency
"""
import os, gc, time, functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

from model import GPT, GPTConfig, TransformerBlock, CausalSelfAttention


def build_model_with_policy(cfg, device, policy_name):
    torch.manual_seed(42)
    model = GPT(cfg)

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    if policy_name == "no_wrap":
        # Worst: entire model as one unit
        wrap_policy = None

    elif policy_name == "block_wrap":
        # Best: each TransformerBlock as one unit
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )

    elif policy_name == "size_1m":
        # Size-based: wrap anything >= 1M params
        wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1_000_000,
        )

    elif policy_name == "size_100k":
        # Very fine: wrap anything >= 100K params
        wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=100_000,
        )

    model = FSDP(
        model,
        auto_wrap_policy  = wrap_policy,
        mixed_precision   = mp_policy,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        device_id         = torch.cuda.current_device(),
        use_orig_params   = True,
    )
    return model


def bench_policy(cfg, device, policy_name, B, T, iters=20):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_model_with_policy(cfg, device, policy_name)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x     = torch.randint(0, 65, (B, T), device=device)
    y     = torch.randint(0, 65, (B, T), device=device)
    ctx   = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        model.clip_grad_norm_(1.0)
        opt.step()

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        model.clip_grad_norm_(1.0)
        opt.step()

    torch.cuda.synchronize()
    dist.barrier()
    ms      = (time.perf_counter() - t0) * 1000 / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    return ms, peak_gb


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    B, T = 32, 256
    cfg  = GPTConfig(vocab_size=65, max_seq_len=T,
                     d_model=384, n_heads=6, n_layers=6, dropout=0.0)

    policies = [
        "no_wrap",       # worst
        "size_100k",     # very fine
        "size_1m",       # moderate
        "block_wrap",    # recommended
    ]

    if is_main:
        print(f"\n{'='*65}")
        print(f"FSDP Wrapping Policy Experiment")
        print(f"world_size={world_size}  B={B}  T={T}")
        print(f"{'='*65}")
        print(f"{'Policy':20s}  {'step_ms':>9}  {'peak_GB':>9}  {'note':>20}")
        print(f"{'─'*65}")

    for policy in policies:
        ms, peak = bench_policy(cfg, device, policy, B, T)

        if is_main:
            notes = {
                "no_wrap":    "⚠ full model AllGather",
                "size_100k":  "too many small AllGathers",
                "size_1m":    "moderate granularity",
                "block_wrap": "✓ recommended",
            }
            print(f"  {policy:20s}  {ms:>9.2f}  {peak:>9.3f}  "
                  f"{notes[policy]:>20}", flush=True)

    if is_main:
        print(f"{'='*65}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 fsdp_wrap_policy_experiment.py
```

---

## 12.11 Lab 7 — CPU Offloading (extreme memory savings)

```bash
cat > fsdp_cpu_offload.py <<'PY'
"""
FSDP CPU Offloading:
  - Unused parameter shards are moved to CPU RAM
  - Only the current FSDP unit's params are on GPU during forward/backward
  - Massive GPU memory savings (at the cost of speed)

Use case: model that barely fits on GPU, training speed is secondary.
"""
import os, gc, time, functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model import GPT, GPTConfig, TransformerBlock


def build_model(cfg, device, cpu_offload=False):
    torch.manual_seed(42)
    model = GPT(cfg)

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    offload = CPUOffload(offload_params=True) if cpu_offload else None

    model = FSDP(
        model,
        auto_wrap_policy  = wrap_policy,
        mixed_precision   = mp_policy,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        device_id         = torch.cuda.current_device(),
        cpu_offload       = offload,
        use_orig_params   = True,
    )
    return model


def measure_step(model, B, T, device, label, iters=10):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x   = torch.randint(0, 65, (B, T), device=device)
    y   = torch.randint(0, 65, (B, T), device=device)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        model.clip_grad_norm_(1.0)
        opt.step()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()

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

    del opt
    return ms, peak


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    B, T = 16, 256
    cfg  = GPTConfig(vocab_size=65, max_seq_len=T,
                     d_model=384, n_heads=6, n_layers=6, dropout=0.0)

    if is_main:
        print(f"\nFSDP CPU Offload comparison  B={B}  T={T}")
        print(f"{'─'*55}")

    for offload in [False, True]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model  = build_model(cfg, device, cpu_offload=offload)
        ms, gb = measure_step(model, B, T, device,
                               label="offload" if offload else "no offload")

        if is_main:
            tag = "CPU offload ON" if offload else "No offload    "
            print(f"  {tag}: step_ms={ms:.1f}  peak_GPU_GB={gb:.3f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if is_main:
        print(f"{'─'*55}")
        print("Note: CPU offload trades speed for GPU memory.")
        print("      Use when model doesn't fit even with FSDP.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 fsdp_cpu_offload.py
```

---

## 12.12 Debug Drills

### Drill 1 — Wrong wrap policy causes hang

```bash
cat > fsdp_wrong_wrap.py <<'PY'
"""
Common FSDP mistake: not wrapping transformer blocks.
The whole model becomes one giant unit → different behavior,
and may cause issues with tied weights (lm_head tied to token_emb).
"""
import os, functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from model import GPT, GPTConfig, TransformerBlock


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    cfg = GPTConfig(vocab_size=65, max_seq_len=64,
                    d_model=128, n_heads=4, n_layers=2, dropout=0.0)

    # ── WRONG: no auto_wrap_policy ────────────────────────────────────────
    torch.manual_seed(42)
    model_bad = GPT(cfg)
    # Without policy: whole model = one FSDP unit
    # This means ALL params gathered at once → no memory savings
    model_bad = FSDP(
        model_bad,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        device_id         = torch.cuda.current_device(),
        use_orig_params   = True,
        # ← no auto_wrap_policy
    )

    # ── CORRECT: wrap each block ──────────────────────────────────────────
    torch.manual_seed(42)
    model_good = GPT(cfg)
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    model_good = FSDP(
        model_good,
        auto_wrap_policy  = wrap_policy,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        device_id         = torch.cuda.current_device(),
        use_orig_params   = True,
    )

    if is_main:
        # Count FSDP units in each model
        bad_units  = sum(1 for m in model_bad.modules()
                         if isinstance(m, FSDP))
        good_units = sum(1 for m in model_good.modules()
                         if isinstance(m, FSDP))
        print(f"\nWrong policy: {bad_units} FSDP units")
        print(f"Right policy: {good_units} FSDP units")
        print(f"  (should be: 1 outer + {cfg.n_layers} blocks = {1+cfg.n_layers})")

    x = torch.randint(0, 65, (4, 64), device=device)
    y = torch.randint(0, 65, (4, 64), device=device)

    # Both should work for small models
    for name, model in [("wrong", model_bad), ("correct", model_good)]:
        _, loss = model(x, y)
        loss.backward()
        if is_main:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
            print(f"  {name}: loss={loss.item():.4f}  peak={peak:.3f}GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 fsdp_wrong_wrap.py
```

---

### Drill 2 — clip_grad_norm_ must be called on FSDP model

```bash
cat > fsdp_grad_clip.py <<'PY'
"""
FSDP gradient clipping:
  WRONG: nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  RIGHT: model.clip_grad_norm_(1.0)

Why: In FSDP, params are sharded. Clipping each rank's shard
     independently gives wrong global gradient norm.
     model.clip_grad_norm_() aggregates across shards correctly.
"""
import os, functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from model import GPT, GPTConfig, TransformerBlock


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = GPTConfig(vocab_size=65, max_seq_len=64,
                    d_model=128, n_heads=4, n_layers=2, dropout=0.0)
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    torch.manual_seed(42)
    model = FSDP(
        GPT(cfg),
        auto_wrap_policy  = wrap_policy,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        device_id         = torch.cuda.current_device(),
        use_orig_params   = True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, 65, (4, 64), device=device)
    y = torch.randint(0, 65, (4, 64), device=device)

    opt.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()

    # WRONG way (gives incorrect global norm on sharded model)
    wrong_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    # RIGHT way (FSDP aggregates shard norms correctly)
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    correct_norm = model.clip_grad_norm_(1.0)

    if rank == 0:
        print(f"\nGradient clipping comparison:")
        print(f"  nn.utils (WRONG for FSDP): norm={wrong_norm:.4f}")
        print(f"  model.clip_grad_norm_:     norm={correct_norm:.4f}")
        print(f"\n  Use model.clip_grad_norm_(max_norm) for FSDP!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 fsdp_grad_clip.py
```

---

### Drill 3 — FSDP forward outside context (summon_full_params)

```bash
cat > fsdp_inspect_params.py <<'PY'
"""
To inspect FULL parameter values from FSDP model (not shard),
you must use FSDP.summon_full_params() context manager.

Outside of forward/backward, params are sharded.
Direct access gives you only the shard.
"""
import os, functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from model import GPT, GPTConfig, TransformerBlock


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg  = GPTConfig(vocab_size=65, max_seq_len=64,
                     d_model=128, n_heads=4, n_layers=2, dropout=0.0)
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock}
    )
    torch.manual_seed(42)
    model = FSDP(
        GPT(cfg),
        auto_wrap_policy  = wrap_policy,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        device_id         = torch.cuda.current_device(),
        use_orig_params   = True,
    )

    # Outside forward: params are sharded
    first_param      = next(model.parameters())
    shard_numel      = first_param.numel()

    # Use summon_full_params to access full params
    with FSDP.summon_full_params(model, writeable=False):
        first_param_full = next(model.parameters())
        full_numel       = first_param_full.numel()

    world_size = dist.get_world_size()
    if rank == 0:
        print(f"\nParameter inspection:")
        print(f"  Shard numel (this rank): {shard_numel:,}")
        print(f"  Full numel:              {full_numel:,}")
        print(f"  Ratio (should be ~{world_size}×): {full_numel/shard_numel:.1f}×")
        print(f"\n  Always use summon_full_params() to see full params!")
        print(f"  Never access model.parameters() directly for inspection.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 fsdp_inspect_params.py
```

---

## 12.13 SLURM Job for FSDP Multi-Node

```bash
cat > jobs/job_gpt_fsdp_16gpu.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=gpt_fsdp_16gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/fsdp_16gpu_%j.out
#SBATCH --error=logs/fsdp_16gpu_%j.err

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/fsdp_module

mkdir -p logs checkpoints_fsdp

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_ADDR MASTER_PORT
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

echo "MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "STRATEGY=FULL_SHARD"

srun bash -c "
  torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_gpt_fsdp.py \
    --strategy=FULL_SHARD \
    --batch_size=64 \
    --max_iters=5000 \
    --ckpt_dir=checkpoints_fsdp/run_16gpu_fsdp
"
SLURM

mkdir -p jobs logs
sbatch jobs/job_gpt_fsdp_16gpu.sh
```

---

## 12.14 Comprehensive Comparison Script

```bash
cat > final_comparison.py <<'PY'
"""
Final comparison: DDP vs FSDP strategies.
Produces a summary table suitable for a research report.
"""
import os, gc, time, functools
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model import GPT, GPTConfig, TransformerBlock


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_config(name, model_builder, device, B, T, iters=30):
    reset()
    model, opt = model_builder()
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    x = torch.randint(0, 65, (B, T), device=device)
    y = torch.randint(0, 65, (B, T), device=device)

    # Warmup
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with ctx: _, loss = model(x, y)
        loss.backward()
        if isinstance(model, FSDP):
            model.clip_grad_norm_(1.0)
        else:
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
        if isinstance(model, FSDP):
            model.clip_grad_norm_(1.0)
        else:
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


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    B, T = 32, 256
    cfg  = GPTConfig(vocab_size=65, max_seq_len=T,
                     d_model=384, n_heads=6, n_layers=6, dropout=0.0)

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    def make_ddp():
        torch.manual_seed(42)
        m   = GPT(cfg).to(device)
        m   = DDP(m, device_ids=[local_rank])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
        return m, opt

    def make_fsdp(strategy):
        def _build():
            torch.manual_seed(42)
            m   = FSDP(GPT(cfg), auto_wrap_policy=wrap_policy,
                       mixed_precision=mp_policy,
                       sharding_strategy=strategy,
                       device_id=torch.cuda.current_device(),
                       use_orig_params=True)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
            return m, opt
        return _build

    configs = [
        ("DDP (ZeRO-0)",              make_ddp),
        ("FSDP NO_SHARD",             make_fsdp(ShardingStrategy.NO_SHARD)),
        ("FSDP SHARD_GRAD_OP (Z-2)",  make_fsdp(ShardingStrategy.SHARD_GRAD_OP)),
        ("FSDP FULL_SHARD (Z-3)",     make_fsdp(ShardingStrategy.FULL_SHARD)),
    ]

    results = []
    for name, builder in configs:
        ms, peak = run_config(name, builder, device, B, T)
        tok_s    = B * T * world_size / (ms / 1000)
        results.append((name, ms, peak, tok_s))

    if is_main:
        baseline_peak = results[0][2]
        baseline_ms   = results[0][1]

        print(f"\n{'='*75}")
        print(f"DDP vs FSDP Final Comparison")
        print(f"world_size={world_size}  GPU={torch.cuda.get_device_name(0)}")
        print(f"B={B}  T={T}  d={cfg.d_model}  L={cfg.n_layers}")
        print(f"{'='*75}")
        print(f"{'Strategy':30s}  {'ms':>7}  {'×spd':>6}  "
              f"{'GB':>7}  {'mem%':>6}  {'tok/s':>9}")
        print(f"{'─'*75}")

        for name, ms, peak, tok_s in results:
            speedup  = baseline_ms / ms
            mem_pct  = peak / baseline_peak * 100
            print(f"  {name:28s}  {ms:>7.2f}  "
                  f"{speedup:>5.2f}×  {peak:>7.3f}  "
                  f"{mem_pct:>5.1f}%  {tok_s:>9.0f}")

        print(f"{'='*75}")
        print(f"\nKey insight: FSDP FULL_SHARD reduces memory by "
              f"~{world_size}× with similar throughput to DDP.")
        print(f"Use FSDP when model doesn't fit on single GPU with DDP.\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=8 final_comparison.py
```

---

## 12.15 Module 12 Pass Criteria

You're done when you can:

1) **Explain the ZeRO stages**:
   - Stage 1: optimizer state sharding (memory formula)
   - Stage 2: + gradient sharding
   - Stage 3: + parameter sharding (what FSDP implements)
   - Byte savings per stage for N params and world_size=8

2) **Explain FSDP internals**:
   - AllGather before forward (why and when)
   - ReduceScatter after backward (how it differs from AllReduce)
   - FlatParameter (what gets flattened and sharded)
   - Why wrapping granularity matters (block vs whole model)

3) **Run successfully**:
   - `torchrun --nproc_per_node=4 fsdp_hello.py` → inspect shards
   - `torchrun --nproc_per_node=8 ddp_vs_fsdp_memory.py` → comparison table
   - `torchrun --nproc_per_node=8 train_gpt_fsdp.py` → training converges
   - `torchrun --nproc_per_node=4 fsdp_checkpoint.py` → checkpoint PASSED
   - `torchrun --nproc_per_node=4 fsdp_grad_clip.py` → correct clip API

4) **Fill your comparison table**:

| Strategy | Mem/GPU | vs DDP | Step ms | tok/s |
|----------|---------|--------|---------|-------|
| DDP (ZeRO-0) | | 100% | | |
| FSDP ZeRO-2 | | ?% | | |
| FSDP ZeRO-3 | | ?% | | |
| FSDP ZeRO-3 + ckpt | | ?% | | |

5) **Know when to use what**:
   - Model fits on 1 GPU → DDP
   - Model fits on 1 GPU but tight → FSDP ZeRO-2
   - Model doesn't fit on 1 GPU → FSDP ZeRO-3
   - Model barely fits even with ZeRO-3 → ZeRO-3 + CPU offload

---

