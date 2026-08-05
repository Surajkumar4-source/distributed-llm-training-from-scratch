# Module 14 — Megatron-LM

## Summary
DDP, FSDP, and DeepSpeed all shard **data** across GPUs (each GPU processes different data, all GPUs run the full model or shards of parameters). Megatron-LM introduces fundamentally different parallelism: **model parallelism** — splitting the model itself across GPUs so that each GPU computes only part of each layer.

- **Theory**: Tensor Parallelism (TP), Pipeline Parallelism (PP), Sequence Parallelism (SP), Expert Parallelism (EP), 3D/4D parallelism
- **From scratch**: implement column-parallel and row-parallel linear layers, pipeline stages, micro-batch scheduling
- **Hands-on**: run real Megatron-LM on your GPT model
- **Combine**: TP + PP + DP = 3D parallelism on 16 GPUs
- **Debug drills**: communication group bugs, pipeline deadlocks, tensor shape mismatches

---

## 14.1 Theory — Why Data Parallelism Alone Is Not Enough

### The scaling wall

```
DDP / FSDP / ZeRO: DATA PARALLELISM
  - Each GPU processes different data
  - All GPUs run same model (or shards of it)
  - Gradient AllReduce syncs after backward
  - Limited by: single layer must fit in GPU memory

Problem with very large models:
  GPT-3 (175B params):
    Single Attention layer (d=12288, 96 heads):
      QKV projection: 12288 × (3×12288) × 2 bytes = 864 MB
      just ONE layer!
    Full model: 350 GB in fp16
    Single GPU (A100 40GB): impossible
    Even ZeRO-3 with 8 GPUs: 350/8 = 43 GB → barely fits, no room for activations
```

### The solution: Model Parallelism

Split the model itself across GPUs:

```
Tensor Parallelism (TP):
  Split individual layer weight matrices across GPUs
  Each GPU computes part of each layer's output
  Small AllReduce within each layer

Pipeline Parallelism (PP):
  Assign different layers to different GPUs
  GPU 0: layers 0-5
  GPU 1: layers 6-11
  GPU 2: layers 12-17
  GPU 3: layers 18-23
  Data flows through GPUs like a pipeline

Sequence Parallelism (SP):
  Split activations along sequence dimension
  Complements Tensor Parallelism
  Reduces activation memory further

Expert Parallelism (EP) / MoE:
  Different "expert" FFN layers on different GPUs
  Routing network decides which expert processes each token
```

### 3D Parallelism (Megatron's signature)

```
Combine all three:

TP × PP × DP = total GPUs

Example: 16 GPUs
  TP=2: split each layer across 2 GPUs
  PP=2: split layers across 2 stages
  DP=4: data parallel replicas

  GPU assignment:
  DP group 0:  [GPU0, GPU1] (TP) - [GPU2, GPU3] (TP)   (PP stages)
  DP group 1:  [GPU4, GPU5] (TP) - [GPU6, GPU7] (TP)
  DP group 2:  [GPU8, GPU9] (TP) - [GPU10, GPU11] (TP)
  DP group 3:  [GPU12,GPU13] (TP) - [GPU14, GPU15] (TP)
```

---

## 14.2 Theory — Tensor Parallelism in Depth

### The key insight: linear layers can be split

A linear layer: `Y = X @ W + b`

where `X: [B, T, in]` and `W: [in, out]`.

This can be split two ways:

### Column Parallel Linear

Split W along the output (column) dimension:

```
W: [in, out]  split into  W1: [in, out/2]  and  W2: [in, out/2]

GPU0:  Y1 = X @ W1    (shape [B, T, out/2])
GPU1:  Y2 = X @ W2    (shape [B, T, out/2])

Both GPUs need full X → need to AllGather X first (or broadcast)
Each GPU produces half the output → no communication needed yet!

Result: [Y1 | Y2] = Y  (concatenate along output dim)
```

The activation can stay split — no communication if the next layer is Row Parallel.

### Row Parallel Linear

Split W along the input (row) dimension:

```
W: [in, out]  split into  W1: [in/2, out]  and  W2: [in/2, out]

GPU0 has X1: [B, T, in/2]  →  Y_partial1 = X1 @ W1    shape [B, T, out]
GPU1 has X2: [B, T, in/2]  →  Y_partial2 = X2 @ W2    shape [B, T, out]

Y = Y_partial1 + Y_partial2  (AllReduce sum across GPUs)

Result: both GPUs have full Y after AllReduce
```

### How TP works in a Transformer (the full picture)

```
Input X: [B, T, d_model]   (same on all TP ranks, after AllGather)

── Attention ──────────────────────────────────────────────────
QKV = X @ W_qkv                [B, T, 3*d_model]
  → Column Parallel: each GPU has [B, T, 3*d_model/TP]
  → Each GPU computes attention with its heads
  → Output: [B, T, d_model/TP]

Attn_out = Attn_out @ W_o      [B, T, d_model]
  → Row Parallel: each GPU has partial [B, T, d_model]
  → AllReduce (sum) → full output [B, T, d_model]
  (+ residual add after AllReduce)

── Feed Forward ────────────────────────────────────────────────
FFN_h = GELU(X @ W1)           [B, T, 4*d_model]
  → Column Parallel: each GPU has [B, T, 4*d_model/TP]

FFN_out = FFN_h @ W2           [B, T, d_model]
  → Row Parallel: each GPU has partial [B, T, d_model]
  → AllReduce (sum) → full output [B, T, d_model]

Communication cost per block: 2 AllReduces (one attn, one FFN)
```

### Sequence Parallelism (extends TP)

With TP, between the Row Parallel AllReduce and Column Parallel input, we have the full activation on all GPUs — redundant!

Sequence Parallelism keeps the activation split along the sequence dimension between layers:

```
Standard TP:
  [Full X]  → ColParallel → [Split Y] → RowParallel → AllReduce → [Full Z]

With SP:
  [Split X (seq)] → AllGather → [Full X] → ColParallel → [Split Y (seq)]
     → ReduceScatter → [Split Z (seq)] → ... next layer

Memory savings: activation memory reduced by TP factor
Communication: AllGather + ReduceScatter ≈ same as AllReduce
```

---

## 14.3 Theory — Pipeline Parallelism in Depth

### Naive pipeline (the problem)

```
Assign layers to GPUs:
  GPU0: layers 0-5    (stage 0)
  GPU1: layers 6-11   (stage 1)
  GPU2: layers 12-17  (stage 2)
  GPU3: layers 18-23  (stage 3)

Naive forward:
  GPU0: forward(batch) → activations → GPU1
  GPU1: wait... → forward → GPU2
  GPU2: wait... → forward → GPU3
  GPU3: forward → loss

GPU Utilization:
  Time: ──────────────────────────────────────────────→
  GPU0: [fwd]..idle..idle..idle..[bwd]..idle..idle.....
  GPU1: .idle.[fwd]..idle..idle.......idle.[bwd]..idle..
  GPU2: ............[fwd]..idle..............idle.[bwd].
  GPU3: ...................[fwd]..bwd..................

Problem: "pipeline bubble" — most GPUs idle at any time
Bubble fraction: (PP-1) / (PP-1 + num_micro_batches)
With PP=4 and 1 micro-batch: 3/4 = 75% idle!
```

### 1F1B Schedule (One Forward One Backward)

Megatron's solution: split the batch into micro-batches and pipeline them:

```
m = number of micro-batches  (should be ≥ PP for good efficiency)
p = pipeline stages

1F1B schedule with PP=4, m=4:

Time:   1    2    3    4    5    6    7    8    9   10   11   12
GPU0: [F1] [F2] [F3] [F4] [B4] [B3] [B2] [B1]
GPU1:      [F1] [F2] [F3] [F4] [B4] [B3] [B2] [B1]
GPU2:           [F1] [F2] [F3] [F4] [B4] [B3] [B2] [B1]
GPU3:                [F1] [F2] [F3] [F4] [B4] [B3] [B2] [B1]

Startup bubble: (PP-1) time slots
Steady state: all GPUs active
Teardown bubble: (PP-1) time slots

Bubble fraction ≈ (PP-1) / (m + PP-1)
With PP=4, m=8: 3/11 ≈ 27% — much better!
With PP=4, m=32: 3/35 ≈ 9% — even better!
```

### Interleaved 1F1B (Megatron-v2)

Further improvement: assign non-contiguous layer groups to each GPU:

```
Standard:  GPU0=[0-5],  GPU1=[6-11],  GPU2=[12-17], GPU3=[18-23]
Interleaved: GPU0=[0-2, 12-14], GPU1=[3-5, 15-17],
             GPU2=[6-8, 18-20],  GPU3=[9-11, 21-23]

Each GPU has 2 "virtual stages"
Bubble fraction ≈ (PP-1) / (m × stages_per_gpu + PP-1)
Much smaller bubble with same PP count
```

---

## 14.4 Theory — Expert Parallelism / Mixture of Experts

### Standard Transformer FFN vs MoE

```
Standard FFN:
  Every token → same FFN weights → output
  Compute: B × T × 8 × d_model² flops

MoE FFN:
  N expert FFNs exist (e.g., 8 experts)
  Router: for each token, pick top-k experts (usually k=1 or k=2)
  Each token processed by only k out of N experts
  Compute: same as standard FFN (only k experts active per token)
  Parameters: N × standard FFN params (N× more params, same compute!)
```

### Expert Parallelism

```
Place each expert on a different GPU:
  GPU0: Expert 0
  GPU1: Expert 1
  GPU2: Expert 2
  GPU3: Expert 3
  ...

For a batch of tokens:
  Router assigns each token to an expert
  All-to-All: send tokens to their assigned expert GPU
  Each GPU runs its expert on its assigned tokens
  All-to-All: return results to original GPUs
```

### 4D Parallelism = TP + PP + DP + EP

```
Modern LLMs (GPT-4, Llama-3, Mixtral) use all four:

TP: splits attention heads + FFN columns across GPUs
PP: splits layers across GPUs
DP: data parallel replicas
EP: different experts on different GPUs
```

---

## 14.5 Setup — Install Megatron-LM

```bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course
mkdir -p megatron_module
cd megatron_module

# Install dependencies
pip install einops regex sentencepiece
pip install pybind11

# Clone Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Install
pip install -e .

# Verify
python -c "import megatron; print('Megatron-LM installed successfully')"

cd ..
cp ../gpt_scratch/shakespeare.txt .
cp ../gpt_scratch/tokenizer.py .
```

---

## 14.6 Lab 1 — Tensor Parallel Linear From Scratch

Build column-parallel and row-parallel linear layers from scratch to understand the math:

```bash
cat > tp_linear.py <<'PY'
"""
Tensor Parallel Linear Layers from scratch.

ColumnParallelLinear:
  - Splits output dimension across TP ranks
  - No communication on input (input replicated across ranks)
  - No communication on output (output is split — feeds into RowParallel)

RowParallelLinear:
  - Splits input dimension across TP ranks
  - Input comes split from ColumnParallel
  - AllReduce output to get full result

Together: one forward/backward through Attention or FFN
with only ONE AllReduce per sub-layer.
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from dataclasses import dataclass


# ─── Process Group Utilities ───────────────────────────────────────────────────

_TP_GROUP = None


def init_tp_group(tp_size):
    """Initialize Tensor Parallel process group."""
    global _TP_GROUP
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size % tp_size == 0, \
        f"world_size {world_size} not divisible by tp_size {tp_size}"

    # Build TP groups: [0..tp-1], [tp..2tp-1], ...
    for i in range(world_size // tp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TP_GROUP = group

    return _TP_GROUP


def get_tp_group():
    return _TP_GROUP


def get_tp_rank():
    return dist.get_rank(group=_TP_GROUP)


def get_tp_world_size():
    return dist.get_world_size(group=_TP_GROUP)


# ─── AllReduce + Split utilities ──────────────────────────────────────────────

def all_reduce(x, group=None):
    """AllReduce sum across TP group."""
    if group is None:
        group = get_tp_group()
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    return x


def split_tensor_along_last_dim(x, tp_size, tp_rank):
    """Split tensor along last dimension for this TP rank."""
    last_dim    = x.shape[-1]
    assert last_dim % tp_size == 0, \
        f"last_dim {last_dim} not divisible by tp_size {tp_size}"
    chunk_size  = last_dim // tp_size
    return x[..., tp_rank * chunk_size : (tp_rank + 1) * chunk_size]


# ─── Custom Autograd Functions ────────────────────────────────────────────────

class CopyToModelParallelRegion(torch.autograd.Function):
    """
    Forward:  identity (input already replicated across TP ranks)
    Backward: AllReduce (gradients from all TP ranks must be summed)

    Used BEFORE ColumnParallelLinear.
    """
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        all_reduce(grad)
        return grad


class ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    Forward:  AllReduce (collect partial sums from all TP ranks)
    Backward: identity (gradient flows back unchanged)

    Used AFTER RowParallelLinear.
    """
    @staticmethod
    def forward(ctx, x):
        return all_reduce(x.clone())

    @staticmethod
    def backward(ctx, grad):
        return grad


class ScatterToModelParallelRegion(torch.autograd.Function):
    """
    Forward:  split tensor, each rank keeps its shard
    Backward: AllGather shards back together

    Used to scatter input to RowParallelLinear.
    """
    @staticmethod
    def forward(ctx, x):
        tp_rank = get_tp_rank()
        tp_size = get_tp_world_size()
        return split_tensor_along_last_dim(x, tp_size, tp_rank)

    @staticmethod
    def backward(ctx, grad):
        # AllGather gradient shards
        tp_size = get_tp_world_size()
        grads   = [torch.zeros_like(grad) for _ in range(tp_size)]
        dist.all_gather(grads, grad.contiguous(), group=get_tp_group())
        return torch.cat(grads, dim=-1)


class GatherFromModelParallelRegion(torch.autograd.Function):
    """
    Forward:  AllGather shards into full tensor
    Backward: split gradient (scatter)

    Used after ColumnParallelLinear when you want full output.
    """
    @staticmethod
    def forward(ctx, x):
        tp_size = get_tp_world_size()
        outputs = [torch.zeros_like(x) for _ in range(tp_size)]
        dist.all_gather(outputs, x.contiguous(), group=get_tp_group())
        return torch.cat(outputs, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        tp_rank = get_tp_rank()
        tp_size = get_tp_world_size()
        return split_tensor_along_last_dim(grad, tp_size, tp_rank)


# ─── Column Parallel Linear ────────────────────────────────────────────────────

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with output (column) dimension split across TP ranks.

    Y = X @ W^T  where W is split along rows (output dimension).

    Each TP rank owns W[rank*out/TP : (rank+1)*out/TP, :]
    Output Y has shape [B, T, out/TP] on each rank.

    gather_output=True:  AllGather output → full [B, T, out]
                         use when this is the LAST TP layer
    gather_output=False: keep split output
                         use when feeding into RowParallelLinear
    """
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, gather_output: bool = False):
        super().__init__()
        tp_size  = get_tp_world_size()
        tp_rank  = get_tp_rank()

        assert out_features % tp_size == 0, \
            f"out_features {out_features} not divisible by tp_size {tp_size}"

        self.in_features   = in_features
        self.out_features  = out_features
        self.out_per_rank  = out_features // tp_size
        self.gather_output = gather_output

        # Each rank has its own shard of the weight
        # Shape: [out/TP, in]
        self.weight = nn.Parameter(
            torch.empty(self.out_per_rank, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_per_rank))
        else:
            self.bias = None

        # Initialize weight (same seed across ranks, then take shard)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure gradient flows back correctly (AllReduce in backward)
        x = CopyToModelParallelRegion.apply(x)

        # Local linear: [B, T, in] @ [in, out/TP] = [B, T, out/TP]
        out = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            # AllGather from all TP ranks: [B, T, out/TP] → [B, T, out]
            out = GatherFromModelParallelRegion.apply(out)

        return out


# ─── Row Parallel Linear ───────────────────────────────────────────────────────

class RowParallelLinear(nn.Module):
    """
    Linear layer with input (row) dimension split across TP ranks.

    Y_partial = X_shard @ W_shard^T  (each rank computes partial output)
    Y = AllReduce(Y_partial)          (sum across ranks = full output)

    input_is_parallel=True:  input already split (from ColumnParallel)
    input_is_parallel=False: input is full, scatter it first
    """
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, input_is_parallel: bool = True):
        super().__init__()
        tp_size  = get_tp_world_size()

        assert in_features % tp_size == 0, \
            f"in_features {in_features} not divisible by tp_size {tp_size}"

        self.in_features      = in_features
        self.out_features     = out_features
        self.in_per_rank      = in_features // tp_size
        self.input_is_parallel= input_is_parallel

        # Shape: [out, in/TP]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_per_rank)
        )
        if bias:
            # Bias is full (not sharded) — added after AllReduce
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # Scatter input to each rank's shard
            x = ScatterToModelParallelRegion.apply(x)

        # Local linear: [B, T, in/TP] @ [in/TP, out] = [B, T, out] (partial)
        out = F.linear(x, self.weight)

        # AllReduce: sum partial outputs → full output
        out = ReduceFromModelParallelRegion.apply(out)

        if self.bias is not None:
            out = out + self.bias

        return out


# ─── Tensor Parallel MLP ──────────────────────────────────────────────────────

class TensorParallelMLP(nn.Module):
    """
    FFN with Tensor Parallelism.

    Standard: FFN(x) = GELU(x @ W1) @ W2

    TP version:
      Col Parallel: W1 split along output → each rank gets 4d/TP cols
      Row Parallel: W2 split along input  → each rank gets 4d/TP rows
      One AllReduce at the end (in RowParallel)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = ColumnParallelLinear(
            d_model, 4 * d_model,
            bias=True, gather_output=False,  # keep split for W2
        )
        self.w2 = RowParallelLinear(
            4 * d_model, d_model,
            bias=True, input_is_parallel=True,  # input comes split from W1
        )
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # [B, T, d] → [B, T, 4d/TP] (split, no AllReduce)
        h = self.act(self.w1(x))
        h = self.drop(h)
        # [B, T, 4d/TP] → [B, T, d] (AllReduce inside RowParallel)
        return self.w2(h)


# ─── Tensor Parallel Self-Attention ───────────────────────────────────────────

class TensorParallelSelfAttention(nn.Module):
    """
    Multi-head attention with Tensor Parallelism.

    Heads split across TP ranks:
      Total heads: n_heads
      Heads per TP rank: n_heads / TP

    QKV projection: Column Parallel (split by heads)
    Output projection: Row Parallel (merge heads)
    """
    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        tp_size  = get_tp_world_size()

        assert n_heads % tp_size == 0, \
            f"n_heads {n_heads} must be divisible by tp_size {tp_size}"

        self.d_model        = d_model
        self.n_heads        = n_heads
        self.n_heads_local  = n_heads // tp_size    # heads on this rank
        self.d_head         = d_model // n_heads

        # QKV: Column Parallel (output split across TP ranks by heads)
        self.c_attn = ColumnParallelLinear(
            d_model, 3 * d_model,
            bias=False, gather_output=False,
        )
        # Output: Row Parallel (input split, AllReduce output)
        self.c_proj = RowParallelLinear(
            d_model, d_model,
            bias=False, input_is_parallel=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop= nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import math
        B, T, C = x.shape

        # QKV: [B, T, 3*d/TP] (split across TP)
        qkv         = self.c_attn(x)
        d_local     = self.n_heads_local * self.d_head
        Q, K, V     = qkv.split(d_local, dim=2)

        # Reshape for multi-head attention
        def split_heads(t):
            return t.view(B, T, self.n_heads_local,
                          self.d_head).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Attention
        scale  = math.sqrt(self.d_head)
        scores = Q @ K.transpose(-2, -1) / scale
        scores = scores.masked_fill(
            self.mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0,
            float('-inf')
        )
        w   = self.attn_drop(F.softmax(scores, dim=-1))
        out = w @ V

        # Merge heads: [B, n_heads_local, T, d_head] → [B, T, d_local]
        out = out.transpose(1, 2).contiguous().view(B, T, d_local)

        # Output projection: AllReduce happens inside
        out = self.resid_drop(self.c_proj(out))
        return out


# ─── Tensor Parallel Transformer Block ────────────────────────────────────────

class TensorParallelBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = TensorParallelSelfAttention(
            d_model, n_heads, max_seq_len, dropout
        )
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = TensorParallelMLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ─── Test ──────────────────────────────────────────────────────────────────────

def test_tp_linear():
    """
    Test that TP linear gives same result as non-TP for forward pass.
    Run with torchrun --nproc_per_node=2 tp_linear.py
    """
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tp_size = world_size   # use all GPUs for TP

    # Initialize TP process group
    init_tp_group(tp_size)
    tp_rank = get_tp_rank()

    torch.manual_seed(42)

    # ── Test 1: ColumnParallelLinear ─────────────────────────────────────
    d_in, d_out = 128, 256

    # Reference: full linear on rank 0
    ref_linear = nn.Linear(d_in, d_out, bias=False).to(device)
    # Broadcast ref weights to all ranks
    dist.broadcast(ref_linear.weight.data, src=0)

    # TP linear: each rank has d_out/tp_size output columns
    col_linear  = ColumnParallelLinear(
        d_in, d_out, bias=False, gather_output=True
    ).to(device)

    # Copy corresponding shard of ref weights to TP linear
    out_per_rank = d_out // tp_size
    with torch.no_grad():
        col_linear.weight.data = ref_linear.weight.data[
            tp_rank * out_per_rank : (tp_rank + 1) * out_per_rank
        ].clone()

    # Forward
    torch.manual_seed(0)
    x           = torch.randn(2, 8, d_in, device=device)
    dist.broadcast(x, src=0)    # same input on all ranks

    with torch.no_grad():
        ref_out = ref_linear(x)       # [2, 8, 256] on all ranks (same)
        tp_out  = col_linear(x)       # [2, 8, 256] after AllGather

    if rank == 0:
        match = torch.allclose(ref_out, tp_out, atol=1e-4)
        print(f"ColumnParallelLinear (gather_output=True): "
              f"matches reference = {match}")
        if not match:
            diff = (ref_out - tp_out).abs().max()
            print(f"  max diff: {diff:.6f}")

    # ── Test 2: Col → Row (standard FFN) ─────────────────────────────────
    tp_mlp = TensorParallelMLP(d_model=128, dropout=0.0).to(device)
    ref_x  = torch.randn(2, 8, 128, device=device)
    dist.broadcast(ref_x, src=0)

    out_tp = tp_mlp(ref_x)

    if rank == 0:
        print(f"TensorParallelMLP output shape: {out_tp.shape}")
        print(f"TensorParallelMLP: forward PASSED")

    # ── Test 3: Full TP block ─────────────────────────────────────────────
    block  = TensorParallelBlock(
        d_model=128, n_heads=4, max_seq_len=16, dropout=0.0
    ).to(device)
    x      = torch.randn(2, 8, 128, device=device)
    dist.broadcast(x, src=0)

    out    = block(x)

    if rank == 0:
        print(f"TensorParallelBlock output shape: {out.shape}")
        print(f"TensorParallelBlock: forward PASSED")
        print(f"\nAll TP tests completed on {world_size} GPUs (TP={tp_size})")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_tp_linear()
PY

torchrun --nproc_per_node=2 tp_linear.py
torchrun --nproc_per_node=4 tp_linear.py
```

---

## 14.7 Lab 2 — Pipeline Parallelism From Scratch

```bash
cat > pipeline_parallel.py <<'PY'
"""
Pipeline Parallelism from scratch.

Implements:
1. Stage assignment: each rank = one pipeline stage
2. Micro-batch scheduling with 1F1B
3. Activation passing between stages via send/recv
4. Gradient synchronization

For simplicity: PP only (no TP), single micro-batch shown first.
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, List

from model import GPTConfig


# ─── Pipeline Stage Assignment ────────────────────────────────────────────────

def get_pipeline_stage_layers(n_layers: int, pp_size: int,
                               pp_rank: int) -> tuple:
    """
    Assign transformer layers to pipeline stages.
    Returns (start_layer, end_layer) for this rank.
    """
    assert n_layers % pp_size == 0, \
        f"n_layers {n_layers} must divide evenly by pp_size {pp_size}"
    layers_per_stage = n_layers // pp_size
    start            = pp_rank * layers_per_stage
    end              = start + layers_per_stage
    return start, end


# ─── Pipeline Stage Module ────────────────────────────────────────────────────

class PipelineStageGPT(nn.Module):
    """
    One pipeline stage of a GPT model.

    First stage (pp_rank=0):
      - Embedding layers
      - First N/pp_size transformer blocks

    Middle stages:
      - N/pp_size transformer blocks

    Last stage (pp_rank=pp_size-1):
      - Last N/pp_size transformer blocks
      - Final LayerNorm + LM head
    """
    def __init__(self, cfg: GPTConfig, pp_rank: int, pp_size: int):
        super().__init__()
        self.cfg     = cfg
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.is_first= (pp_rank == 0)
        self.is_last = (pp_rank == pp_size - 1)

        start, end   = get_pipeline_stage_layers(
            cfg.n_layers, pp_size, pp_rank
        )
        self.layer_start = start
        self.layer_end   = end

        # ── First stage: embeddings ────────────────────────────────────
        if self.is_first:
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.drop_emb  = nn.Dropout(cfg.dropout)

        # ── All stages: transformer blocks ─────────────────────────────
        self.blocks = nn.ModuleList()
        for i in range(start, end):
            self.blocks.append(self._make_block())

        # ── Last stage: final norm + LM head ──────────────────────────
        if self.is_last:
            self.ln_final = nn.LayerNorm(cfg.d_model)
            self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size,
                                       bias=False)
            if self.is_first:
                # Weight tying (only if first and last are same rank)
                self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _make_block(self):
        """Create one transformer block."""
        import math

        class Block(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                d    = cfg.d_model
                h    = cfg.n_heads
                dh   = d // h
                self.ln1   = nn.LayerNorm(d)
                self.c_attn= nn.Linear(d, 3*d, bias=False)
                self.c_proj= nn.Linear(d, d, bias=False)
                self.ln2   = nn.LayerNorm(d)
                self.ffn   = nn.Sequential(
                    nn.Linear(d, 4*d), nn.GELU(),
                    nn.Linear(4*d, d), nn.Dropout(cfg.dropout),
                )
                self.drop  = nn.Dropout(cfg.dropout)
                self.n_heads = h
                self.d_head  = dh
                self.d_model = d
                T = cfg.max_seq_len
                self.register_buffer("mask",
                    torch.tril(torch.ones(T, T)))

            def forward(self, x):
                B, T, C = x.shape
                # Attention
                q, k, v = self.c_attn(self.ln1(x)).split(self.d_model, -1)
                def sh(t):
                    return t.view(B,T,self.n_heads,self.d_head).transpose(1,2)
                q, k, v  = sh(q), sh(k), sh(v)
                sc       = math.sqrt(self.d_head)
                attn     = (q @ k.transpose(-2,-1) / sc).masked_fill(
                    self.mask[:T,:T].unsqueeze(0).unsqueeze(0)==0,float('-inf'))
                attn     = self.drop(F.softmax(attn, dim=-1))
                out      = (attn @ v).transpose(1,2).contiguous().view(B,T,C)
                x        = x + self.c_proj(out)
                # FFN
                x        = x + self.ffn(self.ln2(x))
                return x

        return Block(self.cfg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, targets=None):
        """
        x: token indices (first stage) OR hidden states (other stages)
        """
        if self.is_first:
            B, T = x.shape
            device = x.device
            pos = torch.arange(T, device=device)
            x   = self.drop_emb(
                self.token_emb(x) + self.pos_emb(pos)
            )

        for block in self.blocks:
            x = block(x)

        if self.is_last:
            x      = self.ln_final(x)
            logits = self.lm_head(x)
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, self.cfg.vocab_size),
                    targets.view(-1)
                )
                return logits, loss
            return logits, None

        return x, None   # intermediate stages return hidden state


# ─── Pipeline Communication ────────────────────────────────────────────────────

def send_activations(tensor: torch.Tensor, dst_rank: int):
    """Send activation tensor to next pipeline stage."""
    # Send shape first (so receiver can allocate)
    shape = torch.tensor(tensor.shape, dtype=torch.long)
    dist.send(shape, dst=dst_rank)
    dist.send(tensor.contiguous(), dst=dst_rank)


def recv_activations(src_rank: int, device: torch.device,
                      dtype=torch.float32) -> torch.Tensor:
    """Receive activation tensor from previous pipeline stage."""
    # Receive shape
    shape_buf = torch.zeros(3, dtype=torch.long)
    dist.recv(shape_buf, src=src_rank)
    shape = tuple(shape_buf.tolist())

    # Allocate and receive
    tensor = torch.zeros(*shape, device=device, dtype=dtype)
    dist.recv(tensor, src=src_rank)
    return tensor


# ─── 1F1B Pipeline Schedule ───────────────────────────────────────────────────

class PipelineEngine:
    """
    1F1B pipeline schedule for training.

    Handles:
    - Forward pass communication between stages
    - Backward pass communication between stages
    - Micro-batch scheduling
    """
    def __init__(self, model: PipelineStageGPT, pp_size: int,
                 pp_rank: int, device: torch.device):
        self.model    = model
        self.pp_size  = pp_size
        self.pp_rank  = pp_rank
        self.device   = device
        self.is_first = (pp_rank == 0)
        self.is_last  = (pp_rank == pp_size - 1)
        self.prev_rank= pp_rank - 1 if pp_rank > 0 else None
        self.next_rank= pp_rank + 1 if pp_rank < pp_size - 1 else None

    def forward_step(self, micro_batch_x, micro_batch_y=None):
        """
        Run forward for one micro-batch.
        - If first stage: takes token indices
        - Other stages: receives hidden states from prev stage
        """
        if self.is_first:
            x = micro_batch_x.to(self.device)
        else:
            # Receive from previous stage
            x = recv_activations(self.prev_rank, self.device)
            x.requires_grad_(True)  # need grad for backward

        # Forward through this stage
        out, loss = self.model(x, micro_batch_y)

        # Send to next stage (if not last)
        if not self.is_last:
            send_activations(out.detach(), self.next_rank)

        return out, loss, x if not self.is_first else None

    def backward_step(self, output, grad_output=None, input_tensor=None):
        """
        Run backward for one micro-batch.
        - Last stage: compute loss.backward()
        - Other stages: receive gradient from next stage
        """
        if self.is_last:
            output.backward()
        else:
            # Receive gradient from next stage
            grad = recv_activations(self.next_rank, self.device)
            output.backward(gradient=grad)

        # Send gradient to previous stage
        if not self.is_first and input_tensor is not None:
            if input_tensor.grad is not None:
                send_activations(input_tensor.grad, self.prev_rank)

    def run_1f1b(self, batches_x, batches_y, optimizer):
        """
        Run 1F1B schedule over a list of micro-batches.

        Phase 1: Warmup — forward only (first PP-1 micro-batches)
        Phase 2: Steady state — 1F1B pairs
        Phase 3: Cooldown — backward only
        """
        n_micro    = len(batches_x)
        saved_fwd  = []   # (output, input) pairs for backward

        # ── Warmup phase ─────────────────────────────────────────────────
        n_warmup = min(self.pp_size - 1, n_micro)
        for i in range(n_warmup):
            out, loss, inp = self.forward_step(
                batches_x[i],
                batches_y[i] if self.is_last else None
            )
            saved_fwd.append((out, inp, loss))

        # ── Steady state: 1F1B ────────────────────────────────────────────
        for i in range(n_warmup, n_micro):
            # One forward
            out, loss, inp = self.forward_step(
                batches_x[i],
                batches_y[i] if self.is_last else None
            )

            # One backward (from oldest saved forward)
            old_out, old_inp, old_loss = saved_fwd.pop(0)
            self.backward_step(old_out if not self.is_last else old_loss,
                               input_tensor=old_inp)

            saved_fwd.append((out, inp, loss))

        # ── Cooldown: remaining backwards ────────────────────────────────
        for out, inp, loss in saved_fwd:
            self.backward_step(out if not self.is_last else loss,
                               input_tensor=inp)

        # ── Optimizer step ────────────────────────────────────────────────
        # AllReduce gradients across DP ranks (if using DP + PP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


# ─── Test: naive single micro-batch pipeline ──────────────────────────────────

def test_pipeline():
    """
    Test basic pipeline parallelism.
    Each rank = one pipeline stage.
    Run with: torchrun --nproc_per_node=2 pipeline_parallel.py
    """
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    pp_size = world_size

    cfg = GPTConfig(
        vocab_size=65, max_seq_len=64,
        d_model=128, n_heads=4,
        n_layers=4,      # 4 layers, 2 per stage (for pp=2)
        dropout=0.0,
    )

    # Each rank owns different layers
    model = PipelineStageGPT(cfg, pp_rank=rank, pp_size=pp_size).to(device)

    start, end = get_pipeline_stage_layers(cfg.n_layers, pp_size, rank)
    print(f"rank={rank}: layers [{start}, {end}]  "
          f"is_first={model.is_first}  is_last={model.is_last}",
          flush=True)

    dist.barrier()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ── Simple forward/backward (single micro-batch) ───────────────────────
    B, T = 4, 64
    if rank == 0:   # only first rank has input data
        x = torch.randint(0, 65, (B, T), device=device)
        y = torch.randint(0, 65, (B, T), device=device)
    else:
        x = y = None

    # ── FORWARD ───────────────────────────────────────────────────────────
    if model.is_first:
        hidden, _, _ = model(x), None, None
        # Forward stage 0: get hidden states
        out, _, _ = model.forward(x)
        # Actually just call model forward directly for testing
        out, loss = model(x)
        send_activations(out.detach(), 1)   # send to rank 1

    elif model.is_last:
        # Receive from previous stage
        hidden = recv_activations(rank - 1, device)
        hidden.requires_grad_(True)

        # Forward last stage
        logits, loss = model(hidden, y)

        print(f"rank={rank}: loss={loss.item():.4f}", flush=True)

        # BACKWARD last stage
        loss.backward()

        # Send gradient back to previous stage
        send_activations(hidden.grad, rank - 1)

    else:
        # Middle stage
        hidden = recv_activations(rank - 1, device)
        hidden.requires_grad_(True)
        out, _ = model(hidden)
        send_activations(out.detach(), rank + 1)

        grad = recv_activations(rank + 1, device)
        out.backward(gradient=grad)
        send_activations(hidden.grad, rank - 1)

    if model.is_first:
        # Receive gradient from next stage
        grad = recv_activations(1, device)
        # Backward through embeddings
        out.backward(gradient=grad)

    opt.step()
    opt.zero_grad()

    dist.barrier()
    if rank == 0:
        print(f"\nPipeline test: forward + backward PASSED on {pp_size} stages")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_pipeline()
PY

torchrun --nproc_per_node=2 pipeline_parallel.py
torchrun --nproc_per_node=4 pipeline_parallel.py
```

---

## 14.8 Lab 3 — Process Group Setup for 3D Parallelism

```bash
cat > process_groups_3d.py <<'PY'
"""
Set up process groups for 3D parallelism (TP × PP × DP).

Key insight: each type of parallelism needs its own process group
so collective operations only happen within the right ranks.

Example: 8 GPUs, TP=2, PP=2, DP=2

Rank layout:
  [0, 1]  ← TP group (stage 0, DP replica 0)
  [2, 3]  ← TP group (stage 1, DP replica 0)
  [4, 5]  ← TP group (stage 0, DP replica 1)
  [6, 7]  ← TP group (stage 1, DP replica 1)

DP groups (AllReduce gradients):
  [0, 4], [1, 5]  ← same TP rank, same PP stage, different DP replica
  [2, 6], [3, 7]

PP groups (send/recv activations):
  [0, 2], [1, 3]  ← same TP rank, different PP stage, same DP replica
  [4, 6], [5, 7]
"""
import os
import torch
import torch.distributed as dist


class ParallelConfig:
    """Configuration for 3D parallelism."""
    def __init__(self, tp_size: int, pp_size: int, dp_size: int):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.world_size = tp_size * pp_size * dp_size


class ProcessGroups3D:
    """
    Manages process groups for 3D parallelism.

    Rank ordering convention (Megatron-LM):
      rank = dp_rank * (pp_size * tp_size)
            + pp_rank * tp_size
            + tp_rank
    """
    def __init__(self, cfg: ParallelConfig):
        self.cfg  = cfg
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        assert self.world_size == cfg.world_size, \
            f"world_size {self.world_size} != {cfg.world_size}"

        self._build_groups()

    def _rank(self, dp_rank, pp_rank, tp_rank):
        """Compute global rank from 3D coordinates."""
        return (dp_rank * (self.cfg.pp_size * self.cfg.tp_size)
                + pp_rank * self.cfg.tp_size
                + tp_rank)

    def _coords(self, rank):
        """Decompose global rank into 3D coordinates."""
        tp_rank  = rank % self.cfg.tp_size
        pp_rank  = (rank // self.cfg.tp_size) % self.cfg.pp_size
        dp_rank  = rank // (self.cfg.tp_size * self.cfg.pp_size)
        return dp_rank, pp_rank, tp_rank

    def _build_groups(self):
        """Build all process groups."""
        cfg = self.cfg

        # ── Tensor Parallel groups ────────────────────────────────────────
        # Ranks with same dp_rank and pp_rank, different tp_rank
        self._tp_group = None
        for dp in range(cfg.dp_size):
            for pp in range(cfg.pp_size):
                ranks = [self._rank(dp, pp, tp)
                         for tp in range(cfg.tp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self._tp_group  = group
                    self._tp_ranks  = ranks

        # ── Pipeline Parallel groups ──────────────────────────────────────
        # Ranks with same dp_rank and tp_rank, different pp_rank
        self._pp_group = None
        for dp in range(cfg.dp_size):
            for tp in range(cfg.tp_size):
                ranks = [self._rank(dp, pp, tp)
                         for pp in range(cfg.pp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self._pp_group  = group
                    self._pp_ranks  = ranks

        # ── Data Parallel groups ──────────────────────────────────────────
        # Ranks with same tp_rank and pp_rank, different dp_rank
        self._dp_group = None
        for pp in range(cfg.pp_size):
            for tp in range(cfg.tp_size):
                ranks = [self._rank(dp, pp, tp)
                         for dp in range(cfg.dp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self._dp_group  = group
                    self._dp_ranks  = ranks

    @property
    def tp_rank(self):
        return dist.get_rank(self._tp_group)

    @property
    def pp_rank(self):
        return dist.get_rank(self._pp_group)

    @property
    def dp_rank(self):
        return dist.get_rank(self._dp_group)

    @property
    def tp_group(self):
        return self._tp_group

    @property
    def pp_group(self):
        return self._pp_group

    @property
    def dp_group(self):
        return self._dp_group

    def is_pipeline_first_stage(self):
        return self.pp_rank == 0

    def is_pipeline_last_stage(self):
        return self.pp_rank == self.cfg.pp_size - 1

    def get_next_pp_rank(self):
        """Global rank of next pipeline stage (same TP and DP rank)."""
        if self.is_pipeline_last_stage():
            return None
        return self._pp_ranks[self.pp_rank + 1]

    def get_prev_pp_rank(self):
        """Global rank of previous pipeline stage."""
        if self.is_pipeline_first_stage():
            return None
        return self._pp_ranks[self.pp_rank - 1]

    def print_info(self):
        """Print process group info for this rank."""
        dp_rank, pp_rank, tp_rank = self._coords(self.rank)
        print(
            f"global_rank={self.rank:2d}  "
            f"dp_rank={dp_rank}  pp_rank={pp_rank}  tp_rank={tp_rank}  "
            f"tp_ranks={self._tp_ranks}  "
            f"pp_ranks={self._pp_ranks}  "
            f"dp_ranks={self._dp_ranks}",
            flush=True
        )


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ── Test with different 3D configurations ─────────────────────────────
    configs = [
        (2, 1, world_size//2),   # TP=2, PP=1, DP=world//2
        (1, 2, world_size//2),   # TP=1, PP=2, DP=world//2
        (2, 2, world_size//4) if world_size >= 4 else None,
    ]

    for cfg_params in configs:
        if cfg_params is None:
            continue
        tp, pp, dp = cfg_params
        if tp * pp * dp != world_size:
            continue

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"3D Process Groups: TP={tp}, PP={pp}, DP={dp}, "
                  f"world_size={world_size}")
            print(f"{'='*70}")

        dist.barrier()
        cfg    = ParallelConfig(tp_size=tp, pp_size=pp, dp_size=dp)
        groups = ProcessGroups3D(cfg)

        # Print in rank order
        for r in range(world_size):
            dist.barrier()
            if rank == r:
                groups.print_info()
            dist.barrier()

        # Test AllReduce within each group
        x_tp = torch.ones(1, device=f"cuda:{local_rank}") * rank
        dist.all_reduce(x_tp, op=dist.ReduceOp.SUM, group=groups.tp_group)

        x_dp = torch.ones(1, device=f"cuda:{local_rank}") * rank
        dist.all_reduce(x_dp, op=dist.ReduceOp.SUM, group=groups.dp_group)

        if rank == 0:
            print(f"\nTP AllReduce from rank 0: result={x_tp.item()} "
                  f"(sum of tp_ranks={groups._tp_ranks})")
            print(f"DP AllReduce from rank 0: result={x_dp.item()} "
                  f"(sum of dp_ranks={groups._dp_ranks})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 process_groups_3d.py
torchrun --nproc_per_node=8 process_groups_3d.py
```

---

## 14.9 Lab 4 — MoE (Mixture of Experts) From Scratch

```bash
cat > moe_layer.py <<'PY'
"""
Mixture of Experts (MoE) from scratch.

Replaces the standard FFN with:
  - N expert FFNs
  - A router network that assigns each token to top-k experts
  - Expert Parallelism: each GPU hosts one expert

This is the FFN used in Mixtral, Switch Transformer, etc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from typing import Tuple


class Router(nn.Module):
    """
    Token router for MoE.
    Computes routing weights for each token → each expert.

    Output: top-k expert indices and their weights.
    """
    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.gate      = nn.Linear(d_model, n_experts, bias=False)

        # Router Z-loss: prevents router collapse (all tokens → same expert)
        self.z_loss_coef = 1e-3

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            top_k_indices:  [B, T, top_k] — which experts each token goes to
            top_k_weights:  [B, T, top_k] — routing weights (sum to 1)
            router_loss:    scalar — auxiliary loss to prevent collapse
        """
        B, T, d = x.shape
        x_flat  = x.view(-1, d)         # [B*T, d]

        # Router logits
        logits  = self.gate(x_flat)     # [B*T, n_experts]

        # Z-loss: penalizes large logits (prevents collapse)
        # Z-loss = mean(log(sum(exp(logits)))^2)
        z_loss  = torch.logsumexp(logits, dim=-1).pow(2).mean()
        z_loss  = z_loss * self.z_loss_coef

        # Top-k routing
        probs            = F.softmax(logits, dim=-1)   # [B*T, n_experts]
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Normalize top-k weights to sum to 1
        top_k_weights    = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        top_k_weights    = top_k_weights.view(B, T, self.top_k)
        top_k_indices    = top_k_indices.view(B, T, self.top_k)

        return top_k_indices, top_k_weights, z_loss


class Expert(nn.Module):
    """Single FFN expert."""
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalMoELayer(nn.Module):
    """
    MoE layer where all experts are on the same device.
    (No expert parallelism — for understanding the routing logic)

    For distributed expert parallelism, see DistributedMoELayer below.
    """
    def __init__(self, d_model: int, n_experts: int,
                 top_k: int = 2, expansion: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.router    = Router(d_model, n_experts, top_k)
        self.experts   = nn.ModuleList([
            Expert(d_model, expansion) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            output: [B, T, d_model]
            router_loss: scalar
        """
        B, T, d = x.shape

        # Get routing decisions
        indices, weights, router_loss = self.router(x)
        # indices:  [B, T, top_k]
        # weights:  [B, T, top_k]

        output = torch.zeros_like(x)

        # Process each expert
        for expert_id in range(self.n_experts):
            # Find tokens assigned to this expert
            # expert_mask: [B, T, top_k] bool — True where indices == expert_id
            expert_mask = (indices == expert_id)   # [B, T, top_k]

            if not expert_mask.any():
                continue  # no tokens for this expert

            # For each top_k position, get tokens assigned here
            for k in range(self.top_k):
                # Positions in this batch where expert_id is the k-th choice
                mask_k = expert_mask[:, :, k]   # [B, T]

                if not mask_k.any():
                    continue

                # Extract those tokens
                tokens_k  = x[mask_k]           # [n_tokens_k, d]
                weight_k  = weights[:, :, k][mask_k]  # [n_tokens_k]

                # Run expert
                expert_out = self.experts[expert_id](tokens_k)  # [n_tokens_k, d]

                # Weight and accumulate
                output[mask_k] += weight_k.unsqueeze(-1) * expert_out

        return output, router_loss


class DistributedMoELayer(nn.Module):
    """
    MoE with Expert Parallelism.
    Each GPU (in EP group) hosts one expert.
    Tokens are routed across GPUs via All-to-All.

    Communication pattern:
      1. Router assigns each token to an expert GPU
      2. All-to-All: send tokens to their expert GPUs
      3. Each GPU runs its expert on received tokens
      4. All-to-All: return processed tokens to original GPUs
      5. Combine weighted expert outputs
    """
    def __init__(self, d_model: int, n_experts: int,
                 top_k: int = 1, expansion: int = 4,
                 ep_group=None):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.ep_group  = ep_group
        self.ep_size   = dist.get_world_size(ep_group) if ep_group else 1
        self.ep_rank   = dist.get_rank(ep_group) if ep_group else 0

        assert n_experts == self.ep_size, \
            f"n_experts {n_experts} must equal ep_size {self.ep_size}"

        self.router = Router(d_model, n_experts, top_k)
        # Each GPU hosts exactly ONE expert
        self.expert = Expert(d_model, expansion)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, d = x.shape
        device  = x.device
        n_tok   = B * T
        x_flat  = x.view(n_tok, d)

        # ── Route tokens ──────────────────────────────────────────────────
        indices, weights, router_loss = self.router(x)
        # Use top-1 for simplicity (top-k with k=1)
        expert_ids = indices[:, :, 0].view(n_tok)   # [B*T]
        tok_weights= weights[:, :, 0].view(n_tok)   # [B*T]

        # ── All-to-All: send tokens to their expert GPUs ──────────────────
        # Count tokens per expert for this rank
        tokens_per_expert = torch.zeros(
            self.n_experts, dtype=torch.long, device=device
        )
        for expert_id in range(self.n_experts):
            tokens_per_expert[expert_id] = (expert_ids == expert_id).sum()

        # Reorder tokens by expert
        sorted_indices  = torch.argsort(expert_ids)
        sorted_tokens   = x_flat[sorted_indices]   # [n_tok, d] sorted by expert

        # Split and all_to_all
        splits          = tokens_per_expert.tolist()
        splits          = [int(s) for s in splits]
        split_tensors   = sorted_tokens.split(splits, dim=0)

        # Receive buffer: how many tokens from each GPU?
        recv_counts     = torch.zeros(self.ep_size, dtype=torch.long,
                                       device=device)
        recv_counts[self.ep_rank] = tokens_per_expert[self.ep_rank]
        if self.ep_group is not None:
            dist.all_reduce(recv_counts, group=self.ep_group)
            # Each rank tells others how many tokens it will receive

        # ── Each GPU runs its expert ───────────────────────────────────────
        my_tokens_count = int(tokens_per_expert[self.ep_rank].item())
        my_tokens       = split_tensors[self.ep_rank]   # [my_count, d]

        if my_tokens_count > 0:
            expert_out = self.expert(my_tokens)   # [my_count, d]
        else:
            expert_out = torch.zeros(0, d, device=device)

        # ── All-to-All back: return processed tokens ───────────────────────
        # (simplified: just combine local results for now)
        output = torch.zeros_like(x_flat)

        # Fill in this expert's outputs
        my_expert_mask = (expert_ids == self.ep_rank)
        if my_expert_mask.any():
            # Map back to original positions
            orig_positions = sorted_indices[sum(splits[:self.ep_rank]):
                                             sum(splits[:self.ep_rank+1])]
            if len(orig_positions) > 0 and len(expert_out) > 0:
                output[orig_positions] += (
                    tok_weights[orig_positions].unsqueeze(-1) * expert_out
                )

        # AllReduce to combine all experts' outputs
        if self.ep_group is not None:
            dist.all_reduce(output, op=dist.ReduceOp.SUM,
                             group=self.ep_group)

        return output.view(B, T, d), router_loss


# ─── Test ──────────────────────────────────────────────────────────────────────

def test_moe_local():
    """Test local MoE (all experts on one GPU)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_model   = 128
    n_experts = 4
    top_k     = 2
    B, T      = 4, 32

    moe = LocalMoELayer(
        d_model=d_model, n_experts=n_experts, top_k=top_k
    ).to(device)

    x = torch.randn(B, T, d_model, device=device)
    out, router_loss = moe(x)

    print(f"\nLocal MoE Test:")
    print(f"  Input shape:      {x.shape}")
    print(f"  Output shape:     {out.shape}")
    print(f"  Router loss:      {router_loss.item():.6f}")
    assert out.shape == x.shape
    print(f"  Local MoE: PASSED")

    # Check load balance (each expert should get ~equal tokens)
    with torch.no_grad():
        indices, _, _ = moe.router(x)
        for expert_id in range(n_experts):
            count = (indices == expert_id).sum().item()
            pct   = 100 * count / (B * T * top_k)
            print(f"  Expert {expert_id}: {count} tokens ({pct:.1f}%)")


def test_moe_distributed():
    """Test distributed MoE (Expert Parallelism)."""
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # All GPUs form one EP group
    ep_group   = dist.new_group(list(range(world_size)))
    n_experts  = world_size

    moe = DistributedMoELayer(
        d_model=128, n_experts=n_experts,
        top_k=1, ep_group=ep_group,
    ).to(device)

    B, T = 4, 32
    x    = torch.randn(B, T, 128, device=device)
    # Same input on all ranks
    dist.broadcast(x, src=0)

    out, router_loss = moe(x)

    if rank == 0:
        print(f"\nDistributed MoE (EP={world_size}):")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Router loss:  {router_loss.item():.6f}")
        print(f"  Distributed MoE: PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    if "LOCAL_RANK" in os.environ:
        test_moe_distributed()
    else:
        test_moe_local()
PY

# Local test (no distributed)
python moe_layer.py

# Distributed test
torchrun --nproc_per_node=4 moe_layer.py
```

---

## 14.10 Lab 5 — Real Megatron-LM Training

Now use the actual Megatron-LM library:

```bash
cat > run_megatron_pretrain.sh <<'BASH'
#!/bin/bash
# Run Megatron-LM pretraining on Shakespeare dataset

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/megatron_module

# ── Step 1: Prepare data (Megatron needs tokenized binary format) ──────────
python prepare_megatron_data.py

# ── Step 2: Run pretraining with TP=2, PP=2 ───────────────────────────────
MASTER_ADDR=localhost
MASTER_PORT=29600
GPUS_PER_NODE=8
NNODES=1

# Model config (GPT-small equivalent)
MODEL_ARGS="
    --num-layers 12
    --hidden-size 768
    --num-attention-heads 12
    --seq-length 1024
    --max-position-embeddings 1024
"

# Parallelism config
PARALLEL_ARGS="
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
"

# Training config
TRAIN_ARGS="
    --micro-batch-size 4
    --global-batch-size 64
    --train-iters 5000
    --lr 6e-4
    --min-lr 6e-5
    --lr-decay-style cosine
    --lr-warmup-iters 200
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
"

# Data config
DATA_ARGS="
    --data-path data/shakespeare_text_document
    --vocab-file gpt2-vocab.json
    --merge-file gpt2-merges.txt
    --split 949,50,1
"

# Output config
OUTPUT_ARGS="
    --log-interval 50
    --save-interval 1000
    --eval-interval 500
    --eval-iters 50
    --save checkpoints_megatron
    --load checkpoints_megatron
"

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    Megatron-LM/pretrain_gpt.py \
    $MODEL_ARGS \
    $PARALLEL_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS
BASH

chmod +x run_megatron_pretrain.sh
```

First, prepare data in Megatron format:

```bash
cat > prepare_megatron_data.py <<'PY'
"""
Prepare Shakespeare dataset in Megatron-LM format.
Megatron requires:
  1. JSON lines format (.jsonl)
  2. Tokenized binary format (.bin + .idx)
  3. GPT-2 vocabulary files
"""
import os, json, urllib.request

# Download GPT-2 vocab files
def download_gpt2_vocab():
    files = {
        "gpt2-vocab.json":  "https://huggingface.co/gpt2/resolve/main/vocab.json",
        "gpt2-merges.txt":  "https://huggingface.co/gpt2/resolve/main/merges.txt",
    }
    for fname, url in files.items():
        if not os.path.exists(fname):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, fname)
            print(f"  Done: {fname}")
        else:
            print(f"  Already exists: {fname}")


def shakespeare_to_jsonl():
    """Convert Shakespeare text to JSONL format."""
    os.makedirs("data", exist_ok=True)
    text   = open("shakespeare.txt").read()
    output = "data/shakespeare.jsonl"

    # Split into chunks (~512 chars each)
    chunk_size = 512
    chunks     = [text[i:i+chunk_size]
                  for i in range(0, len(text), chunk_size)]

    with open(output, "w") as f:
        for chunk in chunks:
            if chunk.strip():
                f.write(json.dumps({"text": chunk}) + "\n")

    print(f"Created {output}: {len(chunks)} samples")
    return output


def tokenize_for_megatron(jsonl_path, vocab_file, merge_file):
    """Run Megatron's preprocessing tool."""
    cmd = (
        f"python Megatron-LM/tools/preprocess_data.py "
        f"  --input {jsonl_path} "
        f"  --output-prefix data/shakespeare_text "
        f"  --vocab-file {vocab_file} "
        f"  --merge-file {merge_file} "
        f"  --dataset-impl mmap "
        f"  --tokenizer-type GPT2BPETokenizer "
        f"  --append-eod "
        f"  --workers 4"
    )
    print(f"\nRunning:\n{cmd}\n")
    os.system(cmd)


if __name__ == "__main__":
    download_gpt2_vocab()
    jsonl = shakespeare_to_jsonl()
    tokenize_for_megatron(jsonl, "gpt2-vocab.json", "gpt2-merges.txt")
    print("\nData prepared successfully!")
    print("Files in data/:")
    for f in os.listdir("data"):
        size = os.path.getsize(f"data/{f}") / 1024
        print(f"  {f}: {size:.1f} KB")
PY

python prepare_megatron_data.py
```

---

## 14.11 Lab 6 — TP + PP Combined Training (Custom)

Build a minimal TP+PP combined system without full Megatron:

```bash
cat > tp_pp_combined.py <<'PY'
"""
Combine Tensor Parallelism + Pipeline Parallelism.

Configuration: 4 GPUs
  TP=2: split each layer across 2 GPUs
  PP=2: split layers across 2 stages

  GPU0 (TP=0, PP=0): first stage, TP shard 0
  GPU1 (TP=1, PP=0): first stage, TP shard 1
  GPU2 (TP=0, PP=1): last stage,  TP shard 0
  GPU3 (TP=1, PP=1): last stage,  TP shard 1

Forward pass:
  GPU0+GPU1: forward first stage (TP, 2 AllReduces per block)
  GPU0→GPU2, GPU1→GPU3: send activations (PP send)
  GPU2+GPU3: forward last stage (TP, 2 AllReduces per block)

Backward pass: reverse
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import math

from process_groups_3d import ParallelConfig, ProcessGroups3D
from tp_linear import (
    init_tp_group, get_tp_group, get_tp_rank, get_tp_world_size,
    ColumnParallelLinear, RowParallelLinear,
    CopyToModelParallelRegion, ReduceFromModelParallelRegion,
)


class TPPPBlock(nn.Module):
    """Transformer block with Tensor Parallelism."""
    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        # Attention: Column + Row Parallel
        self.c_attn = ColumnParallelLinear(d_model, 3*d_model,
                                            bias=False,
                                            gather_output=False)
        self.c_proj = RowParallelLinear(d_model, d_model,
                                         bias=False,
                                         input_is_parallel=True)
        self.ln2 = nn.LayerNorm(d_model)
        # FFN: Column + Row Parallel
        self.w1 = ColumnParallelLinear(d_model, 4*d_model,
                                        bias=True,
                                        gather_output=False)
        self.w2 = RowParallelLinear(4*d_model, d_model,
                                     bias=True,
                                     input_is_parallel=True)
        self.drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        tp_size  = get_tp_world_size()
        n_heads_local = self.n_heads // tp_size
        d_local       = n_heads_local * self.d_head

        # Attention
        qkv = self.c_attn(self.ln1(x))       # [B,T,3*d_model/TP]
        Q, K, V = qkv.split(d_local, dim=2)

        def sh(t):
            return t.view(B,T,n_heads_local,self.d_head).transpose(1,2)
        Q, K, V = sh(Q), sh(K), sh(V)
        sc     = math.sqrt(self.d_head)
        attn   = (Q @ K.transpose(-2,-1) / sc).masked_fill(
            self.mask[:T,:T].unsqueeze(0).unsqueeze(0)==0, float('-inf'))
        attn   = self.drop(F.softmax(attn, dim=-1))
        out    = (attn @ V).transpose(1,2).contiguous().view(B,T,d_local)
        x      = x + self.c_proj(out)        # AllReduce inside RowParallel

        # FFN
        h = F.gelu(self.w1(self.ln2(x)))     # [B,T,4*d_model/TP]
        x = x + self.w2(h)                   # AllReduce inside RowParallel
        return x


class TPPPStage(nn.Module):
    """One pipeline stage with TP-parallel blocks."""
    def __init__(self, cfg, pp_rank: int, pp_size: int, n_blocks_per_stage: int):
        super().__init__()
        self.is_first = (pp_rank == 0)
        self.is_last  = (pp_rank == pp_size - 1)

        if self.is_first:
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.drop      = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TPPPBlock(cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.dropout)
            for _ in range(n_blocks_per_stage)
        ])

        if self.is_last:
            self.ln_final = nn.LayerNorm(cfg.d_model)
            self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x, targets=None):
        if self.is_first:
            B, T = x.shape
            pos  = torch.arange(T, device=x.device)
            x    = self.drop(self.token_emb(x) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        if self.is_last:
            logits = self.lm_head(self.ln_final(x))
            loss   = None
            if targets is not None:
                from model import GPTConfig
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            return logits, loss

        return x, None


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)

    assert world_size == 4, "This demo requires exactly 4 GPUs (TP=2, PP=2)"

    # ── Setup 3D process groups ───────────────────────────────────────────
    TP, PP, DP = 2, 2, 1
    cfg_3d     = ParallelConfig(tp_size=TP, pp_size=PP, dp_size=DP)
    groups     = ProcessGroups3D(cfg_3d)

    tp_rank    = groups.tp_rank
    pp_rank    = groups.pp_rank

    # Initialize TP group in tp_linear module
    global _TP_GROUP
    import tp_linear as tpl
    tpl._TP_GROUP = groups.tp_group

    if rank == 0:
        print(f"\n{'='*55}")
        print(f"TP={TP} × PP={PP} combined training test")
        print(f"{'='*55}")

    # Print group info
    dist.barrier()
    for r in range(world_size):
        dist.barrier()
        if rank == r:
            print(f"  rank={rank}: tp_rank={tp_rank} pp_rank={pp_rank}",
                  flush=True)
        dist.barrier()

    # ── Build model ───────────────────────────────────────────────────────
    from model import GPTConfig
    cfg = GPTConfig(
        vocab_size=65, max_seq_len=64,
        d_model=128, n_heads=4,
        n_layers=4, dropout=0.0,
    )
    n_blocks_per_stage = cfg.n_layers // PP

    model = TPPPStage(cfg, pp_rank, PP, n_blocks_per_stage).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B, T  = 2, 32

    # ── Forward + backward (simplified, single micro-batch) ──────────────
    for step in range(3):
        opt.zero_grad(set_to_none=True)

        if model.is_first:
            x  = torch.randint(0, 65, (B, T), device=device)
            y  = torch.randint(0, 65, (B, T), device=device)
            out, _ = model(x)

            # Send to next PP stage (both TP shards send)
            next_pp_rank = groups.get_next_pp_rank()
            shape_t      = torch.tensor(out.shape, device=device)
            dist.send(shape_t, dst=next_pp_rank)
            dist.send(out.contiguous(), dst=next_pp_rank)

        elif model.is_last:
            y    = torch.randint(0, 65, (B, T), device=device)
            # Receive from previous PP stage
            prev_pp_rank = groups.get_prev_pp_rank()
            shape_buf    = torch.zeros(3, dtype=torch.long, device=device)
            dist.recv(shape_buf, src=prev_pp_rank)
            hidden = torch.zeros(*shape_buf.tolist(),
                                  device=device).requires_grad_(True)
            dist.recv(hidden, src=prev_pp_rank)

            logits, loss = model(hidden, y)

            if tp_rank == 0:
                print(f"  step={step} pp={pp_rank} tp={tp_rank}: "
                      f"loss={loss.item():.4f}", flush=True)

            # Backward
            loss.backward()

            # Send gradient back
            if hidden.grad is not None:
                dist.send(hidden.grad.contiguous(), dst=prev_pp_rank)

        if model.is_first:
            # Receive gradient
            grad_buf = torch.zeros_like(out)
            next_pp_rank = groups.get_next_pp_rank()
            dist.recv(grad_buf, src=next_pp_rank)
            out.backward(gradient=grad_buf)

        # AllReduce gradients across DP group
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG,
                                group=groups.dp_group)

        opt.step()

    dist.barrier()
    if rank == 0:
        print(f"\nTP+PP combined training: PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 tp_pp_combined.py
```

---

## 14.12 Debug Drills

### Drill 1 — TP requires matched dimensions

```bash
cat > drill_tp_dimension.py <<'PY'
"""
TP requires out_features divisible by tp_size.
This drill shows the error and correct fix.
"""
import os
import torch
import torch.distributed as dist

from tp_linear import (
    init_tp_group, ColumnParallelLinear, get_tp_world_size
)


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    tp_size = dist.get_world_size()

    init_tp_group(tp_size)

    if rank == 0:
        print(f"\nTP dimension check (tp_size={tp_size})")

    # WRONG: 100 not divisible by tp_size
    try:
        layer = ColumnParallelLinear(64, 100, bias=False).to(device)
        print(f"rank={rank}: WRONG config succeeded (unexpected)")
    except AssertionError as e:
        if rank == 0:
            print(f"WRONG (out=100, tp={tp_size}): {e}")

    # CORRECT: 128 divisible by tp_size
    try:
        layer = ColumnParallelLinear(64, 128, bias=False).to(device)
        x     = torch.randn(2, 8, 64, device=device)
        dist.broadcast(x, src=0)
        out   = layer(x)
        if rank == 0:
            print(f"CORRECT (out=128, tp={tp_size}): "
                  f"output shape {out.shape}")
    except Exception as e:
        if rank == 0:
            print(f"CORRECT config failed: {e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=4 drill_tp_dimension.py
```

---

### Drill 2 — Pipeline deadlock (asymmetric send/recv)

```bash
cat > drill_pp_deadlock.py <<'PY'
"""
Common PP bug: asymmetric send/recv causes deadlock.

If rank 0 does: send then recv
And rank 1 does: send then recv
→ DEADLOCK (both blocking on send, nobody receiving)

Fix: coordinate the order so sends are paired with recvs.
"""
import os
import torch
import torch.distributed as dist


def deadlock_pattern():
    """This WILL deadlock. We will kill it after showing the issue."""
    # Don't actually run this — just show the pattern
    rank = dist.get_rank()
    x    = torch.ones(4, device="cuda")

    if rank == 0:
        dist.send(x, dst=1)     # blocks until rank 1 receives
        dist.recv(x, src=1)     # never reached because rank 1 is also trying to send
    elif rank == 1:
        dist.send(x, dst=0)     # blocks until rank 0 receives (which is blocked above)
        dist.recv(x, src=0)     # deadlock!


def correct_pattern():
    """Coordinate send/recv to avoid deadlock."""
    rank = dist.get_rank()
    x    = torch.ones(4, device="cuda") * rank

    if rank == 0:
        dist.send(x, dst=1)       # rank 0 sends first
        received = torch.zeros(4, device="cuda")
        dist.recv(received, src=1)  # then receives
    elif rank == 1:
        received = torch.zeros(4, device="cuda")
        dist.recv(received, src=0)  # rank 1 receives first
        dist.send(x, dst=0)          # then sends

    print(f"rank={rank}: sent {rank} received {received.tolist()}", flush=True)


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print("\nPipeline communication patterns:")
        print("DEADLOCK pattern (shown as code only):")
        print("  rank 0: send → recv  (rank 1 also tries send → recv)")
        print("  → deadlock: both blocked on send")
        print("\nCORRECT pattern:")

    dist.barrier()
    correct_pattern()

    if rank == 0:
        print("Correct PP communication: PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --nproc_per_node=2 drill_pp_deadlock.py
```

---

### Drill 3 — Router collapse (all tokens → same expert)

```bash
cat > drill_router_collapse.py <<'PY'
"""
MoE router collapse: all tokens routed to same expert.
Causes: imbalanced training, most experts never trained.

Solutions:
  1. Auxiliary load balancing loss
  2. Z-loss (penalizes large router logits)
  3. Expert capacity limits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_layer import LocalMoELayer


def measure_load_balance(moe, x, label):
    """Measure how evenly tokens are distributed across experts."""
    with torch.no_grad():
        indices, weights, _ = moe.router(x)
    B, T, top_k = indices.shape
    total_tokens = B * T * top_k

    print(f"\n  {label}:")
    for expert_id in range(moe.n_experts):
        count  = (indices == expert_id).sum().item()
        pct    = 100 * count / total_tokens
        bar    = "█" * int(pct / 2)
        print(f"    Expert {expert_id}: {count:4d} tokens "
              f"({pct:5.1f}%) {bar}")


device = "cuda" if torch.cuda.is_available() else "cpu"
B, T   = 8, 64
d      = 128

# ── Good initialization (normal distribution) ──────────────────────────────
moe_good = LocalMoELayer(d_model=d, n_experts=4, top_k=2).to(device)
x        = torch.randn(B, T, d, device=device)
measure_load_balance(moe_good, x, "Good init (normal distribution)")

# ── Bad initialization (large values → argmax collapse) ────────────────────
moe_bad  = LocalMoELayer(d_model=d, n_experts=4, top_k=2).to(device)
with torch.no_grad():
    # Bias router strongly toward expert 0
    moe_bad.router.gate.weight.data[0] *= 10.0
measure_load_balance(moe_bad, x, "Bad init (expert 0 bias → collapse)")

# ── After training with z-loss ─────────────────────────────────────────────
print("\n  Training moe_bad with z-loss for 100 steps...")
opt = torch.optim.AdamW(moe_bad.parameters(), lr=1e-3)
for step in range(100):
    opt.zero_grad()
    out, router_loss = moe_bad(x)
    # Task loss (dummy)
    task_loss = out.mean()
    (task_loss + router_loss).backward()
    opt.step()

measure_load_balance(moe_bad, x, "After z-loss training (100 steps)")
print("\nZ-loss helps prevent router collapse by penalizing large logits.")
PY

python drill_router_collapse.py
```

---

## 14.13 SLURM Job for Megatron-style TP+PP Training

```bash
cat > jobs/job_megatron_tp_pp.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=megatron_tp_pp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/megatron_%j.out
#SBATCH --error=logs/megatron_%j.err

source /root/ddl_course/venv/bin/activate
cd /root/ddl_course/megatron_module

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_ADDR MASTER_PORT
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

echo "MASTER=$MASTER_ADDR:$MASTER_PORT"

# 16 GPUs: TP=4, PP=2, DP=2
srun bash -c "
  torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    Megatron-LM/pretrain_gpt.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --micro-batch-size 2 \
    --global-batch-size 64 \
    --train-iters 5000 \
    --lr 6e-4 \
    --min-lr 6e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 200 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --data-path data/shakespeare_text_document \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --split 949,50,1 \
    --log-interval 50 \
    --save-interval 1000 \
    --eval-interval 500 \
    --eval-iters 50 \
    --save checkpoints_megatron \
    --load checkpoints_megatron
"
SLURM

mkdir -p jobs logs
sbatch jobs/job_megatron_tp_pp.sh
```

---

## 14.14 Parallelism Strategy Guide

```bash
cat > parallelism_guide.py <<'PY'
"""
When to use which parallelism strategy.
"""

guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║               PARALLELISM STRATEGY SELECTION GUIDE                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DDP (Data Parallel)                                                         ║
║    When: model fits on 1 GPU                                                 ║
║    Comm: 1 AllReduce per backward (gradients)                               ║
║    Best: batch throughput, scaling efficiency                                ║
║                                                                              ║
║  FSDP / ZeRO (Data Parallel + Parameter Sharding)                           ║
║    When: model almost fits (tight on memory)                                 ║
║    Comm: AllGather + ReduceScatter (same bytes as AllReduce)                ║
║    Best: memory efficiency without model architecture changes               ║
║                                                                              ║
║  Tensor Parallelism                                                          ║
║    When: individual layers are too large                                    ║
║    Comm: 2 AllReduces per transformer block (within layer)                 ║
║    Best: very wide models (large d_model)                                   ║
║    Rule: use within a node (low latency NVLink required)                    ║
║                                                                              ║
║  Pipeline Parallelism                                                        ║
║    When: model is very deep (many layers)                                   ║
║    Comm: activation send/recv between stages (P2P)                         ║
║    Best: models where layers >> GPU count                                   ║
║    Rule: bubble fraction = (PP-1)/(PP-1+micro_batches)                     ║
║         → use many micro-batches to reduce bubble                          ║
║                                                                              ║
║  Sequence Parallelism                                                        ║
║    When: very long sequences (activation memory is bottleneck)              ║
║    Comm: AllGather + ReduceScatter per block                                ║
║    Best: complement to TP, reduces activation memory by TP factor          ║
║                                                                              ║
║  Expert Parallelism (MoE)                                                   ║
║    When: want many parameters without proportional compute cost             ║
║    Comm: 2 All-to-All per MoE layer                                        ║
║    Best: large models with sparse computation (Mixtral, Switch)            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PRACTICAL COMBINATIONS (Megatron-LM style)                                 ║
║                                                                              ║
║  GPT-1 (117M):  DP only                                                     ║
║  GPT-2 (1.5B):  DP only (fits on A100 40GB)                                ║
║  GPT-3 (175B):  TP=8, PP=16, DP=many                                       ║
║  LLaMA-7B:      TP=2-4, DP=many (or ZeRO-3)                               ║
║  LLaMA-70B:     TP=8, PP=4, DP=many                                        ║
║  Mixtral-8x7B:  TP=4, EP=8, DP=many                                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  RULES OF THUMB                                                              ║
║                                                                              ║
║  1. TP communication is expensive → keep TP within NVLink nodes            ║
║  2. PP is good across nodes (low bandwidth needed: just activations)       ║
║  3. DP is always the outermost (most replicas, most bandwidth needed)      ║
║  4. Micro-batches for PP: use ≥ 4 × PP size to keep bubble < 20%          ║
║  5. TP size: use largest that keeps TP comm < 10% of step time             ║
║  6. Total GPUs = TP × PP × DP                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
print(guide)
PY

python parallelism_guide.py
```

---

## 14.15 Module 14 Pass Criteria

You're done when you can:

1) **Explain all parallelism types**:
   - Why data parallelism (DDP/FSDP) alone fails for 175B models
   - Column parallel vs row parallel linear (with AllReduce placement)
   - 1F1B pipeline schedule and bubble fraction formula
   - Sequence parallelism: where AllGather/ReduceScatter go
   - MoE routing and All-to-All communication

2) **Run successfully**:
   - `torchrun --nproc_per_node=4 tp_linear.py` → TP linear PASSED
   - `torchrun --nproc_per_node=2 pipeline_parallel.py` → PP PASSED
   - `torchrun --nproc_per_node=4 process_groups_3d.py` → groups correct
   - `torchrun --nproc_per_node=4 moe_layer.py` → MoE PASSED
   - `torchrun --nproc_per_node=4 tp_pp_combined.py` → TP+PP PASSED

3) **Answer design questions**:
   - 8 GPUs available. Model has 7B params. Which strategy?
   - TP=4, PP=2, DP=? for 16 total GPUs?
   - Bubble fraction with PP=4 and 16 micro-batches?
   - Why is TP best kept within a node?

4) **Understand 3D process groups**:
   - Given world_size=8, TP=2, PP=2, DP=2, compute rank assignment
   - Which ranks share TP group? PP group? DP group?
   - What collective runs on each group?

---
