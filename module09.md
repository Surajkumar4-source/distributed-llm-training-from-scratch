# Module 9 — GPT From Scratch

## Summary
You will build a **complete GPT** in PyTorch from scratch, component by component. Every line is explained. No black boxes.

- **Theory**: tokenizer → embedding → positional encoding → masked self-attention → multi-head attention → feed-forward → LayerNorm → residual → decoder stack → language model head → loss
- **From scratch**: implement each component in isolation, test it, then assemble
- **Training**: train on real text (Shakespeare) on single GPU
- **Visualization**: attention patterns, loss curves, generation
- **This is the model** that Modules 10–16 will distribute across GPUs

---

## 9.1 Theory — What GPT Actually Is

### The big picture

```
Input text:  "The cat sat"

Step 1 — Tokenize
  "The cat sat" → [464, 3797, 3332]

Step 2 — Embed tokens + positions
  [464, 3797, 3332] → tensor of shape [3, d_model]

Step 3 — N × Transformer Decoder Blocks
  Each block:
    LayerNorm → Masked Self-Attention → Residual
    LayerNorm → Feed-Forward          → Residual

Step 4 — Final LayerNorm

Step 5 — Language Model Head (linear projection)
  [3, d_model] → [3, vocab_size]

Step 6 — Loss (cross-entropy on next-token prediction)
  Predict: "cat" from "The"
           "sat" from "The cat"
           "on"  from "The cat sat"
```

GPT is an **autoregressive language model**:
- it predicts the next token given all previous tokens
- trained by teacher-forcing: ground truth tokens are fed as input
- at inference: sample one token, append, repeat

---

## 9.2 Theory — Tokenizer

A tokenizer converts raw text into integer IDs (tokens).

### Types of tokenizers

| Type | Example | Vocabulary size |
|------|---------|-----------------|
| Character-level | `'h','e','l','l','o'` | ~100 |
| Word-level | `'hello','world'` | 50K–200K |
| BPE (Byte Pair Encoding) | `'hel','lo'` | 32K–100K |
| Byte-level BPE (GPT-2) | `'Ġhello'` | 50,257 |

GPT-2/GPT-3 use **byte-level BPE** (tiktoken). For this module we'll build a **character-level tokenizer** first (simplest to understand), then optionally upgrade to tiktoken.

### Character-level tokenizer (from scratch)

```
vocab = {'\n', ' ', '!', ..., 'a', 'b', ..., 'z', ...}
encode("hi") → [34, 51]
decode([34, 51]) → "hi"
```

---

## 9.3 Theory — Token Embedding + Positional Encoding

### Token Embedding
Maps each token ID → a dense vector of size `d_model`.

```python
nn.Embedding(vocab_size, d_model)
# Input:  [B, T]         (batch, sequence length)
# Output: [B, T, d_model]
```

Embedding is just a lookup table: row `i` = embedding vector for token `i`.

### Positional Encoding
Transformers have no recurrence → they don't know token order.
We must add position information.

Two approaches:

**Learned positional embeddings (GPT-2 style)**:
```python
nn.Embedding(max_seq_len, d_model)
# Learn a different vector for each position 0..T-1
```

**Fixed sinusoidal (original Transformer paper)**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

GPT uses **learned** positional embeddings. We'll do the same.

Final input embedding:
```
x = token_embedding(tokens) + position_embedding(positions)
x shape: [B, T, d_model]
```

---

## 9.4 Theory — Self-Attention (The Core)

### Intuition
Each token looks at all other tokens and decides how much to "attend to" each one.

```
"The cat sat on the mat"

When processing "sat":
  attend a lot to "cat" (subject)
  attend a lot to "on" (coming next)
  attend less to "The"
```

### Query, Key, Value
For each token, compute 3 vectors:

```
Q = x @ W_Q    (query: "what am I looking for?")
K = x @ W_K    (key:   "what do I contain?")
V = x @ W_V    (value: "what do I give if attended to?")
```

### Attention scores
```
scores = Q @ K^T / sqrt(d_k)
```

`sqrt(d_k)` stabilizes gradients (prevents softmax saturation for large d_k).

### Causal mask (the "masked" in GPT)
GPT is autoregressive: token at position `t` must not see tokens at positions `t+1, t+2, ...`

```
Mask (T=4):
position:  0  1  2  3
    0:   [ ✓  ✗  ✗  ✗ ]   token 0 sees only itself
    1:   [ ✓  ✓  ✗  ✗ ]   token 1 sees 0,1
    2:   [ ✓  ✓  ✓  ✗ ]   token 2 sees 0,1,2
    3:   [ ✓  ✓  ✓  ✓ ]   token 3 sees all
```

Implemented by setting future positions to `-inf` before softmax → softmax gives them 0 weight.

### Full attention formula
```
Attention(Q,K,V) = softmax( (Q @ K^T / sqrt(d_k)) + mask ) @ V
```

Output shape: `[B, T, d_k]`

### Multi-Head Attention
Run `n_heads` attention operations in parallel, each with different W_Q, W_K, W_V:

```
head_i = Attention(Q_i, K_i, V_i)
MultiHead = concat(head_0, head_1, ..., head_{h-1}) @ W_O
```

- Each head can learn to attend to different aspects of the sequence
- d_k per head = d_model / n_heads (keeps total compute constant)

---

## 9.5 Theory — Feed-Forward Block

After attention, each token position passes through an independent MLP:

```python
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

W1: [d_model, 4 * d_model]   # expand
W2: [4 * d_model, d_model]   # contract
```

The 4× expansion is standard GPT design.

**GELU** (Gaussian Error Linear Unit) is smoother than ReLU and preferred in transformers.

---

## 9.6 Theory — LayerNorm + Residual

### Layer Normalization
Normalizes across the `d_model` dimension for each token independently:

```
LN(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

Unlike BatchNorm:
- normalizes per token (not per batch)
- works well with variable sequence lengths
- no issues at batch_size=1

GPT uses **Pre-LN**: LayerNorm before attention and before FFN (more stable training):
```
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

### Residual connections
Every sub-layer has a residual (skip) connection:
```
output = x + sublayer(x)
```

Why:
- Allows gradients to flow through many layers without vanishing
- Makes it easier to learn identity mappings
- Enables training very deep networks

---

## 9.7 Theory — Full Decoder Block

```
┌─────────────────────────────┐
│         Input x             │  shape: [B, T, d_model]
│             │               │
│      LayerNorm (pre-LN)     │
│             │               │
│    Masked Multi-Head Attn   │
│             │               │
│         + Residual          │
│             │               │
│      LayerNorm (pre-LN)     │
│             │               │
│      Feed-Forward MLP       │
│             │               │
│         + Residual          │
│             │               │
│        Output x             │  shape: [B, T, d_model]
└─────────────────────────────┘
```

Stack N of these blocks = GPT decoder stack.

---

## 9.8 Theory — Language Model Head + Loss

### LM Head
Final linear projection:
```python
nn.Linear(d_model, vocab_size, bias=False)
# Input:  [B, T, d_model]
# Output: [B, T, vocab_size]   (logits over vocabulary)
```

Weight tying: LM head weight = token embedding weight (saves params, improves performance).

### Loss
Cross-entropy on next-token prediction:

```
Input tokens:   [t0, t1, t2, t3]
Target tokens:  [t1, t2, t3, t4]   (shifted by 1)

loss = CrossEntropy(logits[B, T, vocab_size], targets[B, T])
```

---

## 9.9 From Scratch — Build Every Component

### Setup

```bash
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course
pip install tiktoken datasets matplotlib
mkdir -p gpt_scratch
cd gpt_scratch
```

---

### Component 1 — Character-level Tokenizer

```bash
cat > tokenizer.py <<'PY'
"""
Character-level tokenizer.
Simple, transparent, easy to debug.
"""


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.ch2id = {c: i for i, c in enumerate(chars)}
        self.id2ch = {i: c for i, c in enumerate(chars)}
        print(f"CharTokenizer: vocab_size={self.vocab_size}")
        print(f"  chars: {repr(''.join(chars[:20]))}...")

    def encode(self, text: str) -> list[int]:
        return [self.ch2id[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id2ch[i] for i in ids)


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    text = "Hello, World! This is a test of the tokenizer."
    tok = CharTokenizer(text)

    encoded = tok.encode("Hello")
    print(f"\nencode('Hello') → {encoded}")
    decoded = tok.decode(encoded)
    print(f"decode({encoded}) → '{decoded}'")

    # Round-trip test
    for test_str in ["Hello", "World", "test"]:
        assert tok.decode(tok.encode(test_str)) == test_str
    print("\nRound-trip test: PASSED")
PY

python tokenizer.py
```

---

### Component 2 — Self-Attention (single head, from scratch)

```bash
cat > attention.py <<'PY'
"""
Single-head self-attention from scratch.
No nn.MultiheadAttention used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleHeadAttention(nn.Module):
    """
    Single causal self-attention head.
    Input:  [B, T, d_model]
    Output: [B, T, d_head]
    """
    def __init__(self, d_model: int, d_head: int, max_seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_head = d_head

        # Q, K, V projections (no bias, standard for transformers)
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix
        # Register as buffer so it moves with model.to(device)
        # and is saved in state_dict but not a parameter
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape

        Q = self.W_q(x)   # [B, T, d_head]
        K = self.W_k(x)   # [B, T, d_head]
        V = self.W_v(x)   # [B, T, d_head]

        # Attention scores
        scale = math.sqrt(self.d_head)
        scores = Q @ K.transpose(-2, -1) / scale  # [B, T, T]

        # Apply causal mask: fill future positions with -inf
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )

        # Softmax over last dim (over key positions)
        weights = F.softmax(scores, dim=-1)        # [B, T, T]
        weights = self.dropout(weights)

        # Weighted sum of values
        out = weights @ V                          # [B, T, d_head]
        return out


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, d_model, d_head = 2, 8, 64, 32
    x = torch.randn(B, T, d_model)

    attn = SingleHeadAttention(d_model=d_model, d_head=d_head, max_seq_len=T)
    out  = attn(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, T, d_head)
    print("SingleHeadAttention: PASSED")
PY

python attention.py
```

---

### Component 3 — Multi-Head Attention (from scratch)

```bash
cat > multihead_attention.py <<'PY'
"""
Multi-head causal self-attention from scratch.
Input:  [B, T, d_model]
Output: [B, T, d_model]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Efficient implementation:
    Instead of separate W_q, W_k, W_v per head,
    we use one big projection for all heads combined,
    then split into heads.
    This is more GPU-friendly (one big matmul vs many small ones).
    """
    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        # Single big Q, K, V projection for all heads
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

        # Track attention weights for visualization
        self.last_attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # C = d_model

        # ── Project Q, K, V in one shot ──────────────────────────────────
        qkv = self.c_attn(x)           # [B, T, 3*d_model]
        Q, K, V = qkv.split(self.d_model, dim=2)
        # Q, K, V each: [B, T, d_model]

        # ── Split into heads ──────────────────────────────────────────────
        # Reshape: [B, T, d_model] → [B, n_heads, T, d_head]
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head) \
                    .transpose(1, 2)   # [B, n_heads, T, d_head]

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # ── Scaled dot-product attention ──────────────────────────────────
        scale  = math.sqrt(self.d_head)
        scores = Q @ K.transpose(-2, -1) / scale  # [B, n_heads, T, T]

        # Apply causal mask
        scores = scores.masked_fill(
            self.mask[:T, :T].unsqueeze(0).unsqueeze(0) == 0,
            float('-inf')
        )

        weights = F.softmax(scores, dim=-1)       # [B, n_heads, T, T]
        self.last_attn_weights = weights.detach() # save for visualization
        weights = self.attn_drop(weights)

        # Weighted sum
        out = weights @ V                          # [B, n_heads, T, d_head]

        # ── Merge heads ───────────────────────────────────────────────────
        # [B, n_heads, T, d_head] → [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.proj_drop(self.c_proj(out))

        return out


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, d_model, n_heads = 2, 16, 128, 4
    x    = torch.randn(B, T, d_model)
    mha  = MultiHeadAttention(d_model=d_model, n_heads=n_heads,
                              max_seq_len=T)
    out  = mha(x)

    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, T, d_model)

    # Check causality: output at position t should not depend on t+1
    x2    = x.clone()
    x2[:, 5:, :] = 0.0       # zero out positions 5 onwards
    out2  = mha(x2)
    # outputs at positions 0-4 should be identical
    assert torch.allclose(out[:, :5, :], out2[:, :5, :], atol=1e-5), \
        "Causality violated!"

    print("Causal mask test: PASSED")
    print("MultiHeadAttention: PASSED")
PY

python multihead_attention.py
```

---

### Component 4 — Feed-Forward Block

```bash
cat > feedforward.py <<'PY'
"""
Position-wise Feed-Forward Network.
Applied independently at each token position.
"""
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    d_model → 4*d_model → d_model
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, d_model = 2, 16, 128
    x   = torch.randn(B, T, d_model)
    ffn = FeedForward(d_model=d_model)
    out = ffn(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, T, d_model)
    print("FeedForward: PASSED")

    # Count parameters
    n_params = sum(p.numel() for p in ffn.parameters())
    print(f"FFN params: {n_params:,}")
PY

python feedforward.py
```

---

### Component 5 — Transformer Decoder Block

```bash
cat > transformer_block.py <<'PY'
"""
Single Transformer Decoder Block.
Pre-LN architecture (more stable than Post-LN).

x = x + Attention(LN(x))
x = x + FFN(LN(x))
"""
import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: normalize before attention and FFN
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.ffn(self.ln2(x))    # residual + FFN
        return x


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, d_model, n_heads = 2, 16, 128, 4
    x     = torch.randn(B, T, d_model)
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, max_seq_len=T)
    out   = block(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, T, d_model)
    print("TransformerBlock: PASSED")

    n_params = sum(p.numel() for p in block.parameters())
    print(f"Block params: {n_params:,}")
PY

python transformer_block.py
```

---

### Component 6 — Full GPT Model

```bash
cat > gpt.py <<'PY'
"""
Complete GPT model from scratch.
Architecture matches GPT-2 small (with configurable size).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    # Model architecture
    vocab_size:   int   = 65       # character-level default
    max_seq_len:  int   = 256      # context window
    d_model:      int   = 384      # embedding dimension
    n_heads:      int   = 6        # attention heads
    n_layers:     int   = 6        # transformer blocks
    dropout:      float = 0.1

    # GPT-2 small equivalent:
    # vocab_size=50257, max_seq_len=1024
    # d_model=768, n_heads=12, n_layers=12

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ── Embeddings ────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop_emb  = nn.Dropout(config.dropout)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model     = config.d_model,
                n_heads     = config.n_heads,
                max_seq_len = config.max_seq_len,
                dropout     = config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # ── Final LayerNorm ───────────────────────────────────────────────
        self.ln_final = nn.LayerNorm(config.d_model)

        # ── Language model head ───────────────────────────────────────────
        # No bias; weight-tied with token_emb
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: lm_head uses same weights as token_emb
        # This reduces parameters and improves performance
        self.lm_head.weight = self.token_emb.weight

        # ── Initialize weights ────────────────────────────────────────────
        self.apply(self._init_weights)

        # Special scaled init for residual projections (GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / (2 * config.n_layers) ** 0.5)

        n_params = self.count_params()
        print(f"GPT initialized: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def count_params(self) -> int:
        # Don't double-count tied weights
        return sum(p.numel() for p in self.parameters()) \
               - self.lm_head.weight.numel()   # tied, so subtract once

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor = None):
        """
        Args:
            idx:     [B, T] token indices
            targets: [B, T] target token indices (optional, for training)

        Returns:
            logits: [B, T, vocab_size]
            loss:   scalar (if targets provided) else None
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        device = idx.device

        # ── Token + position embeddings ───────────────────────────────────
        tok_emb = self.token_emb(idx)                               # [B,T,d_model]
        pos     = torch.arange(T, device=device)                    # [T]
        pos_emb = self.pos_emb(pos)                                 # [T,d_model]
        x       = self.drop_emb(tok_emb + pos_emb)                 # [B,T,d_model]

        # ── Transformer blocks ────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)

        # ── Final norm + LM head ──────────────────────────────────────────
        x      = self.ln_final(x)                                   # [B,T,d_model]
        logits = self.lm_head(x)                                    # [B,T,vocab]

        # ── Loss ──────────────────────────────────────────────────────────
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Autoregressive generation.
        Args:
            idx:            [B, T] conditioning tokens
            max_new_tokens: how many tokens to generate
            temperature:    sampling temperature (lower = more deterministic)
            top_k:          if set, sample from top-k logits only

        Returns: [B, T + max_new_tokens] token indices
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if sequence is too long
            idx_crop = idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_crop)

            # Take logits at last position
            logits = logits[:, -1, :]               # [B, vocab_size]
            logits = logits / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs     = F.softmax(logits, dim=-1)   # [B, vocab_size]
            next_tok  = torch.multinomial(probs, num_samples=1) # [B, 1]
            idx       = torch.cat([idx, next_tok], dim=1)       # [B, T+1]

        return idx


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    cfg   = GPTConfig(
        vocab_size  = 65,
        max_seq_len = 64,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 4,
        dropout     = 0.0,
    )
    model = GPT(cfg)

    B, T  = 2, 32
    idx   = torch.randint(0, cfg.vocab_size, (B, T))
    tgt   = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, tgt)
    print(f"logits shape: {logits.shape}")
    print(f"loss:         {loss.item():.4f}")
    print(f"expected loss (random init): {torch.log(torch.tensor(cfg.vocab_size)):.4f}")
    assert logits.shape == (B, T, cfg.vocab_size)

    # Generation test
    prompt    = torch.zeros(1, 1, dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10)
    print(f"Generated shape: {generated.shape}")
    print("GPT: PASSED")
PY

python gpt.py
```

---

## 9.10 Download Dataset — Shakespeare

```bash
cat > download_data.py <<'PY'
"""
Download tiny Shakespeare dataset (~1MB).
Classic benchmark for character-level language models.
"""
import os
import urllib.request

URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PATH = "shakespeare.txt"

if not os.path.exists(PATH):
    print("Downloading Shakespeare...")
    urllib.request.urlretrieve(URL, PATH)
    print("Done.")

text = open(PATH, 'r').read()
print(f"Dataset size: {len(text):,} characters")
print(f"Sample:\n{text[:200]}")
PY

python download_data.py
```

---

## 9.11 Full Training Script

```bash
cat > train_gpt.py <<'PY'
"""
Train GPT on Shakespeare (single GPU).
Implements:
  - Dataset + DataLoader
  - Mixed precision (BF16 on A100)
  - Gradient clipping
  - LR warmup + cosine decay
  - Checkpointing
  - Periodic generation
  - TensorBoard logging
"""
import os, time, math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tokenizer import CharTokenizer
from gpt import GPT, GPTConfig


# ─── Hyperconfig ───────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Data
    data_path:     str   = "shakespeare.txt"
    train_split:   float = 0.9

    # Model
    max_seq_len:   int   = 256
    d_model:       int   = 384
    n_heads:       int   = 6
    n_layers:      int   = 6
    dropout:       float = 0.1

    # Training
    batch_size:    int   = 64
    max_iters:     int   = 5000
    eval_interval: int   = 500
    eval_iters:    int   = 50
    log_interval:  int   = 50
    gen_interval:  int   = 500

    # Optimizer
    lr:            float = 6e-4
    weight_decay:  float = 0.1
    beta1:         float = 0.9
    beta2:         float = 0.95
    grad_clip:     float = 1.0

    # LR schedule
    warmup_iters:  int   = 200
    lr_decay_iters:int   = 5000
    min_lr:        float = 6e-5

    # System
    device:        str   = "cuda"
    dtype:         str   = "bfloat16"   # A100 supports bf16 natively
    compile:       bool  = False        # torch.compile (optional)
    checkpoint_dir:str   = "checkpoints"
    log_dir:       str   = "runs/gpt_shakespeare"

    # Generation
    gen_max_tokens:int   = 200
    gen_temperature:float = 0.8
    gen_top_k:     int   = 40


# ─── Dataset ───────────────────────────────────────────────────────────────────
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


# ─── Learning Rate Schedule ────────────────────────────────────────────────────
def get_lr(it: int, cfg: TrainConfig) -> float:
    """
    Warmup for warmup_iters steps,
    then cosine decay down to min_lr.
    """
    # Linear warmup
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters

    # After decay window: return min_lr
    if it > cfg.lr_decay_iters:
        return cfg.min_lr

    # Cosine decay
    progress = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ─── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model, train_loader, val_loader,
                  eval_iters: int, device: str,
                  ctx) -> dict:
    model.eval()
    results = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            x = x.to(device)
            y = y.to(device)
            with ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = TrainConfig()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    device = cfg.device
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.manual_seed(1337)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"dtype:  {cfg.dtype}")

    # ── Mixed precision context ───────────────────────────────────────────
    ptdtype = {"float32": torch.float32,
               "bfloat16": torch.bfloat16,
               "float16": torch.float16}[cfg.dtype]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    # ── Data ─────────────────────────────────────────────────────────────
    text = open(cfg.data_path, 'r').read()
    tok  = CharTokenizer(text)

    data    = torch.tensor(tok.encode(text), dtype=torch.long)
    n_train = int(len(data) * cfg.train_split)
    train_data = data[:n_train]
    val_data   = data[n_train:]

    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")
    print(f"Vocab size:   {tok.vocab_size}")

    train_ds = TextDataset(train_data, cfg.max_seq_len)
    val_ds   = TextDataset(val_data,   cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = GPTConfig(
        vocab_size  = tok.vocab_size,
        max_seq_len = cfg.max_seq_len,
        d_model     = cfg.d_model,
        n_heads     = cfg.n_heads,
        n_layers    = cfg.n_layers,
        dropout     = cfg.dropout,
    )
    model = GPT(model_cfg).to(device)

    if cfg.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Separate weight decay: only apply to 2D+ tensors (weights, not biases/LN)
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]

    optim_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups, lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=True,   # fused AdamW: faster on CUDA
    )

    # GradScaler only for float16 (not needed for bf16)
    use_scaler = (cfg.dtype == "float16")
    scaler     = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(cfg.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    print("\n── Starting training ─────────────────────────────────────")
    print(f"  max_iters: {cfg.max_iters}")
    print(f"  batch_size: {cfg.batch_size}")
    print(f"  seq_len: {cfg.max_seq_len}")
    print(f"  params: {model.count_params()/1e6:.2f}M")
    print("──────────────────────────────────────────────────────────\n")

    train_iter   = iter(train_loader)
    best_val_loss = float('inf')
    t0           = time.time()

    torch.cuda.reset_peak_memory_stats()

    for step in range(cfg.max_iters):
        # ── LR update ─────────────────────────────────────────────────
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Get batch ─────────────────────────────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ── Forward + backward ────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)

        with ctx:
            logits, loss = model(x, y)

        scaler.scale(loss).backward()

        # Gradient clipping (important for transformers)
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()

        # ── Logging ───────────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            t1       = time.time()
            dt_ms    = (t1 - t0) * 1000 / cfg.log_interval
            tok_s    = cfg.batch_size * cfg.max_seq_len * cfg.log_interval \
                       / (t1 - t0)
            peak_gb  = torch.cuda.max_memory_allocated() / 1024**3
            t0       = t1

            writer.add_scalar("train/loss",      loss.item(), step)
            writer.add_scalar("train/lr",         lr,          step)
            writer.add_scalar("train/grad_norm",  grad_norm,   step)
            writer.add_scalar("perf/tokens_per_s", tok_s,      step)
            writer.add_scalar("perf/step_ms",     dt_ms,       step)

            print(
                f"step={step:5d}  "
                f"loss={loss.item():.4f}  "
                f"lr={lr:.2e}  "
                f"grad={grad_norm:.3f}  "
                f"step_ms={dt_ms:.1f}  "
                f"tok/s={tok_s:.0f}  "
                f"peak={peak_gb:.2f}GB",
                flush=True
            )

        # ── Evaluation ────────────────────────────────────────────────
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            metrics = estimate_loss(model, train_loader, val_loader,
                                    cfg.eval_iters, device, ctx)
            writer.add_scalar("eval/train_loss", metrics["train"], step)
            writer.add_scalar("eval/val_loss",   metrics["val"],   step)

            print(f"\n── eval at step {step} ─────────────────────────────")
            print(f"  train loss: {metrics['train']:.4f}")
            print(f"  val   loss: {metrics['val']:.4f}")

            # Save best checkpoint
            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                ckpt = {
                    "step":       step,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "config":     cfg,
                    "model_cfg":  model_cfg,
                    "val_loss":   best_val_loss,
                }
                path = os.path.join(cfg.checkpoint_dir, "best.pt")
                torch.save(ckpt, path)
                print(f"  ✓ saved checkpoint (val_loss={best_val_loss:.4f})")
            print("─────────────────────────────────────────────────────\n")

        # ── Generation sample ─────────────────────────────────────────
        if step % cfg.gen_interval == 0 and step > 0:
            model.eval()
            prompt   = torch.zeros(1, 1, dtype=torch.long, device=device)
            with torch.no_grad():
                generated = model.generate(
                    prompt,
                    max_new_tokens  = cfg.gen_max_tokens,
                    temperature     = cfg.gen_temperature,
                    top_k           = cfg.gen_top_k,
                )
            text_out = tok.decode(generated[0].tolist())
            print(f"── Generation at step {step} ────────────────────────")
            print(text_out)
            print("─────────────────────────────────────────────────────\n")
            writer.add_text("generation", text_out, step)
            model.train()

    print("Training complete.")
    writer.close()


if __name__ == "__main__":
    main()
PY

python train_gpt.py
```

---

## 9.12 Visualize Attention Patterns

```bash
cat > visualize_attention.py <<'PY'
"""
Load trained GPT and visualize attention patterns.
Shows what each attention head attends to.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from tokenizer import CharTokenizer
from gpt import GPT, GPTConfig


def plot_attention(model, tokenizer, text: str,
                   layer_idx: int = 0, save_path: str = "attention.png"):
    model.eval()
    tokens    = tokenizer.encode(text)
    idx       = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    idx       = idx.to(next(model.parameters()).device)

    with torch.no_grad():
        _ = model(idx)

    # Get attention weights from the specified layer
    block   = model.blocks[layer_idx]
    weights = block.attn.last_attn_weights  # [B, n_heads, T, T]
    weights = weights[0].cpu()              # [n_heads, T, T]

    n_heads = weights.shape[0]
    T       = weights.shape[1]
    chars   = [repr(tokenizer.id2ch[t])[1:-1] for t in tokens[:T]]

    fig, axes = plt.subplots(
        2, n_heads // 2,
        figsize=(4 * n_heads // 2, 8),
        squeeze=False
    )
    fig.suptitle(f"Attention Weights — Layer {layer_idx}", fontsize=14)

    for h in range(n_heads):
        ax  = axes[h // (n_heads // 2)][h % (n_heads // 2)]
        im  = ax.imshow(weights[h].numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Head {h}")
        ax.set_xticks(range(T))
        ax.set_xticklabels(chars, rotation=90, fontsize=7)
        ax.set_yticks(range(T))
        ax.set_yticklabels(chars, fontsize=7)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved attention plot to: {save_path}")


def main():
    ckpt_path = "checkpoints/best.pt"

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        print("Train for at least 500 steps first.")
        return

    ckpt   = torch.load(ckpt_path, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text        = open("shakespeare.txt").read()
    tokenizer   = CharTokenizer(text)

    model_cfg   = ckpt["model_cfg"]
    model       = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sample_text = "To be or not to be"
    print(f"Visualizing attention for: '{sample_text}'")

    for layer in range(min(3, model_cfg.n_layers)):
        plot_attention(
            model, tokenizer, sample_text,
            layer_idx = layer,
            save_path = f"attention_layer{layer}.png"
        )

    print("Done. View the .png files.")


if __name__ == "__main__":
    main()
PY

python visualize_attention.py
```

---

## 9.13 Generation Script (interactive)

```bash
cat > generate.py <<'PY'
"""
Load trained GPT and generate text.
"""
import torch, os, sys
from tokenizer import CharTokenizer
from gpt import GPT, GPTConfig


def main():
    ckpt_path = "checkpoints/best.pt"

    if not os.path.exists(ckpt_path):
        print("No checkpoint found. Train first.")
        return

    ckpt      = torch.load(ckpt_path, map_location="cpu")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    text      = open("shakespeare.txt").read()
    tokenizer = CharTokenizer(text)

    model_cfg = ckpt["model_cfg"]
    model     = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded checkpoint: step={ckpt['step']} val_loss={ckpt['val_loss']:.4f}")
    print(f"Model: {model.count_params()/1e6:.2f}M params\n")

    configs = [
        {"temperature": 0.5, "top_k": 10,  "label": "Conservative (temp=0.5, top_k=10)"},
        {"temperature": 0.8, "top_k": 40,  "label": "Balanced    (temp=0.8, top_k=40)"},
        {"temperature": 1.2, "top_k": None,"label": "Creative    (temp=1.2, top_k=None)"},
    ]

    prompt_text = "HAMLET:\n"
    prompt_ids  = tokenizer.encode(prompt_text)
    prompt_t    = torch.tensor(prompt_ids, dtype=torch.long,
                               device=device).unsqueeze(0)

    for cfg in configs:
        print(f"── {cfg['label']} ──────────────────────────────────────────")
        with torch.no_grad():
            generated = model.generate(
                prompt_t,
                max_new_tokens = 300,
                temperature    = cfg["temperature"],
                top_k          = cfg["top_k"],
            )
        out = tokenizer.decode(generated[0].tolist())
        print(out)
        print()


if __name__ == "__main__":
    main()
PY

python generate.py
```

---

## 9.14 Parameter Count and Memory Analysis

```bash
cat > model_analysis.py <<'PY'
"""
Analyze GPT model: parameter count, memory, FLOPs estimate.
"""
import torch
from gpt import GPT, GPTConfig


def analyze_model(cfg: GPTConfig):
    model = GPT(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    # ── Parameter counts ──────────────────────────────────────────────────
    total      = model.count_params()
    emb_params = cfg.vocab_size * cfg.d_model           # token_emb (shared with lm_head)
    pos_params = cfg.max_seq_len * cfg.d_model           # pos_emb

    attn_params_per_block = (
        3 * cfg.d_model * cfg.d_model +  # c_attn (Q,K,V combined)
        cfg.d_model * cfg.d_model         # c_proj
    )
    ffn_params_per_block = (
        cfg.d_model * 4 * cfg.d_model +   # W1
        4 * cfg.d_model * cfg.d_model     # W2
    )
    ln_params_per_block  = 2 * (2 * cfg.d_model)        # 2 LN per block
    block_params         = (attn_params_per_block +
                            ffn_params_per_block +
                            ln_params_per_block)

    print(f"\n── GPT Parameter Analysis ────────────────────────────────")
    print(f"  d_model:       {cfg.d_model}")
    print(f"  n_heads:       {cfg.n_heads}")
    print(f"  n_layers:      {cfg.n_layers}")
    print(f"  max_seq_len:   {cfg.max_seq_len}")
    print(f"  vocab_size:    {cfg.vocab_size}")
    print(f"")
    print(f"  token_emb:     {emb_params/1e6:.3f}M  (shared with lm_head)")
    print(f"  pos_emb:       {pos_params/1e6:.3f}M")
    print(f"  per block:")
    print(f"    attention:   {attn_params_per_block/1e6:.3f}M")
    print(f"    feedforward: {ffn_params_per_block/1e6:.3f}M")
    print(f"    layernorm:   {ln_params_per_block/1e6:.3f}M")
    print(f"    block total: {block_params/1e6:.3f}M")
    print(f"  all blocks:    {block_params*cfg.n_layers/1e6:.3f}M")
    print(f"  TOTAL:         {total/1e6:.3f}M")

    # ── Memory analysis ───────────────────────────────────────────────────
    param_bytes  = total * 4            # float32
    grad_bytes   = total * 4            # float32 gradients
    adam_bytes   = total * 8            # Adam: m + v states
    # Activations rough estimate: B * T * d_model * n_layers * 4 bytes
    # For single batch at training:
    B, T = 4, cfg.max_seq_len
    act_bytes = B * T * cfg.d_model * cfg.n_layers * 4 * 2  # *2 for various intermediates

    total_bytes = param_bytes + grad_bytes + adam_bytes + act_bytes

    print(f"\n── Memory Estimate (B={B}, T={T}) ──────────────────────")
    print(f"  params  (fp32): {param_bytes/1024**3:.3f} GB")
    print(f"  grads   (fp32): {grad_bytes/1024**3:.3f} GB")
    print(f"  adam states:    {adam_bytes/1024**3:.3f} GB")
    print(f"  activations:    {act_bytes/1024**3:.3f} GB  (rough estimate)")
    print(f"  TOTAL:          {total_bytes/1024**3:.3f} GB")

    # ── Actual GPU memory usage ───────────────────────────────────────────
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        x   = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        y   = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n── Actual GPU Peak Memory ───────────────────────────")
        print(f"  Peak: {peak_gb:.3f} GB")

    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    # Our training config
    cfg = GPTConfig(
        vocab_size  = 65,
        max_seq_len = 256,
        d_model     = 384,
        n_heads     = 6,
        n_layers    = 6,
    )
    analyze_model(cfg)

    # GPT-2 small equivalent (for reference, CPU only if no GPU big enough)
    print("\n\n── GPT-2 Small equivalent (reference) ──────────────────")
    cfg2 = GPTConfig(
        vocab_size  = 50257,
        max_seq_len = 1024,
        d_model     = 768,
        n_heads     = 12,
        n_layers    = 12,
    )
    m2 = GPT(cfg2)
    print(f"  params: {m2.count_params()/1e6:.1f}M  "
          f"(GPT-2 small = 124M params)")
PY

python model_analysis.py
```

---

## 9.15 Debug Drills

### Drill 1 — Loss at random init should be ~ln(vocab_size)
```bash
python - <<'PY'
import torch, math
from gpt import GPT, GPTConfig

cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                  d_model=128, n_heads=4, n_layers=2)
model = GPT(cfg)
x     = torch.randint(0, 65, (4, 64))
y     = torch.randint(0, 65, (4, 64))
_, loss = model(x, y)

expected = math.log(65)
print(f"actual loss:   {loss.item():.4f}")
print(f"expected loss: {expected:.4f}  (= log(vocab_size))")
print(f"diff:          {abs(loss.item() - expected):.4f}")
assert abs(loss.item() - expected) < 0.3, "Init loss wrong!"
print("Init loss: PASSED")
PY
```

If initial loss is far from `log(vocab_size)`, weight initialization is wrong.

---

### Drill 2 — Overfit on tiny batch (sanity check)
```bash
cat > overfit_test.py <<'PY'
"""
A model should be able to overfit a tiny dataset.
If it can't, something is wrong with the model or training loop.
"""
import torch
from gpt import GPT, GPTConfig

torch.manual_seed(42)

cfg   = GPTConfig(vocab_size=65, max_seq_len=32,
                  d_model=128, n_heads=4, n_layers=4,
                  dropout=0.0)   # no dropout for overfit test
model = GPT(cfg).to("cuda")
opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Tiny fixed batch
x = torch.randint(0, 65, (4, 32), device="cuda")
y = torch.randint(0, 65, (4, 32), device="cuda")

print("Overfitting on tiny batch (should see loss → ~0):")
for step in range(200):
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    if step % 20 == 0:
        print(f"  step={step:3d}  loss={loss.item():.4f}")

assert loss.item() < 0.1, f"Failed to overfit! Loss={loss.item():.4f}"
print("Overfit test: PASSED")
PY

python overfit_test.py
```

---

### Drill 3 — Causality check (future tokens must not influence past)
Already tested in `multihead_attention.py` — run it again here as a reminder:
```bash
python multihead_attention.py
```

---

### Drill 4 — Gradient flow check (no vanishing/exploding grads)
```bash
cat > grad_check.py <<'PY'
"""Check gradient norms across layers."""
import torch
from gpt import GPT, GPTConfig

cfg   = GPTConfig(vocab_size=65, max_seq_len=64,
                  d_model=256, n_heads=4, n_layers=6,
                  dropout=0.0)
model = GPT(cfg).to("cuda")
x     = torch.randint(0, 65, (4, 64), device="cuda")
y     = torch.randint(0, 65, (4, 64), device="cuda")

_, loss = model(x, y)
loss.backward()

print("Gradient norms by parameter:")
for name, p in model.named_parameters():
    if p.grad is not None:
        gnorm = p.grad.norm().item()
        status = "OK"
        if gnorm < 1e-7: status = "VANISHING"
        if gnorm > 100:  status = "EXPLODING"
        print(f"  {name:50s}  grad_norm={gnorm:.6f}  [{status}]")
PY

python grad_check.py
```

---

## 9.16 TensorBoard — View Training Curves

```bash
# In a separate terminal:
tensorboard --logdir runs/ --port 6006

# Then open in browser:
# http://localhost:6006
```

You'll see:
- `train/loss` — step-level loss
- `eval/train_loss`, `eval/val_loss` — evaluation curves
- `train/lr` — learning rate schedule
- `train/grad_norm` — gradient norm
- `perf/tokens_per_s` — throughput
- `generation` — text samples

---

