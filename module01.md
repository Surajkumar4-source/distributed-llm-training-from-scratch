## Module 1 — Single GPU Training (PyTorch)  
**Goal:** Understand a complete “normal” training loop on **one GPU**: forward → loss → backward → optimizer → (optional) scheduler, plus the *real* bottlenecks: data loading, GPU memory, and utilization.

We’ll build and train a **CNN on CIFAR-10**, then debug performance and memory like a systems person.

---

# 1.1 Theory (in the order it happens every iteration)

### The training iteration lifecycle
For one mini-batch `(x, y)`:

1. **Move batch to GPU**
   - `x = x.to("cuda", non_blocking=True)`
2. **Forward pass**
   - `logits = model(x)`
3. **Loss**
   - `loss = criterion(logits, y)`
4. **Backward pass**
   - `loss.backward()` builds/uses the autograd graph to compute gradients
5. **Optimizer step**
   - `optimizer.step()` updates parameters using the gradients
6. **Zero grads**
   - `optimizer.zero_grad(set_to_none=True)` (important for speed/memory)

### Forward vs Backward (important intuition)
- Forward computes activations.
- Backward uses saved activations + chain rule to compute gradients.
- Backward is often **~2–3×** the cost of forward for many models.

### Autograd graph (what it really is)
- PyTorch tracks operations on tensors with `requires_grad=True`.
- It builds a directed graph of ops; nodes store backward functions.
- When you call `loss.backward()`, autograd traverses graph backward.

### Key vocabulary (must be crystal clear)
- **Epoch**: one full pass over dataset.
- **Iteration/step**: one optimizer update (one mini-batch).
- **Mini-batch**: small subset of data.
- **Learning rate**: step size.
- **Scheduler**: changes LR across steps/epochs (optional but common).

### GPU memory in training (why you run out)
Main components:
- **Parameters** (model weights)
- **Gradients** (same size as parameters)
- **Optimizer state** (e.g., Adam has extra tensors per parameter)
- **Activations** (often the biggest; depends on batch size and network)

PyTorch uses a **caching allocator**:  
- `memory_allocated()` = memory currently used by tensors  
- `memory_reserved()` = memory held by allocator (may be > allocated) 

This explains the common confusion: “I freed tensors but GPU memory doesn’t drop.”

---

# 1.2 Hands-on Lab — Train a CNN on CIFAR-10 (Single GPU)

## Step A — Install dependencies
In your venv:
```bash
pip install torch torchvision torchaudio --upgrade
pip install tqdm
```

## Step B — Create `train_cifar10_single_gpu.py`
Copy-paste this whole script:

```python
import os, time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


@dataclass
class Config:
    data_dir: str = "./data"
    batch_size: int = 256
    num_workers: int = 8
    epochs: int = 10
    lr: float = 0.1
    weight_decay: float = 5e-4
    device: str = "cuda"
    use_amp: bool = True  # mixed precision (recommended on A100)
    print_every: int = 100


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8 -> 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def mem_gb():
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    return alloc, reserv


def main():
    cfg = Config()
    assert torch.cuda.is_available(), "CUDA not available"

    torch.backends.cudnn.benchmark = True

    # Optional: TF32 on Ampere (A100) for faster matmuls with good accuracy in many cases
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = cfg.device
    print("Device:", torch.cuda.get_device_name(0))
    print("TF32 matmul:", torch.backends.cuda.matmul.allow_tf32)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # Warmup memory stats
    torch.cuda.reset_peak_memory_stats()

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()

        t_epoch0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}")

        running_loss = 0.0
        for it, (x, y) in enumerate(pbar):
            t0 = time.time()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accurate step timing: sync before reading time
            torch.cuda.synchronize()
            step_time = time.time() - t0

            running_loss += loss.item()
            global_step += 1

            if (global_step % cfg.print_every) == 0:
                alloc, reserv = mem_gb()
                peak = torch.cuda.max_memory_allocated() / 1024**3
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{running_loss/cfg.print_every:.4f}",
                    "lr": f"{lr:.4g}",
                    "step_s": f"{step_time:.3f}",
                    "alloc_GB": f"{alloc:.2f}",
                    "resv_GB": f"{reserv:.2f}",
                    "peak_GB": f"{peak:.2f}",
                })
                running_loss = 0.0

        scheduler.step()

        acc = evaluate(model, test_loader, device)
        epoch_s = time.time() - t_epoch0
        print(f"Epoch {epoch+1}: acc={acc*100:.2f}% epoch_time={epoch_s:.1f}s")

    print("Done.")


if __name__ == "__main__":
    main()
```

Run:
```bash
python train_cifar10_single_gpu.py
```

---

# 1.3 What to watch while it runs (the “systems” part)

Open another terminal:
```bash
nvidia-smi -l 1
```

You want to see:
- GPU utilization near high %
- memory stable (no leak)
- power draw non-trivial (means GPU actually working)

If GPU util is low, it’s usually:
- DataLoader bottleneck (CPU can’t feed GPU fast enough)
- too small batch / model (GPU underutilized)

---

# 1.4 Debug Lab — common single-GPU bottlenecks

## A) DataLoader bottleneck test
Run two experiments:

1) **Slow loader (simulate worst case)**  
Set:
```python
num_workers=0
```
Run again and observe:
- GPU utilization drops
- step time increases

2) **Faster loader**
Set:
```python
num_workers=8
pin_memory=True
persistent_workers=True
```
Observe improvement.

Why: pinned memory + multiple workers helps overlap CPU work with GPU copies.

## B) GPU memory debugging (allocated vs reserved)
In the log you’ll see:
- `alloc_GB` vs `resv_GB` vs `peak_GB`

Reserved may remain high because of the caching allocator. That’s expected and documented. 

---

# 1.5 (Optional but powerful) Profiling one epoch with `torch.profiler`

This is your first “real” performance microscope.

Add this near the training loop (we’ll keep it minimal here; deeper profiling is Module 15):
- PyTorch’s profiler supports schedules like `wait/warmup/active` to limit overhead. 

If you want, I’ll give you a clean profiler patch next—right now the priority is: **get a working baseline training loop**.

---

# 1.6 Module 1 pass criteria (you’re done when)
You can:
1. Explain what happens in order: **forward → loss → backward → step → zero_grad**
2. Explain the difference between:
   - epoch vs iteration
   - allocated vs reserved GPU memory (PyTorch caching allocator) 
3. Run CIFAR-10 training on 1 GPU and report:
   - accuracy after 1–3 epochs
   - step time
   - peak allocated GB
4. Demonstrate DataLoader bottleneck by changing `num_workers`

---
