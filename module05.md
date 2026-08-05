# Module 5 — DistributedSampler (Deep Dive)

## Summary
This module goes far deeper than "just add DistributedSampler".

- **Theory**: why every GPU must see different non-overlapping data, what happens mathematically when they don't, epoch-level shuffling, drop_last behavior
- **From scratch**: implement your own DistributedSampler without using PyTorch's version
- **Visualization**: print exactly which sample index goes to which GPU, every epoch
- **Experiments**: show training divergence when sampler is wrong
- **Debug drills**: common mistakes (shuffle=True in DataLoader, forgetting set_epoch, uneven dataset sizes)

---

## 5.1 Theory — Why data distribution matters deeply

### What DDP expects from data
Recall from Module 4:
- Each GPU computes gradients on its own mini-batch
- AllReduce **averages** those gradients across all GPUs
- The optimizer step is identical on every GPU

This works correctly only if:
- every GPU sees **different** data each iteration
- across one full epoch, every sample is seen **exactly once** globally
- the distribution of data across GPUs is **statistically similar** (so gradients are meaningful averages)

### What happens without DistributedSampler (the wrong way)

Without a sampler, every GPU uses the **same default sampler**:

```
Dataset: [img0, img1, img2, img3, img4, img5, img6, img7]

GPU0 DataLoader → [img0, img1, img2, img3, img4, img5, img6, img7]
GPU1 DataLoader → [img0, img1, img2, img3, img4, img5, img6, img7]
GPU2 DataLoader → [img0, img1, img2, img3, img4, img5, img6, img7]
...
GPU7 DataLoader → [img0, img1, img2, img3, img4, img5, img6, img7]
```

Every GPU computes gradients on **identical data**.
AllReduce averages them → same result as single GPU.
You get **zero benefit from multiple GPUs** (same as batch_size=256, just repeated 8 times with same examples).

Worse: your effective batch size is still 256, not 256×8=2048.

### What DistributedSampler does (the right way)

DistributedSampler partitions the dataset index space:

```
Dataset: [img0..img49999] (CIFAR-10 train = 50000 samples)

GPU0 → [img0, img8, img16, ...]   (every 8th sample, offset 0)
GPU1 → [img1, img9, img17, ...]   (every 8th sample, offset 1)
GPU2 → [img2, img10, img18, ...]
...
GPU7 → [img7, img15, img23, ...]
```

Now:
- Each GPU sees **unique samples**
- Together they cover the **full dataset**
- Effective global batch size = per_gpu_batch × world_size

### The epoch shuffle problem (why set_epoch matters)

DistributedSampler uses a **seed** to shuffle indices:

```python
seed = self.seed + self.epoch   # epoch changes the shuffle
```

Without `set_epoch(epoch)`:
- epoch stays at 0 forever
- same shuffle every epoch
- every epoch, GPU0 always gets the same subset of the dataset
- model never generalizes properly

With `set_epoch(epoch)`:
- shuffle changes every epoch
- every sample eventually seen by all GPUs across epochs

### drop_last behavior

Dataset size may not divide evenly by world_size:

```
50000 samples / 8 GPUs = 6250 exactly (lucky for CIFAR10)

But general case:
50001 samples / 8 GPUs = 6250 remainder 1
```

With `drop_last=True`:
- drop the last few samples to make it even
- every GPU gets exactly the same number of batches
- **required for DDP**: if one GPU has 1 fewer batch, it will hang at AllReduce because the other 7 GPUs are waiting for a collective that will never come

With `drop_last=False`:
- PyTorch's sampler **pads** the dataset by repeating some samples
- keeps things even but introduces duplicate samples

---

## 5.2 DistributedSampler internals (what PyTorch actually does)

Here is a simplified version of what `torch.utils.data.distributed.DistributedSampler` does internally:

```
1. Take all indices: [0, 1, 2, ..., N-1]

2. If shuffle:
   Generate permutation using torch.Generator seeded with:
   seed + epoch

3. If drop_last:
   Truncate to floor(N / world_size) * world_size

   If NOT drop_last:
   Pad to ceil(N / world_size) * world_size by repeating from start

4. Slice for this rank:
   indices[rank : : world_size]   (strided slice)
   OR equivalently:
   chunk = len(indices) // world_size
   indices[rank*chunk : (rank+1)*chunk]

5. DataLoader iterates over this rank's indices
```

The key line is the **strided slice**:
```
rank=0 gets indices[0::8]  → positions 0,8,16,24,...
rank=1 gets indices[1::8]  → positions 1,9,17,25,...
```

This ensures interleaved coverage of the full shuffled dataset.

---

## 5.3 From Scratch — Implement your own DistributedSampler

This is the most important exercise in this module.

```bash
cat > my_distributed_sampler.py <<'PY'
import math
import torch
from torch.utils.data import Sampler


class MyDistributedSampler(Sampler):
    """
    Custom DistributedSampler built from scratch.
    Replicates the core logic of PyTorch's DistributedSampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas: int,   # world_size
        rank: int,           # this process's rank
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.dataset      = dataset
        self.num_replicas = num_replicas
        self.rank         = rank
        self.shuffle      = shuffle
        self.seed         = seed
        self.drop_last    = drop_last
        self.epoch        = 0   # updated externally each epoch

        # Compute how many samples this rank will see per epoch
        if self.drop_last:
            # floor division: drop leftovers
            self.num_samples = math.floor(len(self.dataset) / self.num_replicas)
        else:
            # ceil division: pad with repeated samples
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        """
        Call this at the start of every epoch.
        Changes the random seed used for shuffling.
        Without this, shuffle is the same every epoch.
        """
        self.epoch = epoch

    def __iter__(self):
        # ── Step 1: generate indices ──────────────────────────────────────
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)   # epoch changes shuffle
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # ── Step 2: pad or drop to make divisible by num_replicas ─────────
        if self.drop_last:
            # drop the last few samples if not divisible
            indices = indices[:self.total_size]
        else:
            # pad by repeating from start
            pad = self.total_size - len(indices)
            indices += indices[:pad]

        assert len(indices) == self.total_size, \
            f"Expected {self.total_size} got {len(indices)}"

        # ── Step 3: slice for this rank using strided indexing ────────────
        my_indices = indices[self.rank : self.total_size : self.num_replicas]

        assert len(my_indices) == self.num_samples, \
            f"Expected {self.num_samples} got {len(my_indices)}"

        return iter(my_indices)

    def __len__(self):
        return self.num_samples


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test (no GPU needed, no distributed needed)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    # Tiny dataset: 20 samples
    N = 20
    ds = TensorDataset(torch.arange(N))

    world_size = 4
    num_epochs = 3

    print("=" * 60)
    print(f"Dataset size={N}, world_size={world_size}")
    print("=" * 60)

    samplers = [
        MyDistributedSampler(
            ds,
            num_replicas=world_size,
            rank=r,
            shuffle=True,
            seed=0,
            drop_last=True,
        )
        for r in range(world_size)
    ]

    for epoch in range(num_epochs):
        print(f"\n── Epoch {epoch} ──────────────────────────────────────")
        all_seen = []

        for r, sampler in enumerate(samplers):
            sampler.set_epoch(epoch)
            my_idx = list(sampler)
            all_seen.extend(my_idx)
            print(f"  rank={r}  indices={my_idx}")

        # Verify: no overlap between ranks
        all_seen_set = set(all_seen)
        print(f"\n  total indices seen: {len(all_seen)}")
        print(f"  unique indices    : {len(all_seen_set)}")
        print(f"  any overlap?      : {len(all_seen) != len(all_seen_set)}")

    print("\n" + "=" * 60)
    print("Compare: what happens WITHOUT set_epoch (same shuffle every epoch)")
    print("=" * 60)
    for epoch in range(3):
        sampler = MyDistributedSampler(ds, num_replicas=4, rank=0,
                                       shuffle=True, seed=0, drop_last=True)
        # deliberately NOT calling set_epoch → epoch stays 0 → same shuffle
        print(f"  epoch={epoch} rank=0 indices={list(sampler)}")
PY
```

Run (no GPU needed, no torchrun needed):
```bash
python my_distributed_sampler.py
```

You will see:
- each epoch: different shuffle, no overlap between ranks
- without `set_epoch`: identical indices every epoch

---

## 5.4 Visualization Lab — which GPU gets which batch

This runs with real distributed processes and prints exactly which sample index goes to which GPU, batch by batch.

```bash
cat > visualize_sampler.py <<'PY'
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def main():
    dist.init_process_group("gloo")   # CPU only, no GPU needed for this demo
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    # Tiny dataset: 32 samples (value = index, easy to track)
    N = 32
    dataset = TensorDataset(torch.arange(N, dtype=torch.float32))

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    loader = DataLoader(dataset, batch_size=4, sampler=sampler)

    for epoch in range(2):
        sampler.set_epoch(epoch)

        # Collect all batches for this rank
        batches = [batch[0].tolist() for batch in loader]

        # Print rank by rank (ordered)
        for r in range(world_size):
            dist.barrier()
            if rank == r:
                print(f"epoch={epoch} rank={r} batches={batches}", flush=True)
            dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run with 4 processes:
```bash
torchrun --nproc_per_node=4 visualize_sampler.py
```

You will see:
- each rank gets different batches
- epoch 0 and epoch 1 have different shuffles
- no sample appears in two ranks in the same epoch

---

## 5.5 Experiment — show what breaks without DistributedSampler

```bash
cat > wrong_sampler_demo.py <<'PY'
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset


def main():
    dist.init_process_group("gloo")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    N = 32
    dataset = TensorDataset(torch.arange(N, dtype=torch.float32))

    # WRONG: no DistributedSampler, shuffle=True in DataLoader
    # Each rank shuffles independently → different order but same data pool
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batches = [b[0].tolist() for b in loader]

    # Collect all batches from all ranks to check overlap
    all_first_batch = [None] * world_size
    my_first = torch.tensor(batches[0])

    # AllGather first batch from each rank
    gathered = [torch.zeros_like(my_first) for _ in range(world_size)]
    dist.all_gather(gathered, my_first)

    for r in range(world_size):
        dist.barrier()
        if rank == r:
            print(f"\nrank={r} first batch={batches[0]}", flush=True)
        dist.barrier()

    if rank == 0:
        print("\n── Overlap analysis ──")
        s0 = set(gathered[0].tolist())
        s1 = set(gathered[1].tolist())
        overlap = s0 & s1
        print(f"rank0 first batch: {sorted(s0)}")
        print(f"rank1 first batch: {sorted(s1)}")
        print(f"overlap          : {sorted(overlap)}")
        print(f"overlap size     : {len(overlap)}/4 samples")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=4 wrong_sampler_demo.py
```

You'll see overlapping sample indices between ranks → this is the bug.

---

## 5.6 Full DDP + Custom Sampler (put it all together)

Now use `MyDistributedSampler` from scratch inside a real DDP training loop on CIFAR-10.

```bash
cat > train_cifar10_custom_sampler.py <<'PY'
import os, time, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms


# ─── Sampler from scratch ──────────────────────────────────────────────────────
class MyDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank,
                 shuffle=True, seed=0, drop_last=True):
        self.dataset      = dataset
        self.num_replicas = num_replicas
        self.rank         = rank
        self.shuffle      = shuffle
        self.seed         = seed
        self.drop_last    = drop_last
        self.epoch        = 0

        if drop_last:
            self.num_samples = math.floor(len(dataset) / num_replicas)
        else:
            self.num_samples = math.ceil(len(dataset) / num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.drop_last:
            indices = indices[:self.total_size]
        else:
            indices += indices[:self.total_size - len(indices)]

        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples


# ─── Model ─────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )
    def forward(self, x): return self.net(x)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    torch.manual_seed(42)

    # ── Data ─────────────────────────────────────────────────────────────────
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])

    if is_main:
        datasets.CIFAR10("./data", train=True,  download=True)
        datasets.CIFAR10("./data", train=False, download=True)
    dist.barrier()

    train_ds = datasets.CIFAR10("./data", train=True,  transform=tfm_train)
    test_ds  = datasets.CIFAR10("./data", train=False, transform=tfm_test)

    # ── Use our custom sampler ────────────────────────────────────────────────
    train_sampler = MyDistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=0, drop_last=True,
    )
    test_sampler = MyDistributedSampler(
        test_ds, num_replicas=world_size, rank=rank,
        shuffle=False, seed=0, drop_last=False,
    )

    train_loader = DataLoader(train_ds, batch_size=256,
                              sampler=train_sampler,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,
                              sampler=test_sampler,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)

    # ── Model + DDP ──────────────────────────────────────────────────────────
    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler    = torch.cuda.amp.GradScaler()

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(5):
        model.train()
        train_sampler.set_epoch(epoch)   # critical

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # ── Evaluation with distributed reduction ────────────────────────
        model.eval()
        correct = torch.zeros(1, device=device)
        total   = torch.zeros(1, device=device)

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x).argmax(1)
                correct += (pred == y).sum()
                total   += y.numel()

        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total,   op=dist.ReduceOp.SUM)

        if is_main:
            print(f"epoch={epoch+1} acc={correct.item()/total.item()*100:.2f}%",
                  flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=8 train_cifar10_custom_sampler.py
```

---

## 5.7 Debug Drills

### Drill 1 — shuffle=True in DataLoader (conflict with sampler)
```python
# WRONG: never do this with DistributedSampler
DataLoader(dataset, batch_size=256, sampler=train_sampler, shuffle=True)
```

PyTorch will raise:
```
ValueError: sampler option is mutually exclusive with shuffle
```

Fix: if you pass a sampler, **remove `shuffle=True`** from DataLoader.

---

### Drill 2 — forget set_epoch (silently wrong)
In `train_cifar10_custom_sampler.py`, comment out:
```python
# train_sampler.set_epoch(epoch)
```

Rerun. Training still works (no crash) but:
- same shuffle every epoch
- rank 0 always sees same subset of data

This is a **silent bug**: no error, but training quality suffers on long runs.

---

### Drill 3 — uneven drop_last=False hang
With `drop_last=False` and a dataset not evenly divisible, one rank may get 1 extra batch. When that rank hits AllReduce for that extra batch, other ranks are not there → hang.

Demonstrate:
```bash
cat > drop_last_hang_demo.py <<'PY'
import os, math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, Sampler


class UnevenSampler(Sampler):
    """Deliberately give rank 0 one extra sample"""
    def __init__(self, rank):
        self.rank = rank

    def __iter__(self):
        if self.rank == 0:
            return iter(range(5))  # 5 samples
        else:
            return iter(range(4))  # 4 samples → different batch count

    def __len__(self):
        return 5 if self.rank == 0 else 4


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    ds      = TensorDataset(torch.arange(10, dtype=torch.float32))
    sampler = UnevenSampler(rank)
    loader  = DataLoader(ds, batch_size=1, sampler=sampler)

    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(nn.Linear(1, 1).to(device), device_ids=[local_rank])
    opt   = torch.optim.SGD(model.parameters(), lr=0.01)

    print(f"rank={rank} will process {len(sampler)} batches", flush=True)

    for i, (x,) in enumerate(loader):
        x = x.to(device)
        loss = model(x).sum()
        loss.backward()    # DDP allreduce fires here
        opt.step()
        opt.zero_grad()
        print(f"rank={rank} step={i} done", flush=True)

    print(f"rank={rank} finished loop", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=2 drop_last_hang_demo.py
```

Rank 0 will hang waiting for rank 1 at the 5th step. Kill:
```bash
pkill -f drop_last_hang_demo.py
pkill -f torchrun
```

**Fix**: always use `drop_last=True` in DDP training.

---

### Drill 4 — verify coverage (every sample seen exactly once)
```bash
cat > verify_coverage.py <<'PY'
import math
import torch
from torch.utils.data import TensorDataset
from my_distributed_sampler import MyDistributedSampler

N          = 50000  # CIFAR-10 size
world_size = 8

ds = TensorDataset(torch.arange(N))

samplers = [
    MyDistributedSampler(ds, num_replicas=world_size, rank=r,
                         shuffle=True, seed=0, drop_last=True)
    for r in range(world_size)
]

for epoch in range(3):
    all_indices = []
    for r, s in enumerate(samplers):
        s.set_epoch(epoch)
        all_indices.extend(list(s))

    unique = set(all_indices)
    expected_total = math.floor(N / world_size) * world_size

    print(f"epoch={epoch} "
          f"total_seen={len(all_indices)} "
          f"unique={len(unique)} "
          f"expected={expected_total} "
          f"duplicates={len(all_indices)-len(unique)} "
          f"dropped={N - expected_total}")
PY
```

Run:
```bash
python verify_coverage.py
```

You should see:
- `duplicates=0` (no sample appears twice)
- `dropped=0` (for CIFAR-10, 50000 divides evenly by 8)

---

## 5.8 Module 5 Pass Criteria

You're done when you can:

1) Explain in your own words:
   - what happens when you skip DistributedSampler
   - why `set_epoch()` is required and what breaks without it
   - what `drop_last=True` does and why DDP needs it

2) Run successfully:
   - `python my_distributed_sampler.py` (standalone test, no GPU)
   - `torchrun --nproc_per_node=4 visualize_sampler.py`
   - `torchrun --nproc_per_node=8 train_cifar10_custom_sampler.py`

3) Reproduce and explain:
   - DataLoader `shuffle=True` + sampler conflict
   - `drop_last=False` hang in DDP
   - silent wrong training from missing `set_epoch`

4) Verify dataset coverage:
   - `python verify_coverage.py` shows `duplicates=0`

---

