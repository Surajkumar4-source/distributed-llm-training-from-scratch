# Module 3 — Distributed Training Fundamentals (the heart)

## Summary
You’ll learn the *real primitives* behind all distributed deep learning:

- What a **process**, **rank**, **world size**, and **process group** are
- How `torchrun` starts N processes and wires them together using env vars
- What “collective communication” means (broadcast/allreduce/allgather/reduce_scatter)
- How and why **deadlocks** happen (and how to debug them)
- Hands-on: write distributed programs **without neural networks** first:
  1) point-to-point `send/recv`
  2) `broadcast`
  3) `all_reduce`
  4) `barrier`
  5) debug drills (intentional hangs)

Everything here is the foundation for DDP, FSDP, DeepSpeed, Megatron.

---

## 3.1 Theory — the objects you must understand

### Process vs thread (why distributed uses processes)
- A **process** has its own Python interpreter, memory space, and CUDA context.
- A **thread** shares memory inside one process.
- PyTorch distributed GPU training is typically **multi-process**: **1 process per GPU**. This avoids Python GIL issues and matches CUDA device ownership well.

### Rank / World Size
- **World size**: total number of processes participating (e.g., 8).
- **Global rank**: ID of each process in `[0, world_size-1]`.
- **Local rank**: GPU index on the current node, usually `[0..(gpus_per_node-1)]`.

`torchrun` sets these for you using environment variables. 

### Backend
Backend = communication engine used by PyTorch:
- **NCCL**: best for CUDA GPU collectives (what you want on A100). 
- **Gloo**: common for CPU collectives (also works for some GPU cases but not the main path).
- **MPI**: depends on how PyTorch was built and your MPI stack.

PyTorch uses a **ProcessGroup** abstraction: `init_process_group(backend="nccl")` creates the group that collective ops run on. 

### Rendezvous (how processes “find” each other)
To form a process group, all ranks need a shared meeting point (rendezvous), usually using:
- `MASTER_ADDR` (hostname/IP of rank 0’s node)
- `MASTER_PORT` (port for the store)
- plus `RANK`, `WORLD_SIZE`, `LOCAL_RANK`

`torchrun` typically sets these automatically. 

### Collective communication (the key primitives)
NCCL provides optimized collectives like:
- **Broadcast**: one rank sends same tensor to everyone
- **AllReduce**: reduce (sum/avg) across ranks and distribute result to all
- **AllGather**: gather tensors from all ranks and give everyone the concatenation
- **ReduceScatter**: reduce across ranks then scatter chunks to ranks

These are the “grammar” of distributed training. 

Important rule: **every rank must call the same collective in the same order**, with compatible tensor sizes/dtypes, or you hang/deadlock. 

### Barrier (synchronization point)
`dist.barrier()` is “everyone wait here until everyone arrives.”
It’s useful for timing and for debugging “who didn’t reach this line.”

---

## 3.2 Practical — Setup patterns (single node, 8 GPUs)

### Golden rule: one process per GPU
In each process you do:
1) read `LOCAL_RANK`
2) set CUDA device
3) init process group
4) run your collectives
5) destroy process group

`torchrun` launches processes and sets env vars like `LOCAL_RANK`, `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`. 

---

# 3.3 Hands-on Labs (no neural networks yet)

All commands below assume:
```bash
source /root/ddl_course/venv/bin/activate
```

## Lab 3.3.1 — Hello ranks + GPU assignment
Create `dist_hello.py`:

```bash
cat > dist_hello.py <<'PY'
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    name = torch.cuda.get_device_name(local_rank)

    # Print in rank order to avoid messy interleaving
    for r in range(world):
        dist.barrier()
        if rank == r:
            print(f"rank={rank}/{world} local_rank={local_rank} gpu={name}", flush=True)
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=8 dist_hello.py
```

You should see ranks 0..7 each bound to a GPU.

---

## Lab 3.3.2 — Broadcast
Create `dist_broadcast.py`:

```bash
cat > dist_broadcast.py <<'PY'
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    x = torch.empty(4, device=device, dtype=torch.float32)

    if rank == 0:
        x[:] = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.float32)
    else:
        x[:] = -1

    dist.broadcast(x, src=0)

    # print rank-wise
    dist.barrier()
    if rank == 0:
        print("After broadcast, all ranks should have [1,2,3,4]")
    dist.barrier()
    print(f"rank {rank} x={x.tolist()}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
```

Run (2 GPUs first):
```bash
torchrun --nproc_per_node=2 dist_broadcast.py
```

Then 8 GPUs:
```bash
torchrun --nproc_per_node=8 dist_broadcast.py
```

---

## Lab 3.3.3 — AllReduce (sum then average)
Create `dist_allreduce.py`:

```bash
cat > dist_allreduce.py <<'PY'
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    x = torch.tensor([rank], device=device, dtype=torch.float32)

    dist.all_reduce(x, op=dist.ReduceOp.SUM)  # everyone gets sum
    if rank == 0:
        print(f"SUM should be {sum(range(world))}")

    dist.barrier()
    print(f"rank {rank} after allreduce SUM: x={x.item()}", flush=True)

    # average
    x_avg = x / world
    dist.barrier()
    print(f"rank {rank} average={x_avg.item():.4f}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=8 dist_allreduce.py
```

---

## Lab 3.3.4 — Point-to-point send/recv (important note)
Point-to-point ops exist, but **NCCL is primarily for collectives**; in practice, for educational send/recv you often use **Gloo** (CPU tensors), because it’s simpler and universally supported.

So here we do a CPU send/recv with Gloo (still teaches ranks correctly). PyTorch supports multiple backends via `init_process_group`. 

Create `dist_send_recv_gloo.py`:

```bash
cat > dist_send_recv_gloo.py <<'PY'
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()

    if rank == 0:
        x = torch.tensor([10, 20, 30], dtype=torch.int64)
        dist.send(x, dst=1)
        print("rank0 sent:", x.tolist(), flush=True)
    elif rank == 1:
        y = torch.empty(3, dtype=torch.int64)
        dist.recv(y, src=0)
        print("rank1 recv:", y.tolist(), flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=2 dist_send_recv_gloo.py
```

---

# 3.4 Deadlocks & Debugging (this is where most people suffer)

## The #1 rule: collectives must match
If rank 0 calls `all_reduce()` but rank 1 calls `broadcast()`, you hang forever.
NCCL also requires matching tensor sizes/dtypes across ranks for a collective. 

### Debug drill: intentionally hang with mismatched collective
Create `dist_hang_demo.py`:

```bash
cat > dist_hang_demo.py <<'PY'
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    x = torch.ones(4, device=device)

    if rank == 0:
        dist.all_reduce(x)      # rank0 does allreduce
    else:
        dist.broadcast(x, src=0)  # rank1 does broadcast -> mismatch -> hang

    print("You will never see this on both ranks.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
PY
```

Run:
```bash
torchrun --nproc_per_node=2 dist_hang_demo.py
```

It will hang. Kill it:
```bash
pkill -f dist_hang_demo.py || true
pkill -f torchrun || true
```

That hang is the same class of bug you’ll see later with DDP/FSDP when ranks diverge.

---

## NCCL debugging knobs (keep these ready)
When something fails or hangs:
```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL torchrun --nproc_per_node=8 dist_allreduce.py
```

This prints NCCL init + collective info and helps you see what it thinks the topology is doing.

---

# 3.5 Module 3 pass criteria
You’re done when you can:

1) Explain and demonstrate:
- rank, local_rank, world_size
- what `torchrun --nproc_per_node=8` actually does

2) Run successfully on 8 GPUs:
- `dist_hello.py`
- `dist_broadcast.py`
- `dist_allreduce.py`

3) Explain (and reproduce) a deadlock:
- mismatched collective order causes hang

---

