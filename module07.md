# Module 7 — Multi-Node Training (2 Nodes × 8 GPUs = 16 A100s)

## Summary
Everything so far ran on **one node**. Now we cross the network boundary.

- **Theory**: cluster architecture, head/compute nodes, InfiniBand vs Ethernet, all environment variables explained deeply
- **Network inspection**: find interfaces, test connectivity, measure inter-node bandwidth
- **Hands-on**: launch your first 16-GPU training job across 2 nodes
- **Convert**: single-node DDP → multi-node DDP (the changes are smaller than you think)
- **Debug drills**: firewall blocks, hostname resolution failures, wrong `MASTER_ADDR`, wrong network interface, rank stall

---

## 7.1 Theory — Cluster Architecture

### Physical layout of a 2-node cluster

```
┌─────────────────────────────────────────────────────────────────┐
│                        HEAD NODE                                │
│  (login node / job scheduler / not used for training itself)    │
│  - you SSH into this machine                                    │
│  - you submit jobs from here                                    │
│  - runs SLURM daemon (slurmctld)                                │
└───────────────────────┬─────────────────────────────────────────┘
                        │ management network (SSH, job control)
           ─────────────┴─────────────
           │                         │
┌──────────▼──────────┐   ┌──────────▼──────────┐
│    COMPUTE NODE 0   │   │    COMPUTE NODE 1   │
│    (scn38-mn or     │   │    (scn38-mn etc.)  │
│     similar)        │   │                     │
│  8× A100 SXM4 40GB  │   │  8× A100 SXM4 40GB  │
│  NVSwitch fabric    │   │  NVSwitch fabric    │
│  SLURM slurmd       │   │  SLURM slurmd       │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           └──────────┬──────────────┘
              High-speed network
              (InfiniBand HDR or
               Ethernet 100GbE)
```

### Head node vs Compute node
- **Head node**: where you SSH, write scripts, submit jobs. Usually has no GPUs (or few).
- **Compute nodes**: where actual training runs. Never SSH directly during jobs — SLURM handles placement.
- The head node **is NOT rank 0** in your training job. One of the compute nodes is.

### What changes going multi-node
On single node, all 8 processes can find each other via shared memory + NVLink.

On multi-node, **processes on different nodes have no shared memory**. They must communicate over the network. This adds:

1. A **rendezvous mechanism** (how processes find each other at startup)
2. A **network transport** for NCCL (InfiniBand or Ethernet)
3. More environment variables to configure correctly
4. Much higher latency for cross-node communication vs NVLink

---

## 7.2 Theory — Network Hardware

### InfiniBand (IB)
Dedicated high-performance network designed for HPC and AI:

```
Specs for common IB generations:
  HDR  (High Data Rate): 200 Gb/s per port  (~25 GB/s)
  HDR100:                100 Gb/s per port  (~12.5 GB/s)
  NDR:                   400 Gb/s per port  (~50 GB/s)
```

Key properties:
- **RDMA** (Remote Direct Memory Access): CPU is bypassed for data transfer. GPU memory → IB NIC → remote GPU memory without CPU involvement.
- **Very low latency**: ~1–2 microseconds
- **High bandwidth**: 25–50 GB/s per link

NCCL uses InfiniBand via **ibverbs** (IB verbs API) or via **UCX**.

### Ethernet (RoCE)
RoCE = RDMA over Converged Ethernet. Runs RDMA semantics over standard Ethernet hardware:

```
100GbE = ~12.5 GB/s
200GbE = ~25 GB/s
400GbE = ~50 GB/s
```

Cheaper than InfiniBand but higher latency, more sensitive to network congestion.

### NCCL transport selection
NCCL automatically selects transport:

```
Priority order (highest to lowest):
1. NVLink (intra-node, same node)
2. PCI P2P (intra-node, no NVLink)
3. InfiniBand (inter-node, IB detected)
4. RoCE / shared memory (inter-node, Ethernet)
5. TCP sockets (fallback, slow)
```

For cross-node traffic, you want IB or RoCE, **not TCP**.

### Why cross-node is slower than intra-node

```
Intra-node (NVLink):  ~300 GB/s
Cross-node (IB HDR):  ~25 GB/s
Cross-node (100GbE):  ~12.5 GB/s

Ratio: NVLink is 10–25× faster than inter-node
```

This means in multi-node training, **cross-node AllReduce is the bottleneck**, not computation.

This is why pipeline parallelism (Module 14) tries to minimize cross-node communication.

---

## 7.3 Theory — Every Environment Variable Explained

This is the complete reference. You must know all of these.

### torchrun sets these automatically:

| Variable | Set by | Meaning | Example |
|----------|--------|---------|---------|
| `RANK` | torchrun | Global rank of this process across all nodes | 0–15 for 16 GPUs |
| `LOCAL_RANK` | torchrun | GPU index on **this node** | 0–7 always |
| `WORLD_SIZE` | torchrun | Total number of processes | 16 |
| `MASTER_ADDR` | **YOU** | Hostname or IP of node running rank 0 | `compute-node-0` |
| `MASTER_PORT` | **YOU** | Port for rendezvous | `29500` |
| `NODE_RANK` | **YOU** (or torchrun with --node_rank) | Index of this node | 0 or 1 |

### Why MASTER_ADDR and MASTER_PORT matter so much
All processes must "meet" somewhere before training starts. This is the **rendezvous**:

```
All 16 processes connect to:
  MASTER_ADDR:MASTER_PORT

Rank 0 acts as the rendezvous server.
All other ranks connect to it.
Once all ranks have connected → training begins.
```

If `MASTER_ADDR` is wrong → no process can connect → all hang at init.
If `MASTER_PORT` is blocked by firewall → same hang.

### RANK vs LOCAL_RANK (must be crystal clear)

```
2 nodes, 8 GPUs each:

Node 0:
  Process 0: RANK=0,  LOCAL_RANK=0,  NODE_RANK=0
  Process 1: RANK=1,  LOCAL_RANK=1,  NODE_RANK=0
  Process 2: RANK=2,  LOCAL_RANK=2,  NODE_RANK=0
  Process 3: RANK=3,  LOCAL_RANK=3,  NODE_RANK=0
  Process 4: RANK=4,  LOCAL_RANK=4,  NODE_RANK=0
  Process 5: RANK=5,  LOCAL_RANK=5,  NODE_RANK=0
  Process 6: RANK=6,  LOCAL_RANK=6,  NODE_RANK=0
  Process 7: RANK=7,  LOCAL_RANK=7,  NODE_RANK=0

Node 1:
  Process 8:  RANK=8,  LOCAL_RANK=0,  NODE_RANK=1
  Process 9:  RANK=9,  LOCAL_RANK=1,  NODE_RANK=1
  Process 10: RANK=10, LOCAL_RANK=2,  NODE_RANK=1
  Process 11: RANK=11, LOCAL_RANK=3,  NODE_RANK=1
  Process 12: RANK=12, LOCAL_RANK=4,  NODE_RANK=1
  Process 13: RANK=13, LOCAL_RANK=5,  NODE_RANK=1
  Process 14: RANK=14, LOCAL_RANK=6,  NODE_RANK=1
  Process 15: RANK=15, LOCAL_RANK=7,  NODE_RANK=1
```

**Rule**: always use `LOCAL_RANK` to set CUDA device. Never use `RANK` for that.

```python
# CORRECT
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# WRONG (crashes on node 1: tries to use GPU 8..15 which don't exist)
torch.cuda.set_device(int(os.environ["RANK"]))
```

---

## 7.4 Pre-Flight Checks — Do These Before Any Multi-Node Job

### Step A — Find your node names and IPs

On head node:
```bash
# See all nodes
sinfo -N -l 2>/dev/null || echo "no SLURM, using manual"

# Get hostname of current node
hostname
hostname -I

# Get all node IPs (if /etc/hosts is configured)
cat /etc/hosts | grep compute
```

Find the IP of the node you want as MASTER:
```bash
# On node 0 (the one that will be rank 0)
hostname -I | awk '{print $1}'
```

### Step B — Test SSH between nodes (passwordless SSH required)
```bash
# From node 0, SSH to node 1 without password
ssh <node1_hostname> "hostname && nvidia-smi -L"
```

If this requires a password, torchrun cannot launch remote processes.

Fix (generate + copy SSH key):
```bash
ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
ssh-copy-id <node1_hostname>
```

### Step C — Test network connectivity + latency
```bash
# Basic ping
ping -c 5 <node1_hostname>

# Check which interface is InfiniBand
ibstat 2>/dev/null || echo "ibstat not found"

# List all network interfaces
ip link show
ip addr show

# Test bandwidth between nodes (if iperf3 available)
# On node 1 (receiver):
iperf3 -s -p 5201

# On node 0 (sender):
iperf3 -c <node1_hostname> -p 5201 -t 10
```

### Step D — Check NCCL can see InfiniBand
```bash
# Check IB devices visible to NCCL
ls /dev/infiniband/ 2>/dev/null || echo "no IB device files"

# Check IB port status
ibstat 2>/dev/null | grep -A 5 "Port 1"

# Check RDMA interfaces
rdma link 2>/dev/null || echo "rdma tool not available"
```

### Step E — Test port availability
```bash
# Check if port 29500 is free on node 0
ss -tulpn | grep 29500

# Or with netstat
netstat -tulpn 2>/dev/null | grep 29500
```

If blocked, choose a different port.

### Step F — Check firewall rules
```bash
# Ubuntu ufw
ufw status

# iptables
iptables -L INPUT -n | grep DROP

# Test if port is reachable from node 1 to node 0
# On node 0:
nc -l -p 29500 &
# On node 1:
nc -zv <node0_ip> 29500
```

---

## 7.5 Network Interface Selection for NCCL

NCCL must use the **right** network interface (the fast one, not the management ethernet).

### Find all interfaces:
```bash
ip addr show
```

Typical server has:
```
lo          127.0.0.1    loopback (ignore)
eth0        10.0.0.1     management (slow, 1GbE)
eth1        192.168.1.1  storage network (ignore)
ib0         10.10.0.1    InfiniBand (USE THIS)
```

### Tell NCCL which interface to use:
```bash
# Force NCCL to use InfiniBand interface
export NCCL_IB_DISABLE=0        # make sure IB is enabled
export NCCL_SOCKET_IFNAME=ib0   # socket fallback interface

# OR: specify interface by name
export NCCL_SOCKET_IFNAME=eth0   # if using ethernet

# Disable IB (force Ethernet, for testing)
export NCCL_IB_DISABLE=1
```

### Check what NCCL actually chose:
```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
torchrun ... 2>&1 | grep -i "NET\|IB\|socket"
```

Look for lines like:
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB
NCCL INFO NET/Socket : Using [0]ib0
```

---

## 7.6 Hands-on — First Multi-Node Test (no neural networks)

### Create `multinode_hello.py`:
```bash
cat > multinode_hello.py <<'PY'
"""
Multi-node hello world.
Run this on 2 nodes to verify:
- All ranks spawn correctly
- Each rank knows its RANK, LOCAL_RANK, NODE_RANK
- AllReduce works cross-node
"""
import os, socket, time
import torch
import torch.distributed as dist


def main():
    # Read all env vars
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank  = int(os.environ.get("GROUP_RANK", 0))  # torchrun sets GROUP_RANK
    master_addr = os.environ.get("MASTER_ADDR", "?")
    master_port = os.environ.get("MASTER_PORT", "?")
    hostname    = socket.gethostname()

    # Init process group
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Print rank info (ordered by rank)
    for r in range(world_size):
        dist.barrier()
        if rank == r:
            print(
                f"rank={rank:2d}/{world_size} "
                f"local_rank={local_rank} "
                f"node_rank={node_rank} "
                f"host={hostname} "
                f"gpu={torch.cuda.get_device_name(local_rank)} "
                f"master={master_addr}:{master_port}",
                flush=True
            )
        dist.barrier()

    # AllReduce test: each rank contributes its rank value
    x = torch.tensor([float(rank)], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    expected = sum(range(world_size))
    if rank == 0:
        print(f"\nAllReduce SUM = {x.item()} (expected {expected})")
        assert abs(x.item() - expected) < 1e-3, "AllReduce gave wrong result!"
        print("AllReduce: PASSED")

    # Broadcast test: rank 0 broadcasts a tensor
    y = torch.zeros(4, device=device)
    if rank == 0:
        y = torch.tensor([10., 20., 30., 40.], device=device)
    dist.broadcast(y, src=0)

    for r in range(world_size):
        dist.barrier()
        if rank == r:
            print(f"rank={rank} broadcast received: {y.tolist()}", flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

---

## 7.7 How to Launch Multi-Node Jobs (3 methods)

### Method 1: torchrun (manual, recommended for learning)

You need **TWO terminals open** (one per node), or use a launch script.

**On Node 0 (rank 0 lives here):**
```bash
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"

torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  multinode_hello.py
```

**On Node 1 (at exactly the same time):**
```bash
export MASTER_ADDR=<node0_ip_or_hostname>   # must match what node 0 used
export MASTER_PORT=29500

torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  multinode_hello.py
```

Both commands must run **simultaneously**. They block waiting for each other.

---

### Method 2: Launch script (one command, SSH-based)

Create `launch_multinode.sh`:
```bash
cat > launch_multinode.sh <<'SCRIPT'
#!/bin/bash
#
# Usage: bash launch_multinode.sh <script.py> [args...]
#
# Edit these:
NODE0="scn38-mn"           # replace with your node 0 hostname
NODE1="scn38-w1"           # replace with your node 1 hostname
MASTER_ADDR=$NODE0
MASTER_PORT=29500
NPROC_PER_NODE=8
NNODES=2
SCRIPT=$1
shift
ARGS="$@"

echo "Launching on $NODE0 and $NODE1"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Script: $SCRIPT $ARGS"

# Launch node 1 in background via SSH
ssh $NODE1 "
  cd $(pwd) &&
  source /root/ddl_course/venv/bin/activate &&
  export MASTER_ADDR=$MASTER_ADDR &&
  export MASTER_PORT=$MASTER_PORT &&
  torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $SCRIPT $ARGS
" &
NODE1_PID=$!

# Launch node 0 in foreground
source /root/ddl_course/venv/bin/activate
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $SCRIPT $ARGS

# Wait for node 1 to finish
wait $NODE1_PID
echo "All nodes done."
SCRIPT

chmod +x launch_multinode.sh
```

Use:
```bash
bash launch_multinode.sh multinode_hello.py
```

---

### Method 3: torchrun with elastic launch (rendezvous via etcd/c10d)

For production, torchrun supports a **rendezvous backend** so you don't need to manually set `--node_rank`:

```bash
# On ALL nodes simultaneously (no --node_rank needed):
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_id=job_001 \
  multinode_hello.py
```

With `c10d` backend, torchrun handles rank assignment automatically. This is the modern recommended way.

---

## 7.8 Multi-Node AllReduce Bandwidth Benchmark

```bash
cat > multinode_allreduce_bench.py <<'PY'
"""
Cross-node AllReduce bandwidth benchmark.
Compare with Module 6 single-node results to see the network cost.
"""
import os, time, socket
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    hostname   = socket.gethostname()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    sizes_mb = [1, 4, 16, 64, 128, 256, 512, 1024]
    iters    = 30
    warmup   = 10

    if rank == 0:
        print(f"\nworld_size={world_size}")
        print(f"{'Size':>8}  {'avg_ms':>8}  {'BW_GBs':>9}  {'note':>10}")
        print("-" * 45)

    for smb in sizes_mb:
        numel = (smb * 1024 * 1024) // 4
        x = torch.ones(numel, device=device, dtype=torch.float32)

        for _ in range(warmup):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()

        ms  = (t1 - t0) * 1000 / iters
        bw  = 2 * smb * (world_size - 1) / world_size / (ms / 1000) / 1000

        if rank == 0:
            note = "latency" if ms < 1 else "bandwidth"
            print(f"{smb:>6}MB  {ms:>8.3f}  {bw:>9.2f}  {note:>10}",
                  flush=True)

    # Cross-node detection: which ranks are on which node
    if rank == 0:
        print(f"\nrank=0 hostname: {hostname}")
    if rank == world_size // 2:
        print(f"rank={world_size//2} hostname: {hostname}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Launch:
```bash
bash launch_multinode.sh multinode_allreduce_bench.py
```

Compare results to single-node Module 6 results. You'll see:
- Cross-node bandwidth is significantly lower
- Latency is higher
- This is the "multi-node tax"

---

## 7.9 Multi-Node DDP Training (full CIFAR-10 on 16 GPUs)

This is the Module 4 + Module 5 DDP training script, unchanged except for how you launch it.

```bash
cat > train_cifar10_multinode.py <<'PY'
"""
Multi-node DDP training on CIFAR-10.
Identical to single-node DDP — only the launch command changes.
"""
import os, time, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms
import socket


# ─── Sampler ──────────────────────────────────────────────────────────────────
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
    hostname   = socket.gethostname()

    torch.cuda.set_device(local_rank)
    device  = torch.device("cuda", local_rank)
    is_main = (rank == 0)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)

    if is_main:
        print(f"world_size={world_size}  "
              f"nodes={world_size//8}  "
              f"gpus_per_node=8")
        print(f"rank0 host: {hostname}")

    # ── Data ─────────────────────────────────────────────────────────────────
    # IMPORTANT: only rank 0 downloads; all ranks must access shared filesystem
    # In multi-node: data_dir must be on a shared NFS/Lustre mount
    data_dir = "./data"  # adjust to shared path in your cluster

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
        datasets.CIFAR10(data_dir, train=True,  download=True)
        datasets.CIFAR10(data_dir, train=False, download=True)
    dist.barrier()  # wait for download before other ranks try to read

    train_ds = datasets.CIFAR10(data_dir, train=True,  transform=tfm_train)
    test_ds  = datasets.CIFAR10(data_dir, train=False, transform=tfm_test)

    train_sampler = MyDistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=0, drop_last=True,
    )
    test_sampler = MyDistributedSampler(
        test_ds, num_replicas=world_size, rank=rank,
        shuffle=False, seed=0, drop_last=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=128,           # per-GPU batch
        sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=128,
        sampler=test_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler    = torch.cuda.amp.GradScaler()

    # ── Train ────────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(5):
        model.train()
        train_sampler.set_epoch(epoch)

        t_epoch = time.time()
        total_loss = 0.0
        n_steps    = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_steps    += 1

        scheduler.step()

        # ── Eval ─────────────────────────────────────────────────────────
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

        epoch_s = time.time() - t_epoch
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3

        if is_main:
            eff_batch = 128 * world_size
            imgs_s    = (len(train_ds) // eff_batch * eff_batch) / epoch_s
            print(
                f"epoch={epoch+1}  "
                f"loss={total_loss/n_steps:.4f}  "
                f"acc={correct.item()/total.item()*100:.2f}%  "
                f"time={epoch_s:.1f}s  "
                f"imgs/s={imgs_s:.0f}  "
                f"peak={peak_gb:.2f}GB",
                flush=True
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Launch on 2 nodes:
```bash
bash launch_multinode.sh train_cifar10_multinode.py
```

---

## 7.10 Shared Filesystem (critical multi-node detail)

In multi-node training, **all nodes must access the same data**. Options:

```
Option 1: NFS (Network File System)
  - Common in small clusters
  - Simple, good for dataset reads
  - Slow for checkpoints (write bandwidth limited)

Option 2: Lustre / GPFS
  - High-performance parallel filesystem
  - Used in large HPC clusters
  - Fast parallel reads and writes

Option 3: Local SSD + pre-copy
  - Copy dataset to each node's local disk before training
  - Fastest data loading
  - Extra step required
```

Check if you have a shared mount:
```bash
df -h
mount | grep nfs
mount | grep lustre
```

If data is **not** on shared storage:
- Download on node 0 only (we already do this with `if is_main: download=True`)
- But then node 1 can't read it
- Fix: copy to each node first:
  ```bash
  # From node 0:
  rsync -avz ./data/ node1:/same/path/data/
  ```

---

## 7.11 Debug Drills — Multi-Node Failures

### Drill 1 — Wrong MASTER_ADDR (most common mistake)
```bash
# Intentionally set wrong IP
MASTER_ADDR=999.999.999.999 \
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=999.999.999.999 --master_port=29500 \
  multinode_hello.py
```

All ranks will hang at `dist.init_process_group()` trying to connect.

Timeout error looks like:
```
RuntimeError: [tcp] connect to MASTER_ADDR:29500 failed after 3 retries
```

Fix: always verify `MASTER_ADDR` is reachable from all nodes:
```bash
# On node 1, test connectivity to node 0
nc -zv $MASTER_ADDR $MASTER_PORT
```

---

### Drill 2 — Firewall blocking MASTER_PORT
```bash
# On node 0: block port 29500 temporarily
iptables -A INPUT -p tcp --dport 29500 -j DROP

# Now try launching → node 1 can't connect to node 0
bash launch_multinode.sh multinode_hello.py

# All ranks on node 1 hang at init

# Fix: open the port
iptables -D INPUT -p tcp --dport 29500 -j DROP
```

---

### Drill 3 — NODE_RANK wrong (both nodes think they're rank 0)
```bash
# Intentionally set node_rank=0 on BOTH nodes
# Node 0: --node_rank=0
# Node 1: --node_rank=0  ← WRONG

# Both nodes try to own ranks 0-7
# Rank collisions → undefined behavior, usually hangs or crashes
```

Fix: node_rank must be **unique per node** (0, 1, 2, ...).

---

### Drill 4 — Wrong network interface (NCCL uses slow management ethernet)
```bash
# Force NCCL to use management ethernet instead of IB
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=eth0 \
bash launch_multinode.sh multinode_allreduce_bench.py
```

You'll see much lower bandwidth. Compare:

| Interface | Bandwidth |
|-----------|-----------|
| eth0 (1GbE management) | ~0.1 GB/s |
| eth1 (100GbE) | ~12 GB/s |
| ib0 (IB HDR) | ~25 GB/s |

Fix: explicitly set the correct interface:
```bash
export NCCL_SOCKET_IFNAME=ib0    # or whatever your IB interface is
export NCCL_IB_DISABLE=0
```

---

### Drill 5 — One node dies during training (partial hang)
Simulate:
```bash
# While training is running on 2 nodes:
# Kill all processes on node 1
ssh node1 "pkill -9 -f train_cifar10_multinode.py"
```

Node 0 will hang at the next AllReduce (waiting for node 1 processes that no longer exist). After NCCL timeout (~30 min default), you'll see:
```
NCCL error: unhandled system error
```

Fix for production: set a reasonable NCCL timeout:
```bash
export NCCL_TIMEOUT=300   # 5 minutes
```

And implement checkpointing + job restart (covered in Module 17).

---

## 7.12 Complete NCCL Environment Variable Reference for Multi-Node

```bash
# ── Network selection ─────────────────────────────────────────
export NCCL_IB_DISABLE=0             # 1=disable InfiniBand
export NCCL_SOCKET_IFNAME=ib0        # network interface for sockets
export NCCL_IB_HCA=mlx5_0           # which IB HCA to use

# ── Debugging ─────────────────────────────────────────────────
export NCCL_DEBUG=INFO               # INFO, WARN, or VERSION
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL

# ── Performance ───────────────────────────────────────────────
export NCCL_ALGO=Ring                # Ring, Tree, CollNet
export NCCL_PROTO=Simple             # Simple, LL, LL128
export NCCL_BUFFSIZE=4194304         # 4MB comm buffer

# ── Timeout ───────────────────────────────────────────────────
export NCCL_TIMEOUT=1800             # seconds (default ~30min)

# ── P2P ───────────────────────────────────────────────────────
export NCCL_P2P_DISABLE=0            # 1=disable P2P (force socket)
export NCCL_P2P_LEVEL=NVL            # NVL, PIX, PXB, NODE, SYS

# ── SHM (shared memory, intra-node) ───────────────────────────
export NCCL_SHM_DISABLE=0            # 1=disable shared mem transport
```

---

## 7.13 Measuring Multi-Node Scaling Efficiency

```bash
cat > multinode_scaling_report.py <<'PY'
"""
Reports scaling metrics for multi-node training.
Run this after training to understand efficiency.
"""
import os, time, socket
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    hostname   = socket.gethostname()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Collect hostname per rank
    is_main = (rank == 0)

    # AllReduce bandwidth test at gradient-typical size (25MB = DDP bucket)
    for size_mb in [1, 25, 100, 512]:
        numel = (size_mb * 1024 * 1024) // 4
        x = torch.ones(numel, device=device)

        # warmup
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()

        iters = 30
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        ms = (time.perf_counter() - t0) * 1000 / iters

        bw = 2 * size_mb * (world_size - 1) / world_size / (ms / 1000) / 1000

        if is_main:
            print(f"allreduce {size_mb:4d}MB: {ms:8.3f}ms  {bw:7.2f} GB/s",
                  flush=True)

    # Node topology report
    for r in range(min(world_size, 4)):  # print first 4 ranks
        dist.barrier()
        if rank == r:
            print(f"rank={rank} node={hostname}", flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY
```

Run:
```bash
bash launch_multinode.sh multinode_scaling_report.py
```

---

## 7.14 Module 7 Pass Criteria

You're done when you can:

1) **Explain clearly**:
   - what `RANK`, `LOCAL_RANK`, `NODE_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` each mean
   - why you use `LOCAL_RANK` for CUDA device (not `RANK`)
   - why cross-node AllReduce is slower than intra-node
   - what RDMA is and why it matters for NCCL

2) **Run successfully on 2 nodes**:
   - `bash launch_multinode.sh multinode_hello.py` → all 16 ranks print correctly
   - `bash launch_multinode.sh multinode_allreduce_bench.py` → bandwidth table
   - `bash launch_multinode.sh train_cifar10_multinode.py` → training runs 5 epochs

3) **Debug and fix**:
   - wrong `MASTER_ADDR` → hang at init
   - firewall block → connection refused
   - wrong `--node_rank` → rank collision

4) **Fill in cross-node vs intra-node bandwidth comparison**:

| Scope | Mechanism | Bandwidth |
|-------|-----------|-----------|
| Intra-node | NVLink / NVSwitch | ~300 GB/s |
| Cross-node | InfiniBand / Ethernet | ~25 GB/s |
| Ratio | | ~12× slower cross-node |

---

