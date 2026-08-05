# Module 8 — SLURM (Very Detailed)

## Summary
Every real cluster uses SLURM. You can't run production multi-node training without understanding it deeply.

- **Theory**: resource manager vs scheduler, partitions, jobs, steps, tasks, GRES, CPU/GPU affinity, memory reservation
- **Every SLURM variable explained**: what each one means, when it's set, how to use it
- **All CLI tools**: `sbatch`, `salloc`, `srun`, `scancel`, `squeue`, `sinfo`, `sacct`, `scontrol`
- **Hands-on**: write 6 production SLURM scripts from scratch
- **Integration**: wrap your multi-node DDP training inside SLURM
- **Debug drills**: job stuck in queue, wrong GPU count, OOM, node failure

---

## 8.1 Theory — What SLURM Actually Is

### Resource Manager vs Scheduler (two jobs, one tool)

```
Resource Manager:
  - Tracks what resources exist on the cluster
    (nodes, CPUs, GPUs, memory, network interfaces)
  - Allocates those resources to jobs exclusively
  - Prevents two jobs from using the same GPU simultaneously

Scheduler:
  - Decides which queued job runs next
  - Uses priority algorithms (FIFO, fair-share, backfill)
  - Considers: job priority, requested resources, available resources,
    reservation windows, user fairshare limits
```

SLURM does both. When you submit `sbatch job.sh`, SLURM:
1. Parses your resource request
2. Queues your job
3. When resources are free → allocates them → launches your job
4. Monitors the job
5. Releases resources when job finishes

### SLURM Daemon Architecture

```
┌─────────────────────────────────────┐
│          HEAD NODE                  │
│  slurmctld  ← central controller   │
│  - manages job queue                │
│  - makes scheduling decisions       │
│  - communicates with all nodes      │
└──────────────┬──────────────────────┘
               │  SLURM internal network
    ┌──────────┴──────────┐
    │                     │
┌───▼────────┐    ┌───────▼────────┐
│ COMPUTE 0  │    │  COMPUTE 1     │
│  slurmd    │    │   slurmd       │
│ - local    │    │  - local       │
│   agent    │    │    agent       │
│ - launches │    │  - launches    │
│   tasks    │    │    tasks       │
└────────────┘    └────────────────┘
```

`slurmctld` = controller daemon (head node)
`slurmd` = worker daemon (every compute node)

---

## 8.2 Theory — Core SLURM Concepts (every term defined)

### Node
A physical machine in the cluster.
```bash
sinfo -N    # list all nodes
```

### Partition
A logical grouping of nodes with shared policies:
- time limits
- access restrictions (who can use it)
- priorities

```
Typical partitions:
  gpu        → general GPU jobs
  gpu_large  → jobs needing full nodes
  debug      → short jobs, fast queue
  interactive→ interactive sessions
```

Think of partitions like "queues" in other systems.

### Job
A unit of work submitted to SLURM.
- Gets a unique `SLURM_JOB_ID`
- Has a state: PENDING → RUNNING → COMPLETED (or FAILED/TIMEOUT/CANCELLED)

### Job Step
A subprocess within a job, launched by `srun`.
A single job can have multiple steps running sequentially or in parallel.

```
Job 12345
  └── Step 0   (srun training script)
  └── Step 1   (srun evaluation)
  └── Step 2   (srun cleanup)
```

### Task
A single process within a step.
`--ntasks=16` means 16 processes total.
`--ntasks-per-node=8` means 8 processes per node.

For DDP training: **1 task per GPU**.

### GRES (Generic Resource)
Resources that aren't CPUs or memory — like GPUs, FPGAs, etc.

```bash
#SBATCH --gres=gpu:8          # 8 GPUs (any type)
#SBATCH --gres=gpu:a100:8     # 8 A100 GPUs specifically
```

### CPU Affinity
Binding processes to specific CPU cores.
For GPU training: bind the DataLoader workers to CPUs on the same NUMA node as the GPU they feed.

### GPU Affinity
Binding processes to specific GPUs.
SLURM + NCCL handle this via `LOCAL_RANK` and `CUDA_VISIBLE_DEVICES`.

### Memory Reservation
```bash
#SBATCH --mem=256G            # total memory per node
#SBATCH --mem-per-cpu=8G      # memory per CPU
#SBATCH --mem-per-gpu=40G     # memory per GPU
```

---

## 8.3 Theory — Every SLURM Environment Variable Explained

When your job runs, SLURM injects these into every process's environment.
You **must** know all of these for multi-node DDP + SLURM integration.

### Node and Allocation

| Variable | Meaning | Example |
|----------|---------|---------|
| `SLURM_JOB_ID` | Unique job ID | `12345` |
| `SLURM_JOB_NAME` | Job name from `--job-name` | `gpt_training` |
| `SLURM_JOB_NODELIST` | List of allocated nodes | `node[01-02]` or `node01,node02` |
| `SLURM_JOB_NUM_NODES` | Number of nodes | `2` |
| `SLURM_NODEID` | Index of **this** node in the job | `0` or `1` |
| `SLURM_JOB_PARTITION` | Partition job is running in | `gpu` |
| `SLURM_SUBMIT_DIR` | Directory where sbatch was called | `/home/user/jobs` |
| `SLURM_SUBMIT_HOST` | Host where sbatch was called | `login-node` |

### Task and Process

| Variable | Meaning | Example |
|----------|---------|---------|
| `SLURM_NTASKS` | Total number of tasks across all nodes | `16` |
| `SLURM_NTASKS_PER_NODE` | Tasks per node | `8` |
| `SLURM_PROCID` | Global task rank (0 to NTASKS-1) | `0..15` |
| `SLURM_LOCALID` | Local task rank on this node (0 to NTASKS_PER_NODE-1) | `0..7` |
| `SLURM_NODEID` | Node index (0 to NUM_NODES-1) | `0` or `1` |

### CPU and Memory

| Variable | Meaning | Example |
|----------|---------|---------|
| `SLURM_CPUS_PER_TASK` | CPUs allocated per task | `8` |
| `SLURM_CPUS_ON_NODE` | Total CPUs on this node | `64` |
| `SLURM_MEM_PER_NODE` | Memory per node in MB | `262144` |
| `SLURM_MEM_PER_CPU` | Memory per CPU in MB | `8192` |

### GPU

| Variable | Meaning | Example |
|----------|---------|---------|
| `SLURM_GPUS` | Total GPUs across all nodes | `16` |
| `SLURM_GPUS_PER_NODE` | GPUs per node | `8` |
| `SLURM_GPUS_PER_TASK` | GPUs per task | `1` |
| `CUDA_VISIBLE_DEVICES` | GPUs visible to this process (set by SLURM) | `0,1,2,3,4,5,6,7` |

### Step (within a job)

| Variable | Meaning | Example |
|----------|---------|---------|
| `SLURM_STEP_ID` | Step number | `0` |
| `SLURM_STEP_NUM_TASKS` | Tasks in this step | `16` |
| `SLURM_STEP_NODELIST` | Nodes in this step | `node[01-02]` |

### How these map to DDP variables

```
SLURM → DDP/torchrun mapping:

SLURM_PROCID          →  RANK  (global rank)
SLURM_LOCALID         →  LOCAL_RANK
SLURM_NODEID          →  NODE_RANK
SLURM_NTASKS          →  WORLD_SIZE
SLURM_JOB_NODELIST[0] →  MASTER_ADDR (first node in list)
```

This is exactly what your SLURM job script must do: read SLURM vars and set DDP vars.

---

## 8.4 All CLI Tools (hands-on reference)

### `sinfo` — cluster state

```bash
# Basic partition + node status
sinfo

# Detailed node view
sinfo -N -l

# Show available GPUs
sinfo -o "%20N %10c %10m %25f %10G"

# Show specific partition
sinfo -p gpu

# Watch live (refresh every 5s)
watch -n 5 sinfo
```

Output columns:
```
PARTITION  AVAIL  TIMELIMIT  NODES  STATE  NODELIST
gpu        up     7-00:00:0  2      idle   node[01-02]
debug      up     1:00:00    1      idle   node03
```

States:
- `idle` → free, ready to accept jobs
- `alloc` → fully allocated to running jobs
- `mix` → partially allocated
- `drain` → admin disabled node
- `down` → node is down

---

### `squeue` — job queue

```bash
# All jobs in queue
squeue

# Your jobs only
squeue -u $USER

# Specific job
squeue -j 12345

# Detailed format
squeue -o "%.10i %.20j %.8u %.8T %.10M %.6D %R"

# Watch live
watch -n 5 "squeue -u $USER"
```

Job states:
```
PD   → PENDING  (waiting for resources)
R    → RUNNING
CG   → COMPLETING (finishing up)
F    → FAILED
TO   → TIMEOUT
CA   → CANCELLED
```

Why is my job PENDING?
```bash
# Shows reason why job is waiting
squeue -j 12345 -o "%i %T %r"
```

Common pending reasons:
```
Resources         → waiting for nodes to free up
Priority          → higher priority jobs ahead
QOSMaxJobsPerUser → hit user job limit
ReqNodeNotAvail   → requested node is down
```

---

### `sbatch` — submit batch job

```bash
# Submit a job script
sbatch job.sh

# Override script options from command line
sbatch --nodes=2 --gres=gpu:8 job.sh

# Submit with a specific job name
sbatch --job-name=my_exp job.sh

# Submit with dependency (run after job 12345 finishes)
sbatch --dependency=afterok:12345 eval.sh

# Submit array job (run same script 10 times with different indices)
sbatch --array=0-9 sweep.sh
```

---

### `salloc` — interactive allocation

```bash
# Get 2 nodes with 8 GPUs each for 2 hours
salloc --nodes=2 --ntasks-per-node=8 --gres=gpu:8 \
       --time=2:00:00 --partition=gpu

# Once allocated, you get a shell with SLURM env vars set
# Then run torchrun manually:
srun --ntasks=16 torchrun ...
```

Good for debugging: you get an interactive session on real nodes.

---

### `srun` — run tasks inside a job

```bash
# Inside sbatch script: run 16 tasks (1 per GPU across 2 nodes)
srun --ntasks=16 --ntasks-per-node=8 python train.py

# Run a command on a specific node
srun --nodelist=node01 hostname

# Run interactively (outside a job, allocates resources)
srun --nodes=1 --gres=gpu:1 --pty bash
```

`srun` is also how SLURM launches your torchrun command in production (we'll see this in the scripts below).

---

### `scancel` — cancel jobs

```bash
# Cancel a specific job
scancel 12345

# Cancel all your jobs
scancel -u $USER

# Cancel all pending jobs only
scancel -u $USER -t PENDING

# Cancel a job step
scancel 12345.0
```

---

### `sacct` — job accounting (past jobs)

```bash
# Your recent jobs
sacct -u $USER --format=JobID,JobName,State,Elapsed,AllocNodes,AllocGRES

# Specific job details
sacct -j 12345 --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocGRES

# Jobs from last 7 days
sacct -u $USER -S $(date -d "7 days ago" +%Y-%m-%dT%H:%M:%S)
```

---

### `scontrol` — inspect and modify

```bash
# Show job details (very verbose)
scontrol show job 12345

# Show node details
scontrol show node node01

# Show partition details
scontrol show partition gpu

# Hold a job (pause it in queue)
scontrol hold 12345

# Release a held job
scontrol release 12345

# Requeue a failed job
scontrol requeue 12345
```

---

## 8.5 Hands-on — Writing Production SLURM Scripts

### Script 1 — Minimal GPU job (sanity test)

```bash
cat > job_minimal.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --partition=gpu
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err

echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "CPUs/task:     $SLURM_CPUS_PER_TASK"
echo "GPUs:          $SLURM_GPUS_PER_NODE"
echo "CUDA_VISIBLE:  $CUDA_VISIBLE_DEVICES"

# Activate venv
source /root/ddl_course/venv/bin/activate

# Run quick GPU test
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0))
x = torch.randn(1000, 1000, device='cuda')
print('alloc GB:', torch.cuda.memory_allocated()/1024**3)
print('GPU test: PASSED')
"

echo "Job complete."
SLURM
```

Create logs directory and submit:
```bash
mkdir -p logs
sbatch job_minimal.sh

# Watch it
watch -n 2 "squeue -u $USER"

# See output when done
tail -f logs/test_gpu_*.out
```

---

### Script 2 — Single-node 8-GPU DDP job

```bash
cat > job_single_node_ddp.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=cifar10_single_node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/single_node_%j.out
#SBATCH --error=logs/single_node_%j.err

# ── Print job info ──────────────────────────────────────────────────────────
echo "=============================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Nodes:      $SLURM_JOB_NUM_NODES"
echo "Tasks:      $SLURM_NTASKS"
echo "GPUs/node:  $SLURM_GPUS_PER_NODE"
echo "CPUs/task:  $SLURM_CPUS_PER_TASK"
echo "=============================="

# ── Activate environment ────────────────────────────────────────────────────
source /root/ddl_course/venv/bin/activate
cd /root/ddl_course

# ── NCCL settings ───────────────────────────────────────────────────────────
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN              # change to INFO for debugging
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Launch with torchrun ────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=29500 \
    train_cifar10_ddp.py

echo "Training complete."
SLURM
```

Submit:
```bash
sbatch job_single_node_ddp.sh
squeue -u $USER
tail -f logs/single_node_*.out
```

---

### Script 3 — Multi-node 2-node DDP job (the important one)

```bash
cat > job_multinode_ddp.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=cifar10_multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/multinode_%j.out
#SBATCH --error=logs/multinode_%j.err

# ── Print full job info ─────────────────────────────────────────────────────
echo "=============================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NUM_NODES"
echo "Node list:    $SLURM_JOB_NODELIST"
echo "Tasks total:  $SLURM_NTASKS"
echo "Tasks/node:   $SLURM_NTASKS_PER_NODE"
echo "GPUs/node:    $SLURM_GPUS_PER_NODE"
echo "CPUs/task:    $SLURM_CPUS_PER_TASK"
echo "Submit dir:   $SLURM_SUBMIT_DIR"
echo "=============================="

# ── Activate environment ────────────────────────────────────────────────────
source /root/ddl_course/venv/bin/activate
cd $SLURM_SUBMIT_DIR

# ── Get MASTER_ADDR from first node in SLURM nodelist ───────────────────────
# scontrol converts nodelist like "node[01-02]" to "node01 node02"
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=$(echo "$NODES" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Nodes:"
echo "$NODES"
echo "=============================="

# ── NCCL settings ───────────────────────────────────────────────────────────
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0        # adjust to your IB interface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Launch: srun launches one process per task ──────────────────────────────
# SLURM_PROCID  → RANK
# SLURM_LOCALID → LOCAL_RANK
# SLURM_NODEID  → NODE_RANK
# These are set automatically by srun for each task

srun \
    --ntasks=$SLURM_NTASKS \
    --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    python -m torch.distributed.run \
        --nproc_per_node=8 \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_cifar10_multinode.py

echo "=============================="
echo "Training complete."
echo "=============================="
SLURM
```

Submit:
```bash
sbatch job_multinode_ddp.sh

# Monitor
watch -n 3 "squeue -u $USER"

# Stream logs live
tail -f logs/multinode_*.out

# Check both stdout and stderr
tail -f logs/multinode_*.out logs/multinode_*.err
```

---

### Script 4 — Better launcher using torchrun with srun (modern approach)

The cleanest modern pattern: use `srun` to launch one `torchrun` per node:

```bash
cat > job_modern_multinode.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=gpt_multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1      # 1 torchrun per node
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64       # all CPUs for DataLoader workers
#SBATCH --mem=480G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/modern_%j.out
#SBATCH --error=logs/modern_%j.err

source /root/ddl_course/venv/bin/activate
cd $SLURM_SUBMIT_DIR

# ── Get master address ──────────────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "MASTER_ADDR=$MASTER_ADDR"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

# ── NCCL settings ───────────────────────────────────────────────────────────
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1   # don't hang forever on error
export OMP_NUM_THREADS=8

# ── Each srun task is one full node; torchrun handles GPUs within node ──────
srun bash -c "
    torchrun \
        --nproc_per_node=8 \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_cifar10_multinode.py
"

echo "Done."
SLURM
```

Key difference from Script 3:
- `--ntasks-per-node=1` → one srun task per node
- `srun bash -c "torchrun ..."` → torchrun spawns 8 processes internally
- `$SLURM_NODEID` inside `bash -c` gives each node its node_rank

---

### Script 5 — Job Array (hyperparameter sweep)

```bash
cat > job_array_sweep.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=lr_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-4              # 5 jobs: indices 0,1,2,3,4
#SBATCH --output=logs/sweep_%A_%a.out  # %A=job ID, %a=array index
#SBATCH --error=logs/sweep_%A_%a.err

source /root/ddl_course/venv/bin/activate
cd $SLURM_SUBMIT_DIR

# Different LR for each array job
LR_VALUES=(0.001 0.01 0.05 0.1 0.2)
LR=${LR_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Learning rate: $LR"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=$((29500 + SLURM_ARRAY_TASK_ID)) \
    train_cifar10_ddp.py --lr=$LR

echo "Array task $SLURM_ARRAY_TASK_ID done."
SLURM
```

Submit:
```bash
sbatch job_array_sweep.sh

# Monitor all array tasks
squeue -u $USER

# Cancel specific array task
scancel 12345_2    # cancel task index 2

# Cancel all pending array tasks
scancel --state=PENDING 12345
```

---

### Script 6 — Production training with checkpointing + auto-resume

```bash
cat > job_production.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=prod_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/prod_%j.out
#SBATCH --error=logs/prod_%j.err
#SBATCH --signal=SIGUSR1@90      # send signal 90s before timeout

source /root/ddl_course/venv/bin/activate
cd $SLURM_SUBMIT_DIR

mkdir -p checkpoints logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
CHECKPOINT_DIR="./checkpoints/job_${SLURM_JOB_ID}"

export MASTER_ADDR MASTER_PORT
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Master addr:  $MASTER_ADDR"
echo "Checkpoint:   $CHECKPOINT_DIR"
echo "========================================"

# Auto-resume: find latest checkpoint
RESUME_CKPT=""
if ls checkpoints/job_*/epoch_*.pt 2>/dev/null | head -1 > /dev/null; then
    RESUME_CKPT=$(ls -t checkpoints/job_*/epoch_*.pt | head -1)
    echo "Resuming from: $RESUME_CKPT"
fi

srun bash -c "
    torchrun \
        --nproc_per_node=8 \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_cifar10_multinode.py \
        ${RESUME_CKPT:+--resume $RESUME_CKPT} \
        --checkpoint_dir $CHECKPOINT_DIR
"

EXIT_CODE=$?
echo "Training exited with code: $EXIT_CODE"

# Auto-resubmit if timeout (exit code 124)
if [ $EXIT_CODE -eq 124 ]; then
    echo "Timeout detected. Resubmitting..."
    sbatch $0     # resubmit this same script
fi
SLURM
```

---

## 8.6 Utility Script — SLURM + torchrun integration helper

This script is the bridge between SLURM variables and torchrun/DDP variables. Save it and source it in all your jobs:

```bash
cat > slurm_setup.sh <<'BASH'
#!/bin/bash
#
# Source this file inside any SLURM job to set up
# DDP environment variables from SLURM variables.
#
# Usage: source slurm_setup.sh

# ── Get master node ──────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

# ── World size = total GPUs = total tasks ────────────────────────────────────
export WORLD_SIZE=$SLURM_NTASKS

# ── NCCL ────────────────────────────────────────────────────────────────────
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}

# ── DataLoader ──────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ── Print summary ────────────────────────────────────────────────────────────
echo "── SLURM → DDP setup ────────────────────────"
echo "  MASTER_ADDR   = $MASTER_ADDR"
echo "  MASTER_PORT   = $MASTER_PORT"
echo "  WORLD_SIZE    = $WORLD_SIZE"
echo "  SLURM_JOB_ID  = $SLURM_JOB_ID"
echo "  NODES         = $SLURM_JOB_NUM_NODES"
echo "  NODELIST      = $SLURM_JOB_NODELIST"
echo "  NTASKS        = $SLURM_NTASKS"
echo "  NTASKS/NODE   = $SLURM_NTASKS_PER_NODE"
echo "  GPUS/NODE     = $SLURM_GPUS_PER_NODE"
echo "  OMP_THREADS   = $OMP_NUM_THREADS"
echo "─────────────────────────────────────────────"
BASH

chmod +x slurm_setup.sh
```

Use in any job:
```bash
source slurm_setup.sh
```

---

## 8.7 Python — Read SLURM variables inside training script

Add this to any training script for useful job metadata logging:

```bash
cat > slurm_info.py <<'PY'
"""
Read all SLURM variables inside a Python training script.
Call print_slurm_info() at the start of main().
"""
import os


def get_slurm_info() -> dict:
    """Returns dict of all relevant SLURM variables."""
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NODEID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS_PER_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_JOB_PARTITION",
        "SLURM_SUBMIT_DIR",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {v: os.environ.get(v, "NOT_SET") for v in slurm_vars}


def print_slurm_info(rank: int = 0):
    """Print SLURM info. Call only on rank 0."""
    if rank != 0:
        return
    info = get_slurm_info()
    print("\n── SLURM Job Info ─────────────────────────────")
    for k, v in info.items():
        print(f"  {k:<30} = {v}")
    print("───────────────────────────────────────────────\n",
          flush=True)


def slurm_to_ddp_env():
    """
    Map SLURM variables to torchrun/DDP variables.
    Call this BEFORE dist.init_process_group() when
    launching with srun instead of torchrun.
    """
    # Only set if not already set (torchrun sets these)
    if "RANK" not in os.environ:
        os.environ["RANK"]       = os.environ.get("SLURM_PROCID", "0")
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")


if __name__ == "__main__":
    print_slurm_info(rank=0)
    slurm_to_ddp_env()
    print("RANK:", os.environ.get("RANK"))
    print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
    print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))
PY
```

Test standalone:
```bash
python slurm_info.py
```

Test inside a SLURM job:
```bash
cat > job_test_vars.sh <<'SLURM'
#!/bin/bash
#SBATCH --job-name=test_vars
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=0:05:00
#SBATCH --output=logs/vars_%j.out

source /root/ddl_course/venv/bin/activate
srun python slurm_info.py
SLURM

sbatch job_test_vars.sh
```

---

## 8.8 Monitoring Jobs in Depth

### Live GPU utilization during a SLURM job
```bash
# From head node, SSH into the running node
NODE=$(squeue -j <JOBID> -o "%N" --noheader)
ssh $NODE "watch -n 1 gpustat"
```

### Job efficiency report (after job finishes)
```bash
seff <JOBID>
```

Output:
```
Job ID: 12345
State: COMPLETED (exit code 0)
Nodes: 2
Cores per node: 64
CPU Utilized: 10:23:45
CPU Efficiency: 78.32%
Job Wall-clock time: 00:55:23
Memory Utilized: 234.56 GB
Memory Efficiency: 91.23%
```

### Detailed step accounting
```bash
sacct -j 12345 \
  --format=JobID,JobName,State,Elapsed,CPUTime,MaxRSS,MaxVMSize,AllocGRES \
  --units=G
```

### Watch queue position
```bash
# How many jobs ahead of yours
squeue --sort=P -u $USER

# Priority of your job
sprio -j 12345
```

---

## 8.9 Debug Drills

### Drill 1 — Job stuck in PENDING
```bash
# Check reason
squeue -j <JOBID> -o "%i %T %r %R"
```

Common reasons + fixes:

| Reason | Meaning | Fix |
|--------|---------|-----|
| `Resources` | Nodes busy | Wait, or request fewer resources |
| `Priority` | Low priority | Wait or ask admin |
| `ReqNodeNotAvail` | Requested node is down | Remove `--nodelist` constraint |
| `QOSMaxJobsPerUser` | Job limit hit | Cancel old jobs |
| `InvalidQOS` | Wrong partition name | Check `sinfo` for valid partitions |

```bash
# Check which nodes are available
sinfo -N --state=idle

# Check your job's QOS limits
sacctmgr show user $USER withassoc

# Check partition limits
scontrol show partition gpu
```

---

### Drill 2 — CUDA_VISIBLE_DEVICES wrong

SLURM sets `CUDA_VISIBLE_DEVICES` automatically.
If it's wrong, your processes bind to wrong GPUs.

```bash
# Check what SLURM assigned
cat > check_cuda_visible.sh <<'SLURM'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=0:02:00
#SBATCH --output=logs/cuda_vis_%j.out

srun bash -c "echo task=\$SLURM_LOCALID CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
SLURM

sbatch check_cuda_visible.sh
```

You should see each task has a different (or correctly mapped) `CUDA_VISIBLE_DEVICES`.

If wrong, SLURM may be mapping GPUs in unexpected order. Check:
```bash
scontrol show node <nodename> | grep Gres
```

---

### Drill 3 — Out of memory (OOM) in SLURM
Job gets killed with:
```
slurmstepd: error: Exceeded job memory limit
```

Fix options:
```bash
# Increase memory request
#SBATCH --mem=480G          # more total node memory

# Or per-GPU
#SBATCH --mem-per-gpu=80G   # 80GB per GPU (A100 40GB VRAM + CPU RAM)
```

---

### Drill 4 — Node failure mid-job

One node crashes. Your job fails with NCCL timeout.

Check what happened:
```bash
sacct -j 12345 --format=JobID,State,ExitCode,Elapsed
scontrol show node <failed_node>
```

If node shows `down` or `drain`:
```bash
# Only admin can drain/undrain nodes
# Report to cluster admin
```

Your training script should checkpoint frequently (Module 17).
Your SLURM script should auto-resubmit on failure (Script 6 above).

---

### Drill 5 — Port conflict between jobs

Two jobs on same node using same `MASTER_PORT`:
```
RuntimeError: Address already in use (port 29500)
```

Fix: derive port from Job ID:
```bash
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_PORT
```

This gives a different port per job, very unlikely to conflict.

---

### Drill 6 — Wrong number of tasks launched

If `srun` launches wrong number of processes:
```bash
# Verify task count before training
echo "Expected: $SLURM_NTASKS tasks"
srun --ntasks=$SLURM_NTASKS echo "task $SLURM_PROCID alive"
```

---

## 8.10 Complete Reference Card

Print this and keep it next to your terminal.

```
── Job lifecycle ──────────────────────────────────────────────
sbatch job.sh              submit batch job
salloc --nodes=2 ...       interactive allocation
squeue -u $USER            list my jobs
scontrol show job <ID>     full job details
scancel <ID>               cancel job
sacct -j <ID>              accounting (after job)
seff <ID>                  efficiency report

── Cluster state ──────────────────────────────────────────────
sinfo                      partition/node overview
sinfo -N -l                node detail list
scontrol show node <name>  full node details
scontrol show partition gpu partition details

── Most used #SBATCH options ──────────────────────────────────
--nodes=N                  number of nodes
--ntasks-per-node=N        tasks (processes) per node
--gres=gpu:8               8 GPUs per node
--gres=gpu:a100:8          8 A100 GPUs specifically
--cpus-per-task=N          CPU cores per task
--mem=256G                 total RAM per node
--time=HH:MM:SS            wall time limit
--partition=gpu            which partition
--output=logs/%j.out       stdout (%j=jobID)
--error=logs/%j.err        stderr
--job-name=name            job name
--dependency=afterok:ID    run after another job
--array=0-9                job array

── SLURM variables for DDP ────────────────────────────────────
SLURM_JOB_NODELIST  → parse for MASTER_ADDR
SLURM_NODEID        → NODE_RANK
SLURM_PROCID        → RANK (if using srun directly)
SLURM_LOCALID       → LOCAL_RANK (if using srun directly)
SLURM_NTASKS        → WORLD_SIZE

── Get MASTER_ADDR ────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

── Port that won't conflict ───────────────────────────────────
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
```

---

## 8.11 Module 8 Pass Criteria

You're done when you can:

1) **Explain every concept**:
   - partition, job, step, task, GRES, CPU affinity
   - difference between `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NODEID`
   - why `SLURM_PROCID` = `RANK` and `SLURM_LOCALID` = `LOCAL_RANK`

2) **Use all CLI tools**:
   - submit, monitor, cancel, inspect jobs
   - read `seff` output and explain CPU/memory efficiency

3) **Write from scratch** (without looking):
   - single-node 8-GPU DDP job script
   - multi-node 2-node DDP job script
   - job array for hyperparameter sweep
   - `MASTER_ADDR` extraction from `SLURM_JOB_NODELIST`

4) **Debug**:
   - job stuck PENDING → read reason → fix
   - port conflict → derive port from job ID
   - wrong `CUDA_VISIBLE_DEVICES` → inspect + fix
   - node failure → auto-resubmit script

---

