## How we’ll run the course (so it actually works)

### What you’ll do for each module
1. Read the short theory notes I give you  
2. Copy/paste + run the lab code  
3. Capture outputs (logs, `nvidia-smi`, timing)  
4. Do the debugging drills  
5. Answer a few “explain in your own words” questions (to lock it in)

### Your end result
By the time we reach Module 17, you’ll have your own training system that can run on:
- **1 GPU → 8 GPUs (single node) → 16 GPUs (2 nodes)**  
- with **SLURM**, checkpoints, mixed precision, logging, profiling

---

# Module 0 — Foundations (START HERE)

## Module 0 Goal
Understand what actually happens inside a GPU during training, and set up a clean environment so later distributed issues don’t become mystery problems.

---

## 0.1 Theory — “Mental model” of GPU training (the minimum you must know)

### CPU vs GPU (why GPU wins)
- **CPU**: few powerful cores, great at branchy logic, low-latency tasks.
- **GPU**: many smaller cores, great at doing the *same operation on lots of data* (matrix multiply, convolutions).

Deep learning training is mostly big linear algebra (GEMMs), so GPUs shine.

### CUDA programming model (your first key idea)
- You launch a **kernel** on the GPU.
- A kernel runs many threads in parallel:
  - **Thread**
  - grouped into **Warps** (typically 32 threads executed together)
  - grouped into **Blocks**
  - blocks scheduled on **SMs** (Streaming Multiprocessors)

### SMs, CUDA cores, Tensor Cores (what matters for DL)
- **CUDA cores** do general math.
- **Tensor Cores** accelerate matrix operations (FP16/BF16/TF32) massively.
- Most DL speedups come from:
  - using **Tensor Cores**
  - keeping GPU busy (high utilization)
  - avoiding memory stalls / data loader stalls

### GPU memory hierarchy (why memory becomes the bottleneck)
From fastest/smallest → slowest/largest:
- Registers (per-thread)
- Shared memory / L1 cache (per-SM)
- L2 cache (device-wide)
- HBM (GPU VRAM)
- Host RAM (CPU)
- Disk / network storage

Training often bottlenecks on:
- **HBM bandwidth**
- **PCIe / NVLink transfers**
- **CPU → GPU input pipeline**

### PCIe vs NVLink vs InfiniBand (why multi-GPU is hard)
- **PCIe**: general interconnect; decent bandwidth.
- **NVLink**: much faster GPU-GPU within a node (depends on topology).
- **InfiniBand**: fast networking between nodes (low latency + high bandwidth).

### CUDA Streams + Events (why “GPU utilization” can lie)
- Operations are queued on **streams**.
- If you don’t synchronize, CPU timing can be misleading.
- Events let you time GPU work correctly.

---

## 0.2 Practical — Environment + Verification (Hands-on)

Below is a **clean checklist**. You’ll run it in order.

### Step A — Verify GPU + driver
```bash
nvidia-smi
```

You should see:
- GPU name (A100)
- driver version
- memory size
- running processes (maybe none)

Also check topology (important later):
```bash
nvidia-smi topo -m
```

### Step B — Install / verify CUDA toolkit (basic check)
```bash
nvcc --version
```

If `nvcc` is missing, CUDA toolkit is not installed (driver might still be installed).

### Step C — Create Python env + install PyTorch (CUDA build)
Recommended:
```bash
python3 -m venv ddl_env
source ddl_env/bin/activate
pip install --upgrade pip
```

Then install PyTorch with CUDA support (we’ll choose the correct command based on your system—do not guess randomly).  
For now, just verify what you have:

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count());"
```

### Step D — GPU monitoring tools
Install `nvtop`:
```bash
sudo apt-get update
sudo apt-get install -y nvtop
```

Run:
```bash
nvtop
```

Keep it open during training later.

### Step E — Measure GPU memory + utilization from PyTorch
Create `check_gpu.py`:

```python
import torch, time

assert torch.cuda.is_available()
device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))
print("Total VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)

# Warmup
x = torch.randn(8192, 8192, device=device, dtype=torch.float16)
w = torch.randn(8192, 8192, device=device, dtype=torch.float16)

torch.cuda.synchronize()
t0 = time.time()
y = x @ w
torch.cuda.synchronize()
t1 = time.time()

print("Matmul time (s):", t1 - t0)
print("Allocated (GB):", torch.cuda.memory_allocated() / 1024**3)
print("Reserved  (GB):", torch.cuda.memory_reserved() / 1024**3)
```

Run:
```bash
python check_gpu.py
```

This confirms:
- CUDA works
- You can allocate on GPU
- Timing uses `torch.cuda.synchronize()` correctly

---

## 0.3 Mini Project — Vector Addition in CUDA

### Goal
Write the simplest real CUDA program so you understand:
- kernel launch
- memory copy H→D and D→H
- grid/block/thread indexing

Create `vector_add.cu`:

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20; // ~1M
    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // quick check
    printf("h_c[0]=%f h_c[n-1]=%f\n", h_c[0], h_c[n-1]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

Compile + run:
```bash
nvcc -O2 vector_add.cu -o vector_add
./vector_add
```

Expected:
- `h_c[0]=3.000000` and `h_c[n-1]=3.000000`

### What you should learn from this
- GPU doesn’t “see” CPU memory directly: you copied data using `cudaMemcpy`.
- `<<<blocks, threads>>>` controls parallelism.
- Index math (`blockIdx`, `blockDim`, `threadIdx`) is the heart of CUDA.

---

## 0.4 Debugging drills (do these on purpose)
1. Change `threads = 1024` and see what happens (some GPUs limit threads/block).  
2. Remove `cudaDeviceSynchronize()` and check if results become unreliable (timing definitely will).  
3. Set `n` to something not divisible by block size and confirm the boundary condition protects you.

---

## 0.5 Module 0 “pass criteria” (you’re done when)
You can:
- explain **warp/block/SM** in simple terms
- explain why **HBM / PCIe / NVLink** matter
- run:
  - `nvidia-smi`
  - `nvidia-smi topo -m`
  - PyTorch CUDA check script
  - CUDA vector add

---

## Next step
Run these 3 commands (copy/paste):

```bash
nvidia-smi
nvidia-smi topo -m
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count());"
```

