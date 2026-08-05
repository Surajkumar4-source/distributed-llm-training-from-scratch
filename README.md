# Distributed Deep Learning and LLM Training

> A comprehensive course on distributed deep learning systems, covering the complete journey from single-GPU training to large-scale multi-node LLM training.

---

## Overview

Modern deep learning systems no longer train on a single GPU. State-of-the-art language models are built using clusters of GPUs connected through high-speed interconnects, where computation, communication, and memory management become equally important.

This course develops a deep understanding of the algorithms, systems, and software that enable large-scale model training. Rather than treating distributed frameworks as black boxes, every major concept is introduced from first principles, implemented in minimal code, and then related to production systems.

The course combines theoretical foundations with practical implementations to explain how modern frameworks such as PyTorch Distributed, FSDP, DeepSpeed, and Megatron-LM scale neural network training across multiple GPUs and multiple machines.

---

## Learning Objectives

After completing this course, you will be able to:

- Explain GPU architecture and the CUDA execution model.
- Understand the complete lifecycle of a distributed training iteration.
- Implement distributed communication using collective operations.
- Train neural networks with Distributed Data Parallel (DDP).
- Scale training across multiple GPUs and multiple nodes.
- Optimize memory using mixed precision, activation checkpointing, and Fully Sharded Data Parallel (FSDP).
- Understand DeepSpeed ZeRO and Megatron-LM parallelism strategies.
- Profile distributed workloads and identify performance bottlenecks.
- Read, modify, and extend production-scale LLM training code.

---

## Learning Philosophy

Every topic in this course follows the same progression.

1. Theory
2. Mathematical intuition
3. Minimal implementation
4. Hands-on experiments
5. Visualization
6. Debugging
7. Performance analysis
8. Production implementation

The objective is to understand not only *how* distributed training works, but *why* modern frameworks are designed the way they are.

---

## Course Modules

| Module | Topics |
|---------|--------|
| **00** | GPU Architecture and CUDA Foundations |
| **01** | Single GPU Training with PyTorch |
| **02** | Scaling Beyond One GPU |
| **03** | Distributed Training Fundamentals |
| **04** | Distributed Data Parallel (DDP) |
| **05** | Distributed Data Loading |
| **06** | Multi-GPU Single-Node Training |
| **07** | Multi-Node Distributed Training |
| **08** | SLURM and Cluster Computing |
| **09** | GPT From Scratch |
| **10** | Distributed GPT Training |
| **11** | Memory Optimization Techniques |
| **12** | Fully Sharded Data Parallel (FSDP) |
| **13** | DeepSpeed and ZeRO |
| **14** | Megatron-LM and Model Parallelism |
| **15** | Profiling and Performance Engineering |
| **16** | Production LLM Training Systems |
| **17** | Capstone Project |

---

## Topics Covered

### GPU Systems

- CPU vs GPU Architecture
- CUDA Programming Model
- Streaming Multiprocessors
- CUDA Cores
- Tensor Cores
- GPU Memory Hierarchy
- CUDA Streams
- CUDA Events
- PCIe
- NVLink
- InfiniBand
- NUMA
- DMA

### Distributed Systems

- Processes and Threads
- Process Groups
- Ranks
- World Size
- Collective Communication
- Broadcast
- Reduce
- AllReduce
- ReduceScatter
- AllGather
- Synchronization
- NCCL
- Gloo
- MPI

### Large Language Model Training

- Transformer Architecture
- Self-Attention
- Multi-Head Attention
- Feed Forward Networks
- Layer Normalization
- Decoder-Only Models
- GPT Training
- Mixed Precision
- Gradient Accumulation
- Activation Checkpointing
- FSDP
- DeepSpeed ZeRO
- Tensor Parallelism
- Pipeline Parallelism
- Sequence Parallelism
- Mixture of Experts

### Performance Engineering

- GPU Profiling
- Memory Analysis
- Communication Overlap
- Scaling Efficiency
- Throughput Optimization
- Performance Benchmarking

---

## 📁 Repository Structure

```

distributed-llm-training-course/
│
├── README.md                    ← you are here
├── requirements.txt             ← all Python dependencies
├── SETUP.md                     ← detailed environment setup
├── CONTRIBUTING.md              ← how to contribute
├── LICENSE                      ← MIT
│
├── module00/                    ← Foundations
│   ├── README.md                ← module-specific instructions
│   ├── theory/                  ← concept notes + diagrams
│   │   └── gpu_architecture.md
│   ├── labs/                    ← hands-on exercises
│   │   ├── vector_add.cu
│   │   ├── check_gpu.py
│   │   └── verify_cuda.py
│   └── solutions/               ← reference solutions
│
├── module01/                    ← Single GPU Training
│   ├── README.md
│   ├── labs/
│   │   ├── train_cifar10.py
│   │   └── profile_dataloader.py
│   └── solutions/
│
├── module02/                    ← Why Multiple GPUs
│   ├── README.md
│   └── labs/
│       ├── bench_allreduce.py
│       └── scaling_analysis.py
│
├── module03/                    ← Distributed Fundamentals
│   ├── README.md
│   └── labs/
│       ├── dist_hello.py
│       ├── dist_broadcast.py
│       ├── dist_allreduce.py
│       ├── dist_send_recv.py
│       └── dist_hang_demo.py
│
├── module04/                    ← DDP
│   ├── README.md
│   └── labs/
│       ├── train_cifar10_ddp.py
│       ├── inspect_ddp_buckets.py
│       ├── ddp_correctness.py
│       └── checkpoint_test.py
│
├── module05/                    ← DistributedSampler
│   ├── README.md
│   └── labs/
│       ├── my_distributed_sampler.py
│       ├── visualize_sampler.py
│       ├── wrong_sampler_demo.py
│       └── verify_coverage.py
│
├── module06/                    ← Multi-GPU Single Node
│   ├── README.md
│   └── labs/
│       ├── check_p2p.py
│       ├── p2p_bandwidth.py
│       ├── bench_allreduce_msgsize.py
│       ├── bench_scaling_efficiency.py
│       └── ddp_grad_accumulation.py
│
├── module07/                    ← Multi-Node Training
│   ├── README.md
│   └── labs/
│       ├── multinode_hello.py
│       ├── launch_multinode.sh
│       ├── multinode_allreduce_bench.py
│       └── train_cifar10_multinode.py
│
├── module08/                    ← SLURM
│   ├── README.md
│   └── labs/
│       ├── job_minimal.sh
│       ├── job_single_node_ddp.sh
│       ├── job_multinode_ddp.sh
│       ├── job_array_sweep.sh
│       ├── slurm_setup.sh
│       └── slurm_info.py
│
├── module09/                    ← GPT From Scratch
│   ├── README.md
│   └── labs/
│       ├── tokenizer.py
│       ├── attention.py
│       ├── multihead_attention.py
│       ├── feedforward.py
│       ├── transformer_block.py
│       ├── gpt.py
│       ├── train_gpt.py
│       ├── overfit_test.py
│       ├── grad_check.py
│       ├── visualize_attention.py
│       └── generate.py
│
├── module10/                    ← GPT with DDP
│   ├── README.md
│   └── labs/
│       ├── train_gpt_ddp.py
│       ├── bench_gpt_scaling.py
│       ├── check_model_sync.py
│       ├── checkpoint_test.py
│       └── ddp_correctness.py
│
├── module11/                    ← Memory Optimization
│   ├── README.md
│   └── labs/
│       ├── measure_memory_components.py
│       ├── mixed_precision_lab.py
│       ├── gpt_with_checkpointing.py
│       ├── memory_reduction_experiments.py
│       ├── memory_profiler_lab.py
│       └── oom_handling.py
│
├── module12/                    ← FSDP
│   ├── README.md
│   └── labs/
│       ├── toy_zero1.py
│       ├── fsdp_hello.py
│       ├── ddp_vs_fsdp_memory.py
│       ├── fsdp_checkpoint.py
│       ├── train_gpt_fsdp.py
│       ├── fsdp_wrap_policy_experiment.py
│       └── fsdp_cpu_offload.py
│
├── module13/                    ← DeepSpeed
│   ├── README.md
│   ├── ds_configs/
│   │   ├── zero1.json
│   │   ├── zero2.json
│   │   ├── zero3.json
│   │   ├── zero3_offload.json
│   │   └── zero_infinity.json
│   └── labs/
│       ├── ds_hello.py
│       ├── ds_zero_stages.py
│       ├── train_gpt_deepspeed.py
│       ├── ds_checkpoint_lab.py
│       └── ds_offload_lab.py
│
├── module14/                    ← Megatron-LM
│   ├── README.md
│   └── labs/
│       ├── tp_linear.py
│       ├── pipeline_parallel.py
│       ├── process_groups_3d.py
│       ├── moe_layer.py
│       └── tp_pp_combined.py
│
├── module15/                    ← Profiling
│   ├── README.md
│   └── labs/
│       ├── profile_basic.py
│       ├── profile_training.py
│       ├── profile_ddp_comm.py
│       ├── nsys_profile_target.py
│       ├── ncu_target.py
│       ├── nccl_profile.py
│       ├── memory_profiler.py
│       ├── profile_attention_comparison.py
│       ├── find_bottlenecks.py
│       └── profiling_report.py
│
├── module16/                    ← Production Training
│   ├── README.md
│   └── labs/
│       ├── study_nanogpt.py
│       ├── mfu_tracker.py
│       ├── study_hf_trainer.py
│       ├── train_with_hf.py
│       ├── study_megatron.py
│       ├── production_trainer.py
│       └── patterns_reference.py
│
└── module17/                    ← Capstone Project
    ├── README.md
    ├── train.py                 ← entry point
    ├── src/
    │   ├── config.py
    │   ├── model.py
    │   ├── data.py
    │   ├── distributed.py
    │   ├── optimizer.py
    │   ├── metrics.py
    │   ├── checkpoint.py
    │   ├── profiler.py
    │   └── trainer.py
    ├── configs/
    │   ├── small.yaml
    │   ├── medium.yaml
    │   └── large.yaml
    ├── jobs/
    │   ├── job_1gpu.sh
    │   ├── job_8gpu.sh
    │   ├── job_16gpu.sh
    │   └── job_sweep.sh
    ├── scripts/
    │   ├── benchmark_scaling.sh
    │   ├── generate.py
    │   └── performance_report.py
    └── tests/
        ├── test_model.py
        ├── test_distributed.py
        └── test_checkpoint.py
