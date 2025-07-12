# Vynapse

> A Rust-native deep learning and neuroevolution engine built as a hybrid learning runtime — unifying the optimization paradigms of PyTorch, TensorFlow, DEAP, and NEAT into one modular, high-performance system.

---

## 🚀 Project Description

Vynapse is a multi-paradigm machine learning engine written entirely in safe Rust. It bridges **gradient-based learning** and **evolutionary optimization** within a unified execution and graph infrastructure. Inspired by the flexibility of PyTorch, the structural clarity of TensorFlow, and the adaptive search power of DEAP and NEAT, Vynapse supports:

- **SGD-style training with reverse-mode autodiff**
- **Topology-evolving neural networks using NEAT**
- **Population-based weight evolution in DEAP style**
- **Static graph execution with forward/backward scheduling**

Every training mode operates on the same core tensor and graph runtime, enabling flexible, hybrid workflows that can evolve, fine-tune, and deploy networks interchangeably.

---

## 🧠 What Makes Vynapse Different?

- **Hybrid Optimization**: Combine evolution and gradient descent in a single training loop (Lamarckian or Baldwinian modes).
- **Neuroevolution-Native**: Evolve both weights and graph structures with speciation, compatibility distance, and reproduction strategies.
- **Autodiff Engine**: Full reverse-mode autodiff with support for eager and static graph execution.
- **Rust Performance**: Built from the ground up in safe, parallel Rust — no Python bindings, no unsafe blocks.
- **Runtime Unification**: A single graph engine powers static, dynamic, differentiable, and evolutionary workflows.

---

## 🗺️ Roadmap Overview

- [x] Milestone 1: Weight Evolution (DEAP-style) *(currently in progress)*
- [ ] Milestone 2: Multi-Mode Trainers (DEAP, NEAT, SGD, Static)
- [ ] Milestone 3: Autodiff & Speciation
- [ ] Milestone 4: Hybrid Learning + Dataset API
- [ ] Milestone 5: GPU, WASM, Distributed Runtime
- [ ] Milestone 6: Research & Exploration

---

## 📍 Milestone 1 – *Genesis* (Weight-Evolution MVP)  
> Prove the core concept by evolving static network weights on XOR.

### 🛠️ Systems Foundations
- [ ] Establish a multi-crate workspace (`core`, `cli`, `math`)
- [ ] Design a project-wide error hierarchy with recoverable and fatal classes
- [ ] Integrate a fast, seedable PRNG and structured log sink

### 📐 Numeric & Memory Primitives
- [ ] Draft a contiguous, heap-allocated tensor abstraction (row-major, shape + stride metadata)
- [ ] Implement element-wise operations and small fixed-size matrix multiplication using cache-aware loop order
- [ ] Add invariant checks for dimension mismatches and overflow guarding

### 🧬 Evolutionary Pipeline (DEAP-Style)
- [ ] Specify genome layout for a fixed 2-2-1 MLP (flat f32 buffer)
- [ ] Create Gaussian mutation with adaptive σ decay and optional per-weight masking
- [ ] Implement (μ + λ) and (μ, λ) selection strategies with tournament and roulette variants
- [ ] Build a generation scheduler with early-exit on convergence

### 🔁 Experiment Runner
- [ ] Provide CLI workflow to scaffold XOR configuration, inject hyper-parameters, and stream fitness stats
- [ ] Persist per-generation CSV logs and a compressed binary checkpoint of the best genome

### ✔️ Acceptance Criteria
- [ ] Converge to XOR loss < 1e-2 in < 5 s on 8-thread desktop CPU
- [ ] Achieve deterministic results on fixed seed; variance report < ±5 %

---

## 📍 Milestone 2 – *Quad-Core Minimal* (DEAP, NEAT, SGD, Static-Graph)  
> Train the *same* XOR network with four distinct optimisation paradigms.

### 🏗️ Graph & Tensor Layer
- [ ] Extend tensor core with broadcast semantics, strided slices, and BLAS-style GEMM micro-kernel (naïve C = αAB + βC)
- [ ] Introduce a DAG representation for computation graphs (node op, edge tensor ref) with acyclic validation
- [ ] Implement DOT and JSON exporters for graph visualisation and diffing

### 🔄 Trainer Implementations
- [ ] Port DEAP logic to pluggable `Trainer` interface
- [ ] Implement minimal NEAT topology mutation and feed-forward executor
- [ ] Add manual back-prop pipeline with simple optimiser for SGD mode
- [ ] Create declarative static-graph builder and reversed gradient pass

### 🔧 Config & CLI
- [ ] Build YAML/TOML experiment loader with spawn-time validation
- [ ] Add unified CLI flag `--mode={deap, neat, sgd, static}` with hyper-parameter overrides

### 📊 Benchmark Harness
- [ ] Collect wall-clock time, CPU utilisation, and memory footprint per mode
- [ ] Generate Markdown comparison table via post-run script

### ✔️ Acceptance Criteria
- [ ] All four modes hit loss < 2e-2 within configurable budget
- [ ] Runtime variance between repeated runs < 10 %

---

## 📍 Milestone 3 – *Autodiff & Dynamic Graphs*  
> Introduce full reverse-mode autodiff and robust NEAT speciation.

### ⚙️ Reverse-Mode Engine
- [ ] Design tape data-structure with parent indices, saved intermediates, and gradient function pointer
- [ ] Implement zero-grad, gradient accumulation (Σ from multiple downstream nodes), and retention policy
- [ ] Provide optimisers: SGD, momentum, Adam, RMSprop, and gradient clipping (L2 norm ceiling)

### 🧩 Dynamic Eager Runtime
- [ ] Overload tensor operations to record tape entries transparently (eager mode)
- [ ] Add runtime shape inference, alignment checks, and broadcast expansion
- [ ] Provide `trace!` utility converting eager runs to static DAG for re-execution

### 🧬 NEAT 2.0
- [ ] Implement compatibility distance metric (c₁, c₂, c₃) with user-defined weights
- [ ] Add species management: bucket assignment, stagnation detection, elite carry-over, dynamic σ scaling
- [ ] Parallelise population evaluation with a scoped thread-pool and work-stealing

### 📈 Task Benchmark Suite
- [ ] Integrate 2-Spiral classification and CartPole reinforcement learning tasks
- [ ] Auto-generate loss / reward plots and termination statistics

### ✔️ Acceptance Criteria
- [ ] Back-prop finite-difference error < 1e-4 relative tolerance across ops
- [ ] Reach ≥ 80 % code-coverage target in CI with deterministic, seed-locked tests

---

## 📍 Milestone 4 – *Hybrid Optimiser & Dataset Fabric*  
> Merge evolutionary search with gradient descent; add real-world datasets and experiment automation.

### 🔀 Hybrid Learning Layer
- [ ] Implement Lamarckian / Baldwinian pipeline: evolve population → fine-tune top-k → fitness re-evaluation → merge
- [ ] Support composite fitness weighting (accuracy, parameter count, FLOPs, latency)

### 🗄️ Dataset Management
- [ ] Create memory-mapped CSV ingestion and image-folder loader with optional caching
- [ ] Implement mini-batch shuffler, deterministic epoch splits, and lazy prefetch into thread-local buffers
- [ ] Add augmentation hooks (crop, flip, noise) for images

### 📊 Observability & Telemetry
- [ ] Integrate hierarchical tracing spans with wall-time and CPU counters
- [ ] Expose Prometheus metrics endpoint guarded by CLI flag
- [ ] Provide live progress UI via terminal dashboard

### 🖥️ Experiment Workflow
- [ ] Build CLI helpers for dataset splitting, provenance hashing, and cached preprocessing
- [ ] Define YAML recipe schema for reproducible experiment sweeps

### ✔️ Benchmarks & Targets
- [ ] Achieve MNIST ≥ 98 % test accuracy (SGD) and ≥ 96 % via hybrid in ≤ 10 epochs
- [ ] Ensure Prometheus scrape latency < 10 ms and live UI refresh interval ≤ 0.5 s

---

## 📍 Milestone 5 – *GPU, WASM & Distributed Scale-Out*  
> Accelerate with GPUs, run in browsers, and scale neuroevolution across nodes.

### ⚡ GPU Compute
- [ ] Add WGSL compute kernels for matmul, activation, and reduction ops
- [ ] Provide GPU memory pool with pinned host buffers and asynchronous copy queues
- [ ] Implement optional CUDA PTX backend with occupancy querying

### 🌐 WASM Runtime
- [ ] Port tensor and graph engine to `no_std` with custom allocator
- [ ] Deliver browser demo running neuroevolution of XOR inside a WebWorker
- [ ] Build WASI command-line binary for edge deployments

### 🌍 Distributed Evolution
- [ ] Define gRPC protocol for remote genome evaluation jobs
- [ ] Implement node heartbeat, cluster fault detection, and checkpoint-based resume
- [ ] Support island model with periodic genome migration and fitness sharing

### 🛠️ CI / CD & Packaging
- [ ] Configure GitHub Actions for Linux, macOS, Windows, wasm32, and CUDA targets
- [ ] Automate release bundling, version tagging, changelog generation, and crates.io publication
- [ ] Enforce dual MIT / Apache-2.0 licensing with automated header checks

### ✔️ Acceptance Criteria
- [ ] Achieve ≥ 10× speed-up over CPU baseline on 512 × 512 matmul using WGSL
- [ ] Complete 1 000 XOR generations in < 5 s in browser demo
- [ ] Demonstrate near-linear scaling to 32 worker processes in distributed mode

---

## 📍 Milestone 6 – *Research & Stretch Ventures*  
> Exploratory paths after core roadmap completion.

| Track | Objectives |
|-------|------------|
| **Differentiable NEAT** | Back-prop through mutable topologies for joint weight+structure optimisation |
| **Reinforcement Learning** | Add Gymnasium bridge, policy gradients, neuro-evolutionary actor/critic hybrids |
| **Meta-Evolution** | Evolve optimiser hyper-parameters or hyper-networks that generate task-specific sub-nets |
| **Hardware Autotuning** | Perform Bayesian search for GPU tile sizes, vector widths, register blocking |
| **Mobile Inference** | Produce Flutter / iOS / Android demos via `wgpu-rs` and cross-compiled WASM |
| **Compiler Fusion** | Integrate Xyntra-generated fused kernels as node implementations for accelerated inference |
| **Sparse Execution** | Implement graph-level dynamic pruning, activation gating, and runtime re-wiring |

---
