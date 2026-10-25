# Contributing to sched-sim 🤝

Thank you for your interest in contributing to `sched-sim`! We welcome contributions from systems engineers, performance researchers, and open-source developers.

Our mission is to build a high-fidelity, CPU-first **discrete-event simulator for LLM serving schedulers** (continuous batching, chunked prefill, radix prefix caching, and disaggregation) that runs in seconds on any developer workstation using first-principles roofline models calibrated against real silicon.

---

## 🛠️ Development Setup

`sched-sim` uses [Astral's `uv`](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

```bash
# 1. Fork and clone the repository
git clone https://github.com/Prajwal7-2002/sched-sim.git
cd sched-sim

# 2. Sync all dependencies into an isolated virtual environment
uv sync

# 3. Verify hardware presets and runnable examples
uv run python examples/01_radix_prefix_caching.py
uv run python examples/02_simulate_schedulers.py
uv run python examples/03_perfetto_trace_export.py
```

---

## 🏛️ Core Architectural Invariants

Before submitting a Pull Request, please ensure your contribution adheres to these fundamental design invariants:

1. **Deterministic & 100% CPU-First**:
   * `sched-sim` simulates hours of multi-GPU serving traffic in **seconds on any standard CPU**.
   * It must never require a physical GPU or proprietary CUDA compiler to execute or test.
2. **First-Principles Analytical Roofline Mechanics**:
   * Step latencies must be derived from mathematical hardware bounds (effective TFLOPS, DRAM bandwidth, KV-cache byte transfer, PCIe/NVLink interconnect delay, and kernel launch overheads).
   * Arbitrary synthetic sleep calls or uncalibrated magic delays are strictly forbidden.
3. **Paged Virtual Memory Integrity**:
   * Memory management must faithfully reflect real serving systems (like vLLM and SGLang): 16-token physical blocks, virtual-to-physical block tables, zero-copy prefix sharing, and reference-counted LRU eviction.
4. **Catapult / Perfetto Standard Observability**:
   * Timeline traces must conform to the Chromium Trace Event format loadable directly at [ui.perfetto.dev](https://ui.perfetto.dev).

---

## 🧪 Testing & Code Quality

All changes must pass our strict automated test and linting gates before merge:

```bash
# Run unit and integration tests (must complete in < 2s)
uv run pytest

# Check code coverage (target: >= 85%)
uv run pytest --cov=sched_sim --cov-report=term-missing

# Lint checks (Ruff)
uv run ruff check .

# Format checks (Ruff)
uv run ruff format --check .

# Validate documentation build
uv run mkdocs build --strict
```

---

## 🎯 Where We Need Help (High-Impact Areas)

We actively welcome contributions in the following areas:

### 1. New Scheduling Policies (`src/sched_sim/schedulers/`)
* [ ] **Slack-Aware Earliest Deadline First (EDF)**: Priority scheduling that calculates request slack time ($T_{\text{deadline}} - T_{\text{estimated\_finish}}$) to prioritize tightest requests without starving long context jobs.
* [ ] **Fair-Share Deficit Round Robin (DRR)**: Multi-tenant scheduling that enforces fair token allocation across multiple tenant API keys.
* [ ] **Speculative Decoding Schedulers**: Schedulers that dynamically adjust batch sizes based on speculative token acceptance rates.

### 2. Modern Hardware Presets (`src/sched_sim/hardware/presets.py`)
* [ ] **NVIDIA H200 SXM (141GB HBM3e)**: 4.8 TB/s memory bandwidth profile.
* [ ] **NVIDIA Blackwell B200 (192GB HBM3e)**: 8.0 TB/s memory bandwidth profile.
* [ ] **AMD Instinct MI300X (192GB HBM3)**: 5.3 TB/s memory bandwidth profile.
* [ ] **Apple Silicon M3 / M4 Max**: Unified memory bandwidth modeling for edge serving.

### 3. Real-World Trace Loaders (`src/sched_sim/workloads/`)
* [ ] **Azure Public LLM Serving Traces**: Native loader for Microsoft's published Azure LLM serving arrival traces.
* [ ] **Multimodal Request Distributions**: Modeling interleaved image/audio token counts with vision encoder prefill overheads.

### 4. Interconnect & Topology Simulation
* [ ] **Tensor Parallelism (TP=2, 4, 8)**: Analytical modeling of Megatron-style All-Reduce latency over NVLink (900 GB/s) vs PCIe Gen4/Gen5.
* [ ] **Pipeline Parallelism (PP)**: Modeling bubble overhead and cross-node KV-transfer delays.

---

## 📬 Pull Request Conventions

To maintain a clean git history and automated release pipeline:

### 1. Branch Naming
Create a descriptive branch off `main`:
* `feat/short-feature-name` (e.g. `feat/h200-hardware-preset`)
* `fix/short-bug-description` (e.g. `fix/lru-leaf-eviction-leak`)
* `perf/short-perf-description` (e.g. `perf/faster-hash-chaining`)
* `docs/short-doc-description` (e.g. `docs/contributing-guide`)
* `test/short-test-description` (e.g. `test/disaggregated-network-delay`)

### 2. Conventional Commit Messages
Write clear, imperative commit messages:
* `feat: add NVIDIA H200 SXM hardware preset`
* `fix: correct token count accounting during preemption recompute`
* `docs: update quickstart guide with perfetto trace loading instructions`

### 3. No Unrelated Trailing Metadata
Do not include automated AI co-author trailers (`Co-authored-by:`). Commits should represent your direct contribution.

---

## 📄 License

By contributing to `sched-sim`, you agree that your contributions will be licensed under the project's [Apache 2.0 License](https://github.com/Prajwal7-2002/sched-sim/blob/main/LICENSE).
