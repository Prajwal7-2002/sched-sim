# sched-sim: Development Log

Chronological record of architectural milestones, benchmark validations, and engineering changelog.

---

## [0.2.0] - 2026-10-25

### Flagship Parity Milestone: Radix Prefix Caching, Perfetto Timeline & Silicon Validation

#### Added
- **Radix-Tree Prefix Cache (`src/sched_sim/memory/prefix_cache.py`)**:
  - Implemented SGLang/vLLM-style Radix Tree with chained 16-token rolling block hashing: $H_i = \text{hash}(H_{i-1}, \text{tokens}_{16i:16(i+1)})$.
  - Zero-copy physical block sharing across requests with identical prompt prefixes.
  - Reference counting (`ref_count`) with automatic LRU eviction of unreferenced leaf blocks when GPU VRAM is constrained.
  - Skip prefill compute FLOPs for warm prefix hits, dropping TTFT to near-zero ($<5\text{ ms}$).
- **Cache-Aware Scheduler (`src/sched_sim/schedulers/cache_aware.py`)**:
  - Prefix-affinity priority queue: prioritizes requests with warm cached prefixes, with Shortest-Job-First tie-breaking on remaining prefill.
- **Perfetto / Chromium Trace Exporter (`src/sched_sim/visualizer/perfetto.py`)**:
  - Export simulation execution to native Chromium Catapult Trace Event JSON (`--trace-out trace.json`).
  - Drag-and-drop support into **[ui.perfetto.dev](https://ui.perfetto.dev)**:
    - Microsecond GPU forward-pass execution slices (`gpu_pass`).
    - Per-request arrival, TTFT, and completion timelines (`requests`).
    - Dynamic memory counter tracks (`KV_Cache_Utilization` and `Active_Batch_Concurrency`).
- **Empirical Ground-Truth Silicon Validation Suite (`benchmarks/`)**:
  - Committed `benchmarks/validate_flagships.py` and `benchmarks/RESULTS.md` validating `sched-sim` against physical silicon measurements from `servebench` and `mini-infer` on RTX 3050 Laptop GPU:
    - Ollama C++/CUDA: TPOT error **7.5%**, TTFT error **7.9%**, E2E error **6.3%**.
    - mini-infer Python eager: TPOT error **3.3%**, E2E error **3.8%**.
- **Architectural Documentation**:
  - Created `flow.md` ("How It Flows") covering DES min-heap queue, request state machine, and roofline mechanics.
  - Created `decisions.md` documenting 8 core Architecture Decision Records (ADRs).
- **Test Suite Expansion**:
  - Expanded test suite from 27 to **34 unit tests** covering prefix chained hashing, block sharing, LRU eviction, cache-aware scheduling, and Perfetto trace schema with **87% test coverage**.

#### Changed
- Bumped project version to `0.2.0` in `pyproject.toml` and `src/sched_sim/__init__.py`.
- Updated CLI defaults to **NVIDIA A100 SXM4 (80GB)** with Llama-3-8B.
- Formatted with Ruff across all 45 files.

---

## [0.1.0] - 2026-10-24

### Initial Release: Discrete-Event LLM Serving Simulation Engine

#### Added
- **Core Discrete-Event Simulation Engine**:
  - Virtual clock progression backed by Python `heapq` priority queue.
  - Deterministic event scheduling (`ARRIVAL`, `ITERATION`).
- **Analytical Roofline Latency Model**:
  - Analytical computation of prefill (compute-bound), decode (memory-bound), and mixed forward passes.
  - Kernel launch overhead and network interconnect delay modeling for PD Disaggregation.
- **5 Core Scheduling Policies**:
  - First-Come First-Served (`FCFS`).
  - Shortest-Job-First (`SJF`).
  - Sarathi-Serve Chunked Prefill (`Chunked`).
  - Earliest Deadline First with selective drop (`SLO-Aware`).
  - Splitwise / DistServe Prefill-Decode Disaggregation (`Disaggregated`).
- **Paged KV-Cache Memory Management**:
  - 16-token physical block allocator with virtual block tables.
  - Preemption subsystem supporting Recompute and PCIe CPU Host DRAM swapping.
- **Workload Generator & Trace Replay**:
  - Synthetic arrival processes: Poisson, Bursty Gamma, and Constant.
  - Bundled LMSYS Chatbot Arena and RAG stress traces.
- **Visualizer & CLI**:
  - Rich terminal percentiles, ASCII Gantt iteration timeline, and JSON/Markdown export.
  - Load saturation sweeper (`sched-sim sweep`).
  - MkDocs Material documentation site.
