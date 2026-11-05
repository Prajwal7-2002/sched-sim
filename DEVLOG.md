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

## 2026-11-05: v0.2.3 - Out-of-sample validation against real serving runs

- **Problem**: `benchmarks/validate_flagships.py` derived bandwidth (Ollama) or launch overhead (mini-infer)
  from the same measured TPOT it then "predicted", so the published 3-15% errors were in-sample fits. The
  mini-infer trace also came from a version whose prefix cache was broken. The CI test read files from sibling
  repos that do not exist in CI, so it never ran there.
- **New data** (`benchmarks/data/`): servebench closed-loop sweeps (1/2/4 users, ~286-token prompts, 128 output
  tokens, 24 requests per level) on an RTX 3050 laptop: Ollama 0.35.1 with Qwen2.5 0.5B/1.5B/3B (q4_k_m) and
  mini-infer 0.2.9 with 0.5B/1.5B (bf16). Re-measuring mini-infer exposed a KV-pool leak (fixed in mini-infer 0.2.9).
- **Method**: fixed inputs (architecture, bytes per weight from the served file, 121.5 GB/s measured copy
  bandwidth); three engine constants fitted on the smallest model's 1- and 2-user runs only; everything else is
  held out. The simulator now runs the same closed loop as servebench (arrivals from simulated completions);
  replaying recorded arrival times made simulated queues grow without bound.
- **Model additions**: `weight_bytes_per_param` (4-bit weights with an fp16 KV-cache) and
  `per_sequence_overhead_ms` (batch of N costs more than batch of 1). Fixed T4 (300 -> 320 GB/s) and A100 SXM4
  80 GB (1935 -> 2039 GB/s) presets.
- **Results** (held out): Ollama TPOT within +/-12% on 6 of 7 rows (1.5B at 4 users +84%); TTFT +9% for 0.5B but
  +34% to +100% for 1.5B/3B (prefill efficiency fitted on 0.5B does not transfer). mini-infer TPOT -13% to +28%;
  TTFT -56% to -69% because mini-infer prefills each request in its own forward pass, unlike the simulator's
  fused batches.
- **Tests**: `tests/test_validation_benchmark.py` now runs on the bundled data and asserts the published bounds.
