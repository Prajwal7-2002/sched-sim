<p align="center">
  <img src="./assets/banner.svg" alt="sched-sim banner" width="100%">
</p>

# sched-sim ⏱️

[![CI](https://github.com/Prajwal7-2002/sched-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwal7-2002/sched-sim/actions)
[![PyPI](https://img.shields.io/pypi/v/sched-sim.svg?color=blue)](https://pypi.org/project/sched-sim/)
[![Docs](https://github.com/Prajwal7-2002/sched-sim/actions/workflows/docs.yml/badge.svg)](https://prajwal7-2002.github.io/sched-sim/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/CPU-100%25_Deterministic-green.svg)](#)
[![Calibrated](https://img.shields.io/badge/Calibrated_With-servebench_%26_mini--infer-teal.svg)](#)

A high-fidelity, CPU-first **discrete-event simulator (DES) for LLM serving schedulers**.

In modern inference clusters (vLLM, TGI, TensorRT-LLM, SGLang, DistServe), **the scheduling policy is the single largest determinant of cluster throughput, latency, and hardware cost**. While continuous batching and Paged KV-caching provide the mechanism for dynamic execution, the scheduler determines:
1. Which requests enter the next forward iteration.
2. How large prompt prefills are chunked to prevent decode starvation.
3. What happens when the physical KV-cache runs out of memory (swapping to CPU DRAM vs eviction and recomputation).
4. When to disaggregate prefill and decode phases onto specialized GPU worker pools (Splitwise / DistServe architecture).

Testing these policies on real GPU clusters (e.g. 8x H100 nodes) is **slow, noisy, and expensive (\$20–\$50/hr)**. `sched-sim` simulates hours of multi-GPU serving traffic in **seconds on any standard CPU**, using analytical performance models calibrated on, and checked out of sample against, real serving runs measured with **[servebench](https://github.com/Prajwal7-2002/servebench)** and **[mini-infer](https://github.com/Prajwal7-2002/mini-infer)**.

---

## ⚡ Core Architecture

```
                          ┌────────────────────────────────────────────────────────┐
                          │                   Arrival Process                      │
                          │   (Poisson, Bursty Gamma, LMSYS / ShareGPT Traces)    │
                          └───────────────────────────┬────────────────────────────┘
                                                      │ Request arrives (prompt_len, output_len)
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │       Waiting Queue         │
                                       │  (FCFS, SJF, Priority, SLO) │
                                       └──────────────┬──────────────┘
                                                      │ Iteration Scheduler
                                                      ▼
                       ┌──────────────────────────────────────────────────────────────┐
                       │               Continuous Batching Step (Iteration)           │
                       │  ┌────────────────────────────┐ ┌──────────────────────────┐ │
                       │  │   Prefill Tokens (Chunked) │ │ Decode Tokens (1/req)    │ │
                       │  └────────────────────────────┘ └──────────────────────────┘ │
                       │       ▲                                      ▲               │
                       │       │ Paged KV Allocator                   │ Memory Check  │
                       │       │                                      │               │
                       │       ▼                                      ▼               │
                       │  ┌──────────────────────────┐   ┌──────────────────────────┐ │
                       │  │   Paged KV-Cache Pool    │   │   Preemption Subsystem   │ │
                       │  │ (Physical 16-tok Blocks) │   │ (Recompute vs CPU Swap)  │ │
                       │  └──────────────────────────┘   └──────────────────────────┘ │
                       └──────────────────────────────────────────────────────────────┘
```

---

## ⏱️ Quickstart in 30 Seconds

### Installation

```bash
# Install via pip
pip install sched-sim

# Or via uv
uv add sched-sim

# Or direct from source for development
git clone https://github.com/Prajwal7-2002/sched-sim.git
cd sched-sim
uv sync
```

### 1. Run a Simulation Scenario
Simulate **Chunked Prefill** serving a long-context RAG workload on an **NVIDIA A100 SXM4 (80GB)** cluster running Llama-3-8B:
```bash
sched-sim run --scheduler chunked --workload rag --hardware a100 --duration 10
```

> **Runs 100% on your CPU**: You do **not** need an A100 or any GPU to run `sched-sim`. It simulates hours of multi-GPU serving traffic in seconds on any standard CPU. The `--hardware` flag tells the discrete-event engine which cluster architecture to mathematically simulate (`a100`, `h100`, `t4`, `rtx3050`, or a custom JSON profile).

### 2. Compare Schedulers Side-by-Side (Identical Trace)
Benchmark all 5 scheduling policies on the exact same Poisson arrival trace:
```bash
sched-sim compare --schedulers fcfs,sjf,chunked,slo,disaggregated --workload rag --hardware a100
```

### 3. Replay Real-World Conversation Traces
Replay Chatbot Arena conversation traces through the simulator:
```bash
sched-sim trace ./traces/lmsys_sample.json --scheduler chunked
```

### 4. Sweep Load to Discover Saturation Knees
Discover the exact request rate where each scheduler hits its Goodput cliff:
```bash
sched-sim sweep --schedulers fcfs,chunked,slo,disaggregated --workload rag --min-rate 2 --max-rate 18 --step 4
```

### 5. Check Silicon Hardware Presets
Inspect calibrated hardware profiles:
```bash
sched-sim info
```

---

## 💡 Runnable Python Examples

Explore self-contained runnable scripts under [`examples/`](examples/):

* **[`examples/01_radix_prefix_caching.py`](examples/01_radix_prefix_caching.py)**: SGLang-style Radix tree prefix caching, prompt matching, and zero-copy block sharing.
* **[`examples/02_simulate_schedulers.py`](examples/02_simulate_schedulers.py)**: Side-by-side simulation comparing FCFS vs Cache-Aware scheduling on shared RAG contexts (69.5% TTFT reduction).
* **[`examples/03_perfetto_trace_export.py`](examples/03_perfetto_trace_export.py)**: Programmatic execution and Catapult JSON trace export for [ui.perfetto.dev](https://ui.perfetto.dev).

```bash
python examples/01_radix_prefix_caching.py
python examples/02_simulate_schedulers.py
python examples/03_perfetto_trace_export.py
```

---

## 🔬 The 6 Core Schedulers

| Scheduler | Architecture & Policy | Key Trade-Off |
| :--- | :--- | :--- |
| **`Cache-Aware`** | **Radix-Tree prefix caching** (SGLang / vLLM APC). Shares physical 16-token KV blocks with LRU eviction and prefix-affinity prioritization. | **Near-zero TTFT** for RAG, multi-turn chat, and shared system prompts. |
| **`Chunked`** | Token budgeting (Sarathi-Serve / vLLM). Slices prompt prefills into 512-token chunks. | Eliminates TPOT latency spikes and ensures steady iteration times. |
| **`SLO-Aware`** | Dynamic deadline scheduling (EDF) with selective dropping of unrecoverable requests. | Maximizes **Goodput** (% of requests meeting both TTFT and TPOT SLOs). |
| **`Disaggregated`**| Splitwise / DistServe architecture: separates compute-bound Prefill workers from memory-bound Decode workers. | Zero decode interference; models KV-cache interconnect transfer delay. |
| **`SJF`** | Shortest-Job-First (shortest prompt first). | Minimizes average waiting time and mean TTFT, but starves heavy RAG prompts. |
| **`FCFS`** | First-Come First-Served continuous batching with atomic prefills. | Simple, but large prompts cause Head-of-Line blocking and decode starvation. |

---

## 🎨 Interactive Perfetto / Chromium Timeline Visualization

Export native Chromium / Catapult Trace Event JSON files and inspect discrete forward-pass iteration slices and per-request lifecycles in **[ui.perfetto.dev](https://ui.perfetto.dev)**:

```bash
sched-sim run --scheduler cache_aware --workload rag --hardware a100 --trace-out trace.json
```

Drag and drop `trace.json` directly into **Perfetto** to visually analyze:
- **Prefill vs. Decode vs. Mixed Slices**: Microsecond-accurate forward pass durations on the GPU accelerator.
- **Request Streams**: Exact arrival, TTFT first-token timestamps, and completion times.
- **Counter Tracks**: Real-time GPU KV-cache memory utilization (%) and concurrent batch stream counts.

---

## 🛠️ Hardware Calibration (Analytical Roofline Model)

`sched-sim` estimates each step's latency from roofline mechanics plus two measured engine overheads:

1. **Prefill Latency ($T_{\text{prefill}}$)**: Compute-bound matrix multiplication:
   $$T_{\text{prefill}} = \max\left(\frac{2 \cdot P \cdot N}{\text{Effective TFLOPS}}, \frac{\text{Weights}}{\text{Bandwidth}}\right) + T_{\text{launch}}$$

2. **Decode Latency ($T_{\text{decode}}$)**: Memory-bandwidth bound weight and KV-cache gather:
   $$T_{\text{decode}} = \max\left(\frac{P \cdot b_w + L_{\text{total}} \cdot \text{KV\_bytes\_per\_token}}{\text{DRAM Bandwidth}}, \frac{2 P B}{\text{Effective TFLOPS}}\right) + T_{\text{launch}} + B \cdot T_{\text{seq}}$$

   $b_w$: bytes per weight (`weight_bytes_per_param`, e.g. ~0.6-0.8 for 4-bit GGUF; defaults to `dtype_bytes`). $T_{\text{seq}}$: per-sequence cost (`per_sequence_overhead_ms`: sampling, per-sequence attention and bookkeeping), so a batch of $B$ costs more than a batch of 1.

3. **Interconnect Transfer ($T_{\text{transfer}}$)**: Network serialization delay for PD Disaggregation:
   $$T_{\text{transfer}} = \frac{L_{\text{prompt}} \cdot \text{KV\_bytes\_per\_token}}{\text{Network Bandwidth}} + T_{\text{net\_latency}}$$

---

## 🔬 Validation Against Real Serving Runs (RTX 3050 6GB Laptop)

Measured with [servebench](https://github.com/Prajwal7-2002/servebench): closed loop at 1, 2 and 4 users, ~286-token prompts,
128 output tokens, 24 requests per level. Raw data in [`benchmarks/data/`](benchmarks/data/); full table in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

**Method (out of sample).** Model architecture, bytes per weight and memory bandwidth (121.5 GB/s, measured with a
device-to-device copy) are fixed inputs. Three engine constants (step overhead, per-sequence overhead, prefill efficiency)
are fitted on the **smallest model's 1- and 2-user runs only**. Every other row is a prediction that used no data from
that row. The simulator runs the same closed loop as servebench.

| Engine | Held-out rows | Decode TPOT error | Prefill TTFT error |
| :--- | :--- | :--- | :--- |
| **Ollama 0.35.1** (llama.cpp, 4-bit) | 0.5B at 4 users; 1.5B and 3B at 1/2/4 users | **within ±12% on 6 of 7 rows** (1.5B at 4 users: +84%) | +9% for 0.5B; **+34% to +100% for 1.5B/3B** |
| **mini-infer 0.2.9** (PyTorch, bf16) | 0.5B at 4 users; 1.5B at 1/2/4 users | −13% to +28% (engine is CPU-bound and noisy) | **−56% to −69%** |

What this tells you:
* **Decode under batching is predicted well** for an efficient engine: one 0.5B calibration predicts the 3B model's
  per-token latency within ±5% at every load.
* **Prefill does not transfer across model sizes.** Prefill efficiency fitted on 0.5B (27% of peak) is too low for
  larger models, whose bigger matrix multiplications use the GPU better, so TTFT is overpredicted. Calibrate prefill
  on a model close in size to the one you simulate.
* **The simulator assumes vLLM-style fused batches.** mini-infer prefills each waiting request in its own forward
  pass, so real TTFT under load is much higher than the fused-batch prediction. Simulated scheduler comparisons are
  only as good as that assumption for your engine.
* Earlier versions showed 3-15% "validation errors" that were in-sample: the script derived bandwidth and launch
  overhead from the same measured TPOT it then reported. That table has been replaced (v0.2.3).

Reproduce: `uv run python benchmarks/validate_flagships.py` (CI runs the same check in `tests/test_validation_benchmark.py`).

---

## 📊 Telemetry & Reports

Every run outputs comprehensive telemetry:
* **Latency Percentiles**: TTFT (p50, p90, p99 ms), TPOT (p50, p90, p99 ms/tok), End-to-End latency.
* **Goodput Ratio**: Exact percentage of requests meeting both TTFT and TPOT Service Level Objectives.
* **Preemptions & Recomputations**: Total memory thrashing events and discarded tokens.
* **Visual Gantt Timeline**: Live ASCII/Unicode GPU iteration breakdowns.
* **Export**: JSON (`--output report.json`) and Markdown (`--output report.md`).

---

## 🧪 Testing

Run the deterministic test suite:
```bash
uv run pytest
```

---

## 🤝 Contributing

Contributions from systems engineers and researchers are welcome! Check out the [Contributing Guide](CONTRIBUTING.md) for architectural invariants, testing standards, and high-impact areas where we need help (new scheduling policies, modern hardware presets, and interconnect topologies).

---

## 📄 License

Apache 2.0. See [LICENSE](LICENSE) for details.
