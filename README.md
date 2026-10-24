# sched-sim ⏱️

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

Testing these policies on real GPU clusters (e.g. 8x H100 nodes) is **slow, noisy, and expensive ($20–$50/hr)**. `sched-sim` simulates hours of multi-GPU serving traffic in **seconds on any standard CPU**, using validated analytical performance models calibrated against empirical silicon measurements from **[servebench](https://github.com/Prajwal7-2002/servebench)** and **[mini-infer](https://github.com/Prajwal7-2002/mini-infer)**.

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

### Install
```bash
# Direct from source with uv (fastest)
git clone https://github.com/Prajwal7-2002/sched-sim.git
cd sched-sim
uv sync

# Or with standard pip
pip install -e .
```

### 1. Run a Simulation Scenario
Simulate **Chunked Prefill** serving a long-context RAG workload on an **RTX 3050 6GB Laptop GPU**:
```bash
sched-sim run --scheduler chunked --workload rag --hardware rtx3050 --duration 10
```

### 2. Compare Schedulers Side-by-Side (Identical Trace)
Benchmark all 5 scheduling policies on the exact same Poisson arrival trace:
```bash
sched-sim compare --schedulers fcfs,sjf,chunked,slo,disaggregated --workload rag --hardware a100
```

### 3. Check Silicon Hardware Presets
Inspect calibrated hardware profiles:
```bash
sched-sim info
```

---

## 🔬 The 5 Core Schedulers

| Scheduler | Architecture & Policy | Key Trade-Off |
| :--- | :--- | :--- |
| **`FCFS`** | First-Come First-Served continuous batching with atomic prefills. | Simple, but large prompts cause Head-of-Line blocking and decode starvation. |
| **`SJF`** | Shortest-Job-First (shortest prompt first). | Minimizes average waiting time and mean TTFT, but starves heavy RAG prompts. |
| **`Chunked`** | Token budgeting (Sarathi-Serve / vLLM). Slices prompt prefills into 512-token chunks. | Eliminates TPOT latency spikes and ensures steady iteration times. |
| **`SLO-Aware`** | Dynamic deadline scheduling (EDF) with selective dropping of unrecoverable requests. | Maximizes **Goodput** (% of requests meeting both TTFT and TPOT SLOs). |
| **`Disaggregated`**| Splitwise / DistServe architecture: separates compute-bound Prefill workers from memory-bound Decode workers. | Zero decode interference; models KV-cache interconnect transfer delay. |

---

## 🛠️ Hardware Calibration (Analytical Roofline Model)

`sched-sim` avoids arbitrary delays. It computes exact step latencies from first-principles roofline mechanics calibrated with empirical silicon data:

1. **Prefill Latency ($T_{\text{prefill}}$)**: Compute-bound matrix multiplication:
   $$T_{\text{prefill}} = \max\left(\frac{2 \cdot P \cdot N}{\text{Effective TFLOPS}}, \frac{\text{Weights}}{\text{Bandwidth}}\right) + T_{\text{launch}}$$

2. **Decode Latency ($T_{\text{decode}}$)**: Memory-bandwidth bound weight and KV-cache gather:
   $$T_{\text{decode}} = \frac{P \cdot \text{dtype} + L_{\text{total}} \cdot \text{KV\_bytes\_per\_token}}{\text{DRAM Bandwidth (GB/s)}} + T_{\text{launch}}$$

3. **Interconnect Transfer ($T_{\text{transfer}}$)**: Network serialization delay for PD Disaggregation:
   $$T_{\text{transfer}} = \frac{L_{\text{prompt}} \cdot \text{KV\_bytes\_per\_token}}{\text{Network Bandwidth}} + T_{\text{net\_latency}}$$

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

## 📄 License

Apache 2.0. See [LICENSE](LICENSE) for details.
