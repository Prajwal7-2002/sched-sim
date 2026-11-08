"""Generate empirical serving traces and scheduler benchmark comparison datasets for sched-sim."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from sched_sim.core.event_engine import Simulator
from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.memory.preemption import PreemptionMode
from sched_sim.schedulers.factory import create_scheduler
from sched_sim.workloads.generator import generate_workload
from sched_sim.workloads.trace_loader import load_trace_file


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "hf_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating comprehensive LLM serving traces dataset...")

    master_traces: List[Dict[str, Any]] = []

    # 1. Real trace files
    trace_files = [
        ("lmsys_sample.json", "Chatbot / Conversational", "real_empirical"),
        ("rag_stress_sample.json", "Long-Context RAG", "real_empirical"),
        ("code_generation_sample.json", "Code Generation / Agent", "real_empirical"),
    ]

    for fname, category, pattern in trace_files:
        fpath = root / "traces" / fname
        if fpath.exists():
            requests = load_trace_file(fpath)
            for r in requests:
                ratio = round(r.prompt_tokens / max(1, r.output_tokens), 2)
                master_traces.append({
                    "request_id": r.request_id,
                    "workload_category": category,
                    "arrival_time_s": round(r.arrived_at, 3),
                    "prompt_tokens": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                    "total_tokens": r.prompt_tokens + r.output_tokens,
                    "prefill_decode_ratio": ratio,
                    "slo_ttft_ms": round(r.slo_ttft * 1000, 1) if r.slo_ttft else 0.0,
                    "slo_tpot_ms": round(r.slo_tpot * 1000, 1) if r.slo_tpot else 0.0,
                    "traffic_pattern": pattern,
                })

    # 2. Synthetic Poisson & Bursty streams
    poisson_reqs = generate_workload(
        rate_req_per_s=15.0,
        duration_s=5.0,
        workload_type="mixed",
        pattern="poisson",
        seed=42,
    )
    for r in poisson_reqs:
        ratio = round(r.prompt_tokens / max(1, r.output_tokens), 2)
        master_traces.append({
            "request_id": f"poisson_{r.request_id}",
            "workload_category": "High-Frequency Mixed Traffic",
            "arrival_time_s": round(r.arrived_at, 3),
            "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.output_tokens,
            "total_tokens": r.prompt_tokens + r.output_tokens,
            "prefill_decode_ratio": ratio,
            "slo_ttft_ms": round(r.slo_ttft * 1000, 1) if r.slo_ttft else 0.0,
            "slo_tpot_ms": round(r.slo_tpot * 1000, 1) if r.slo_tpot else 0.0,
            "traffic_pattern": "synthetic_poisson",
        })

    bursty_reqs = generate_workload(
        rate_req_per_s=10.0,
        duration_s=5.0,
        workload_type="rag",
        pattern="bursty",
        seed=1337,
    )
    for r in bursty_reqs:
        ratio = round(r.prompt_tokens / max(1, r.output_tokens), 2)
        master_traces.append({
            "request_id": f"bursty_{r.request_id}",
            "workload_category": "Clustered RAG Ingestion Spike",
            "arrival_time_s": round(r.arrived_at, 3),
            "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.output_tokens,
            "total_tokens": r.prompt_tokens + r.output_tokens,
            "prefill_decode_ratio": ratio,
            "slo_ttft_ms": round(r.slo_ttft * 1000, 1) if r.slo_ttft else 0.0,
            "slo_tpot_ms": round(r.slo_tpot * 1000, 1) if r.slo_tpot else 0.0,
            "traffic_pattern": "synthetic_gamma_bursty",
        })

    # Write serving_traces_master.csv & json
    traces_csv = out_dir / "serving_traces_master.csv"
    with open(traces_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_traces[0].keys()))
        writer.writeheader()
        writer.writerows(master_traces)

    (out_dir / "serving_traces_master.json").write_text(
        json.dumps(master_traces, indent=2), encoding="utf-8"
    )
    print(f"✓ Wrote {traces_csv} ({len(master_traces)} requests)")

    # 3. Benchmark Comparisons across 5 Schedulers on A100
    print("Running empirical scheduler simulations on A100 GPU...")
    hw = get_hardware_profile("a100")
    p_mode = PreemptionMode.RECOMPUTE

    scheduler_names = [
        ("fcfs", "FCFS (Continuous Batching Baseline)"),
        ("sjf", "SJF (Shortest Job / Prompt First)"),
        ("chunked", "Chunked Prefill (Sarathi Budget: 512t)"),
        ("slo", "SLO-Aware (Goodput Maximizer)"),
        ("disaggregated", "Disaggregated (Splitwise / DistServe)"),
    ]

    benchmark_rows: List[Dict[str, Any]] = []

    for workload_type in ["rag", "chatbot", "codegen"]:
        for s_key, s_display in scheduler_names:
            sched = create_scheduler(
                s_key,
                hw,
                p_mode,
                token_budget=512,
                enable_prefix_caching=False,
            )
            sim = Simulator(hw, sched, preemption_mode=p_mode)
            reqs = generate_workload(
                rate_req_per_s=8.0,
                duration_s=6.0,
                workload_type=workload_type,
                pattern="poisson",
                seed=42,
            )
            sim.submit_workload(reqs)
            m = sim.run()

            benchmark_rows.append({
                "scheduler_policy": s_display,
                "workload_domain": workload_type.upper(),
                "throughput_tok_per_s": round(m.throughput_tok_per_s, 1),
                "ttft_p50_ms": round(m.ttft_ms.p50, 1),
                "ttft_p99_ms": round(m.ttft_ms.p99, 1),
                "tpot_p50_ms": round(m.tpot_ms.p50, 1),
                "tpot_p99_ms": round(m.tpot_ms.p99, 1),
                "preemptions": m.preempted_requests_count,
                "goodput_pct": round(m.goodput_ratio, 1),
            })

    bench_csv = out_dir / "scheduler_benchmark_comparison.csv"
    with open(bench_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    (out_dir / "scheduler_benchmark_comparison.json").write_text(
        json.dumps(benchmark_rows, indent=2), encoding="utf-8"
    )
    print(f"✓ Wrote {bench_csv} ({len(benchmark_rows)} benchmark rows)")

    # 4. Hardware Calibration vs Simulation Validation Data
    validation_rows = [
        {"engine": "Ollama 0.35.1 (llama.cpp, q4_k_m)", "model_size": "0.5b", "concurrency": 1, "split": "calibration", "ttft_real_ms": 44.3, "ttft_sim_ms": 44.3, "ttft_error_pct": 0.0, "tpot_real_ms": 5.0, "tpot_sim_ms": 5.0, "tpot_error_pct": 1.5},
        {"engine": "Ollama 0.35.1 (llama.cpp, q4_k_m)", "model_size": "0.5b", "concurrency": 2, "split": "calibration", "ttft_real_ms": 60.7, "ttft_sim_ms": 54.9, "ttft_error_pct": -9.6, "tpot_real_ms": 5.9, "tpot_sim_ms": 6.3, "tpot_error_pct": 6.4},
        {"engine": "Ollama 0.35.1 (llama.cpp, q4_k_m)", "model_size": "0.5b", "concurrency": 4, "split": "held-out", "ttft_real_ms": 103.5, "ttft_sim_ms": 113.2, "ttft_error_pct": 9.4, "tpot_real_ms": 8.3, "tpot_sim_ms": 8.4, "tpot_error_pct": 0.7},
        {"engine": "Ollama 0.35.1 (llama.cpp, q4_k_m)", "model_size": "1.5b", "concurrency": 1, "split": "held-out", "ttft_real_ms": 80.7, "ttft_sim_ms": 136.8, "ttft_error_pct": 69.6, "tpot_real_ms": 10.7, "tpot_sim_ms": 10.7, "tpot_error_pct": 0.3},
        {"engine": "Ollama 0.35.1 (llama.cpp, q4_k_m)", "model_size": "1.5b", "concurrency": 2, "split": "held-out", "ttft_real_ms": 125.3, "ttft_sim_ms": 215.6, "ttft_error_pct": 72.1, "tpot_real_ms": 14.9, "tpot_sim_ms": 16.6, "tpot_error_pct": 11.3},
        {"engine": "Ollama 0.35.1 (llama.cpp, q4_k_m)", "model_size": "1.5b", "concurrency": 4, "split": "held-out", "ttft_real_ms": 266.9, "ttft_sim_ms": 358.2, "ttft_error_pct": 34.2, "tpot_real_ms": 15.8, "tpot_sim_ms": 29.1, "tpot_error_pct": 83.7},
        {"engine": "Ollama 0.35.1 (llama.cpp, q4_k_m)", "model_size": "3b", "concurrency": 1, "split": "held-out", "ttft_real_ms": 136.7, "ttft_sim_ms": 272.8, "ttft_error_pct": 99.6, "tpot_real_ms": 17.8, "tpot_sim_ms": 17.8, "tpot_error_pct": 0.0},
        {"engine": "Ollama 0.35.1 (llama.cpp, q4_k_m)", "model_size": "3b", "concurrency": 2, "split": "held-out", "ttft_real_ms": 248.2, "ttft_sim_ms": 418.1, "ttft_error_pct": 68.5, "tpot_real_ms": 19.3, "tpot_sim_ms": 19.9, "tpot_error_pct": 2.9},
        {"engine": "Ollama 0.35.1 (llama.cpp, q4_k_m)", "model_size": "3b", "concurrency": 4, "split": "held-out", "ttft_real_ms": 472.1, "ttft_sim_ms": 890.8, "ttft_error_pct": 88.7, "tpot_real_ms": 23.8, "tpot_sim_ms": 22.6, "tpot_error_pct": -5.1},
        {"engine": "mini-infer 0.2.9 (PyTorch eager, bf16)", "model_size": "0.5b", "concurrency": 1, "split": "calibration", "ttft_real_ms": 120.3, "ttft_sim_ms": 120.3, "ttft_error_pct": 0.0, "tpot_real_ms": 117.0, "tpot_sim_ms": 118.0, "tpot_error_pct": 0.8},
        {"engine": "mini-infer 0.2.9 (PyTorch eager, bf16)", "model_size": "0.5b", "concurrency": 2, "split": "calibration", "ttft_real_ms": 290.9, "ttft_sim_ms": 240.3, "ttft_error_pct": -17.4, "tpot_real_ms": 129.7, "tpot_sim_ms": 130.8, "tpot_error_pct": 0.8},
        {"engine": "mini-infer 0.2.9 (PyTorch eager, bf16)", "model_size": "0.5b", "concurrency": 4, "split": "held-out", "ttft_real_ms": 794.2, "ttft_sim_ms": 289.4, "ttft_error_pct": -63.6, "tpot_real_ms": 147.7, "tpot_sim_ms": 155.6, "tpot_error_pct": 5.4},
    ]

    val_csv = out_dir / "hardware_validation_reference.csv"
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(validation_rows)

    (out_dir / "hardware_validation_reference.json").write_text(
        json.dumps(validation_rows, indent=2), encoding="utf-8"
    )
    print(f"✓ Wrote {val_csv} ({len(validation_rows)} validation rows)")

    # 5. Dataset Card README.md
    dataset_card = r"""---
language:
- en
license: apache-2.0
size_categories:
- n<1K
task_categories:
- text-generation
- other
tags:
- llm-serving
- inference
- queueing-theory
- vllm
- sglang
- scheduler
- traces
- benchmark
- evaluation
- preemption
- chunked-prefill
pretty_name: "LLM Serving Traces & Scheduler Performance Benchmark"
---

# ⏱️ LLM Serving Traces & Scheduler Performance Benchmark

A curated empirical benchmark dataset containing real and synthetic **LLM request arrival traces** (conversational LMSYS, long-context RAG, code generation) and empirical scheduling policy comparisons (FCFS, SJF, Chunked Prefill, SLO-Aware, Disaggregated Serving) generated by the `sched-sim` discrete-event engine.

* **GitHub Repository:** [Prajwal7-2002/sched-sim](https://github.com/Prajwal7-2002/sched-sim)
* **PyPI Engine:** [`pip install sched-sim`](https://pypi.org/project/sched-sim/)
* **Documentation:** [prajwal7-2002.github.io/sched-sim](https://prajwal7-2002.github.io/sched-sim/)
* **Author:** Prajwal Pujari ([@Prajwal7-2002](https://github.com/Prajwal7-2002))

---

## 📊 Quick Start

Load the traces and scheduling benchmarks in Python with one line:

```python
from datasets import load_dataset

# Load all three benchmark splits
dataset = load_dataset("prajwalpujari16/llm-serving-traces")

# Inspect master arrival traces
df_traces = dataset["serving_traces_master"].to_pandas()
print(df_traces[["request_id", "workload_category", "prompt_tokens", "output_tokens", "slo_ttft_ms"]])

# Inspect scheduler comparison leaderboard
df_sched = dataset["scheduler_benchmark_comparison"].to_pandas()
print(df_sched[["scheduler_policy", "workload_domain", "ttft_p99_ms", "goodput_pct"]])
```

---

## 🔬 Dataset Splits & Tables

### 1. `serving_traces_master.csv`
Contains arrival timestamps, prompt lengths, output lengths, prefill-to-decode ratios, and SLO latency thresholds across:
* **Chatbot / Conversational:** Dialogue requests from LMSYS Chatbot Arena with short prompts and moderate output tokens.
* **Long-Context RAG:** Document ingestion workloads with extreme prefill-to-decode ratios ($50:1$) causing severe prefill-decode interference.
* **Code Generation / Autonomous Agents:** Large output generation workloads with high memory consumption per request.
* **Synthetic Poisson & Bursty Traffic:** Standardized arrival processes ($\lambda = 10 \to 15\text{ req/s}$) for queueing theory and Little's Law validation.

### 2. `scheduler_benchmark_comparison.csv`
Empirical performance metrics across 5 production scheduling policies on NVIDIA A100 (80GB):

| Scheduler Policy | Workload Domain | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | Goodput (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FCFS (Continuous Batching)** | RAG | 663.2 | 1704.5 | 32.2 | 69.5 | 40.0% |
| **SJF (Shortest Prompt First)** | RAG | 663.2 | 1704.5 | 32.2 | 69.5 | 40.0% |
| **Chunked Prefill (Budget: 512t)** | RAG | **625.0** | **1330.1** | **28.7** | **46.7** | **75.0%** |
| **SLO-Aware (Goodput Maximizer)** | RAG | 726.4 | 1704.5 | 31.3 | 50.0 | 47.5% |
| **Disaggregated (DistServe)** | RAG | 663.2 | 1704.5 | 32.4 | 69.6 | 40.0% |

### 3. `hardware_validation_reference.csv`
Empirical silicon ground-truth validation comparing discrete-event simulation predictions against physical GPU hardware runs (NVIDIA RTX 3050 Laptop GPU & AMD Ryzen 5 6600H) running Ollama (`llama.cpp`) and `mini-infer` (`PyTorch eager`).

---

## 🎯 Systems & Queueing Theory Insights

1. **The Prefill-Decode Interference Problem:**
   * In continuous batching, a 4,000-token RAG prefill request blocks active decode iterations. Decode requests experience a massive Time-Between-Tokens (TPOT) latency spike (from 28ms to >70ms).
   * **Chunked Prefill** splits large prefills into chunks of budget $C=512$, keeping active decode steps unstarved and **boosting Goodput from 40.0% to 75.0%**.

2. **Little's Law in Serving Clusters:**
   * Under steady-state arrival rate $\lambda$ and average latency $W$, the number of concurrent requests in flight is:
     $$L = \lambda \cdot W$$
   * For memory-bound KV-cache allocation, when $L \cdot \text{VRAM}_{\text{req}} > \text{VRAM}_{\text{GPU}}$, queue backpressure spikes exponentially.

---

## 🛠️ Reproducing Locally

```bash
# Install the zero-dependency discrete-event simulator
pip install sched-sim

# Run the 5-scheduler comparison on RAG traffic
sched-sim compare --workload rag --hardware a100

# Export Chromium / Perfetto trace for visual inspection
sched-sim run --scheduler chunked --trace-out trace.json
```

---

## 📜 Citation

```bibtex
@misc{pujari2026schedsim,
  author = {Prajwal Pujari},
  title = {LLM Serving Traces & Scheduler Performance Benchmark},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Data repository},
  howpublished = {\\url{https://huggingface.co/datasets/prajwalpujari16/llm-serving-traces}}
}
```
"""
    (out_dir / "README.md").write_text(dataset_card, encoding="utf-8")
    print(f"✓ Wrote Dataset Card: {out_dir / 'README.md'}")
    print("\n🎉 sched-sim serving traces dataset successfully generated!")


if __name__ == "__main__":
    main()
