"""Build the sched-sim Hugging Face dataset.

- serving_traces_master: synthetic Poisson and bursty workloads plus the small hand-written sample
  traces bundled in traces/ (illustrative shapes, not recorded from real users).
- scheduler_benchmark_comparison: simulated results (A100 preset), not measurements.
- hardware_validation_reference: real runs vs simulation, read from
  benchmarks/validation_results.json.
"""

import csv
import json
from pathlib import Path
from typing import Any

from sched_sim.core.event_engine import Simulator
from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.memory.preemption import PreemptionMode
from sched_sim.schedulers.factory import create_scheduler
from sched_sim.workloads.generator import generate_workload
from sched_sim.workloads.trace_loader import load_trace_file


TOLERANCE_PCT = 12.0  # band used for the "within ±X%" summary of held-out rows


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(h for _, h in cols) + " |\n"
    sep = "| " + " | ".join(":---" if i == 0 else "---:" for i in range(len(cols))) + " |\n"
    body = "".join("| " + " | ".join(str(r[k]) for k, _ in cols) + " |\n" for r in rows)
    return head + sep + body


def build_dataset_card(
    traces: list[dict[str, Any]],
    bench: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    hw: Any,
) -> str:
    from collections import Counter

    pattern_counts = Counter(t["traffic_pattern"] for t in traces)
    held = [v for v in validation if v["split"] == "held-out"]
    tpot_ok = sum(abs(v["tpot_error_pct"]) <= TOLERANCE_PCT for v in held)
    ttft_ok = sum(abs(v["ttft_error_pct"]) <= TOLERANCE_PCT for v in held)
    rag = [b for b in bench if b["workload_domain"] == "RAG"]

    yaml = """---
language:
- en
license: apache-2.0
size_categories:
- n<1K
task_categories:
- other
tags:
- llm-serving
- inference
- scheduler
- simulation
- traces
- chunked-prefill
pretty_name: "sched-sim: workload traces, simulated scheduler results and validation against real runs"
configs:
- config_name: serving_traces
  data_files:
  - split: train
    path: serving_traces_master.json
  default: true
- config_name: scheduler_benchmarks
  data_files:
  - split: train
    path: scheduler_benchmark_comparison.json
- config_name: hardware_validation
  data_files:
  - split: train
    path: hardware_validation_reference.json
---
"""
    return (
        yaml
        + f"""
# sched-sim: workload traces, simulated scheduler results and validation

Three tables from the [`sched-sim`](https://github.com/Prajwal7-2002/sched-sim) discrete-event
simulator of LLM serving schedulers. Read what each one is before using it: only the validation table
contains real measurements.

| Config | What it is | Rows |
|:---|:---|---:|
| `serving_traces` | Workload shapes: synthetic arrivals plus small hand-written samples. **Not recorded from real users.** | {len(traces)} |
| `scheduler_benchmarks` | **Simulated** results for 5 schedulers. Not measured on hardware. | {len(bench)} |
| `hardware_validation` | **Real runs** on one laptop GPU compared with the simulator's predictions | {len(validation)} |

```python
from datasets import load_dataset

traces = load_dataset("prajwalpujari16/llm-serving-traces", "serving_traces", split="train")
sim = load_dataset("prajwalpujari16/llm-serving-traces", "scheduler_benchmarks", split="train")
val = load_dataset("prajwalpujari16/llm-serving-traces", "hardware_validation", split="train")
```

## 1. `serving_traces`

| Traffic pattern | Requests |
|:---|---:|
"""
        + "".join(f"| `{k}` | {v} |\n" for k, v in sorted(pattern_counts.items()))
        + f"""
`synthetic_*` rows come from sched-sim's workload generator. `illustrative_sample` rows are the small
hand-written traces bundled in the repo (`traces/`), shaped like chat, RAG and code workloads; they are
not sampled from LMSYS or any real service.

## 2. `scheduler_benchmarks` (simulated)

Simulated with the `a100` hardware preset ({hw.name}, model profile `{hw.model_name}`), Poisson
arrivals at 8 requests/s for 6 s, seed 42, recompute preemption, token budget 512. These are the
simulator's outputs, not hardware measurements. RAG workload:

"""
        + _table(
            rag,
            [
                ("scheduler_policy", "Scheduler"),
                ("ttft_p50_ms", "TTFT p50 (ms)"),
                ("ttft_p99_ms", "TTFT p99 (ms)"),
                ("tpot_p50_ms", "TPOT p50 (ms)"),
                ("tpot_p99_ms", "TPOT p99 (ms)"),
                ("goodput_pct", "Goodput (%)"),
            ],
        )
        + f"""
All workloads (RAG, chatbot, codegen) are in the file.

## 3. `hardware_validation` (real runs vs simulation)

Measured with `servebench` on an NVIDIA GeForce RTX 3050 6 GB Laptop GPU (AMD Ryzen 5 6600H): closed
loop, ~286-token prompts, 128 output tokens, 24 requests per level. **calibration** rows were used to
fit each engine's constants; **held-out** rows were not used and are true predictions.

Held-out rows: TPOT within ±{TOLERANCE_PCT:.0f}% in **{tpot_ok} of {len(held)}**; TTFT within
±{TOLERANCE_PCT:.0f}% in **{ttft_ok} of {len(held)}**. Decode-speed (TPOT) predictions are closer
than time-to-first-token (TTFT) predictions, which are off by a wide margin on most held-out rows.

"""
        + _table(
            validation,
            [
                ("engine", "Engine"),
                ("model_size", "Model"),
                ("concurrency", "Users"),
                ("split", "Row"),
                ("ttft_real_ms", "TTFT real (ms)"),
                ("ttft_sim_ms", "TTFT sim (ms)"),
                ("ttft_error_pct", "TTFT err (%)"),
                ("tpot_real_ms", "TPOT real (ms)"),
                ("tpot_sim_ms", "TPOT sim (ms)"),
                ("tpot_error_pct", "TPOT err (%)"),
            ],
        )
        + r"""
## Limitations

- Validation covers one laptop GPU, two engines and models up to 3B. Other hardware is untested.
- Scheduler comparisons are simulated; differences between policies are the model's predictions.
- Traces are synthetic or hand-written, so they show workload shapes, not real user behaviour.

## Reproduce

```bash
pip install sched-sim
sched-sim compare --workload rag --hardware a100          # simulated comparison
uv run python benchmarks/validate_flagships.py            # validation (from a repo checkout)
uv run python scripts/generate_traces_dataset.py          # rebuild this dataset
```

## Citation

```bibtex
@misc{pujari2026schedsim,
  author = {Prajwal Pujari},
  title = {sched-sim: LLM Serving Workload Traces, Simulated Scheduler Results and Validation},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/prajwalpujari16/llm-serving-traces}}
}
```
"""
    )


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "hf_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating comprehensive LLM serving traces dataset...")

    master_traces: list[dict[str, Any]] = []

    # 1. Bundled sample trace files (hand-written, illustrative)
    trace_files = [
        ("lmsys_sample.json", "Chatbot / Conversational", "illustrative_sample"),
        ("rag_stress_sample.json", "Long-Context RAG", "illustrative_sample"),
        ("code_generation_sample.json", "Code Generation / Agent", "illustrative_sample"),
    ]

    for fname, category, pattern in trace_files:
        fpath = root / "traces" / fname
        if fpath.exists():
            requests = load_trace_file(fpath)
            for r in requests:
                ratio = round(r.prompt_tokens / max(1, r.output_tokens), 2)
                master_traces.append(
                    {
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
                    }
                )

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
        master_traces.append(
            {
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
            }
        )

    bursty_reqs = generate_workload(
        rate_req_per_s=10.0,
        duration_s=5.0,
        workload_type="rag",
        pattern="bursty",
        seed=1337,
    )
    for r in bursty_reqs:
        ratio = round(r.prompt_tokens / max(1, r.output_tokens), 2)
        master_traces.append(
            {
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
            }
        )

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
    print("Simulating 5 schedulers on the A100 preset (simulated, not measured)...")
    hw = get_hardware_profile("a100")
    p_mode = PreemptionMode.RECOMPUTE

    scheduler_names = [
        ("fcfs", "FCFS (Continuous Batching Baseline)"),
        ("sjf", "SJF (Shortest Job / Prompt First)"),
        ("chunked", "Chunked Prefill (Sarathi Budget: 512t)"),
        ("slo", "SLO-Aware (Goodput Maximizer)"),
        ("disaggregated", "Disaggregated (Splitwise / DistServe)"),
    ]

    benchmark_rows: list[dict[str, Any]] = []

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

            benchmark_rows.append(
                {
                    "scheduler_policy": s_display,
                    "workload_domain": workload_type.upper(),
                    "throughput_tok_per_s": round(m.throughput_tok_per_s, 1),
                    "ttft_p50_ms": round(m.ttft_ms.p50, 1),
                    "ttft_p99_ms": round(m.ttft_ms.p99, 1),
                    "tpot_p50_ms": round(m.tpot_ms.p50, 1),
                    "tpot_p99_ms": round(m.tpot_ms.p99, 1),
                    "preemptions": m.preempted_requests_count,
                    "goodput_pct": round(m.goodput_ratio, 1),
                }
            )

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
    # Read every row produced by benchmarks/validate_flagships.py (never hand-copied).
    raw_validation = json.loads(
        (root / "benchmarks" / "validation_results.json").read_text(encoding="utf-8")
    )
    validation_rows = [
        {
            "engine": r["engine"],
            "model_size": r["model"],
            "concurrency": r["users"],
            "split": "held-out" if r["held_out"] else "calibration",
            "ttft_real_ms": r["real_ttft_ms"],
            "ttft_sim_ms": r["sim_ttft_ms"],
            "ttft_error_pct": r["err_ttft_ms_pct"],
            "tpot_real_ms": r["real_tpot_ms"],
            "tpot_sim_ms": r["sim_tpot_ms"],
            "tpot_error_pct": r["err_tpot_ms_pct"],
        }
        for r in raw_validation
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

    # 5. Dataset Card README.md (every table and count is computed from the rows above)
    dataset_card = build_dataset_card(master_traces, benchmark_rows, validation_rows, hw)
    (out_dir / "README.md").write_text(dataset_card, encoding="utf-8")
    print(f"✓ Wrote Dataset Card: {out_dir / 'README.md'}")
    print("\n🎉 sched-sim serving traces dataset successfully generated!")


if __name__ == "__main__":
    main()
