"""Export simulation metrics to JSON, CSV, and Markdown formats."""

import csv
import json
from pathlib import Path
from sched_sim.core.metrics import SimulationMetrics


def export_metrics_json(metrics: SimulationMetrics, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(metrics.model_dump_json(indent=2))


def export_metrics_markdown(metrics: SimulationMetrics, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# sched-sim Benchmark Report: {metrics.scheduler_name}

- **Hardware**: {metrics.hardware_name}
- **Duration**: {metrics.simulation_duration_s:.2f} s
- **Total Requests**: {metrics.total_requests} (Completed: {metrics.completed_requests}, Dropped: {metrics.dropped_requests})
- **Preemptions**: {metrics.preempted_requests_count} ({metrics.total_recomputed_tokens} recomputed tokens)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Throughput (Tokens/s)** | **{metrics.throughput_tok_per_s:.1f} tok/s** |
| **Throughput (Requests/s)** | **{metrics.throughput_req_per_s:.1f} req/s** |
| **SLO Goodput** | **{metrics.goodput_ratio:.1f}%** |
| **TTFT (p50 / p90 / p99)** | {metrics.ttft_ms.p50} / {metrics.ttft_ms.p90} / {metrics.ttft_ms.p99} ms |
| **TPOT (p50 / p90 / p99)** | {metrics.tpot_ms.p50} / {metrics.tpot_ms.p90} / {metrics.tpot_ms.p99} ms/tok |
| **E2E Latency (p50 / p99)** | {metrics.e2e_ms.p50} / {metrics.e2e_ms.p99} ms |
| **GPU Utilization** | {metrics.gpu_utilization_pct:.1f}% |
| **Avg Batch Size** | {metrics.avg_batch_size:.1f} streams |
| **Peak KV Cache Utilization**| {metrics.peak_kv_utilization_pct:.1f}% |
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
