"""Chromium / Perfetto Trace Event exporter for interactive GPU schedule visualization.

Outputs standard Catapult / Perfetto JSON format loadable directly into https://ui.perfetto.dev.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sched_sim.core.event_engine import IterationRecord
    from sched_sim.core.metrics import SimulationMetrics
    from sched_sim.core.request import Request


def export_perfetto_trace(
    metrics: "SimulationMetrics",
    iteration_records: list["IterationRecord"],
    all_requests: list["Request"],
    output_path: str | Path,
) -> None:
    """Export discrete forward pass execution records to Perfetto trace JSON format.

    Args:
        metrics: Aggregate simulation metrics.
        iteration_records: Forward pass timeline slices.
        all_requests: Full list of simulation requests with lifecycle records.
        output_path: Destination filepath for trace JSON.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trace_events: list[dict[str, Any]] = [
        # Process and thread metadata
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "args": {"name": f"GPU Accelerator ({metrics.hardware_name})"},
        },
        {
            "name": "thread_name",
            "ph": "M",
            "pid": 1,
            "tid": 1,
            "args": {"name": f"Forward Pass ({metrics.scheduler_name})"},
        },
        {
            "name": "process_name",
            "ph": "M",
            "pid": 2,
            "args": {"name": "Inference Request Queue"},
        },
        {
            "name": "thread_name",
            "ph": "M",
            "pid": 2,
            "tid": 1,
            "args": {"name": "Request Lifecycles"},
        },
    ]

    # 1. Forward Pass Slices on GPU
    for record in iteration_records:
        ts_us = round(record.start_time * 1e6)
        dur_us = max(1.0, round(record.duration * 1e6))

        if record.prefill_tokens > 0 and record.decode_count > 0:
            name = f"Mixed [Prefill {record.prefill_tokens}t + Decode {record.decode_count}s]"
            cat = "gpu_mixed"
        elif record.prefill_tokens > 0:
            name = f"Prefill [{record.prefill_tokens} tokens]"
            cat = "gpu_prefill"
        else:
            name = f"Decode [Batch {record.decode_count}]"
            cat = "gpu_decode"

        trace_events.append(
            {
                "name": name,
                "cat": cat,
                "ph": "X",  # Complete event
                "ts": ts_us,
                "dur": dur_us,
                "pid": 1,
                "tid": 1,
                "args": {
                    "iteration": record.iteration_index,
                    "prefill_tokens": record.prefill_tokens,
                    "decode_streams": record.decode_count,
                    "kv_utilization_pct": round(record.kv_utilization * 100, 1),
                    "active_request_ids": record.active_request_ids[:16],
                },
            }
        )

        # Counter: KV Cache memory utilization
        trace_events.append(
            {
                "name": "KV_Cache_Utilization",
                "ph": "C",
                "ts": ts_us,
                "pid": 1,
                "args": {"utilization_pct": round(record.kv_utilization * 100, 1)},
            }
        )

        # Counter: Active batch concurrency
        trace_events.append(
            {
                "name": "Active_Batch_Concurrency",
                "ph": "C",
                "ts": ts_us,
                "pid": 1,
                "args": {"streams": len(record.active_request_ids)},
            }
        )

    # 2. Per-request duration spans
    sim_duration = metrics.simulation_duration_s
    for req in all_requests:
        req_start_us = round(req.arrived_at * 1e6)
        end_time = req.completed_at if req.completed_at is not None else sim_duration
        req_dur_us = max(1.0, round((end_time - req.arrived_at) * 1e6))

        trace_events.append(
            {
                "name": f"{req.request_id} ({req.state.value})",
                "cat": "requests",
                "ph": "X",
                "ts": req_start_us,
                "dur": req_dur_us,
                "pid": 2,
                "tid": 1,
                "args": {
                    "request_id": req.request_id,
                    "prompt_tokens": req.prompt_tokens,
                    "output_tokens": req.output_tokens,
                    "cached_prefix_tokens": req.cached_prefix_tokens,
                    "ttft_ms": round(req.ttft * 1000, 2)
                    if req.ttft is not None
                    else None,
                    "tpot_ms": round(req.tpot * 1000, 2)
                    if req.tpot is not None
                    else None,
                    "slo_met": req.slo_met,
                },
            }
        )

    trace_data = {"traceEvents": trace_events}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)
