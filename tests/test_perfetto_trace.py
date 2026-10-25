"""Unit tests for Perfetto / Chromium trace exporter."""

import json
from pathlib import Path

from sched_sim.core.event_engine import Simulator
from sched_sim.core.request import Request
from sched_sim.hardware.presets import PRESETS
from sched_sim.schedulers.chunked import ChunkedPrefillScheduler
from sched_sim.visualizer.perfetto import export_perfetto_trace


def test_perfetto_trace_export(tmp_path: Path):
    hw = PRESETS["a100"]
    from sched_sim.hardware.latency_model import RooflineLatencyModel
    from sched_sim.memory.block_manager import BlockManager
    from sched_sim.memory.preemption import PreemptionHandler, PreemptionMode

    bm = BlockManager(hw, block_size=16)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    scheduler = ChunkedPrefillScheduler(
        bm, lm, ph, max_batch_size=4, max_token_budget=128
    )
    sim = Simulator(hw, scheduler)

    reqs = [
        Request(
            request_id="req_001", arrived_at=0.0, prompt_tokens=256, output_tokens=10
        ),
        Request(
            request_id="req_002", arrived_at=0.05, prompt_tokens=128, output_tokens=5
        ),
    ]
    sim.submit_workload(reqs)
    metrics = sim.run()

    trace_file = tmp_path / "test_trace.json"
    export_perfetto_trace(metrics, sim.iteration_records, sim.all_requests, trace_file)

    assert trace_file.exists()

    with open(trace_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "traceEvents" in data
    events = data["traceEvents"]
    assert len(events) > 5

    # Check for metadata events
    process_names = [
        e["args"]["name"] for e in events if e.get("name") == "process_name"
    ]
    assert any("GPU Accelerator" in name for name in process_names)
    assert any("Inference Request Queue" in name for name in process_names)

    # Check for complete slice events (ph == 'X')
    gpu_slices = [
        e for e in events if e.get("ph") == "X" and e.get("cat", "").startswith("gpu_")
    ]
    assert len(gpu_slices) > 0
    first_slice = gpu_slices[0]
    assert "ts" in first_slice
    assert "dur" in first_slice
    assert first_slice["dur"] > 0

    # Check for request lifecycle slices
    req_slices = [
        e for e in events if e.get("ph") == "X" and e.get("cat") == "requests"
    ]
    assert len(req_slices) == 2
    assert "req_001" in req_slices[0]["name"]

    # Check for Counter events (ph == 'C')
    counters = [e for e in events if e.get("ph") == "C"]
    assert len(counters) > 0
    assert any(c["name"] == "KV_Cache_Utilization" for c in counters)
    assert any(c["name"] == "Active_Batch_Concurrency" for c in counters)
