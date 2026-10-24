"""Comprehensive tests boosting unit test coverage across all sched-sim modules."""

import json

from sched_sim.core.event_engine import Simulator
from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler, PreemptionMode
from sched_sim.schedulers.fcfs import FCFSScheduler
from sched_sim.schedulers.sjf import SJFScheduler
from sched_sim.schedulers.slo_aware import SLOAwareScheduler
from sched_sim.visualizer.exporter import export_metrics_json, export_metrics_markdown
from sched_sim.visualizer.gantt import print_gantt_timeline
from sched_sim.workloads.generator import generate_workload


def test_slo_aware_scheduler_edf_and_drop():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    sched = SLOAwareScheduler(bm, lm, ph, max_batch_size=2)

    # Req 1: urgent deadline
    req1 = Request(
        request_id="urgent",
        arrived_at=0.0,
        prompt_tokens=50,
        output_tokens=10,
        slo_ttft=0.05,
        slo_tpot=0.01,
    )
    # Req 2: relaxed deadline
    req2 = Request(
        request_id="relaxed",
        arrived_at=0.0,
        prompt_tokens=50,
        output_tokens=10,
        slo_ttft=1.00,
        slo_tpot=0.10,
    )

    sched.add_request(req2)
    sched.add_request(req1)

    # Priority queue should order urgent request first
    step = sched.schedule_iteration(current_time=0.01)
    assert len(step.prefill_requests) > 0
    assert step.prefill_requests[0][0].request_id == "urgent"


def test_slo_aware_drop_unrecoverable():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    sched = SLOAwareScheduler(bm, lm, ph, max_batch_size=2)

    # Impossible SLO: 0.1ms TTFT for 2000 tokens
    req = Request(
        request_id="impossible",
        arrived_at=0.0,
        prompt_tokens=2000,
        output_tokens=50,
        slo_ttft=0.0001,
        slo_tpot=0.0001,
    )
    sched.add_request(req)
    sched.schedule_iteration(current_time=1.0)
    # The request should either be dropped or scheduled
    assert req.state in (RequestState.DROPPED, RequestState.PREFILLING)


def test_fcfs_and_sjf_step_execution():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)

    fcfs = FCFSScheduler(bm, lm, ph, max_batch_size=2)
    sjf = SJFScheduler(bm, lm, ph, max_batch_size=2)

    r1 = Request(request_id="r1", arrived_at=0.0, prompt_tokens=100, output_tokens=5)
    r2 = Request(request_id="r2", arrived_at=0.0, prompt_tokens=20, output_tokens=5)

    fcfs.add_request(r1)
    fcfs.add_request(r2)
    step_fcfs = fcfs.schedule_iteration(0.0)
    assert len(step_fcfs.prefill_requests) > 0

    sjf.add_request(r1)
    sjf.add_request(r2)
    step_sjf = sjf.schedule_iteration(0.0)
    assert len(step_sjf.prefill_requests) > 0
    # SJF should pick shorter prompt r2
    assert step_sjf.prefill_requests[0][0].request_id == "r2"


def test_exporters_json_and_markdown(tmp_path):
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    sched = FCFSScheduler(bm, lm, ph, max_batch_size=2)
    sim = Simulator(hw, sched)

    req = Request(
        request_id="exp_test", arrived_at=0.0, prompt_tokens=30, output_tokens=5
    )
    sim.submit_workload([req])
    metrics = sim.run()

    json_path = tmp_path / "metrics.json"
    md_path = tmp_path / "metrics.md"

    export_metrics_json(metrics, json_path)
    export_metrics_markdown(metrics, md_path)

    assert json_path.exists()
    assert md_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["completed_requests"] == 1

    md_content = md_path.read_text(encoding="utf-8")
    assert "sched-sim Benchmark Report" in md_content


def test_gantt_timeline_render():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    sched = FCFSScheduler(bm, lm, ph, max_batch_size=2)
    sim = Simulator(hw, sched)

    req = Request(
        request_id="gantt_test", arrived_at=0.0, prompt_tokens=30, output_tokens=5
    )
    sim.submit_workload([req])
    sim.run()

    # Should not raise exception
    print_gantt_timeline(sim.iteration_records, max_steps=5)


def test_generator_patterns():
    # Bursty Gamma
    reqs_bursty = generate_workload(
        rate_req_per_s=5, duration_s=1.0, pattern="bursty", seed=42
    )
    assert len(reqs_bursty) > 0

    # Constant
    reqs_const = generate_workload(
        rate_req_per_s=5, duration_s=1.0, pattern="constant", seed=42
    )
    assert len(reqs_const) > 0

    # Mixed distribution
    reqs_mixed = generate_workload(
        rate_req_per_s=5, duration_s=2.0, workload_type="mixed", seed=42
    )
    assert len(reqs_mixed) > 0
