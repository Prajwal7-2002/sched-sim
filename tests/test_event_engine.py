"""Tests for discrete-event simulator engine."""

from sched_sim.core.event_engine import Simulator
from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler
from sched_sim.schedulers.fcfs import FCFSScheduler


def test_request_lifecycle_and_metrics():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm)
    sched = FCFSScheduler(bm, lm, ph)
    sim = Simulator(hw, sched)

    req = Request(
        request_id="req_001",
        arrived_at=0.0,
        prompt_tokens=100,
        output_tokens=10,
        slo_ttft=0.5,
        slo_tpot=0.05,
    )
    sim.submit_workload([req])
    metrics = sim.run()

    assert req.state == RequestState.COMPLETED
    assert req.ttft is not None and req.ttft > 0
    assert req.tpot is not None and req.tpot > 0
    assert req.e2e_latency is not None and req.e2e_latency > req.ttft
    assert metrics.completed_requests == 1
    assert metrics.dropped_requests == 0
    assert metrics.throughput_tok_per_s > 0
