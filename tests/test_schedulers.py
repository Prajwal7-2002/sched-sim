"""Tests comparing scheduling policies (FCFS, SJF, SLO-Aware, Disaggregated)."""

from sched_sim.core.event_engine import Simulator
from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler
from sched_sim.schedulers.disaggregated import DisaggregatedScheduler
from sched_sim.schedulers.sjf import SJFScheduler
from sched_sim.schedulers.slo_aware import SLOAwareScheduler


def test_sjf_prioritizes_shorter_prompts():
    hw = get_hardware_profile("rtx3050")

    # Request 1: Long prompt (1000 tokens)
    # Request 2: Short prompt (50 tokens) - both arrive at time 0
    req_long = Request(
        request_id="long", arrived_at=0.0, prompt_tokens=1000, output_tokens=5
    )
    req_short = Request(
        request_id="short", arrived_at=0.0, prompt_tokens=50, output_tokens=5
    )

    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm)
    sjf = SJFScheduler(bm, lm, ph, max_batch_size=1)  # only 1 request at a time

    sjf.add_request(req_long)
    sjf.add_request(req_short)

    decision = sjf.schedule_iteration(0.0)
    # In SJF, the short prompt must be scheduled first!
    assert decision.prefill_requests[0][0].request_id == "short"


def test_slo_aware_drops_expired_requests():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm)
    slo = SLOAwareScheduler(bm, lm, ph, max_batch_size=1)

    # Request arrived long ago and missed its SLO deadline
    req_late = Request(
        request_id="late",
        arrived_at=0.0,
        prompt_tokens=50,
        output_tokens=10,
        slo_ttft=0.10,  # 100ms deadline
    )
    slo.add_request(req_late)

    # At current_time = 0.5s (500ms), it's way past the 100ms deadline
    decision = slo.schedule_iteration(current_time=0.5)
    assert req_late in decision.dropped_requests
    assert req_late.state == RequestState.DROPPED


def test_disaggregated_scheduler():
    hw = get_hardware_profile("a100")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm)
    disagg = DisaggregatedScheduler(bm, lm, ph, max_batch_size=4)
    sim = Simulator(hw, disagg)

    req = Request(
        request_id="req_pd", arrived_at=0.0, prompt_tokens=200, output_tokens=10
    )
    sim.submit_workload([req])
    metrics = sim.run()

    assert req.state == RequestState.COMPLETED
    assert metrics.completed_requests == 1
