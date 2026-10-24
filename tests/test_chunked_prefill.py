"""Tests for Chunked Prefill token budgeting."""

from sched_sim.core.event_engine import Simulator
from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler
from sched_sim.schedulers.chunked import ChunkedPrefillScheduler


def test_chunked_prefill_slices_large_prompt():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm)

    # Budget of 256 tokens per iteration
    chunked = ChunkedPrefillScheduler(
        bm, lm, ph, max_batch_size=4, max_token_budget=256
    )

    # Request with 600 prompt tokens (requires 3 iterations: 256 + 256 + 88)
    req = Request(
        request_id="large_prompt", arrived_at=0.0, prompt_tokens=600, output_tokens=5
    )
    chunked.add_request(req)

    # Iteration 1
    d1 = chunked.schedule_iteration(0.0)
    assert len(d1.prefill_requests) == 1
    req1, tokens1 = d1.prefill_requests[0]
    assert tokens1 == 256
    req1.prefilled_tokens += tokens1
    assert not req1.is_prefill_complete

    # Iteration 2
    d2 = chunked.schedule_iteration(0.1)
    assert len(d2.prefill_requests) == 1
    req2, tokens2 = d2.prefill_requests[0]
    assert tokens2 == 256
    req2.prefilled_tokens += tokens2
    assert not req2.is_prefill_complete

    # Iteration 3
    d3 = chunked.schedule_iteration(0.2)
    assert len(d3.prefill_requests) == 1
    req3, tokens3 = d3.prefill_requests[0]
    assert tokens3 == 88
    req3.prefilled_tokens += tokens3
    assert req3.is_prefill_complete


def test_chunked_prefill_simulation_e2e():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm)
    chunked = ChunkedPrefillScheduler(bm, lm, ph, max_token_budget=256)
    sim = Simulator(hw, chunked)

    req = Request(
        request_id="e2e_chunk", arrived_at=0.0, prompt_tokens=500, output_tokens=10
    )
    sim.submit_workload([req])
    metrics = sim.run()

    assert req.state == RequestState.COMPLETED
    assert metrics.completed_requests == 1
    # Check that multiple iterations occurred
    assert len(sim.iteration_records) > 2
