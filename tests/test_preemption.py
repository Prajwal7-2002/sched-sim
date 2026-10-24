"""Tests for preemption mechanics (recompute vs swap)."""

from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler, PreemptionMode


def test_recomputation_preemption():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw, block_size=16)
    lm = RooflineLatencyModel(hw)
    handler = PreemptionHandler(bm, lm, mode=PreemptionMode.RECOMPUTE)

    req1 = Request(request_id="req1", arrived_at=0.0, prompt_tokens=32, output_tokens=10)
    req2 = Request(request_id="req2", arrived_at=0.1, prompt_tokens=32, output_tokens=10)

    bm.allocate(req1, 32)
    bm.allocate(req2, 32)
    req1.decoded_tokens = 5
    req2.decoded_tokens = 1  # victim because fewer tokens generated

    victim, penalty = handler.preempt_victim([req1, req2])
    assert victim == req2
    assert victim.state == RequestState.PREEMPTED
    assert victim.preemptions_count == 1
    assert victim.recomputed_tokens > 0
    # Victim's GPU blocks are freed
    assert victim.request_id not in bm.gpu_block_tables


def test_swap_preemption():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw, block_size=16)
    lm = RooflineLatencyModel(hw)
    handler = PreemptionHandler(bm, lm, mode=PreemptionMode.SWAP)

    req = Request(request_id="req_swap_test", arrived_at=0.0, prompt_tokens=32, output_tokens=10)
    bm.allocate(req, 32)

    victim, penalty = handler.preempt_victim([req])
    assert victim == req
    assert penalty > 0.0  # transfer delay over PCIe
    assert victim.request_id in bm.cpu_block_tables
