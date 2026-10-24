"""Tests for Paged KV-Cache BlockManager."""

from sched_sim.core.request import Request
from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.memory.block_manager import BlockManager


def test_block_allocation_and_free():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw, block_size=16)

    initial_free = bm.num_free_gpu_blocks
    req = Request(request_id="req_test", arrived_at=0.0, prompt_tokens=35, output_tokens=10)

    # 35 tokens require ceil(35 / 16) = 3 blocks
    assert bm.can_allocate(req, 35)
    success = bm.allocate(req, 35)
    assert success
    assert bm.num_free_gpu_blocks == initial_free - 3

    # Free blocks
    bm.free(req)
    assert bm.num_free_gpu_blocks == initial_free


def test_append_token_block_expansion():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw, block_size=16)

    req = Request(request_id="req_expand", arrived_at=0.0, prompt_tokens=15, output_tokens=5)
    bm.allocate(req, 15)  # 1 block allocated (holds 16)
    initial_blocks = len(bm.gpu_block_tables[req.request_id])
    assert initial_blocks == 1

    # Add 1st decode token (16 tokens total): still fits in block 1
    req.prefilled_tokens = 15
    req.decoded_tokens = 0
    bm.append_slot(req)
    assert len(bm.gpu_block_tables[req.request_id]) == 1

    # Add 2nd decode token (17 tokens total): must allocate block 2
    req.decoded_tokens = 1
    bm.append_slot(req)
    assert len(bm.gpu_block_tables[req.request_id]) == 2


def test_swapping_to_cpu():
    hw = get_hardware_profile("rtx3050")
    bm = BlockManager(hw, block_size=16)

    req = Request(request_id="req_swap", arrived_at=0.0, prompt_tokens=32, output_tokens=10)
    bm.allocate(req, 32)
    assert len(bm.gpu_block_tables[req.request_id]) == 2

    # Swap to CPU
    assert bm.can_swap_out(req)
    count = bm.swap_out(req)
    assert count == 2
    assert req.request_id not in bm.gpu_block_tables
    assert req.request_id in bm.cpu_block_tables

    # Swap back to GPU
    assert bm.can_swap_in(req)
    bm.swap_in(req)
    assert req.request_id in bm.gpu_block_tables
    assert req.request_id not in bm.cpu_block_tables
