"""Tests for roofline latency model."""

from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.hardware.latency_model import RooflineLatencyModel


def test_roofline_prefill_and_decode():
    hw = get_hardware_profile("rtx3050")
    model = RooflineLatencyModel(hw)

    t_prefill_100 = model.compute_prefill_step_time(100)
    t_prefill_1000 = model.compute_prefill_step_time(1000)

    # Longer prompt requires more FLOPs, therefore higher latency
    assert t_prefill_1000 > t_prefill_100
    assert t_prefill_100 > model.launch_overhead_s

    # Decode latency scales with active context length in KV cache
    t_decode_small_ctx = model.compute_decode_step_time(batch_size=4, total_context_tokens=200)
    t_decode_large_ctx = model.compute_decode_step_time(batch_size=4, total_context_tokens=4000)

    assert t_decode_large_ctx > t_decode_small_ctx


def test_kv_transfer_time():
    hw = get_hardware_profile("a100")
    model = RooflineLatencyModel(hw)

    t_transfer_512 = model.compute_kv_transfer_time(512)
    t_transfer_2048 = model.compute_kv_transfer_time(2048)

    assert t_transfer_2048 > t_transfer_512
