"""Unit tests for CacheAwareScheduler and end-to-end prefix caching simulation."""

from sched_sim.core.event_engine import Simulator
from sched_sim.core.request import Request
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.hardware.presets import PRESETS
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler, PreemptionMode
from sched_sim.schedulers.cache_aware import CacheAwareScheduler
from sched_sim.schedulers.fcfs import FCFSScheduler


def test_cache_aware_scheduler_priority_ordering():
    hw = PRESETS["a100"]
    bm = BlockManager(hw, block_size=16, enable_prefix_caching=True)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    scheduler = CacheAwareScheduler(bm, lm, ph, max_batch_size=8)

    # Prime the prefix cache with a warm system prompt
    prime_req = Request(
        request_id="prime_000",
        arrived_at=0.0,
        prompt_tokens=64,
        output_tokens=5,
        prefix_key="system_ctx",
    )
    bm.allocate(prime_req, 64, current_time=0.0)

    # Now create two waiting requests:
    # req_cold arrives earlier (t=1.0) but has NO cached prefix
    req_cold = Request(
        request_id="req_cold",
        arrived_at=1.0,
        prompt_tokens=64,
        output_tokens=5,
        prefix_key="unique_cold_context",
    )
    # req_warm arrives later (t=2.0) but shares the warm system prompt
    req_warm = Request(
        request_id="req_warm",
        arrived_at=2.0,
        prompt_tokens=64,
        output_tokens=5,
        prefix_key="system_ctx",
    )

    scheduler.add_request(req_cold)
    scheduler.add_request(req_warm)

    # Check that sorting prioritizes req_warm over req_cold despite later arrival
    scheduler._sort_waiting_queue()
    assert scheduler.waiting_queue[0].request_id == "req_warm"
    assert scheduler.waiting_queue[1].request_id == "req_cold"


def test_e2e_simulation_prefix_caching_ttft_drop():
    hw = PRESETS["a100"]

    # Scenario 1: Run 2 identical requests under standard FCFS without prefix caching
    bm_fcfs = BlockManager(hw, block_size=16, enable_prefix_caching=False)
    lm_fcfs = RooflineLatencyModel(hw)
    ph_fcfs = PreemptionHandler(bm_fcfs, lm_fcfs, PreemptionMode.RECOMPUTE)
    sched_fcfs = FCFSScheduler(bm_fcfs, lm_fcfs, ph_fcfs, max_batch_size=4)
    sim_fcfs = Simulator(hw, sched_fcfs)

    reqs_fcfs = [
        Request(
            request_id="req_1",
            arrived_at=0.0,
            prompt_tokens=256,
            output_tokens=10,
            prefix_key="shared_rag_doc",
        ),
        Request(
            request_id="req_2",
            arrived_at=0.01,
            prompt_tokens=256,
            output_tokens=10,
            prefix_key="shared_rag_doc",
        ),
    ]
    sim_fcfs.submit_workload(reqs_fcfs)
    metrics_fcfs = sim_fcfs.run()

    # Scenario 2: Run identical 2 requests under CacheAwareScheduler with Radix prefix caching
    bm_cache = BlockManager(hw, block_size=16, enable_prefix_caching=True)
    lm_cache = RooflineLatencyModel(hw)
    ph_cache = PreemptionHandler(bm_cache, lm_cache, PreemptionMode.RECOMPUTE)
    sched_cache = CacheAwareScheduler(bm_cache, lm_cache, ph_cache, max_batch_size=4)
    sim_cache = Simulator(hw, sched_cache)

    reqs_cache = [
        Request(
            request_id="req_1",
            arrived_at=0.0,
            prompt_tokens=256,
            output_tokens=10,
            prefix_key="shared_rag_doc",
        ),
        Request(
            request_id="req_2",
            arrived_at=0.01,
            prompt_tokens=256,
            output_tokens=10,
            prefix_key="shared_rag_doc",
        ),
    ]
    sim_cache.submit_workload(reqs_cache)
    metrics_cache = sim_cache.run()

    # req_2 in cache-aware simulation should have 256 cached prefix tokens
    assert reqs_cache[1].cached_prefix_tokens == 256

    # Mean TTFT should be significantly lower under Cache-Aware due to prefill skipping
    assert metrics_cache.ttft_ms.mean < metrics_fcfs.ttft_ms.mean
    assert bm_cache.prefix_cache.total_hits >= 1
