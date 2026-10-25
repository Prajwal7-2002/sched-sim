"""Example 2: Comparative Discrete-Event Simulation.

Simulates and benchmarks FCFS vs Cache-Aware (Radix Prefix Caching) on an A100 GPU:
1. Generates a multi-turn chat/RAG workload sharing a 512-token prompt prefix.
2. Measures TTFT (Time To First Token) reduction.
3. Quantifies throughput speedup and KV-cache memory savings.
"""

from sched_sim.core.event_engine import Simulator
from sched_sim.core.request import Request
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.hardware.presets import PRESETS
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler, PreemptionMode
from sched_sim.schedulers.cache_aware import CacheAwareScheduler
from sched_sim.schedulers.fcfs import FCFSScheduler


def create_workload(
    num_requests: int = 8, prompt_len: int = 512, output_len: int = 32
) -> list[Request]:
    """Create a realistic RAG / multi-turn workload sharing a common document context."""
    requests = []
    for i in range(num_requests):
        req = Request(
            request_id=f"rag_req_{i:02d}",
            arrived_at=i * 0.05,  # Arrive in rapid succession (50ms intervals)
            prompt_tokens=prompt_len,
            output_tokens=output_len,
            prefix_key="enterprise_q4_earnings_pdf",  # Shared context
        )
        requests.append(req)
    return requests


def run_fcfs_simulation(hw, requests) -> tuple:
    bm = BlockManager(hw, block_size=16, enable_prefix_caching=False)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    scheduler = FCFSScheduler(bm, lm, ph, max_batch_size=8)
    sim = Simulator(hw, scheduler)
    sim.submit_workload(requests)
    metrics = sim.run()
    return metrics, bm


def run_cache_aware_simulation(hw, requests) -> tuple:
    bm = BlockManager(hw, block_size=16, enable_prefix_caching=True)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    scheduler = CacheAwareScheduler(bm, lm, ph, max_batch_size=8)
    sim = Simulator(hw, scheduler)
    sim.submit_workload(requests)
    metrics = sim.run()
    return metrics, bm


def main() -> None:
    print("=" * 70)
    print("Example 2: Discrete-Event Simulation (FCFS vs Cache-Aware)")
    print("=" * 70)

    hw = PRESETS["a100"]
    num_reqs = 8
    prompt_len = 512
    out_len = 32

    print(f"\n[Setup] Hardware: {hw.name}")
    print(
        f"        Workload: {num_reqs} requests, {prompt_len} prompt tokens (Shared Context), {out_len} output tokens"
    )

    # Run Baseline: FCFS without caching
    print("\n[1/2] Running Baseline (FCFS, No Prefix Caching)...")
    reqs_baseline = create_workload(num_reqs, prompt_len, out_len)
    metrics_fcfs, _ = run_fcfs_simulation(hw, reqs_baseline)

    # Run Cache-Aware with Radix prefix caching
    print("[2/2] Running Cache-Aware Scheduler (SGLang Radix Tree)...")
    reqs_cache = create_workload(num_reqs, prompt_len, out_len)
    metrics_cache, bm_cache = run_cache_aware_simulation(hw, reqs_cache)

    # Results Comparison Table
    print("\n" + "=" * 70)
    print(f"{'Metric':<32} | {'FCFS (Baseline)':<16} | {'Cache-Aware':<16}")
    print("-" * 70)
    print(
        f"{'Mean TTFT (ms)':<32} | {metrics_fcfs.ttft_ms.mean:<16.2f} | {metrics_cache.ttft_ms.mean:<16.2f}"
    )
    print(
        f"{'p50 TTFT (ms)':<32} | {metrics_fcfs.ttft_ms.p50:<16.2f} | {metrics_cache.ttft_ms.p50:<16.2f}"
    )
    print(
        f"{'p99 TTFT (ms)':<32} | {metrics_fcfs.ttft_ms.p99:<16.2f} | {metrics_cache.ttft_ms.p99:<16.2f}"
    )
    print(
        f"{'Total Serving Time (s)':<32} | {metrics_fcfs.simulation_duration_s:<16.2f} | {metrics_cache.simulation_duration_s:<16.2f}"
    )
    print(
        f"{'Token Throughput (tok/s)':<32} | {metrics_fcfs.throughput_tok_per_s:<16.2f} | {metrics_cache.throughput_tok_per_s:<16.2f}"
    )
    print(
        f"{'Peak KV Utilization (%)':<32} | {metrics_fcfs.peak_kv_utilization_pct:<16.2f} | {metrics_cache.peak_kv_utilization_pct:<16.2f}"
    )

    cache = bm_cache.prefix_cache
    if cache:
        print(f"{'Prefix Cache Hit Rate':<32} | {'0.0%':<16} | {cache.hit_rate:<16.1%}")
        print(f"{'Total Cached Prefix Hits':<32} | {'0':<16} | {cache.total_hits:<16d}")
    print("=" * 70)

    ttft_speedup = (
        (metrics_fcfs.ttft_ms.mean - metrics_cache.ttft_ms.mean)
        / metrics_fcfs.ttft_ms.mean
        * 100
    )
    print(
        f"\n[INSIGHT] Cache-Aware scheduling reduced Mean TTFT by {ttft_speedup:.1f}%"
    )
    print("          via zero-copy KV-cache reuse and prefill skipping!")


if __name__ == "__main__":
    main()
