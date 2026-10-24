"""Comprehensive latency, throughput, and SLO metrics calculation for simulation runs."""

import math
from pydantic import BaseModel, Field
from sched_sim.core.request import Request, RequestState


def percentile(values: list[float], p: float) -> float:
    """Calculate the p-th percentile (0 <= p <= 100) using linear interpolation."""
    if not values:
        return 0.0
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


class Percentiles(BaseModel):
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    max: float = 0.0


class SimulationMetrics(BaseModel):
    """Holistic performance summary of an LLM serving scheduler execution."""

    scheduler_name: str
    hardware_name: str
    simulation_duration_s: float
    total_requests: int
    completed_requests: int
    dropped_requests: int
    preempted_requests_count: int
    total_recomputed_tokens: int

    # Throughput
    throughput_req_per_s: float
    throughput_tok_per_s: float

    # Percentiles (in milliseconds for human readability)
    ttft_ms: Percentiles
    tpot_ms: Percentiles
    e2e_ms: Percentiles

    # Cluster efficiency
    goodput_ratio: float = 0.0  # Percentage of requests meeting SLOs
    gpu_busy_time_s: float = 0.0
    gpu_utilization_pct: float = 0.0
    avg_batch_size: float = 0.0
    peak_kv_utilization_pct: float = 0.0
    total_iterations: int = 0


def compute_metrics(
    scheduler_name: str,
    hardware_name: str,
    duration_s: float,
    all_requests: list[Request],
    total_iterations: int,
    gpu_busy_time_s: float,
    iteration_batch_sizes: list[int],
    peak_kv_utilization: float,
) -> SimulationMetrics:
    """Aggregate simulation trace into structured statistical metrics."""
    completed = [r for r in all_requests if r.state == RequestState.COMPLETED]
    dropped = [r for r in all_requests if r.state == RequestState.DROPPED]

    ttft_list = sorted([r.ttft * 1000.0 for r in completed if r.ttft is not None])
    tpot_list = sorted([r.tpot * 1000.0 for r in completed if r.tpot is not None])
    e2e_list = sorted([r.e2e_latency * 1000.0 for r in completed if r.e2e_latency is not None])

    def calc_percentiles(arr: list[float]) -> Percentiles:
        if not arr:
            return Percentiles()
        return Percentiles(
            p50=round(percentile(arr, 50), 2),
            p90=round(percentile(arr, 90), 2),
            p95=round(percentile(arr, 95), 2),
            p99=round(percentile(arr, 99), 2),
            mean=round(sum(arr) / len(arr), 2),
            max=round(max(arr), 2),
        )

    # Tokens generated
    total_generated_tokens = sum(r.decoded_tokens for r in completed)
    throughput_tok_s = (total_generated_tokens / duration_s) if duration_s > 0 else 0.0
    throughput_req_s = (len(completed) / duration_s) if duration_s > 0 else 0.0

    # Goodput
    slo_evaluated = [r for r in completed if r.slo_ttft is not None or r.slo_tpot is not None]
    if slo_evaluated:
        goodput = sum(1 for r in slo_evaluated if r.slo_met) / len(slo_evaluated)
    else:
        goodput = 1.0 if completed else 0.0

    preemptions = sum(r.preemptions_count for r in all_requests)
    recomputed = sum(r.recomputed_tokens for r in all_requests)
    gpu_util = (gpu_busy_time_s / duration_s * 100.0) if duration_s > 0 else 0.0
    avg_batch = (sum(iteration_batch_sizes) / len(iteration_batch_sizes)) if iteration_batch_sizes else 0.0

    return SimulationMetrics(
        scheduler_name=scheduler_name,
        hardware_name=hardware_name,
        simulation_duration_s=round(duration_s, 3),
        total_requests=len(all_requests),
        completed_requests=len(completed),
        dropped_requests=len(dropped),
        preempted_requests_count=preemptions,
        total_recomputed_tokens=recomputed,
        throughput_req_per_s=round(throughput_req_s, 2),
        throughput_tok_per_s=round(throughput_tok_s, 2),
        ttft_ms=calc_percentiles(ttft_list),
        tpot_ms=calc_percentiles(tpot_list),
        e2e_ms=calc_percentiles(e2e_list),
        goodput_ratio=round(goodput * 100.0, 1),
        gpu_busy_time_s=round(gpu_busy_time_s, 3),
        gpu_utilization_pct=round(min(100.0, gpu_util), 1),
        avg_batch_size=round(avg_batch, 2),
        peak_kv_utilization_pct=round(peak_kv_utilization * 100.0, 1),
        total_iterations=total_iterations,
    )
