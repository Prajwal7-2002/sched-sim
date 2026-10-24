"""Synthetic workload arrival generator supporting Poisson, Bursty, and Constant traffic."""

import random

from sched_sim.core.request import Request
from sched_sim.workloads.distributions import DISTRIBUTIONS


def generate_workload(
    rate_req_per_s: float = 10.0,
    duration_s: float = 10.0,
    workload_type: str = "chatbot",
    pattern: str = "poisson",
    seed: int = 42,
) -> list[Request]:
    """Generate a reproducible stream of requests.

    Args:
        rate_req_per_s: Target arrival frequency (lambda).
        duration_s: Total time span of the arrival stream.
        workload_type: Distribution preset ('chatbot', 'rag', 'codegen', 'mixed').
        pattern: Arrival time pattern ('poisson', 'bursty', 'constant').
        seed: Random seed for deterministic simulation.
    """
    random.seed(seed)
    requests: list[Request] = []
    current_time = 0.0
    req_index = 0

    while current_time < duration_s:
        # Determine inter-arrival delta
        if pattern == "constant":
            delta = 1.0 / rate_req_per_s
        elif pattern == "bursty":
            # Gamma distribution creates arrival clustering (burstiness)
            shape = 0.4
            scale = (1.0 / rate_req_per_s) / shape
            delta = random.gammavariate(shape, scale)
        else:
            # Standard Poisson process (exponential inter-arrival)
            delta = random.expovariate(rate_req_per_s)

        current_time += delta
        if current_time >= duration_s:
            break

        req_index += 1

        # Select distribution
        if workload_type == "mixed":
            # 65% chatbot, 25% rag, 10% codegen
            r = random.random()
            if r < 0.65:
                preset = DISTRIBUTIONS["chatbot"]
            elif r < 0.90:
                preset = DISTRIBUTIONS["rag"]
            else:
                preset = DISTRIBUTIONS["codegen"]
        else:
            preset = DISTRIBUTIONS.get(workload_type.lower(), DISTRIBUTIONS["chatbot"])

        prompt_len, output_len = preset.sample_lengths()

        requests.append(
            Request(
                request_id=f"req_{req_index:04d}",
                arrived_at=round(current_time, 4),
                prompt_tokens=prompt_len,
                output_tokens=output_len,
                slo_ttft=preset.slo_ttft_s,
                slo_tpot=preset.slo_tpot_s,
            )
        )

    return requests
