"""Preemption policies: Recomputation vs. Host-DRAM Swapping."""

from enum import Enum

from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.memory.block_manager import BlockManager


class PreemptionMode(str, Enum):
    RECOMPUTE = "recompute"
    SWAP = "swap"


class PreemptionHandler:
    """Handles victim selection and preemption when GPU VRAM blocks are exhausted."""

    def __init__(
        self,
        block_manager: BlockManager,
        latency_model: RooflineLatencyModel,
        mode: PreemptionMode = PreemptionMode.RECOMPUTE,
    ):
        self.block_manager = block_manager
        self.latency_model = latency_model
        self.mode = mode

    def preempt_victim(
        self,
        active_requests: list[Request],
    ) -> tuple[Request | None, float]:
        """Select a victim request to preempt and free GPU memory.

        Victim policy: Pick the request with the least tokens generated (youngest decode stream),
        minimizing the amount of discarded progress or swapped data.
        Returns (preempted_request, delay_penalty_seconds).
        """
        if not active_requests:
            return None, 0.0

        # Sort by decoded tokens ascending (victim is the least progressed)
        victims = sorted(
            active_requests, key=lambda r: (r.decoded_tokens, -r.arrived_at)
        )
        victim = victims[0]

        victim.preemptions_count += 1
        victim.state = RequestState.PREEMPTED

        if self.mode == PreemptionMode.SWAP and self.block_manager.can_swap_out(victim):
            num_blocks = len(
                self.block_manager.gpu_block_tables.get(victim.request_id, [])
            )
            self.block_manager.swap_out(victim)
            swap_penalty = self.latency_model.compute_swap_time(
                num_blocks, self.block_manager.block_size
            )
            return victim, swap_penalty
        else:
            # Recompute fallback: discard KV blocks and record tokens that must be recomputed later
            victim.recomputed_tokens += victim.current_context_len
            self.block_manager.free(victim)
            victim.prefilled_tokens = 0
            victim.decoded_tokens = 0
            return victim, 0.0
