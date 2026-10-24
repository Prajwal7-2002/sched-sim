"""Chunked Prefill scheduler (Sarathi-Serve / vLLM token budgeting)."""

from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler
from sched_sim.schedulers.base import BaseScheduler, SchedulingDecision


class ChunkedPrefillScheduler(BaseScheduler):
    """Interleaves chunked prompt prefill tokens with decode tokens to bound TPOT jitter.

    Uses an iteration token budget (e.g. 512 or 1024 tokens) to ensure the GPU
    memory bandwidth and tensor cores remain smoothly saturated without starving decode streams.
    """

    def __init__(
        self,
        block_manager: BlockManager,
        latency_model: RooflineLatencyModel,
        preemption_handler: PreemptionHandler,
        max_batch_size: int = 32,
        max_token_budget: int = 512,
    ):
        super().__init__(
            block_manager, latency_model, preemption_handler, max_batch_size
        )
        self.max_token_budget = max_token_budget
        self.prefill_in_progress: list[Request] = []

    @property
    def name(self) -> str:
        return f"Chunked Prefill (Budget: {self.max_token_budget}t)"

    def add_request(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self.waiting_queue.append(request)

    def schedule_iteration(self, current_time: float) -> SchedulingDecision:
        decision = SchedulingDecision()
        tokens_remaining_budget = self.max_token_budget

        # Step 1: Decode requests have highest priority (1 token per active stream)
        still_running: list[Request] = []
        for req in list(self.running_batch):
            if tokens_remaining_budget <= 0:
                break

            if self.block_manager.append_slot(req):
                decision.decode_requests.append(req)
                still_running.append(req)
                tokens_remaining_budget -= 1
            else:
                victim, _ = self.preemption_handler.preempt_victim(self.running_batch)
                if victim:
                    decision.preempted_requests.append(victim)
                    if victim in still_running:
                        still_running.remove(victim)
                    if victim in decision.decode_requests:
                        decision.decode_requests.remove(victim)
                        tokens_remaining_budget += 1
                    self.add_request(victim)

                    if self.block_manager.append_slot(req):
                        decision.decode_requests.append(req)
                        still_running.append(req)
                        tokens_remaining_budget -= 1
                    else:
                        self.add_request(req)

        self.running_batch = still_running

        # Step 2: Fill remaining token budget with chunked prefills
        # First priority: finish any partially prefilled request
        active_prefills = list(self.prefill_in_progress)
        self.prefill_in_progress.clear()

        # Combine active partially prefilled requests with waiting queue
        candidates = active_prefills + self.waiting_queue
        admitted: list[Request] = []

        for req in candidates:
            if tokens_remaining_budget <= 0:
                if req in active_prefills:
                    self.prefill_in_progress.append(req)
                break

            needed_prefill = req.remaining_prefill_tokens
            if needed_prefill <= 0:
                continue

            chunk_size = min(needed_prefill, tokens_remaining_budget)

            # Check if block manager can allocate for this chunk
            if self.block_manager.can_allocate(req, chunk_size):
                self.block_manager.allocate(req, chunk_size)
                decision.prefill_requests.append((req, chunk_size))
                req.state = RequestState.PREFILLING
                tokens_remaining_budget -= chunk_size

                if req.prefilled_tokens + chunk_size < req.prompt_tokens:
                    # Request still has more prompt chunks for subsequent iterations
                    self.prefill_in_progress.append(req)
                admitted.append(req)
            else:
                if req in active_prefills:
                    self.prefill_in_progress.append(req)
                break

        for req in admitted:
            if req in self.waiting_queue:
                self.waiting_queue.remove(req)

        return decision

    def has_pending_work(self) -> bool:
        return bool(
            self.waiting_queue
            or self.running_batch
            or self.prefill_in_progress
            or self.swapped_queue
        )
