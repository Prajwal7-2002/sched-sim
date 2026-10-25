"""Prefix-Cache-Aware scheduler prioritizing warm KV-cache prefixes and shortest remaining prefill."""

from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler
from sched_sim.memory.prefix_cache import PrefixCache
from sched_sim.schedulers.base import BaseScheduler, SchedulingDecision


class CacheAwareScheduler(BaseScheduler):
    """Schedules requests by prioritizing warm Radix prefix-cache hits and short remaining prefills.

    Minimizes TTFT and avoids cache thrashing by grouping requests that share prompt
    prefixes (e.g. system instructions, few-shot examples, RAG documents).
    """

    def __init__(
        self,
        block_manager: BlockManager,
        latency_model: RooflineLatencyModel,
        preemption_handler: PreemptionHandler,
        max_batch_size: int = 32,
    ):
        super().__init__(
            block_manager, latency_model, preemption_handler, max_batch_size
        )
        # Ensure prefix caching is active in block manager
        self.block_manager.enable_prefix_caching = True
        if self.block_manager.prefix_cache is None:
            self.block_manager.prefix_cache = PrefixCache(
                block_size=self.block_manager.block_size
            )

    @property
    def name(self) -> str:
        return "Cache-Aware (Prefix Affinity)"

    def add_request(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self.waiting_queue.append(request)

    def _sort_waiting_queue(self) -> None:
        """Sort waiting requests by warm prefix match length (desc), then remaining prompt (asc)."""
        cache = self.block_manager.prefix_cache

        def priority_key(req: Request) -> tuple[int, int, float]:
            if cache is not None:
                matched_blocks, _ = cache.match_prefix(req.get_token_ids())
                hit_tokens = len(matched_blocks) * self.block_manager.block_size
            else:
                hit_tokens = 0
            remaining_prefill = max(0, req.prompt_tokens - hit_tokens)
            # -hit_tokens prioritizes warmest cache hits first
            # remaining_prefill breaks ties with Shortest Job First
            # req.arrived_at breaks ties with FCFS
            return (-hit_tokens, remaining_prefill, req.arrived_at)

        self.waiting_queue.sort(key=priority_key)

    def schedule_iteration(self, current_time: float) -> SchedulingDecision:
        decision = SchedulingDecision()

        # Step 1: Schedule active decode requests
        still_running: list[Request] = []
        for req in list(self.running_batch):
            if self.block_manager.append_slot(req):
                decision.decode_requests.append(req)
                still_running.append(req)
            else:
                victim, _ = self.preemption_handler.preempt_victim(self.running_batch)
                if victim:
                    decision.preempted_requests.append(victim)
                    if victim == req:
                        self.waiting_queue.append(victim)
                    else:
                        if victim in still_running:
                            still_running.remove(victim)
                        if victim in decision.decode_requests:
                            decision.decode_requests.remove(victim)
                        self.waiting_queue.append(victim)

                        if self.block_manager.append_slot(req):
                            decision.decode_requests.append(req)
                            still_running.append(req)
                        else:
                            self.waiting_queue.append(req)

        self.running_batch = still_running

        # Step 2: Sort waiting queue by prefix affinity
        self._sort_waiting_queue()

        # Step 3: Admit requests that fit in batch and memory
        admitted: list[Request] = []
        for req in self.waiting_queue:
            if len(self.running_batch) + len(admitted) >= self.max_batch_size:
                break

            needed_tokens = req.prompt_tokens
            if self.block_manager.can_allocate(req, needed_tokens):
                self.block_manager.allocate(req, needed_tokens, current_time)
                uncached_tokens = max(0, req.prompt_tokens - req.cached_prefix_tokens)
                decision.prefill_requests.append((req, uncached_tokens))
                req.state = RequestState.PREFILLING
                admitted.append(req)

        for req in admitted:
            self.waiting_queue.remove(req)

        return decision
