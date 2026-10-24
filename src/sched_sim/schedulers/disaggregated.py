"""Prefill-Decode (PD) Disaggregated Scheduler (Splitwise / DistServe architecture)."""

from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler
from sched_sim.schedulers.base import BaseScheduler, SchedulingDecision


class DisaggregatedScheduler(BaseScheduler):
    """Physically separates Prefill workers from Decode workers with simulated KV network transfer."""

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
        # Queue of requests currently in transit across the interconnect (req, ready_at_timestamp)
        self.transferring_queue: list[tuple[Request, float]] = []

    @property
    def name(self) -> str:
        return "Disaggregated (Splitwise / DistServe)"

    def add_request(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self.waiting_queue.append(request)

    def schedule_iteration(self, current_time: float) -> SchedulingDecision:
        decision = SchedulingDecision()

        # Step 0: Check if any transferring KV-caches have finished crossing the interconnect
        still_transferring: list[tuple[Request, float]] = []
        for req, ready_at in self.transferring_queue:
            if current_time >= ready_at:
                # Ready for decode worker
                req.state = RequestState.DECODING
                self.running_batch.append(req)
            else:
                still_transferring.append((req, ready_at))
        self.transferring_queue = still_transferring

        # Step 1: Decode Worker Pool (processes active decode streams)
        still_running: list[Request] = []
        for req in list(self.running_batch):
            if self.block_manager.append_slot(req):
                decision.decode_requests.append(req)
                still_running.append(req)
            else:
                victim, _ = self.preemption_handler.preempt_victim(self.running_batch)
                if victim:
                    decision.preempted_requests.append(victim)
                    if victim in still_running:
                        still_running.remove(victim)
                    if victim in decision.decode_requests:
                        decision.decode_requests.remove(victim)
                    self.add_request(victim)

                    if self.block_manager.append_slot(req):
                        decision.decode_requests.append(req)
                        still_running.append(req)
                    else:
                        self.add_request(req)

        self.running_batch = still_running

        # Step 2: Prefill Worker Pool (admits waiting requests independently)
        admitted: list[Request] = []
        for req in self.waiting_queue:
            if (
                len(self.running_batch) + len(admitted) + len(self.transferring_queue)
                >= self.max_batch_size
            ):
                break

            needed_tokens = req.prompt_tokens
            if self.block_manager.can_allocate(req, needed_tokens):
                self.block_manager.allocate(req, needed_tokens)
                decision.prefill_requests.append((req, needed_tokens))
                req.state = RequestState.PREFILLING
                admitted.append(req)

        for req in admitted:
            self.waiting_queue.remove(req)

        return decision

    def register_prefill_completion(
        self, request: Request, current_time: float
    ) -> None:
        """Called when a prefill finishes on the prefill worker, initiating KV transfer to decode worker."""
        transfer_delay = self.latency_model.compute_kv_transfer_time(
            request.prompt_tokens
        )
        ready_timestamp = current_time + transfer_delay
        self.transferring_queue.append((request, ready_timestamp))

    def has_pending_work(self) -> bool:
        return bool(
            self.waiting_queue
            or self.running_batch
            or self.transferring_queue
            or self.swapped_queue
        )
