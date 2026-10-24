"""First-Come First-Served (FCFS) continuous batching scheduler."""

from sched_sim.core.request import Request, RequestState
from sched_sim.schedulers.base import BaseScheduler, SchedulingDecision


class FCFSScheduler(BaseScheduler):
    """Classical FCFS continuous-batching policy with atomic prompt prefills."""

    @property
    def name(self) -> str:
        return "FCFS (Continuous Batching)"

    def add_request(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self.waiting_queue.append(request)

    def schedule_iteration(self, current_time: float) -> SchedulingDecision:
        decision = SchedulingDecision()

        # Step 1: Schedule active decode requests
        still_running: list[Request] = []
        for req in list(self.running_batch):
            # Check if block allocation succeeds for the next decode token
            if self.block_manager.append_slot(req):
                decision.decode_requests.append(req)
                still_running.append(req)
            else:
                # Need to preempt someone to free memory
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

                        # Retry allocating slot for req after victim was freed
                        if self.block_manager.append_slot(req):
                            decision.decode_requests.append(req)
                            still_running.append(req)
                        else:
                            self.waiting_queue.append(req)

        self.running_batch = still_running

        # Step 2: Admit waiting requests if batch capacity and KV blocks allow
        admitted: list[Request] = []
        for req in self.waiting_queue:
            if len(self.running_batch) + len(admitted) >= self.max_batch_size:
                break

            needed_tokens = req.prompt_tokens
            if self.block_manager.can_allocate(req, needed_tokens):
                self.block_manager.allocate(req, needed_tokens)
                decision.prefill_requests.append((req, needed_tokens))
                req.state = RequestState.PREFILLING
                admitted.append(req)

        # Remove admitted requests from waiting queue
        for req in admitted:
            self.waiting_queue.remove(req)

        return decision
