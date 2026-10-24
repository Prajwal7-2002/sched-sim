"""Shortest-Job-First (SJF) / Shortest-Prompt-First scheduler."""

from sched_sim.core.request import Request, RequestState
from sched_sim.schedulers.base import BaseScheduler, SchedulingDecision


class SJFScheduler(BaseScheduler):
    """Shortest-Job-First scheduler: prioritizes shortest prompt arrivals to minimize average TTFT."""

    @property
    def name(self) -> str:
        return "SJF (Shortest Prompt First)"

    def add_request(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self.waiting_queue.append(request)
        # Keep waiting queue sorted by prompt tokens ascending
        self.waiting_queue.sort(key=lambda r: (r.prompt_tokens, r.arrived_at))

    def schedule_iteration(self, current_time: float) -> SchedulingDecision:
        decision = SchedulingDecision()

        # Step 1: Schedule active decode streams
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

        # Step 2: Admit shortest prompt waiting requests
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

        for req in admitted:
            self.waiting_queue.remove(req)

        return decision
