"""SLO-Aware Deadline Scheduler to maximize Goodput under heavy load."""

from sched_sim.core.request import Request, RequestState
from sched_sim.schedulers.base import BaseScheduler, SchedulingDecision


class SLOAwareScheduler(BaseScheduler):
    """Dynamically prioritizes requests by Earliest Deadline First (EDF) and sheds hopeless requests."""

    @property
    def name(self) -> str:
        return "SLO-Aware (Goodput Maximizer)"

    def add_request(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self.waiting_queue.append(request)

    def _calculate_slack(self, req: Request, current_time: float) -> float:
        """Calculate time remaining until the request violates its SLO deadline."""
        if req.slo_ttft is not None and req.first_token_at is None:
            deadline = req.arrived_at + req.slo_ttft
            return deadline - current_time

        if req.slo_tpot is not None and req.first_token_at is not None:
            # Target completion based on TPOT
            deadline = req.first_token_at + (req.output_tokens * req.slo_tpot)
            return deadline - current_time

        # Default high slack if no SLO set
        return 999.0

    def schedule_iteration(self, current_time: float) -> SchedulingDecision:
        decision = SchedulingDecision()

        # Step 0: Check for SLO violations and selectively shed unrecoverable requests
        for req in list(self.waiting_queue):
            slack = self._calculate_slack(req, current_time)
            if slack < -0.05:  # Overdue by > 50ms without being scheduled
                req.state = RequestState.DROPPED
                decision.dropped_requests.append(req)
                self.waiting_queue.remove(req)

        for req in list(self.running_batch):
            slack = self._calculate_slack(req, current_time)
            if slack < -0.10:  # Running decode has blown past acceptable bounds
                req.state = RequestState.DROPPED
                self.block_manager.free(req)
                decision.dropped_requests.append(req)
                self.running_batch.remove(req)

        # Sort remaining waiting queue by earliest deadline (smallest slack first)
        self.waiting_queue.sort(key=lambda r: self._calculate_slack(r, current_time))

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

        # Step 2: Admit waiting requests with highest deadline urgency
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
