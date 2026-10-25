"""Discrete-event simulation engine for LLM continuous batching."""

import heapq
from typing import Any

from sched_sim.core.metrics import SimulationMetrics, compute_metrics
from sched_sim.core.request import Request, RequestState
from sched_sim.hardware.profile import HardwareProfile
from sched_sim.memory.preemption import PreemptionMode
from sched_sim.schedulers.base import BaseScheduler
from sched_sim.schedulers.disaggregated import DisaggregatedScheduler


class EventType:
    ARRIVAL = "ARRIVAL"
    ITERATION = "ITERATION"


class IterationRecord:
    """Snapshot of a single forward pass iteration for Gantt visualization."""

    def __init__(
        self,
        iteration_index: int,
        start_time: float,
        duration: float,
        prefill_tokens: int,
        decode_count: int,
        active_request_ids: list[str],
        kv_utilization: float,
    ):
        self.iteration_index = iteration_index
        self.start_time = start_time
        self.duration = duration
        self.prefill_tokens = prefill_tokens
        self.decode_count = decode_count
        self.active_request_ids = active_request_ids
        self.kv_utilization = kv_utilization

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


class Simulator:
    """Discrete-Event Simulator driving continuous-batching queueing dynamics."""

    def __init__(
        self,
        hardware: HardwareProfile,
        scheduler: BaseScheduler,
        preemption_mode: PreemptionMode = PreemptionMode.RECOMPUTE,
        block_size: int = 16,
    ):
        self.hardware = hardware
        self.scheduler = scheduler
        self.latency_model = scheduler.latency_model
        self.block_manager = scheduler.block_manager
        self.preemption_handler = scheduler.preemption_handler

        self.current_time: float = 0.0
        self.event_queue: list[tuple[float, int, str, Any]] = []
        self.event_counter: int = 0
        self.gpu_is_busy: bool = False

        # Output tracking
        self.all_requests: list[Request] = []
        self.iteration_records: list[IterationRecord] = []
        self.iteration_batch_sizes: list[int] = []
        self.gpu_busy_time_s: float = 0.0
        self.peak_kv_utilization: float = 0.0

    def add_event(self, timestamp: float, event_type: str, payload: Any = None) -> None:
        self.event_counter += 1
        heapq.heappush(
            self.event_queue, (timestamp, self.event_counter, event_type, payload)
        )

    def submit_workload(self, requests: list[Request]) -> None:
        """Schedule arrivals for an entire batch of incoming requests."""
        self.all_requests.extend(requests)
        for req in requests:
            self.add_event(req.arrived_at, EventType.ARRIVAL, req)

    def run(self, max_duration_s: float | None = None) -> SimulationMetrics:
        """Run the simulation to completion or until max_duration_s."""
        while self.event_queue:
            timestamp, _, event_type, payload = heapq.heappop(self.event_queue)
            self.current_time = timestamp

            if max_duration_s is not None and self.current_time >= max_duration_s:
                break

            if event_type == EventType.ARRIVAL:
                self._handle_arrival(payload)
            elif event_type == EventType.ITERATION:
                self._handle_iteration()

        # Track any uncompleted requests at end of simulation duration
        sim_duration = max(0.001, self.current_time)
        return compute_metrics(
            scheduler_name=self.scheduler.name,
            hardware_name=self.hardware.name,
            duration_s=sim_duration,
            all_requests=self.all_requests,
            total_iterations=len(self.iteration_records),
            gpu_busy_time_s=self.gpu_busy_time_s,
            iteration_batch_sizes=self.iteration_batch_sizes,
            peak_kv_utilization=self.peak_kv_utilization,
        )

    def _handle_arrival(self, request: Request) -> None:
        self.scheduler.add_request(request)
        # If GPU was idle, kick off an iteration step immediately
        if not self.gpu_is_busy:
            self.gpu_is_busy = True
            self.add_event(self.current_time, EventType.ITERATION)

    def _handle_iteration(self) -> None:
        decision = self.scheduler.schedule_iteration(self.current_time)

        if decision.is_empty:
            # Check if there is still pending work waiting for transfer or blocks
            if self.scheduler.has_pending_work():
                # Re-check in a small step (e.g. 5ms) for pending transfers or completions
                self.add_event(self.current_time + 0.005, EventType.ITERATION)
            else:
                self.gpu_is_busy = False
            return

        # Calculate forward pass execution time
        num_prefill = decision.total_prefill_tokens
        num_decode = decision.total_decode_requests
        total_decode_context = sum(
            r.current_context_len for r in decision.decode_requests
        )

        if num_prefill > 0 and num_decode > 0:
            step_time = self.latency_model.compute_mixed_step_time(
                num_prefill, num_decode, total_decode_context
            )
        elif num_prefill > 0:
            step_time = self.latency_model.compute_prefill_step_time(num_prefill)
        elif num_decode > 0:
            step_time = self.latency_model.compute_decode_step_time(
                num_decode, total_decode_context
            )
        else:
            # 100% prefix cache hit with no active decodes: only cache lookup / launch overhead
            step_time = max(0.0005, self.latency_model.launch_overhead_s)

        step_end_time = self.current_time + step_time
        self.gpu_busy_time_s += step_time

        # Update peak memory utilization
        current_kv_util = self.block_manager.gpu_utilization
        self.peak_kv_utilization = max(self.peak_kv_utilization, current_kv_util)

        # Record telemetry
        active_ids = [req.request_id for req, _ in decision.prefill_requests] + [
            req.request_id for req in decision.decode_requests
        ]
        self.iteration_batch_sizes.append(len(active_ids))
        self.iteration_records.append(
            IterationRecord(
                iteration_index=len(self.iteration_records) + 1,
                start_time=self.current_time,
                duration=step_time,
                prefill_tokens=num_prefill,
                decode_count=num_decode,
                active_request_ids=active_ids,
                kv_utilization=current_kv_util,
            )
        )

        # Advance prefill requests
        for req, tokens_prefilled in decision.prefill_requests:
            req.prefilled_tokens += tokens_prefilled
            if req.is_prefill_complete:
                if req.first_token_at is None:
                    req.first_token_at = step_end_time

                if isinstance(self.scheduler, DisaggregatedScheduler):
                    self.scheduler.register_prefill_completion(req, step_end_time)
                else:
                    req.state = RequestState.DECODING
                    self.scheduler.running_batch.append(req)

        # Advance decode requests
        for req in decision.decode_requests:
            req.decoded_tokens += 1
            req.token_timestamps.append(step_end_time)

            if req.decoded_tokens >= req.output_tokens:
                req.state = RequestState.COMPLETED
                req.completed_at = step_end_time
                self.block_manager.free(req)
                if req in self.scheduler.running_batch:
                    self.scheduler.running_batch.remove(req)

        # Schedule next iteration step
        if self.scheduler.has_pending_work() or self.event_queue:
            self.add_event(step_end_time, EventType.ITERATION)
        else:
            self.gpu_is_busy = False
