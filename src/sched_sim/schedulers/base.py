"""Abstract Base Scheduler interface and decision primitives."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from sched_sim.core.request import Request
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler


class SchedulingDecision(BaseModel):
    """The allocation and execution plan for a single continuous-batching forward pass."""

    # List of (request, tokens_to_prefill)
    prefill_requests: list[tuple[Request, int]] = Field(default_factory=list)
    decode_requests: list[Request] = Field(default_factory=list)
    preempted_requests: list[Request] = Field(default_factory=list)
    dropped_requests: list[Request] = Field(default_factory=list)

    @property
    def total_prefill_tokens(self) -> int:
        return sum(tokens for _, tokens in self.prefill_requests)

    @property
    def total_decode_requests(self) -> int:
        return len(self.decode_requests)

    @property
    def total_tokens(self) -> int:
        return self.total_prefill_tokens + self.total_decode_requests

    @property
    def is_empty(self) -> bool:
        return not self.prefill_requests and not self.decode_requests


class BaseScheduler(ABC):
    """Abstract base class for all continuous-batching iteration schedulers."""

    def __init__(
        self,
        block_manager: BlockManager,
        latency_model: RooflineLatencyModel,
        preemption_handler: PreemptionHandler,
        max_batch_size: int = 32,
    ):
        self.block_manager = block_manager
        self.latency_model = latency_model
        self.preemption_handler = preemption_handler
        self.max_batch_size = max_batch_size

        self.waiting_queue: list[Request] = []
        self.running_batch: list[Request] = []
        self.swapped_queue: list[Request] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier of the scheduler policy."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Enqueue an incoming request into the scheduler's waiting pool."""

    @abstractmethod
    def schedule_iteration(self, current_time: float) -> SchedulingDecision:
        """Select which requests to prefill, which to decode, and execute memory checks."""

    def has_pending_work(self) -> bool:
        """Returns True if there are requests currently queued or generating."""
        return bool(self.waiting_queue or self.running_batch or self.swapped_queue)
