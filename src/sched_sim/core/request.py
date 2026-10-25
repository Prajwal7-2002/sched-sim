"""Request data models and lifecycle tracking for LLM serving simulation."""

from enum import Enum

from pydantic import BaseModel, Field


class RequestState(str, Enum):
    WAITING = "WAITING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    PREEMPTED = "PREEMPTED"
    COMPLETED = "COMPLETED"
    DROPPED = "DROPPED"


class Request(BaseModel):
    """Represents a single inference request entering the serving cluster."""

    request_id: str
    arrived_at: float
    prompt_tokens: int
    output_tokens: int

    # Service Level Objectives (SLOs) in seconds
    slo_ttft: float | None = None
    slo_tpot: float | None = None

    # Lifecycle state
    state: RequestState = RequestState.WAITING
    prefilled_tokens: int = 0
    decoded_tokens: int = 0

    # Timing records
    first_token_at: float | None = None
    completed_at: float | None = None
    token_timestamps: list[float] = Field(default_factory=list)

    # Preemption records
    preemptions_count: int = 0
    recomputed_tokens: int = 0
    swapped_blocks_count: int = 0

    # Prefix Caching support
    prompt_token_ids: list[int] = Field(default_factory=list)
    prefix_key: str | None = None
    shared_prefix_tokens: int = 0
    cached_prefix_tokens: int = 0

    def get_token_ids(self) -> list[int]:
        """Returns the token ID sequence for prefix matching in Radix tree.

        If explicit prompt_token_ids are provided, returns them.
        If prefix_key is set, generates deterministic pseudo-tokens
        shared across all requests with that prefix_key.
        Otherwise, returns a deterministic unique sequence for this request.
        """
        if self.prompt_token_ids:
            return self.prompt_token_ids

        if self.prefix_key:
            shared_len = (
                self.shared_prefix_tokens
                if self.shared_prefix_tokens > 0
                else self.prompt_tokens
            )
            base_id = (hash(self.prefix_key) & 0x7FFFFFFF) % 100000
            req_hash = (hash(self.request_id) & 0x7FFFFFFF) % 100000
            shared_count = min(self.prompt_tokens, shared_len)
            tokens = [base_id + (i % 2000) for i in range(shared_count)]
            if len(tokens) < self.prompt_tokens:
                tokens.extend(
                    req_hash + i for i in range(self.prompt_tokens - len(tokens))
                )
            return tokens

        req_hash = (hash(self.request_id) & 0x7FFFFFFF) % 100000
        return [req_hash + i for i in range(self.prompt_tokens)]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def current_context_len(self) -> int:
        return self.prefilled_tokens + self.decoded_tokens

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.prefilled_tokens)

    @property
    def remaining_decode_tokens(self) -> int:
        return max(0, self.output_tokens - self.decoded_tokens)

    @property
    def is_prefill_complete(self) -> bool:
        return self.prefilled_tokens >= self.prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.state in (RequestState.COMPLETED, RequestState.DROPPED)

    @property
    def ttft(self) -> float | None:
        """Time-to-First-Token in seconds."""
        if self.first_token_at is not None:
            return max(0.0, self.first_token_at - self.arrived_at)
        return None

    @property
    def tpot(self) -> float | None:
        """Time-Per-Output-Token in seconds per token."""
        if (
            self.decoded_tokens > 1
            and self.completed_at is not None
            and self.first_token_at is not None
        ):
            return max(
                0.0,
                (self.completed_at - self.first_token_at) / (self.decoded_tokens - 1),
            )
        elif self.decoded_tokens == 1 and self.ttft is not None:
            return self.ttft
        return None

    @property
    def e2e_latency(self) -> float | None:
        """End-to-End latency in seconds."""
        if self.completed_at is not None:
            return max(0.0, self.completed_at - self.arrived_at)
        return None

    @property
    def slo_met(self) -> bool:
        """Evaluates whether all specified SLO targets were satisfied."""
        if self.state == RequestState.DROPPED:
            return False
        if not self.is_finished:
            return False

        if self.slo_ttft is not None:
            t = self.ttft
            if t is None or t > self.slo_ttft:
                return False

        if self.slo_tpot is not None:
            p = self.tpot
            if p is None or p > self.slo_tpot:
                return False

        return True
