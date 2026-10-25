"""Scheduler factory and registry for extensible LLM serving algorithms.

Follows the Open-Closed Principle (OCP) and Factory Pattern, allowing new scheduling
policies to register dynamically without modifying CLI or presentation modules.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler, PreemptionMode
from sched_sim.schedulers.base import BaseScheduler
from sched_sim.schedulers.cache_aware import CacheAwareScheduler
from sched_sim.schedulers.chunked import ChunkedPrefillScheduler
from sched_sim.schedulers.disaggregated import DisaggregatedScheduler
from sched_sim.schedulers.fcfs import FCFSScheduler
from sched_sim.schedulers.sjf import SJFScheduler
from sched_sim.schedulers.slo_aware import SLOAwareScheduler

if TYPE_CHECKING:
    from sched_sim.hardware.profile import HardwareProfile

SchedulerBuilder = Callable[..., BaseScheduler]


class SchedulerRegistry:
    """Central registry for discovering and instantiating scheduling policies."""

    _registry: ClassVar[dict[str, type[BaseScheduler]]] = {}
    _aliases: ClassVar[dict[str, str]] = {}

    @classmethod
    def register(
        cls, name: str, aliases: list[str] | None = None
    ) -> Callable[[type[BaseScheduler]], type[BaseScheduler]]:
        """Decorator to register a scheduler class under a canonical name and optional aliases."""

        def decorator(subclass: type[BaseScheduler]) -> type[BaseScheduler]:
            canonical = name.lower().replace("-", "").replace("_", "")
            cls._registry[canonical] = subclass
            if aliases:
                for alias in aliases:
                    cls._aliases[alias.lower().replace("-", "").replace("_", "")] = (
                        canonical
                    )
            return subclass

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseScheduler]:
        """Look up a scheduler class by canonical name or alias.

        Raises:
            KeyError: If the scheduler name is not recognized.
        """
        cleaned = name.lower().replace("-", "").replace("_", "")
        # First check canonical registry
        if cleaned in cls._registry:
            return cls._registry[cleaned]

        # Check aliases mapping
        if cleaned in cls._aliases:
            return cls._registry[cls._aliases[cleaned]]

        # Fuzzy lookup for common prefixes/substrings
        for key, subclass in cls._registry.items():
            if key in cleaned or cleaned in key:
                return subclass

        raise KeyError(f"Unknown scheduler '{name}'. Available: {cls.list_available()}")

    @classmethod
    def list_available(cls) -> list[str]:
        """Return sorted list of all registered canonical scheduler names."""
        return sorted(cls._registry.keys())


# Register core built-in schedulers
SchedulerRegistry.register("fcfs", aliases=["firstcomefirstserved", "fifo"])(
    FCFSScheduler
)
SchedulerRegistry.register("sjf", aliases=["shortest", "shortestjobfirst"])(
    SJFScheduler
)
SchedulerRegistry.register(
    "chunked", aliases=["chunkedprefill", "sarathi", "sarathiserve"]
)(ChunkedPrefillScheduler)
SchedulerRegistry.register("slo", aliases=["sloaware", "deadline", "edf"])(
    SLOAwareScheduler
)
SchedulerRegistry.register(
    "disaggregated", aliases=["disagg", "splitwise", "distserve"]
)(DisaggregatedScheduler)
SchedulerRegistry.register(
    "cacheaware", aliases=["cache", "prefix", "radix", "prefixcache", "sglang"]
)(CacheAwareScheduler)


def create_scheduler(
    name: str,
    hardware: "HardwareProfile",
    preemption_mode: PreemptionMode = PreemptionMode.RECOMPUTE,
    token_budget: int = 512,
    max_batch_size: int = 32,
    enable_prefix_caching: bool = False,
) -> BaseScheduler:
    """Factory creating a fully configured scheduler instance.

    Args:
        name: Name or alias of the scheduler policy (e.g. 'fcfs', 'chunked', 'cache_aware').
        hardware: Silicon hardware specification for roofline modeling.
        preemption_mode: Memory preemption policy (RECOMPUTE or SWAP).
        token_budget: Maximum batched tokens per forward step for chunked prefill.
        max_batch_size: Maximum concurrent sequences in forward iteration.
        enable_prefix_caching: Enable Radix tree prefix caching in the BlockManager.

    Returns:
        Instantiated and configured BaseScheduler subclass.
    """
    key = name.lower().replace("-", "").replace("_", "")
    is_cache_aware = "cache" in key or "prefix" in key or "radix" in key
    use_prefix_cache = enable_prefix_caching or is_cache_aware

    block_manager = BlockManager(hardware, enable_prefix_caching=use_prefix_cache)
    latency_model = RooflineLatencyModel(hardware)
    preemption_handler = PreemptionHandler(
        block_manager, latency_model, preemption_mode
    )

    scheduler_cls = SchedulerRegistry.get(name)

    # Special handling for schedulers with extra hyperparameters
    if issubclass(scheduler_cls, ChunkedPrefillScheduler):
        return ChunkedPrefillScheduler(
            block_manager,
            latency_model,
            preemption_handler,
            max_batch_size=max_batch_size,
            max_token_budget=token_budget,
        )

    return scheduler_cls(
        block_manager,
        latency_model,
        preemption_handler,
        max_batch_size=max_batch_size,
    )
