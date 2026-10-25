"""Unit tests for the SchedulerRegistry and create_scheduler factory."""

import pytest

from sched_sim.hardware.presets import get_hardware_profile
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionMode
from sched_sim.schedulers.base import BaseScheduler, SchedulingDecision
from sched_sim.schedulers.cache_aware import CacheAwareScheduler
from sched_sim.schedulers.chunked import ChunkedPrefillScheduler
from sched_sim.schedulers.disaggregated import DisaggregatedScheduler
from sched_sim.schedulers.factory import SchedulerRegistry, create_scheduler
from sched_sim.schedulers.fcfs import FCFSScheduler
from sched_sim.schedulers.sjf import SJFScheduler
from sched_sim.schedulers.slo_aware import SLOAwareScheduler


def test_registry_list_available():
    """Verify built-in schedulers are all registered and discoverable."""
    available = SchedulerRegistry.list_available()
    expected = ["cacheaware", "chunked", "disaggregated", "fcfs", "sjf", "slo"]
    for name in expected:
        assert name in available


def test_registry_get_canonical_and_aliases():
    """Verify registry resolves both canonical names and common aliases/kebab-case."""
    assert SchedulerRegistry.get("fcfs") is FCFSScheduler
    assert SchedulerRegistry.get("fifo") is FCFSScheduler
    assert SchedulerRegistry.get("firstcomefirstserved") is FCFSScheduler

    assert SchedulerRegistry.get("sjf") is SJFScheduler
    assert SchedulerRegistry.get("shortest") is SJFScheduler

    assert SchedulerRegistry.get("chunked") is ChunkedPrefillScheduler
    assert SchedulerRegistry.get("chunked-prefill") is ChunkedPrefillScheduler
    assert SchedulerRegistry.get("sarathi") is ChunkedPrefillScheduler

    assert SchedulerRegistry.get("slo") is SLOAwareScheduler
    assert SchedulerRegistry.get("slo-aware") is SLOAwareScheduler
    assert SchedulerRegistry.get("edf") is SLOAwareScheduler

    assert SchedulerRegistry.get("disaggregated") is DisaggregatedScheduler
    assert SchedulerRegistry.get("disagg") is DisaggregatedScheduler
    assert SchedulerRegistry.get("splitwise") is DisaggregatedScheduler

    assert SchedulerRegistry.get("cacheaware") is CacheAwareScheduler
    assert SchedulerRegistry.get("cache_aware") is CacheAwareScheduler
    assert SchedulerRegistry.get("cache-aware") is CacheAwareScheduler
    assert SchedulerRegistry.get("radix") is CacheAwareScheduler
    assert SchedulerRegistry.get("sglang") is CacheAwareScheduler


def test_registry_unknown_scheduler_raises():
    """Verify looking up an unrecorded policy raises KeyError with candidate list."""
    with pytest.raises(KeyError, match="Unknown scheduler 'nonexistent'"):
        SchedulerRegistry.get("nonexistent")


def test_custom_scheduler_registration():
    """Verify third-party / experimental schedulers can be registered dynamically via decorator."""

    @SchedulerRegistry.register("mock_policy", aliases=["mock_alias"])
    class MockCustomScheduler(BaseScheduler):
        @property
        def name(self) -> str:
            return "mock"

        def schedule(self, current_time: float) -> SchedulingDecision:
            return SchedulingDecision()

    assert SchedulerRegistry.get("mock_policy") is MockCustomScheduler
    assert SchedulerRegistry.get("mock_alias") is MockCustomScheduler
    assert "mockpolicy" in SchedulerRegistry.list_available()


def test_create_scheduler_instances():
    """Verify create_scheduler constructs ready-to-run scheduler instances with correct configurations."""
    hw = get_hardware_profile("a100")

    # FCFS
    s_fcfs = create_scheduler("fcfs", hw, max_batch_size=16)
    assert isinstance(s_fcfs, FCFSScheduler)
    assert s_fcfs.max_batch_size == 16
    assert isinstance(s_fcfs.block_manager, BlockManager)

    # Chunked prefill with custom token budget
    s_chunked = create_scheduler("chunked", hw, token_budget=256, max_batch_size=8)
    assert isinstance(s_chunked, ChunkedPrefillScheduler)
    assert s_chunked.max_token_budget == 256
    assert s_chunked.max_batch_size == 8

    # Cache-aware auto-activates prefix caching
    s_cache = create_scheduler("cache_aware", hw)
    assert isinstance(s_cache, CacheAwareScheduler)
    assert s_cache.block_manager.prefix_cache is not None

    # Swapping preemption mode
    s_slo = create_scheduler("slo", hw, preemption_mode=PreemptionMode.SWAP)
    assert isinstance(s_slo, SLOAwareScheduler)
    assert s_slo.preemption_handler.mode == PreemptionMode.SWAP
