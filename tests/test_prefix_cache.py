"""Unit tests for Radix-Tree Prefix Cache, block sharing, and LRU eviction."""

from sched_sim.core.request import Request
from sched_sim.hardware.presets import PRESETS
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.prefix_cache import PrefixCache


def test_prefix_cache_chained_hashing():
    cache = PrefixCache(block_size=16)
    tokens_a = list(range(32))  # 2 blocks
    tokens_b = list(range(32))  # Identical tokens
    tokens_c = list(range(16)) + [999] * 16  # Diverges in second block

    hashes_a = cache.compute_block_hashes(tokens_a)
    hashes_b = cache.compute_block_hashes(tokens_b)
    hashes_c = cache.compute_block_hashes(tokens_c)

    assert len(hashes_a) == 2
    # Identical tokens produce identical chained hashes
    assert hashes_a[0][0] == hashes_b[0][0]
    assert hashes_a[1][0] == hashes_b[1][0]

    # Divergent second block produces identical first hash, different second hash
    assert hashes_a[0][0] == hashes_c[0][0]
    assert hashes_a[1][0] != hashes_c[1][0]


def test_prefix_cache_insert_and_match():
    cache = PrefixCache(block_size=16)
    tokens = list(range(48))  # 3 blocks of 16 tokens
    block_ids = [101, 102, 103]

    cache.insert_blocks("req_1", tokens, block_ids, current_time=0.0)

    # Query matching prefix
    matched_blocks, matched_nodes = cache.match_prefix(tokens)
    assert matched_blocks == [101, 102, 103]
    assert len(matched_nodes) == 3

    # Query partial prefix (32 tokens)
    partial_matched, _ = cache.match_prefix(tokens[:32])
    assert partial_matched == [101, 102]

    # Query non-matching prefix
    non_matching, _ = cache.match_prefix([500] * 32)
    assert non_matching == []


def test_prefix_cache_lru_eviction():
    cache = PrefixCache(block_size=16)
    # Insert 3 chained blocks for req_1
    cache.insert_blocks("req_1", list(range(48)), [1, 2, 3], current_time=1.0)
    # Insert 1 separate block for req_2
    cache.insert_blocks("req_2", [99] * 16, [4], current_time=0.5)

    # Release both requests so ref_count becomes 0
    cache.release_request("req_1")
    cache.release_request("req_2")

    # req_2 was accessed at t=0.5, block 3 was accessed at t=1.0
    # Evict 1 block -> should evict block 4 first (oldest LRU leaf)
    freed = cache.evict_lru(1)
    assert freed == [4]

    # Evict next block -> should evict leaf block 3 (leaf of 1->2->3)
    freed_next = cache.evict_lru(1)
    assert freed_next == [3]


def test_block_manager_prefix_caching_sharing():
    hw = PRESETS["a100"]
    bm = BlockManager(hw, block_size=16, enable_prefix_caching=True)

    # Create two requests sharing a 64-token system prompt prefix
    req1 = Request(
        request_id="req_001",
        arrived_at=0.0,
        prompt_tokens=64,
        output_tokens=10,
        prefix_key="shared_system_prompt",
    )
    req2 = Request(
        request_id="req_002",
        arrived_at=1.0,
        prompt_tokens=64,
        output_tokens=10,
        prefix_key="shared_system_prompt",
    )

    initial_free = bm.num_free_gpu_blocks

    # Allocate req1: needs 4 blocks (64 tokens / 16)
    assert bm.allocate(req1, 64, current_time=0.0)
    assert bm.num_free_gpu_blocks == initial_free - 4
    assert req1.cached_prefix_tokens == 0

    # Allocate req2 with identical prefix: should hit 4 cached blocks!
    assert bm.allocate(req2, 64, current_time=1.0)
    # No new physical blocks consumed!
    assert bm.num_free_gpu_blocks == initial_free - 4
    # req2 got 64 cached tokens
    assert req2.cached_prefix_tokens == 64
    assert req2.prefilled_tokens == 64

    # Both requests share the exact block table pointers
    assert bm.gpu_block_tables["req_001"] == bm.gpu_block_tables["req_002"]

    # Free req1: blocks are kept alive in cache because req2 is still referencing them
    bm.free(req1)
    assert bm.num_free_gpu_blocks == initial_free - 4

    # Free req2: ref_count becomes 0, but still retained in prefix cache until evicted
    bm.free(req2)
    assert bm.num_free_gpu_blocks == initial_free - 4
    assert bm.prefix_cache.hit_rate > 0.0
