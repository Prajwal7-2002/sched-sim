"""Example 1: Radix Prefix Caching & Zero-Copy Block Sharing.

Demonstrates SGLang-style prefix caching using BlockManager and Request:
1. Initializing hardware memory with prefix caching enabled.
2. Allocating memory for Request 1 with a system prompt.
3. Automatically matching and re-using prefix blocks for Request 2.
"""

from sched_sim.core.request import Request
from sched_sim.hardware.presets import PRESETS
from sched_sim.memory.block_manager import BlockManager


def main() -> None:
    print("=" * 65)
    print("Example 1: Radix Prefix Caching & Zero-Copy Block Sharing")
    print("=" * 65)

    # 1. Initialize BlockManager with A100 preset and prefix caching enabled
    hardware = PRESETS["a100"]
    block_mgr = BlockManager(
        hardware=hardware, block_size=16, enable_prefix_caching=True
    )

    print(f"\n[1] Hardware: {hardware.name} ({hardware.vram_gb} GB VRAM)")
    print(f"    Total GPU KV-Cache Blocks : {block_mgr.total_gpu_blocks}")
    print(f"    Tokens Per Block          : {block_mgr.block_size}")

    # 2. Define shared 64-token system prompt (4 physical blocks of 16)
    system_prompt = list(range(1000, 1064))
    user_query1 = [2001, 2002, 2003, 2004]

    req1 = Request(
        request_id="req_001",
        arrived_at=0.0,
        prompt_tokens=len(system_prompt) + len(user_query1),
        output_tokens=32,
        prompt_token_ids=system_prompt + user_query1,
    )

    print(
        f"\n[2] Request 1 Arrives ({req1.prompt_tokens} prompt tokens: 64 sys + 4 query)"
    )
    allocated1 = block_mgr.allocate(
        req1, initial_tokens=req1.prompt_tokens, current_time=0.0
    )
    assert allocated1, "Failed to allocate Request 1"

    req1_blocks = block_mgr.gpu_block_tables[req1.request_id]
    print(f"    Allocated Physical Blocks: {req1_blocks}")
    print(f"    Cached Prefix Tokens     : {req1.cached_prefix_tokens} (Cold cache)")

    # 3. Request 2 arrives with identical system prompt + new query
    user_query2 = [3001, 3002, 3003, 3004, 3005, 3006]
    req2 = Request(
        request_id="req_002",
        arrived_at=1.2,
        prompt_tokens=len(system_prompt) + len(user_query2),
        output_tokens=32,
        prompt_token_ids=system_prompt + user_query2,
    )

    print(
        f"\n[3] Request 2 Arrives ({req2.prompt_tokens} prompt tokens: 64 shared sys + 6 query)"
    )
    allocated2 = block_mgr.allocate(
        req2, initial_tokens=req2.prompt_tokens, current_time=1.2
    )
    assert allocated2, "Failed to allocate Request 2"

    req2_blocks = block_mgr.gpu_block_tables[req2.request_id]
    print(f"    Allocated Physical Blocks: {req2_blocks}")
    print(f"    Cached Prefix Tokens     : {req2.cached_prefix_tokens} tokens matched!")
    print(f"    Shared Physical Block IDs: {req2_blocks[:4]}")

    # Verify zero-copy sharing
    assert req2.cached_prefix_tokens == 64, (
        f"Expected 64 cached tokens, got {req2.cached_prefix_tokens}"
    )
    assert req2_blocks[:4] == req1_blocks[:4], (
        "Shared blocks must reuse identical physical block IDs!"
    )

    # 4. Telemetry
    cache = block_mgr.prefix_cache
    assert cache is not None
    print("\n[4] Prefix Cache Performance Telemetry:")
    print(f"    Total Lookups            : {cache.total_queries}")
    print(f"    Cache Hits               : {cache.total_hits}")
    print(f"    Hit Ratio                : {cache.hit_rate:.1%}")
    print(f"    Total Matched Tokens     : {cache.total_matched_tokens} tokens")
    print(f"    Active Cached Blocks     : {len(cache.node_map)}")

    print(
        "\n[SUCCESS] SGLang-style Radix Prefix Caching verified with 100% shared block hits!"
    )


if __name__ == "__main__":
    main()
