"""Radix-Tree based Prefix Caching for physical KV-cache block sharing and LRU eviction."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sched_sim.core.request import Request


class RadixTreeNode:
    """A node in the prefix tree representing a cached 16-token physical block."""

    def __init__(
        self,
        block_id: int,
        block_hash: int,
        tokens: tuple[int, ...],
        parent: "RadixTreeNode | None" = None,
        depth: int = 0,
    ):
        self.block_id = block_id
        self.block_hash = block_hash
        self.tokens = tokens
        self.parent = parent
        self.depth = depth
        self.children: dict[int, RadixTreeNode] = {}
        self.ref_count: int = 0  # Active reader requests
        self.last_accessed: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_evictable(self) -> bool:
        return self.ref_count == 0 and self.is_leaf


class PrefixCache:
    """Radix-Tree prefix cache inspired by SGLang RadixAttention and vLLM automatic prefix caching.

    Enables physical block sharing across requests with identical prompt prefixes,
    drastically reducing TTFT and memory footprint for multi-turn chats, RAG, and few-shot workloads.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.root = RadixTreeNode(
            block_id=-1, block_hash=0, tokens=(), parent=None, depth=0
        )
        self.active_request_nodes: dict[str, list[RadixTreeNode]] = {}
        self.node_map: dict[int, RadixTreeNode] = {}  # block_id -> node

        # Telemetry
        self.total_queries: int = 0
        self.total_hits: int = 0
        self.total_matched_blocks: int = 0
        self.total_matched_tokens: int = 0
        self.total_evicted_blocks: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_hits / self.total_queries

    def compute_block_hashes(
        self, tokens: list[int]
    ) -> list[tuple[int, tuple[int, ...]]]:
        """Compute chained rolling hashes for each 16-token block."""
        blocks: list[tuple[int, tuple[int, ...]]] = []
        num_full_blocks = len(tokens) // self.block_size
        prev_hash = 0

        for i in range(num_full_blocks):
            block_tokens = tuple(
                tokens[i * self.block_size : (i + 1) * self.block_size]
            )
            # Hash depends on prior block context + current block tokens
            curr_hash = hash((prev_hash, block_tokens))
            blocks.append((curr_hash, block_tokens))
            prev_hash = curr_hash

        return blocks

    def match_prefix(self, tokens: list[int]) -> tuple[list[int], list[RadixTreeNode]]:
        """Walk down the Radix tree and find the longest cached prefix."""
        block_hashes = self.compute_block_hashes(tokens)
        matched_blocks: list[int] = []
        matched_nodes: list[RadixTreeNode] = []

        curr = self.root
        for b_hash, _ in block_hashes:
            if b_hash in curr.children:
                curr = curr.children[b_hash]
                matched_blocks.append(curr.block_id)
                matched_nodes.append(curr)
            else:
                break

        return matched_blocks, matched_nodes

    def acquire_prefix(self, request: "Request", current_time: float) -> list[int]:
        """Acquire matched prefix blocks for an incoming request, bumping ref counts."""
        self.total_queries += 1
        tokens = request.get_token_ids()
        matched_blocks, matched_nodes = self.match_prefix(tokens)

        if matched_blocks:
            self.total_hits += 1
            self.total_matched_blocks += len(matched_blocks)
            self.total_matched_tokens += len(matched_blocks) * self.block_size

            for node in matched_nodes:
                node.ref_count += 1
                node.last_accessed = current_time

            self.active_request_nodes[request.request_id] = list(matched_nodes)

        return matched_blocks

    def insert_blocks(
        self,
        request_id: str,
        tokens: list[int],
        block_ids: list[int],
        current_time: float,
    ) -> None:
        """Register newly allocated physical blocks into the Radix tree."""
        block_hashes = self.compute_block_hashes(tokens)
        num_blocks = min(len(block_hashes), len(block_ids))

        curr = self.root
        tracked_nodes = self.active_request_nodes.setdefault(request_id, [])

        for i in range(num_blocks):
            b_hash, b_tokens = block_hashes[i]
            b_id = block_ids[i]

            if b_hash in curr.children:
                child = curr.children[b_hash]
                child.last_accessed = current_time
                if child not in tracked_nodes:
                    child.ref_count += 1
                    tracked_nodes.append(child)
                curr = child
            else:
                new_node = RadixTreeNode(
                    block_id=b_id,
                    block_hash=b_hash,
                    tokens=b_tokens,
                    parent=curr,
                    depth=curr.depth + 1,
                )
                new_node.ref_count = 1
                new_node.last_accessed = current_time
                curr.children[b_hash] = new_node
                self.node_map[b_id] = new_node
                tracked_nodes.append(new_node)
                curr = new_node

    def release_request(self, request_id: str) -> None:
        """Release reference counts for all cached blocks used by the request."""
        nodes = self.active_request_nodes.pop(request_id, [])
        for node in nodes:
            if node.ref_count > 0:
                node.ref_count -= 1

    def evict_lru(self, num_blocks_needed: int) -> list[int]:
        """Evict unreferenced leaf nodes in LRU order to reclaim physical blocks."""
        freed_blocks: list[int] = []

        while len(freed_blocks) < num_blocks_needed:
            # Find all evictable leaf nodes (ref_count == 0 and no children)
            evictable_leaves = [
                node for node in self.node_map.values() if node.is_evictable
            ]

            if not evictable_leaves:
                break  # No more evictable blocks

            # Pick oldest accessed leaf
            evictable_leaves.sort(key=lambda n: n.last_accessed)
            victim = evictable_leaves[0]

            # Detach victim from parent
            parent = victim.parent
            if parent and victim.block_hash in parent.children:
                del parent.children[victim.block_hash]

            freed_blocks.append(victim.block_id)
            del self.node_map[victim.block_id]
            self.total_evicted_blocks += 1

        return freed_blocks
