"""Simulated Paged KV-Cache memory allocator with block tables and swapping."""

import math

from sched_sim.core.request import Request
from sched_sim.hardware.profile import HardwareProfile


class BlockManager:
    """Manages physical 16-token memory blocks across GPU VRAM and CPU Host memory."""

    def __init__(
        self,
        hardware: HardwareProfile,
        block_size: int = 16,
        safety_buffer_gb: float = 0.5,
        cpu_memory_gb: float = 32.0,
    ):
        self.hardware = hardware
        self.block_size = block_size
        self.kv_bytes_per_token = hardware.kv_bytes_per_token
        self.bytes_per_block = self.block_size * self.kv_bytes_per_token

        # Calculate available GPU VRAM for KV-cache after model weights and safety buffer
        available_vram_gb = max(
            0.5, hardware.vram_gb - hardware.model_weights_gb - safety_buffer_gb
        )
        available_vram_bytes = available_vram_gb * (1024**3)
        self.total_gpu_blocks = max(
            16, int(available_vram_bytes // self.bytes_per_block)
        )

        # Calculate available CPU RAM blocks for swap preemption
        available_cpu_bytes = cpu_memory_gb * (1024**3)
        self.total_cpu_blocks = max(
            64, int(available_cpu_bytes // self.bytes_per_block)
        )

        # Free block pools
        self.free_gpu_blocks: list[int] = list(range(self.total_gpu_blocks))
        self.free_cpu_blocks: list[int] = list(range(self.total_cpu_blocks))

        # Block tables: request_id -> list of block IDs
        self.gpu_block_tables: dict[str, list[int]] = {}
        self.cpu_block_tables: dict[str, list[int]] = {}

    @property
    def num_free_gpu_blocks(self) -> int:
        return len(self.free_gpu_blocks)

    @property
    def num_free_cpu_blocks(self) -> int:
        return len(self.free_cpu_blocks)

    @property
    def num_used_gpu_blocks(self) -> int:
        return self.total_gpu_blocks - len(self.free_gpu_blocks)

    @property
    def gpu_utilization(self) -> float:
        """Percentage of GPU KV-cache blocks currently in use."""
        return self.num_used_gpu_blocks / self.total_gpu_blocks

    def blocks_required(self, num_tokens: int) -> int:
        """Calculate how many physical blocks are required to hold N tokens."""
        return math.ceil(num_tokens / self.block_size)

    def can_allocate(self, request: Request, additional_tokens: int) -> bool:
        """Check if GPU VRAM has enough free blocks for the requested token expansion."""
        current_blocks = len(self.gpu_block_tables.get(request.request_id, []))
        total_tokens = request.current_context_len + additional_tokens
        needed_blocks = self.blocks_required(total_tokens)
        new_blocks = max(0, needed_blocks - current_blocks)
        return len(self.free_gpu_blocks) >= new_blocks

    def allocate(self, request: Request, initial_tokens: int) -> bool:
        """Allocate initial physical blocks for a new request."""
        req_id = request.request_id
        if req_id in self.gpu_block_tables:
            return True

        needed = self.blocks_required(initial_tokens)
        if len(self.free_gpu_blocks) < needed:
            return False

        allocated = [self.free_gpu_blocks.pop() for _ in range(needed)]
        self.gpu_block_tables[req_id] = allocated
        return True

    def append_slot(self, request: Request) -> bool:
        """Allocate an additional token slot during autoregressive decode.

        Allocates a new block only if the current context has crossed the 16-token boundary.
        """
        req_id = request.request_id
        if req_id not in self.gpu_block_tables:
            return self.allocate(request, 1)

        current_tokens = request.current_context_len
        current_blocks = len(self.gpu_block_tables[req_id])
        needed_blocks = self.blocks_required(current_tokens + 1)

        if needed_blocks > current_blocks:
            if not self.free_gpu_blocks:
                return False  # OOM / need preemption
            new_block = self.free_gpu_blocks.pop()
            self.gpu_block_tables[req_id].append(new_block)

        return True

    def free(self, request: Request) -> None:
        """Free all physical blocks assigned to this request across GPU and CPU."""
        req_id = request.request_id
        if req_id in self.gpu_block_tables:
            for b in self.gpu_block_tables[req_id]:
                self.free_gpu_blocks.append(b)
            del self.gpu_block_tables[req_id]

        if req_id in self.cpu_block_tables:
            for b in self.cpu_block_tables[req_id]:
                self.free_cpu_blocks.append(b)
            del self.cpu_block_tables[req_id]

    def can_swap_out(self, request: Request) -> bool:
        """Check if CPU DRAM has enough space to hold swapped GPU blocks."""
        req_id = request.request_id
        blocks = self.gpu_block_tables.get(req_id, [])
        return len(self.free_cpu_blocks) >= len(blocks)

    def swap_out(self, request: Request) -> int:
        """Swap request's KV blocks from GPU VRAM to CPU DRAM."""
        req_id = request.request_id
        gpu_blocks = self.gpu_block_tables.pop(req_id, [])
        count = len(gpu_blocks)

        cpu_blocks = [self.free_cpu_blocks.pop() for _ in range(count)]
        self.cpu_block_tables[req_id] = cpu_blocks

        # Return blocks back to GPU free list
        self.free_gpu_blocks.extend(gpu_blocks)
        request.swapped_blocks_count += count
        return count

    def can_swap_in(self, request: Request) -> bool:
        """Check if GPU VRAM has enough space to bring swapped blocks back."""
        req_id = request.request_id
        cpu_blocks = self.cpu_block_tables.get(req_id, [])
        return len(self.free_gpu_blocks) >= len(cpu_blocks)

    def swap_in(self, request: Request) -> int:
        """Swap request's KV blocks from CPU DRAM back to GPU VRAM."""
        req_id = request.request_id
        cpu_blocks = self.cpu_block_tables.pop(req_id, [])
        count = len(cpu_blocks)

        gpu_blocks = [self.free_gpu_blocks.pop() for _ in range(count)]
        self.gpu_block_tables[req_id] = gpu_blocks

        # Return blocks back to CPU free list
        self.free_cpu_blocks.extend(cpu_blocks)
        return count
