"""Analytical roofline latency model for LLM inference execution steps."""

from sched_sim.hardware.profile import HardwareProfile


class RooflineLatencyModel:
    """Computes exact execution time for prefill, decode, and mixed forward passes."""

    def __init__(self, hardware: HardwareProfile):
        self.hardware = hardware
        self.bandwidth_bytes_s = hardware.memory_bandwidth_gb_s * 1e9
        self.compute_flops_s = hardware.effective_tflops * 1e12
        self.weights_bytes = (
            hardware.parameter_count_billions * 1e9 * hardware.dtype_bytes
        )
        self.launch_overhead_s = hardware.kernel_launch_overhead_ms / 1000.0

    def compute_prefill_step_time(self, num_prefill_tokens: int) -> float:
        """Calculate forward pass execution time for prompt prefill.

        Prefill is predominantly compute-bound for moderate to large token counts.
        """
        if num_prefill_tokens <= 0:
            return 0.0

        # FLOPs = 2 * params * tokens (standard transformer forward pass)
        total_flops = (
            2.0 * (self.hardware.parameter_count_billions * 1e9) * num_prefill_tokens
        )
        compute_time = total_flops / self.compute_flops_s

        # Weights memory streaming time
        memory_time = self.weights_bytes / self.bandwidth_bytes_s

        # Roofline model: bounded by the slower bottleneck
        execution_time = max(memory_time, compute_time) + self.launch_overhead_s
        return execution_time

    def compute_decode_step_time(
        self, batch_size: int, total_context_tokens: int
    ) -> float:
        """Calculate forward pass execution time for autoregressive token decoding.

        Decode is memory-bandwidth bound: weights are streamed once from DRAM,
        plus the accumulated KV-cache tensors for all active sequences in the batch.
        """
        if batch_size <= 0:
            return 0.0

        # Memory transferred: model weights + active KV-cache
        kv_cache_bytes = total_context_tokens * self.hardware.kv_bytes_per_token
        total_bytes = self.weights_bytes + kv_cache_bytes
        memory_time = total_bytes / self.bandwidth_bytes_s

        # Compute FLOPs for 1 token per sequence in the active batch
        decode_flops = 2.0 * (self.hardware.parameter_count_billions * 1e9) * batch_size
        compute_time = decode_flops / self.compute_flops_s

        execution_time = max(memory_time, compute_time) + self.launch_overhead_s
        return execution_time

    def compute_mixed_step_time(
        self,
        num_prefill_tokens: int,
        num_decode_requests: int,
        total_decode_context_tokens: int,
    ) -> float:
        """Calculate execution time for an interleaved batch (e.g. Chunked Prefill).

        In an interleaved iteration, chunked prompt tokens and decode tokens share
        the same forward pass, reading weights exactly once while saturating both
        memory bandwidth and tensor core compute.
        """
        total_tokens = num_prefill_tokens + num_decode_requests
        if total_tokens <= 0:
            return 0.0

        # KV Cache bytes read during attention
        kv_cache_bytes = total_decode_context_tokens * self.hardware.kv_bytes_per_token
        total_bytes = self.weights_bytes + kv_cache_bytes
        memory_time = total_bytes / self.bandwidth_bytes_s

        # FLOPs = 2 * params * (prefill_tokens + decode_tokens)
        total_flops = (
            2.0 * (self.hardware.parameter_count_billions * 1e9) * total_tokens
        )
        compute_time = total_flops / self.compute_flops_s

        execution_time = max(memory_time, compute_time) + self.launch_overhead_s
        return execution_time

    def compute_kv_transfer_time(self, num_tokens: int) -> float:
        """Calculate network transfer latency of KV-cache tensors for PD Disaggregation.

        Models transferring KV cache across network (e.g. 100 Gbps Ethernet / InfiniBand / PCIe).
        """
        if num_tokens <= 0:
            return 0.0

        tensor_bytes = num_tokens * self.hardware.kv_bytes_per_token
        network_bytes_s = (
            self.hardware.network_bandwidth_gbps * 1e9
        ) / 8.0  # bits to bytes
        transfer_time = (tensor_bytes / network_bytes_s) + (
            self.hardware.network_latency_ms / 1000.0
        )
        return transfer_time

    def compute_swap_time(self, num_blocks: int, block_size: int = 16) -> float:
        """Calculate time to swap physical KV blocks between GPU VRAM and CPU Host DRAM over PCIe."""
        if num_blocks <= 0:
            return 0.0

        block_bytes = num_blocks * block_size * self.hardware.kv_bytes_per_token
        pcie_bytes_s = self.hardware.pcie_bandwidth_gb_s * 1e9
        return block_bytes / pcie_bytes_s
