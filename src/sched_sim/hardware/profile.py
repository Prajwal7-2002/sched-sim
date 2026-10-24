"""Hardware profile schema representing GPU silicon and interconnect capabilities."""

from pydantic import BaseModel, Field


class HardwareProfile(BaseModel):
    """Specification of the target accelerator silicon, memory subsystem, and interconnect."""

    name: str = "NVIDIA GeForce RTX 3050 Laptop GPU"
    vram_gb: float = 6.0
    memory_bandwidth_gb_s: float = 168.0
    fp16_tflops: float = 9.0
    tensor_core_efficiency: float = 0.50
    kernel_launch_overhead_ms: float = 0.08
    pcie_bandwidth_gb_s: float = 16.0
    network_bandwidth_gbps: float = 100.0
    network_latency_ms: float = 0.02

    # Model specifications to run on this hardware
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    parameter_count_billions: float = 1.54
    num_layers: int = 28
    hidden_size: int = 1536
    num_heads: int = 12
    num_kv_heads: int = 2  # GQA (Grouped Query Attention)
    head_dim: int = 128
    dtype_bytes: int = 2  # bfloat16 = 2 bytes

    @property
    def model_weights_gb(self) -> float:
        """Raw model weight memory in gigabytes."""
        return (self.parameter_count_billions * 1e9 * self.dtype_bytes) / (1024**3)

    @property
    def kv_bytes_per_token(self) -> int:
        """KV-cache memory required per token across all layers (2 * layers * kv_heads * head_dim * bytes)."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def effective_tflops(self) -> float:
        """Effective achievable TFLOPS taking into account real tensor core efficiency."""
        return self.fp16_tflops * self.tensor_core_efficiency
