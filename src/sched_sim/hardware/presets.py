"""Predefined hardware and model configurations for realistic simulation."""

from sched_sim.hardware.profile import HardwareProfile


PRESETS: dict[str, HardwareProfile] = {
    "rtx3050": HardwareProfile(
        name="RTX 3050 Laptop (6GB)",
        vram_gb=6.0,
        memory_bandwidth_gb_s=168.0,
        fp16_tflops=9.0,
        tensor_core_efficiency=0.48,
        kernel_launch_overhead_ms=0.10,
        pcie_bandwidth_gb_s=16.0,
        network_bandwidth_gbps=10.0,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        parameter_count_billions=1.54,
        num_layers=28,
        hidden_size=1536,
        num_heads=12,
        num_kv_heads=2,
        head_dim=128,
        dtype_bytes=2,
    ),
    "rtx3050_05b": HardwareProfile(
        name="RTX 3050 Laptop (6GB) · 0.5B",
        vram_gb=6.0,
        memory_bandwidth_gb_s=168.0,
        fp16_tflops=9.0,
        tensor_core_efficiency=0.48,
        kernel_launch_overhead_ms=0.08,
        pcie_bandwidth_gb_s=16.0,
        network_bandwidth_gbps=10.0,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        parameter_count_billions=0.49,
        num_layers=24,
        hidden_size=896,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        dtype_bytes=2,
    ),
    "t4": HardwareProfile(
        name="NVIDIA T4 (16GB)",
        vram_gb=16.0,
        memory_bandwidth_gb_s=300.0,
        fp16_tflops=65.0,
        tensor_core_efficiency=0.45,
        kernel_launch_overhead_ms=0.08,
        pcie_bandwidth_gb_s=16.0,
        network_bandwidth_gbps=25.0,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        parameter_count_billions=1.54,
        num_layers=28,
        hidden_size=1536,
        num_heads=12,
        num_kv_heads=2,
        head_dim=128,
        dtype_bytes=2,
    ),
    "a100": HardwareProfile(
        name="NVIDIA A100 SXM4 (80GB)",
        vram_gb=80.0,
        memory_bandwidth_gb_s=1935.0,
        fp16_tflops=312.0,
        tensor_core_efficiency=0.52,
        kernel_launch_overhead_ms=0.05,
        pcie_bandwidth_gb_s=64.0,
        network_bandwidth_gbps=100.0,
        model_name="Meta-Llama-3-8B-Instruct",
        parameter_count_billions=8.03,
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
    ),
    "h100": HardwareProfile(
        name="NVIDIA H100 SXM5 (80GB)",
        vram_gb=80.0,
        memory_bandwidth_gb_s=3350.0,
        fp16_tflops=989.0,
        tensor_core_efficiency=0.55,
        kernel_launch_overhead_ms=0.03,
        pcie_bandwidth_gb_s=128.0,
        network_bandwidth_gbps=400.0,
        model_name="Meta-Llama-3-70B-Instruct",
        parameter_count_billions=70.6,
        num_layers=80,
        hidden_size=8192,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
    ),
}


def get_hardware_profile(name_or_key: str) -> HardwareProfile:
    """Retrieve hardware profile by key (e.g., 'rtx3050', 'a100') or default to rtx3050."""
    key = name_or_key.lower().replace("-", "").replace(" ", "")
    for k, v in PRESETS.items():
        if k == key or k.replace("_", "") == key:
            return v
    # fallback
    return PRESETS["rtx3050"]
