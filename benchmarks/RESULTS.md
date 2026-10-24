# Empirical Validation Report: sched-sim vs Real Silicon

Ground-truth empirical validation comparing `sched-sim` analytical roofline predictions against measured benchmarks from **`servebench`** and **`mini-infer`** on physical hardware (**AMD Ryzen 5 6600H + NVIDIA RTX 3050 6GB Laptop GPU**).

## Validation Results Matrix

| Target Serving Engine | Model | Metric | Real Hardware (`servebench`) | `sched-sim` Predicted | Error % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ollama (C++/CUDA runtime)** | Qwen 2.5 0.5B | **TTFT** | 44.33 ms | 47.81 ms | **7.9%** |
| | | **TPOT** | 4.96 ms | 5.33 ms | **7.5%** |
| | | **E2E** | 627.5 ms | 666.73 ms | **6.3%** |
| **mini-infer (PyTorch eager runtime)** | Qwen 2.5 0.5B | **TTFT** | 121.8 ms | 140.6 ms | **15.4%** |
| | | **TPOT** | 103.24 ms | 106.67 ms | **3.3%** |
| | | **E2E** | 3322.29 ms | 3447.26 ms | **3.8%** |

## Analytical Methodology

1. **Compute-bound phases (Prefill)**: Modeled via standard FLOPs calculation ($2 \times P \times N$) bounded by effective Tensor Core TFLOPS plus kernel launch overhead.
2. **Memory-bound phases (Decode)**: Modeled via model weight streaming and physical 16-token Paged KV-cache reads bounded by achievable DRAM memory bandwidth (168 GB/s).
3. **Runtime Overheads**: `sched-sim calibrate` accurately accounts for C++/CUDA runtime efficiency vs Python/PyTorch eager op-dispatch latency.
