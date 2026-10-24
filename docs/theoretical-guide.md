# Theoretical Guide: Queueing Theory, Roofline Models, & Network Physics

This document details the mathematical models used inside `sched-sim`'s discrete-event simulation engine.

---

## 1. Roofline Latency Modeling

### Prefill Stage (Compute-Bound)
During prompt prefill, the model processes $N$ input tokens simultaneously in a batched matrix multiplication. The execution time is bounded by:

$$T_{\text{prefill}}(N) = T_{\text{launch}} + \max\left(\frac{2 \cdot P \cdot N}{\text{Effective TFLOPS} \times 10^{12}}, \frac{P \cdot \text{dtype\_bytes}}{\text{DRAM Bandwidth}}\right)$$

Where:
* $P$ = Parameter count (e.g., $1.54 \times 10^9$ for Qwen2.5-1.5B).
* $\text{Effective TFLOPS} = \text{Peak TFLOPS} \times \eta_{\text{efficiency}}$.
* $T_{\text{launch}}$ = CUDA kernel dispatch latency (~0.05–0.10 ms).

### Decode Stage (Memory-Bandwidth Bound)
Autoregressive token generation generates 1 token per sequence. In each iteration step, the GPU must stream all model weights from DRAM plus the accumulated Key-Value tensors across the active batch $B$:

$$T_{\text{decode}}(B, L_{\text{total}}) = T_{\text{launch}} + \frac{P \cdot \text{dtype\_bytes} + L_{\text{total}} \cdot D_{\text{kv\_token}}}{\text{DRAM Bandwidth (bytes/s)}}$$

Where $D_{\text{kv\_token}}$ is the bytes per token across all layers:
$$D_{\text{kv\_token}} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times \text{dtype\_bytes}$$

---

## 2. Prefill-Decode Disaggregation (PD Split) Networking

In disaggregated serving (DistServe, Splitwise), prefill workers stream finished KV-caches to decode workers over high-speed interconnects (PCIe Gen4/5, NVLink, 100GbE):

$$T_{\text{transfer}} = \frac{L_{\text{prompt}} \cdot D_{\text{kv\_token}}}{\text{Interconnect Bandwidth (bytes/s)}} + \text{Latency}_{\text{network}}$$

When network bandwidth is sufficiently high (e.g. 100–400 Gbps), transfer latency is a negligible fraction of decode time, completely isolating decode streams from prefill latency spikes.

---

## 3. Queueing Dynamics ($M/G/1$ Queues)

In an idealized serving system with Poisson arrivals (rate $\lambda$) and general service distribution $S$, the expected waiting time in queue $W_q$ is given by the **Pollaczek–Khinchine (P-K) formula**:

$$W_q = \frac{\lambda \cdot \mathbb{E}[S^2]}{2(1 - \rho)}$$

Where:
* $\rho = \lambda \cdot \mathbb{E}[S]$ is the cluster utilization.
* $\mathbb{E}[S^2] = \text{Var}(S) + (\mathbb{E}[S])^2$ represents the second moment of service time.

### Why Variance Kills Latency:
In asymmetric workloads (such as RAG, where prompt lengths range from 50 to 5,000 tokens), the variance $\text{Var}(S)$ is enormous. The P-K formula proves that high variance in service time explodes waiting time $W_q$ even when average utilization $\rho$ is moderate!

**Chunked Prefill** directly dampens $\text{Var}(S)$ by enforcing uniform chunk execution steps, driving queue waiting times down dramatically.
