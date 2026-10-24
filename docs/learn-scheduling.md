# Learning Guide: The Schedulers That Power LLM Serving

This hands-on guide explains the operating system principles, queueing theory, and silicon mechanics behind Large Language Model (LLM) serving schedulers.

---

## 1. The Two Fundamental Phases of LLM Inference

Every generative inference request passes through two distinct execution regimes:

```
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│ Phase 1: Prompt Prefill                      │ Phase 2: Autoregressive Token Decode         │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ • Computes KV embeddings for the full prompt │ • Generates tokens one-by-one sequentially   │
│ • Compute-Bound (Matrix Multiplication)     │ • Memory-Bandwidth Bound (Weight Streaming)  │
│ • Saturates Tensor Cores at high TFLOPS      │ • Starves Tensor Cores, limited by DRAM GB/s│
│ • Determines Time-To-First-Token (TTFT)      │ • Determines Time-Per-Output-Token (TPOT)    │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 2. The Core Problem: Head-of-Line (HoL) Blocking & Decode Starvation

In naive First-Come First-Served (**FCFS**) continuous batching, prompt prefills are executed atomically:

```text
Iteration 1: [Decode 5 streams] -> 30 ms
Iteration 2: [Decode 5 streams] -> 30 ms
Iteration 3: >>> A 4,000-token RAG prompt arrives! <<<
             [Full Prefill 4000t] -> 450 ms stall!
Iteration 4: [Decode 6 streams] -> 32 ms
```

During Iteration 3, **all 5 active user streams stall for nearly half a second**. Their inter-token latency (TPOT) spikes from 30 ms to 450 ms, creating severe user-perceived stutter and violating Service Level Objectives (SLOs).

---

## 3. Hands-On Experiment 1: Witnessing Head-of-Line Blocking

Run the bundled RAG stress trace on FCFS vs. Chunked Prefill:

```bash
sched-sim compare --schedulers fcfs,chunked --workload rag --hardware rtx3050 --duration 10
```

### Expected Output:
* In **FCFS**, TPOT p99 spikes to **>400 ms** due to atomic RAG prefill interruptions.
* In **Chunked Prefill**, TPOT p99 stays bounded under **60 ms**, because 4,000-token prompts are sliced into 512-token chunks that share iterations with ongoing decodes.

---

## 4. How Chunked Prefill (Sarathi-Serve) Works

Chunked prefill introduces a **token budget** ($B_{\text{budget}}$, e.g. 512 tokens) per iteration forward pass:

$$\text{Budget Available for Prefill} = B_{\text{budget}} - \text{Active Decode Streams}$$

1. High-priority decode streams take 1 token each.
2. The remaining budget is filled with a chunk of the waiting prompt.
3. The GPU streams model weights **exactly once**, using the prefill tokens to keep Tensor Cores compute-saturated while stepping decode tokens simultaneously.

---

## 5. Little's Law in LLM Serving

In queueing theory, **Little's Law** states:

$$L = \lambda \cdot W$$

Where:
* $L$ = Average number of requests in the serving cluster (in-flight batch).
* $\lambda$ = Request arrival rate (requests per second).
* $W$ = Average end-to-end response time.

When $\lambda$ exceeds the cluster's physical processing capacity ($\rho = \lambda / \mu > 1.0$), queue length $L$ explodes to infinity, memory blocks are exhausted, and the scheduler must **preempt** active sequences (either by recomputation or swapping to CPU DRAM).

---

## 6. Hands-On Experiment 2: Finding the Saturation Knee (`sched-sim sweep`)

Sweep arrival rates to discover when a cluster hits its Goodput cliff:

```bash
sched-sim sweep --schedulers fcfs,chunked,slo --hardware rtx3050 --min-rate 2 --max-rate 18 --step 4
```

### Observation:
* Below 6 req/s, all schedulers deliver near 100% Goodput.
* Beyond 10 req/s, FCFS Goodput collapses rapidly under 40% due to cascading queue delays.
* SLO-Aware scheduling maintains higher Goodput by shedding unrecoverable requests and prioritizing near-deadline queries.

---

## 7. Systems Glossary

| Term | Definition |
| :--- | :--- |
| **TTFT** | **Time-To-First-Token**: Latency from request submission until the first response token is returned. |
| **TPOT** | **Time-Per-Output-Token**: Average generation latency per subsequent token. |
| **ITL** | **Inter-Token Latency**: Time difference between consecutive emitted tokens. |
| **MBU** | **Memory Bandwidth Utilization**: Ratio of achieved DRAM throughput to theoretical silicon peak. |
| **MFU** | **Model FLOPs Utilization**: Ratio of achieved compute operations to theoretical Tensor Core peak. |
| **Goodput** | The percentage of successfully served requests that satisfied both TTFT and TPOT SLO deadlines. |
| **Disaggregation**| Separating Prefill and Decode phases onto separate GPU pools connected by a fast network link (Splitwise/DistServe). |
