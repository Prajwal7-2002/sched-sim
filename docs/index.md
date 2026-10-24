# sched-sim: Discrete-Event LLM Serving Scheduler Simulator

Welcome to the documentation for **`sched-sim`**, a high-fidelity discrete-event simulator for Large Language Model (LLM) continuous-batching schedulers.

## Why sched-sim?

In modern LLM inference clusters (vLLM, TGI, TensorRT-LLM, SGLang, DistServe), **the scheduling policy is the single largest determinant of cluster throughput, latency, and hardware cost**.

While continuous batching and Paged KV-caching provide the mechanism for dynamic execution, the scheduler determines:
1. Which requests enter the next forward iteration.
2. How large prompt prefills are chunked to prevent decode starvation.
3. What happens when physical KV-cache memory runs out (swapping vs recomputation).
4. When to disaggregate prefill and decode phases onto specialized GPU worker pools.

Testing these policies on real GPU clusters (e.g. 8x H100 nodes) is **slow, noisy, and expensive ($20–$50/hr)**. `sched-sim` simulates hours of multi-GPU serving traffic in **seconds on any standard CPU**, using validated analytical performance models calibrated against empirical silicon measurements from **[servebench](https://github.com/Prajwal7-2002/servebench)** and **[mini-infer](https://github.com/Prajwal7-2002/mini-infer)**.

---

## Quick Navigation

* **[Learning Guide: The Schedulers That Power LLM Serving](learn-scheduling.md)** — Core concepts, Little's Law, queueing theory, and hands-on experiments.
* **[Theoretical Guide](theoretical-guide.md)** — Mathematical derivations of $M/G/1$ queues, roofline limits, and KV-cache network transfer physics.
* **[Calibration Guide](calibration-guide.md)** — How to feed real silicon benchmarks from `servebench` and `mini-infer` into `sched-sim`.
