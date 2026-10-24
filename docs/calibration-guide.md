# Hardware Calibration Guide: Feeding Real Silicon Benchmarks into sched-sim

`sched-sim` grounds its discrete-event simulation in verified analytical performance models. This guide shows how to calibrate simulation parameters using benchmark exports from **`servebench`** and **`mini-infer`**.

---

## 1. Capturing Empirical Silicon Data with `servebench`

Run a quick load-test against your local inference server (e.g. `mini-infer`, Ollama, vLLM):

```bash
# Export benchmark metrics to JSON
servebench run --url http://127.0.0.1:8000/v1 --model qwen2.5:0.5b --concurrency 4 --requests 100 --out-json mini_infer_benchmark.json
```

`servebench` produces microsecond-precision empirical distributions of:
* TTFT mean and percentiles ($N \ge 100$).
* TPOT mean and percentiles.
* Memory Bandwidth Utilization (MBU).

---

## 2. Running `sched-sim calibrate`

Point `sched-sim` to the exported JSON file:

```bash
sched-sim calibrate --servebench-result ./mini_infer_benchmark.json --output-preset local_calibrated.json
```

### What Calibration Fits:
1. **Kernel Launch Overhead ($T_{\text{launch}}$)**: Fitted from base driver and kernel dispatch latency (typically 0.05–0.15 ms).
2. **Tensor Core Efficiency ($\eta$)**: Fitted from measured prefill TFLOPS vs theoretical peak (typically 40–55% on consumer GPUs, up to 65% on H100).

---

## 3. Replaying Simulations with Your Custom Preset

Use the calibrated preset directly in simulations:

```bash
sched-sim run --hardware local_calibrated.json --scheduler chunked --duration 10
```
