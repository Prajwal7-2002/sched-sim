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

---

## 4. Validation against real runs (RTX 3050 6GB Laptop GPU)

Calibration fits each engine's constants to a few measured points, so the error on those points says
nothing about accuracy: it is small by construction. What matters is **held-out** rows that were not
used for fitting.

`benchmarks/validate_flagships.py` measures Ollama and mini-infer with `servebench` on an AMD Ryzen 5
6600H + NVIDIA RTX 3050 6GB Laptop GPU, fits on the smallest model at 1 and 2 users, and predicts the
rest. Full table: [`benchmarks/RESULTS.md`](https://github.com/Prajwal7-2002/sched-sim/blob/main/benchmarks/RESULTS.md).

On the 11 held-out rows (Ollama 0.5B/1.5B/3B, mini-infer 0.5B/1.5B, 1 to 4 users):
* **TPOT** (time per output token) was within ±12% in 7 of 11 rows.
* **TTFT** (time to first token) was within ±12% in 1 of 11 rows; it was overestimated by 34–100% for
  larger Ollama models and underestimated by 56–69% for mini-infer.

So the simulator is useful for comparing schedulers and for decode-speed estimates on hardware it has
been calibrated on, but its TTFT predictions are not reliable yet. Calibrate on your own hardware and
check against held-out runs before trusting absolute numbers.

