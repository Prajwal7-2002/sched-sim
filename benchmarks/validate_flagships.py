"""Out-of-sample validation of sched-sim against real serving runs.

Data: servebench exports in ``benchmarks/data/`` (closed-loop sweeps at 1, 2 and 4
concurrent users, ~286-token prompts, 128 output tokens, 24 requests per level) recorded
on an RTX 3050 6 GB laptop.

Method, per engine:
  1. Fixed inputs, not fitted: model architecture, bytes per weight (from the served
     weight file), and memory bandwidth measured independently with a device-to-device
     copy (121.5 GB/s on the tested laptop).
  2. Calibrate three engine constants on the *smallest model only*, using only its
     1-user and 2-user runs:
       - step overhead  = TPOT(1 user) - weight streaming time - per-sequence overhead
       - per-sequence   = TPOT(2 users) - TPOT(1 user)
       - prefill efficiency from TTFT(1 user)
  3. Predict everything else (the 4-user run of that model, and every run of the larger
     models) and report the error. Those rows are held out: nothing was fitted on them.

servebench drives a closed loop (a user sends the next request when the previous one
finishes), so the simulation does the same: arrivals are solved from the *simulated*
completions. The only real timing used is the client's median gap between requests.

    uv run python benchmarks/validate_flagships.py
"""

from __future__ import annotations

import json
from pathlib import Path

from sched_sim.core.event_engine import Simulator
from sched_sim.core.request import Request
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.hardware.profile import HardwareProfile
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler, PreemptionMode
from sched_sim.schedulers.fcfs import FCFSScheduler

DATA = Path(__file__).parent / "data"
MEASURED_BANDWIDTH_GB_S = 121.5  # device-to-device copy on the tested RTX 3050 laptop
FP16_TFLOPS = 24.0  # RTX 3050 laptop, dense FP16 Tensor Core peak at ~1.5 GHz

# Qwen2.5 architectures (from each model's config.json).
ARCH = {
    "0.5b": {
        "parameter_count_billions": 0.494,
        "num_layers": 24,
        "hidden_size": 896,
        "num_heads": 14,
        "num_kv_heads": 2,
        "head_dim": 64,
    },
    "1.5b": {
        "parameter_count_billions": 1.544,
        "num_layers": 28,
        "hidden_size": 1536,
        "num_heads": 12,
        "num_kv_heads": 2,
        "head_dim": 128,
    },
    "3b": {
        "parameter_count_billions": 3.086,
        "num_layers": 36,
        "hidden_size": 2048,
        "num_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 128,
    },
}

# (export file, architecture, bytes per weight). Ollama: GGUF file size / parameters.
ENGINES = {
    "Ollama 0.35.1 (llama.cpp, q4_k_m)": [
        ("ollama-qwen2.5-0.5b.json", "0.5b", 397e6 / 0.494e9),
        ("ollama-qwen2.5-1.5b.json", "1.5b", 986e6 / 1.544e9),
        ("ollama-qwen2.5-coder-3b.json", "3b", 1929e6 / 3.086e9),
    ],
    "mini-infer 0.2.9 (PyTorch eager, bf16)": [
        ("mini-infer-qwen2.5-0.5b.json", "0.5b", 2.0),
        ("mini-infer-qwen2.5-1.5b.json", "1.5b", 2.0),
    ],
}


def _profile(arch: str, bytes_per_weight: float, **engine) -> HardwareProfile:
    return HardwareProfile(
        name="RTX 3050 Laptop (6GB)",
        memory_bandwidth_gb_s=MEASURED_BANDWIDTH_GB_S,
        fp16_tflops=FP16_TFLOPS,
        weight_bytes_per_param=bytes_per_weight,
        dtype_bytes=2,
        **ARCH[arch],
        **engine,
    )


def _measured(level: dict) -> dict:
    s = level["summary"]
    return {"ttft_ms": s["ttft"]["mean"] * 1e3, "tpot_ms": s["tpot"]["mean"] * 1e3}


def _run_once(hw: HardwareProfile, level: dict, arrivals: list[float]):
    bm = BlockManager(hw)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    sched = FCFSScheduler(bm, lm, ph, max_batch_size=int(level["value"]))
    sim = Simulator(hw, sched, preemption_mode=PreemptionMode.RECOMPUTE)
    reqs = [
        Request(
            request_id=f"r{i}",
            arrived_at=t,
            prompt_tokens=r["prompt_tokens"],
            output_tokens=r["output_tokens"],
            slo_ttft=0.5,
            slo_tpot=0.05,
        )
        for i, (r, t) in enumerate(zip(_ok(level), arrivals, strict=True))
    ]
    sim.submit_workload(reqs)
    return sim.run(), reqs


def _ok(level: dict) -> list[dict]:
    return sorted((r for r in level["requests"] if r["ok"]), key=lambda r: r["start"])


def _simulate(hw: HardwareProfile, level: dict) -> dict:
    """Closed-loop simulation: with c users, request i starts when an earlier one finishes.

    Arrivals are solved by fixed-point iteration from the simulated completions; the only
    real timing used is the client's median gap between one request ending and the next
    starting (HTTP and scheduling on the client side).
    """
    c = int(level["value"])
    reqs = _ok(level)
    ends = sorted(r["start"] + r["e2e"] for r in reqs)
    gaps = [max(0.0, reqs[i]["start"] - ends[i - c]) for i in range(c, len(reqs))]
    gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    arrivals = [r["start"] for r in reqs]
    for _ in range(50):
        metrics, sim_reqs = _run_once(hw, level, arrivals)
        done = sorted(r.completed_at for r in sim_reqs)
        new = arrivals[:c] + [done[i - c] + gap for i in range(c, len(reqs))]
        if max(abs(x - y) for x, y in zip(new, arrivals, strict=True)) < 1e-6:
            break
        arrivals = new
    return {"ttft_ms": metrics.ttft_ms.mean, "tpot_ms": metrics.tpot_ms.mean}


def _levels(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(lv["value"]): lv for lv in data["levels"]}


def calibrate(levels: dict[int, dict], arch: str, bytes_per_weight: float) -> dict:
    """Engine constants from the 1- and 2-user runs of one model."""
    hw0 = _profile(arch, bytes_per_weight)
    weight_ms = (
        hw0.parameter_count_billions
        * 1e9
        * bytes_per_weight
        / (hw0.memory_bandwidth_gb_s * 1e9)
        * 1e3
    )
    one, two = _measured(levels[1]), _measured(levels[2])
    per_seq = max(0.0, two["tpot_ms"] - one["tpot_ms"])
    overhead = max(0.0, one["tpot_ms"] - weight_ms - per_seq)
    prompt = sum(r["prompt_tokens"] for r in levels[1]["requests"]) / len(
        levels[1]["requests"]
    )
    flops = 2 * hw0.parameter_count_billions * 1e9 * prompt
    prefill_s = max(1e-4, (one["ttft_ms"] - overhead) / 1e3)
    efficiency = min(1.0, flops / (prefill_s * FP16_TFLOPS * 1e12))
    return {
        "kernel_launch_overhead_ms": round(overhead, 3),
        "per_sequence_overhead_ms": round(per_seq, 3),
        "tensor_core_efficiency": round(efficiency, 4),
    }


def main() -> list[dict]:
    rows = []
    for engine, runs in ENGINES.items():
        available = [(f, a, b) for f, a, b in runs if (DATA / f).exists()]
        if not available:
            print(f"skip {engine}: no data")
            continue
        cal_file, cal_arch, cal_bpw = available[0]
        constants = calibrate(_levels(DATA / cal_file), cal_arch, cal_bpw)
        print(f"\n{engine}\n  calibrated on {cal_arch} (1 and 2 users): {constants}")
        for f, arch, bpw in available:
            hw = _profile(arch, bpw, **constants)
            for users, level in _levels(DATA / f).items():
                real, sim = _measured(level), _simulate(hw, level)
                held_out = not (f == cal_file and users in (1, 2))
                row = {
                    "engine": engine,
                    "model": arch,
                    "users": users,
                    "held_out": held_out,
                    **{f"real_{k}": round(v, 1) for k, v in real.items()},
                    **{f"sim_{k}": round(v, 1) for k, v in sim.items()},
                    **{
                        f"err_{k}_pct": round((sim[k] - real[k]) / real[k] * 100, 1)
                        for k in real
                    },
                }
                rows.append(row)
                tag = "held-out" if held_out else "calibration"
                print(
                    f"  {arch:4s} {users} users [{tag:11s}] TTFT real {real['ttft_ms']:7.1f} sim {sim['ttft_ms']:7.1f} ({row['err_ttft_ms_pct']:+6.1f}%)"
                    f" | TPOT real {real['tpot_ms']:6.1f} sim {sim['tpot_ms']:6.1f} ({row['err_tpot_ms_pct']:+6.1f}%)"
                )
    return rows


def write_report(rows: list[dict], path: Path) -> None:
    """Write the results table as Markdown (benchmarks/RESULTS.md)."""
    lines = [
        "# sched-sim validation against real serving runs",
        "",
        "Generated by `uv run python benchmarks/validate_flagships.py`; method in the script's",
        "docstring. Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU, AMD Ryzen 5 6600H.",
        "Workload: closed loop, ~286-token prompts, 128 output tokens, 24 requests per level.",
        "",
        "**calibration** rows were used to fit the engine's three constants (smallest model,",
        "1 and 2 users). **held-out** rows were not used in any way and are true predictions.",
        "",
        "| Engine | Model | Users | Row | TTFT real | TTFT sim | TTFT err | TPOT real | TPOT sim | TPOT err |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        kind = "held-out" if r["held_out"] else "calibration"
        lines.append(
            f"| {r['engine']} | {r['model']} | {r['users']} | {kind} "
            f"| {r['real_ttft_ms']} ms | {r['sim_ttft_ms']} ms | {r['err_ttft_ms_pct']:+.1f}% "
            f"| {r['real_tpot_ms']} ms | {r['sim_tpot_ms']} ms | {r['err_tpot_ms_pct']:+.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    results = main()
    here = Path(__file__).parent
    (here / "validation_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    write_report(results, here / "RESULTS.md")
