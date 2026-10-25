"""Example 3: Exporting Chromium / Perfetto Interactive Timeline Traces.

Simulates an LLM serving workload and exports the iteration-level execution records
to Perfetto Trace JSON format (Catapult format).
The resulting JSON file can be dropped directly into https://ui.perfetto.dev to inspect
forward pass slices, KV-cache utilization, and individual request lifecycles.
"""

from pathlib import Path

from sched_sim.core.event_engine import Simulator
from sched_sim.core.request import Request
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.hardware.presets import PRESETS
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler, PreemptionMode
from sched_sim.schedulers.cache_aware import CacheAwareScheduler
from sched_sim.visualizer.perfetto import export_perfetto_trace


def main() -> None:
    print("=" * 70)
    print("Example 3: Perfetto Interactive Trace Exporter")
    print("=" * 70)

    # 1. Initialize Hardware and Engine
    hw = PRESETS["a100"]
    bm = BlockManager(hw, block_size=16, enable_prefix_caching=True)
    lm = RooflineLatencyModel(hw)
    ph = PreemptionHandler(bm, lm, PreemptionMode.RECOMPUTE)
    scheduler = CacheAwareScheduler(bm, lm, ph, max_batch_size=8)
    sim = Simulator(hw, scheduler)

    # 2. Build multi-turn conversational workload
    requests = []
    for i in range(12):
        # Even requests share Context A, Odd requests share Context B
        ctx_key = "system_agent_prompt" if (i % 2 == 0) else "coding_assistant_rag"
        req = Request(
            request_id=f"turn_{i:02d}",
            arrived_at=i * 0.04,  # Staggered 40ms arrivals
            prompt_tokens=384,
            output_tokens=24,
            prefix_key=ctx_key,
        )
        requests.append(req)

    print(f"\n[1] Submitting {len(requests)} requests to Simulator ({hw.name})...")
    sim.submit_workload(requests)

    # 3. Execute Simulation
    metrics = sim.run()
    print(f"[2] Simulation completed in {metrics.simulation_duration_s:.3f}s:")
    print(
        f"    - Completed Requests : {metrics.completed_requests}/{metrics.total_requests}"
    )
    print(f"    - Mean TTFT          : {metrics.ttft_ms.mean:.2f} ms")
    print(f"    - Token Throughput   : {metrics.throughput_tok_per_s:.2f} tok/s")
    print(
        f"    - Iteration Slices   : {len(sim.iteration_records)} forward passes recorded"
    )

    # 4. Export Perfetto Trace
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "perfetto_trace.json"

    export_perfetto_trace(
        metrics=metrics,
        iteration_records=sim.iteration_records,
        all_requests=requests,
        output_path=trace_path,
    )

    file_size_kb = trace_path.stat().st_size / 1024
    print(f"\n[3] Exported Perfetto trace to: {trace_path} ({file_size_kb:.1f} KB)")

    # 5. Visualizer Instructions
    print("\n" + "=" * 70)
    print("HOW TO VISUALIZE YOUR GPU SCHEDULE IN PERFETTO:")
    print("-" * 70)
    print("1. Open your browser and navigate to:  https://ui.perfetto.dev")
    print("2. Click 'Open trace file' in the sidebar.")
    print(f"3. Select: {trace_path.resolve()}")
    print("4. Inspect the timeline tracks:")
    print("   • GPU Forward Passes (Prefill vs Decode phases with batch sizes)")
    print("   • Request Lifecycles (Arrival -> First Token -> Finish)")
    print("   • Memory Counters (GPU KV-Cache Utilization % over time)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
