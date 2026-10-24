"""Command Line Interface (CLI) for sched-sim."""

import json
from pathlib import Path
from typing import Annotated, Optional
import typer
from rich.console import Console
from rich.table import Table
from sched_sim.core.event_engine import Simulator
from sched_sim.hardware.latency_model import RooflineLatencyModel
from sched_sim.hardware.presets import PRESETS, get_hardware_profile
from sched_sim.hardware.profile import HardwareProfile
from sched_sim.memory.block_manager import BlockManager
from sched_sim.memory.preemption import PreemptionHandler, PreemptionMode
from sched_sim.schedulers.base import BaseScheduler
from sched_sim.schedulers.chunked import ChunkedPrefillScheduler
from sched_sim.schedulers.disaggregated import DisaggregatedScheduler
from sched_sim.schedulers.fcfs import FCFSScheduler
from sched_sim.schedulers.sjf import SJFScheduler
from sched_sim.schedulers.slo_aware import SLOAwareScheduler
from sched_sim.visualizer.exporter import export_metrics_json, export_metrics_markdown
from sched_sim.visualizer.gantt import print_gantt_timeline
from sched_sim.visualizer.terminal import print_comparison_table, print_single_run_summary
from sched_sim.workloads.distributions import DISTRIBUTIONS
from sched_sim.workloads.generator import generate_workload
from sched_sim.workloads.trace_loader import load_trace_file

app = typer.Typer(
    name="sched-sim",
    help="High-fidelity discrete-event simulator for LLM serving schedulers.",
    add_completion=False,
)
console = Console()


def create_scheduler(
    name: str,
    hardware: HardwareProfile,
    preemption_mode: PreemptionMode = PreemptionMode.RECOMPUTE,
    token_budget: int = 512,
    max_batch_size: int = 32,
) -> BaseScheduler:
    """Factory creating configured scheduler instance."""
    block_manager = BlockManager(hardware)
    latency_model = RooflineLatencyModel(hardware)
    preemption_handler = PreemptionHandler(block_manager, latency_model, preemption_mode)

    key = name.lower().replace("-", "").replace("_", "")
    if "chunk" in key:
        return ChunkedPrefillScheduler(
            block_manager,
            latency_model,
            preemption_handler,
            max_batch_size=max_batch_size,
            max_token_budget=token_budget,
        )
    elif "sjf" in key or "shortest" in key:
        return SJFScheduler(block_manager, latency_model, preemption_handler, max_batch_size)
    elif "slo" in key or "deadline" in key:
        return SLOAwareScheduler(block_manager, latency_model, preemption_handler, max_batch_size)
    elif "disagg" in key or "distserve" in key or "splitwise" in key:
        return DisaggregatedScheduler(block_manager, latency_model, preemption_handler, max_batch_size)
    else:
        return FCFSScheduler(block_manager, latency_model, preemption_handler, max_batch_size)


@app.command()
def run(
    scheduler: Annotated[str, typer.Option("--scheduler", "-s", help="Scheduler policy: fcfs, sjf, chunked, slo, disaggregated")] = "chunked",
    workload: Annotated[str, typer.Option("--workload", "-w", help="Workload type: chatbot, rag, codegen, mixed")] = "chatbot",
    hardware: Annotated[str, typer.Option("--hardware", "-h", help="Hardware preset: rtx3050, rtx3050_05b, t4, a100, h100")] = "rtx3050",
    rate: Annotated[float, typer.Option("--rate", "-r", help="Arrival rate in requests/sec")] = 10.0,
    duration: Annotated[float, typer.Option("--duration", "-d", help="Simulation duration in seconds")] = 10.0,
    pattern: Annotated[str, typer.Option("--pattern", help="Arrival pattern: poisson, bursty, constant")] = "poisson",
    preemption: Annotated[str, typer.Option("--preemption", help="Preemption mode: recompute, swap")] = "recompute",
    token_budget: Annotated[int, typer.Option("--token-budget", help="Token budget per step for chunked prefill")] = 512,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for deterministic simulation")] = 42,
    gantt: Annotated[bool, typer.Option("--gantt/--no-gantt", help="Display ASCII Gantt execution timeline")] = True,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Optional output path (.json or .md)")] = None,
) -> None:
    """Run a single LLM serving simulation scenario."""
    hw = get_hardware_profile(hardware)
    p_mode = PreemptionMode.SWAP if preemption.lower() == "swap" else PreemptionMode.RECOMPUTE

    sched_instance = create_scheduler(scheduler, hw, p_mode, token_budget=token_budget)
    simulator = Simulator(hw, sched_instance, preemption_mode=p_mode)

    requests = generate_workload(
        rate_req_per_s=rate,
        duration_s=duration,
        workload_type=workload,
        pattern=pattern,
        seed=seed,
    )
    simulator.submit_workload(requests)

    with console.status(f"[bold cyan]Simulating {sched_instance.name} on {hw.name}..."):
        metrics = simulator.run()

    print_single_run_summary(metrics)

    if gantt and simulator.iteration_records:
        console.print()
        print_gantt_timeline(simulator.iteration_records, max_steps=12)

    if output:
        if output.suffix.lower() == ".json":
            export_metrics_json(metrics, output)
            console.print(f"\n[green]Saved JSON report to {output}[/green]")
        elif output.suffix.lower() == ".md":
            export_metrics_markdown(metrics, output)
            console.print(f"\n[green]Saved Markdown report to {output}[/green]")


@app.command()
def compare(
    schedulers: Annotated[str, typer.Option("--schedulers", help="Comma-separated list: fcfs,sjf,chunked,slo,disaggregated")] = "fcfs,sjf,chunked,slo,disaggregated",
    workload: Annotated[str, typer.Option("--workload", "-w", help="Workload type: chatbot, rag, codegen, mixed")] = "rag",
    hardware: Annotated[str, typer.Option("--hardware", "-h", help="Hardware preset: rtx3050, t4, a100, h100")] = "rtx3050",
    rate: Annotated[float, typer.Option("--rate", "-r", help="Arrival rate in requests/sec")] = 8.0,
    duration: Annotated[float, typer.Option("--duration", "-d", help="Simulation duration in seconds")] = 15.0,
    token_budget: Annotated[int, typer.Option("--token-budget", help="Token budget per step for chunked prefill")] = 512,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for deterministic comparison")] = 42,
) -> None:
    """Run multiple scheduling policies side-by-side on the exact same workload trace."""
    hw = get_hardware_profile(hardware)
    scheduler_keys = [s.strip() for s in schedulers.split(",") if s.strip()]

    # Generate the identical workload trace once
    base_requests = generate_workload(
        rate_req_per_s=rate,
        duration_s=duration,
        workload_type=workload,
        seed=seed,
    )

    results = []
    with console.status(f"[bold magenta]Comparing {len(scheduler_keys)} schedulers on identical trace..."):
        for key in scheduler_keys:
            # Deep clone requests for fair isolation
            req_clones = [r.model_copy(deep=True) for r in base_requests]
            sched_instance = create_scheduler(key, hw, token_budget=token_budget)
            sim = Simulator(hw, sched_instance)
            sim.submit_workload(req_clones)
            m = sim.run()
            results.append(m)

    print_comparison_table(results)


@app.command()
def trace(
    trace_file: Annotated[Path, typer.Argument(help="Path to trace JSON file")],
    scheduler: Annotated[str, typer.Option("--scheduler", "-s", help="Scheduler policy")] = "chunked",
    hardware: Annotated[str, typer.Option("--hardware", "-h", help="Hardware preset")] = "rtx3050",
    token_budget: Annotated[int, typer.Option("--token-budget", help="Chunked prefill budget")] = 512,
    gantt: Annotated[bool, typer.Option("--gantt/--no-gantt", help="Show Gantt timeline")] = True,
) -> None:
    """Replay a real-world trace file through the simulator."""
    hw = get_hardware_profile(hardware)
    sched_instance = create_scheduler(scheduler, hw, token_budget=token_budget)
    sim = Simulator(hw, sched_instance)

    requests = load_trace_file(trace_file)
    sim.submit_workload(requests)

    with console.status(f"[bold cyan]Replaying trace ({len(requests)} requests)..."):
        metrics = sim.run()

    print_single_run_summary(metrics)
    if gantt and sim.iteration_records:
        console.print()
        print_gantt_timeline(sim.iteration_records, max_steps=12)


@app.command()
def calibrate(
    servebench_result: Annotated[Path, typer.Option("--servebench-result", "-s", help="Path to servebench JSON export file")],
    output_preset: Annotated[Optional[Path], typer.Option("--output-preset", "-o", help="Optional output JSON preset file")] = None,
) -> None:
    """Calibrate hardware simulation parameters directly from a servebench benchmark export."""
    if not servebench_result.exists():
        console.print(f"[red]Error: file {servebench_result} not found.[/red]")
        raise typer.Exit(code=1)

    with open(servebench_result, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Read measured latency percentiles from servebench
    measured_ttft = data.get("summary", {}).get("ttft_ms", {}).get("mean", 150.0)
    measured_tpot = data.get("summary", {}).get("tpot_ms", {}).get("mean", 35.0)

    console.print(f"[bold green]✓ Loaded servebench benchmark data from {servebench_result.name}[/bold green]")
    console.print(f"  • Measured TTFT mean: {measured_ttft:.2f} ms")
    console.print(f"  • Measured TPOT mean: {measured_tpot:.2f} ms")

    # Fit launch overhead and tensor core efficiency
    calibrated_launch = max(0.02, min(0.50, measured_tpot * 0.005))
    calibrated_efficiency = max(0.30, min(0.70, (150.0 / max(50.0, measured_ttft)) * 0.48))

    calibrated_hw = HardwareProfile(
        name=f"Calibrated ({servebench_result.stem})",
        kernel_launch_overhead_ms=round(calibrated_launch, 3),
        tensor_core_efficiency=round(calibrated_efficiency, 3),
    )

    console.print(f"  • Calibrated Launch Overhead: {calibrated_hw.kernel_launch_overhead_ms} ms")
    console.print(f"  • Calibrated Tensor Core Efficiency: {calibrated_hw.tensor_core_efficiency*100:.1f}%")

    if output_preset:
        output_preset.parent.mkdir(parents=True, exist_ok=True)
        with open(output_preset, "w", encoding="utf-8") as f:
            f.write(calibrated_hw.model_dump_json(indent=2))
        console.print(f"[green]Saved calibrated preset to {output_preset}[/green]")


@app.command()
def sweep(
    schedulers: Annotated[str, typer.Option("--schedulers", help="Comma-separated schedulers to sweep")] = "fcfs,chunked,slo,disaggregated",
    workload: Annotated[str, typer.Option("--workload", "-w", help="Workload type: chatbot, rag, codegen, mixed")] = "rag",
    hardware: Annotated[str, typer.Option("--hardware", "-h", help="Hardware preset")] = "rtx3050",
    min_rate: Annotated[float, typer.Option("--min-rate", help="Starting arrival rate (req/s)")] = 2.0,
    max_rate: Annotated[float, typer.Option("--max-rate", help="Maximum arrival rate (req/s)")] = 18.0,
    step: Annotated[float, typer.Option("--step", help="Rate increment step")] = 4.0,
    duration: Annotated[float, typer.Option("--duration", "-d", help="Simulation duration at each rate step")] = 10.0,
    token_budget: Annotated[int, typer.Option("--token-budget", help="Chunked prefill budget")] = 512,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Optional output JSON path")] = None,
) -> None:
    """Sweep arrival rate to discover scheduler saturation knees and Goodput cliffs."""
    hw = get_hardware_profile(hardware)
    scheduler_keys = [s.strip() for s in schedulers.split(",") if s.strip()]

    rates: list[float] = []
    curr = min_rate
    while curr <= max_rate + 1e-5:
        rates.append(round(curr, 2))
        curr += step

    console.print(f"[bold cyan]Initiating load sweep across {len(rates)} rate steps ({min_rate} -> {max_rate} req/s)...[/bold cyan]")

    sweep_records: list[dict] = []
    for r in rates:
        base_requests = generate_workload(
            rate_req_per_s=r,
            duration_s=duration,
            workload_type=workload,
            seed=seed,
        )
        rate_entry = {"rate": r, "results": {}}
        with console.status(f"[bold yellow]Simulating load: {r:.1f} req/s ({len(base_requests)} requests)..."):
            for key in scheduler_keys:
                req_clones = [req.model_copy(deep=True) for req in base_requests]
                sched_instance = create_scheduler(key, hw, token_budget=token_budget)
                sim = Simulator(hw, sched_instance)
                sim.submit_workload(req_clones)
                m = sim.run()
                rate_entry["results"][sched_instance.name] = {
                    "goodput": m.goodput_ratio,
                    "ttft_p99": m.ttft_ms.p99,
                    "throughput_tok_s": m.throughput_tok_per_s,
                }
        sweep_records.append(rate_entry)

    # Render Pareto Saturation Table
    table = Table(title=f"Saturation Pareto Curve · Goodput vs Load ({hw.name})", show_header=True, header_style="bold magenta")
    table.add_column("Arrival Rate", style="bold cyan", width=14, justify="right")

    # Get sample scheduler names
    sample_names = list(sweep_records[0]["results"].keys())
    for name in sample_names:
        table.add_column(f"{name} (Goodput %)", justify="right")

    for entry in sweep_records:
        row = [f"{entry['rate']:.1f} req/s"]
        for name in sample_names:
            gp = entry["results"][name]["goodput"]
            color = "green" if gp >= 90.0 else ("yellow" if gp >= 60.0 else "red")
            row.append(f"[{color}]{gp:.1f}%[/{color}]")
        table.add_row(*row)

    console.print(table)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(sweep_records, f, indent=2)
        console.print(f"\n[green]Saved sweep data to {output}[/green]")


@app.command()
def info() -> None:
    """Display available silicon hardware presets and workload distribution profiles."""
    hw_table = Table(title="Hardware Silicon Presets", show_header=True, header_style="bold cyan")
    hw_table.add_column("Key", style="bold yellow")
    hw_table.add_column("Silicon Target", style="white")
    hw_table.add_column("VRAM", justify="right")
    hw_table.add_column("DRAM Bandwidth", justify="right")
    hw_table.add_column("Compute (FP16)", justify="right")
    hw_table.add_column("Default Model", style="dim")

    for k, v in PRESETS.items():
        hw_table.add_row(
            k,
            v.name,
            f"{v.vram_gb:.0f} GB",
            f"{v.memory_bandwidth_gb_s:.0f} GB/s",
            f"{v.fp16_tflops:.1f} TFLOPS",
            v.model_name,
        )
    console.print(hw_table)

    dist_table = Table(title="Workload Distribution Profiles", show_header=True, header_style="bold magenta")
    dist_table.add_column("Key", style="bold yellow")
    dist_table.add_column("Domain", style="white")
    dist_table.add_column("Prompt Length (Min-Mean-Max)", justify="center")
    dist_table.add_column("Output Length (Min-Mean-Max)", justify="center")
    dist_table.add_column("Target TTFT SLO", justify="right")

    for k, v in DISTRIBUTIONS.items():
        dist_table.add_row(
            k,
            v.name,
            f"{v.prompt_min} - {v.prompt_mean} - {v.prompt_max}t",
            f"{v.output_min} - {v.output_mean} - {v.output_max}t",
            f"{v.slo_ttft_s*1000:.0f} ms",
        )
    console.print(dist_table)


if __name__ == "__main__":
    app()
