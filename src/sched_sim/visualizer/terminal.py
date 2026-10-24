"""Rich terminal rendering for simulation results and side-by-side comparisons."""

from rich.console import Console
from rich.table import Table

from sched_sim.core.metrics import SimulationMetrics

console = Console()


def print_single_run_summary(metrics: SimulationMetrics) -> None:
    """Print an executive summary panel for a single simulation run."""
    table = Table(
        title=f"Execution Telemetry - {metrics.scheduler_name}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Category", style="dim", width=24)
    table.add_column("Metric", style="bold white", width=22)
    table.add_column("Value", style="green", width=20)

    # Core Stats
    table.add_row("Cluster Silicon", "Target Accelerator", metrics.hardware_name)
    table.add_row(
        "Simulation Horizon",
        "Simulated Duration",
        f"{metrics.simulation_duration_s:.2f} s",
    )
    table.add_row(
        "Workload Volume",
        "Completed / Total",
        f"{metrics.completed_requests} / {metrics.total_requests}",
    )
    table.add_row(
        "Workload Volume", "Dropped (SLO Violations)", f"{metrics.dropped_requests}"
    )
    table.add_row(
        "Memory Preemption",
        "Preemptions / Recomputes",
        f"{metrics.preempted_requests_count} ({metrics.total_recomputed_tokens} tok)",
    )

    table.add_section()
    # Throughput & Goodput
    table.add_row(
        "Serving Economics",
        "Throughput (Tokens/s)",
        f"{metrics.throughput_tok_per_s:.1f} tok/s",
    )
    table.add_row(
        "Serving Economics",
        "Throughput (Requests/s)",
        f"{metrics.throughput_req_per_s:.1f} req/s",
    )
    table.add_row(
        "Serving Economics", "SLO Goodput Ratio", f"{metrics.goodput_ratio:.1f}%"
    )

    table.add_section()
    # Latencies
    table.add_row(
        "Latency Percentiles",
        "TTFT (p50 / p90 / p99)",
        f"{metrics.ttft_ms.p50} / {metrics.ttft_ms.p90} / {metrics.ttft_ms.p99} ms",
    )
    table.add_row(
        "Latency Percentiles",
        "TPOT (p50 / p90 / p99)",
        f"{metrics.tpot_ms.p50} / {metrics.tpot_ms.p90} / {metrics.tpot_ms.p99} ms/tok",
    )
    table.add_row(
        "Latency Percentiles",
        "End-to-End (p50 / p99)",
        f"{metrics.e2e_ms.p50} / {metrics.e2e_ms.p99} ms",
    )

    table.add_section()
    # Hardware Saturation
    table.add_row(
        "Hardware Saturation",
        "GPU Active Utilization",
        f"{metrics.gpu_utilization_pct:.1f}%",
    )
    table.add_row(
        "Hardware Saturation",
        "Average Batch Size",
        f"{metrics.avg_batch_size:.1f} streams",
    )
    table.add_row(
        "Hardware Saturation",
        "Peak KV Cache Util",
        f"{metrics.peak_kv_utilization_pct:.1f}%",
    )

    console.print(table)


def print_comparison_table(metrics_list: list[SimulationMetrics]) -> None:
    """Print a side-by-side comparative leaderboard of multiple schedulers."""
    table = Table(
        title="LLM Serving Scheduler Leaderboard",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Scheduler Policy", style="bold cyan", width=26)
    table.add_column("Throughput", justify="right")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p99", justify="right")
    table.add_column("TPOT p50", justify="right")
    table.add_column("TPOT p99", justify="right")
    table.add_column("Preemptions", justify="right")
    table.add_column("Goodput", justify="right", style="bold green")

    for m in metrics_list:
        table.add_row(
            m.scheduler_name,
            f"{m.throughput_tok_per_s:.1f} tok/s",
            f"{m.ttft_ms.p50:.1f} ms",
            f"{m.ttft_ms.p99:.1f} ms",
            f"{m.tpot_ms.p50:.1f} ms",
            f"{m.tpot_ms.p99:.1f} ms",
            str(m.preempted_requests_count),
            f"{m.goodput_ratio:.1f}%",
        )

    console.print(table)
