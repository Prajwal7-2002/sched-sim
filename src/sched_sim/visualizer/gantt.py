"""ASCII / Unicode Gantt timeline visualizer for GPU iterations."""

from rich.console import Console
from rich.table import Table
from sched_sim.core.event_engine import IterationRecord

console = Console()


def print_gantt_timeline(records: list[IterationRecord], max_steps: int = 15) -> None:
    """Print an execution timeline of the first N forward pass iterations."""
    table = Table(title="GPU Iteration Step Timeline (Gantt Trace)", show_header=True, header_style="bold yellow")
    table.add_column("Step", style="dim", width=6)
    table.add_column("Time", style="cyan", width=12)
    table.add_column("Duration", style="white", width=10)
    table.add_column("Phase Breakdown", style="bold magenta", width=24)
    table.add_column("KV Memory", style="green", width=12)
    table.add_column("Active Requests (Tokens)", style="white")

    slice_records = records[:max_steps]
    for r in slice_records:
        if r.prefill_tokens > 0 and r.decode_count > 0:
            phase = f"[MIXED] {r.prefill_tokens}p + {r.decode_count}d"
        elif r.prefill_tokens > 0:
            phase = f"[PREFILL] {r.prefill_tokens} tok"
        else:
            phase = f"[DECODE] {r.decode_count} streams"

        time_str = f"{r.start_time*1000.0:8.1f} ms"
        dur_str = f"{r.duration*1000.0:6.1f} ms"
        req_sample = ", ".join(r.active_request_ids[:4])
        if len(r.active_request_ids) > 4:
            req_sample += f" (+{len(r.active_request_ids)-4})"

        table.add_row(
            f"#{r.iteration_index:03d}",
            time_str,
            dur_str,
            phase,
            f"{r.kv_utilization*100.0:.1f}%",
            req_sample,
        )

    console.print(table)
