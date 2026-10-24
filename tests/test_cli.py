"""Tests for sched-sim Typer CLI."""

from typer.testing import CliRunner
from sched_sim.cli import app

runner = CliRunner()


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Hardware Silicon Presets" in result.output
    assert "rtx3050" in result.output.lower()
    assert "Workload Distribution Profiles" in result.output


def test_cli_run_fast():
    result = runner.invoke(
        app,
        ["run", "--scheduler", "chunked", "--workload", "chatbot", "--duration", "2", "--rate", "5", "--no-gantt"],
    )
    assert result.exit_code == 0
    assert "Execution Telemetry" in result.output
    assert "Throughput (Tokens/s)" in result.output


def test_cli_compare_fast():
    result = runner.invoke(
        app,
        ["compare", "--schedulers", "fcfs,chunked", "--duration", "2", "--rate", "5"],
    )
    assert result.exit_code == 0
    assert "LLM Serving Scheduler Leaderboard" in result.output
    assert "FCFS" in result.output
    assert "Chunked Prefill" in result.output
