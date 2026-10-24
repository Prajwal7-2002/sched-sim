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
        [
            "run",
            "--scheduler",
            "chunked",
            "--workload",
            "chatbot",
            "--duration",
            "2",
            "--rate",
            "5",
            "--no-gantt",
        ],
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


def test_cli_trace_fast(tmp_path):
    trace_file = "traces/lmsys_sample.json"
    result = runner.invoke(
        app,
        ["trace", trace_file, "--scheduler", "chunked", "--no-gantt"],
    )
    assert result.exit_code == 0
    assert "Execution Telemetry" in result.output
    assert "Completed / Total" in result.output


def test_cli_sweep_fast():
    result = runner.invoke(
        app,
        [
            "sweep",
            "--schedulers",
            "fcfs,chunked",
            "--min-rate",
            "2",
            "--max-rate",
            "4",
            "--step",
            "2",
            "--duration",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Saturation Pareto Curve" in result.output
    assert "Arrival Rate" in result.output


def test_cli_calibrate(tmp_path):
    import json

    dummy_bench = tmp_path / "mock_servebench.json"
    dummy_bench.write_text(
        json.dumps({"summary": {"ttft_ms": {"mean": 120.0}, "tpot_ms": {"mean": 28.5}}})
    )
    out_preset = tmp_path / "calibrated_preset.json"

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--servebench-result",
            str(dummy_bench),
            "--output-preset",
            str(out_preset),
        ],
    )
    assert result.exit_code == 0
    assert "Loaded servebench benchmark data" in result.output
    assert out_preset.exists()


def test_cli_calibrate_nested_levels_and_run(tmp_path):
    import json

    from sched_sim.hardware.presets import get_hardware_profile

    dummy_bench = tmp_path / "servebench_np4.json"
    dummy_bench.write_text(
        json.dumps(
            {
                "config": {"model": "qwen2.5:0.5b"},
                "levels": [
                    {
                        "summary": {
                            "ttft": {"mean": 0.045},  # 45 ms in seconds
                            "tpot": {"mean": 0.005},  # 5 ms in seconds
                        }
                    }
                ],
            }
        )
    )
    out_preset = tmp_path / "calibrated_05b.json"

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--servebench-result",
            str(dummy_bench),
            "--output-preset",
            str(out_preset),
        ],
    )
    assert result.exit_code == 0
    assert "Loaded servebench benchmark data" in result.output
    assert out_preset.exists()

    # Verify get_hardware_profile can load custom JSON
    hw = get_hardware_profile(str(out_preset))
    assert hw.parameter_count_billions == 0.49

    # Verify sched-sim run can consume the custom json profile directly
    run_result = runner.invoke(
        app,
        [
            "run",
            "--hardware",
            str(out_preset),
            "--duration",
            "1",
            "--rate",
            "2",
            "--no-gantt",
        ],
    )
    assert run_result.exit_code == 0
    assert "Execution Telemetry" in run_result.output
