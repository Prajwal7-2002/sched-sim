"""Integration tests ensuring all runnable examples execute cleanly in CI."""

import runpy
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def test_example_01_radix_prefix_caching():
    script_path = EXAMPLES_DIR / "01_radix_prefix_caching.py"
    assert script_path.exists(), f"Missing example: {script_path}"
    runpy.run_path(str(script_path), run_name="__main__")


def test_example_02_simulate_schedulers():
    script_path = EXAMPLES_DIR / "02_simulate_schedulers.py"
    assert script_path.exists(), f"Missing example: {script_path}"
    runpy.run_path(str(script_path), run_name="__main__")


def test_example_03_perfetto_trace_export(tmp_path, monkeypatch):
    script_path = EXAMPLES_DIR / "03_perfetto_trace_export.py"
    assert script_path.exists(), f"Missing example: {script_path}"
    # Run in tmp_path to keep workspace clean
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script_path), run_name="__main__")
    assert (tmp_path / "examples" / "output" / "perfetto_trace.json").exists()
