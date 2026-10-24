import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.validate_flagships import (
    run_mini_infer_validation,
    run_ollama_validation,
)


def test_cross_flagship_benchmark_execution():
    root = Path(__file__).resolve().parent.parent.parent
    ollama_trace = root / "servebench" / "results" / "quick-ollama-np4.json"
    mini_infer_trace = root / "mini-infer" / "benchmarks" / "mini_infer_benchmark.json"

    if ollama_trace.exists():
        res_ollama = run_ollama_validation(ollama_trace)
        assert res_ollama["requests"] > 0
        assert res_ollama["error_pct"]["tpot"] < 15.0  # Under 15% error

    if mini_infer_trace.exists():
        res_mini = run_mini_infer_validation(mini_infer_trace)
        assert res_mini["requests"] > 0
        assert res_mini["error_pct"]["tpot"] < 15.0  # Under 15% error
