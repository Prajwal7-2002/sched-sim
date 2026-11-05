"""Out-of-sample validation against recorded serving runs (benchmarks/data/).

Engine constants are calibrated only on the smallest model's 1- and 2-user runs; every
other row is held out. The bounds below are the published accuracy (README), so a change
that makes predictions worse fails CI. Rows the model does not predict well (prefill of
larger models, decode under load for larger models) are documented, not asserted.
"""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.validate_flagships import main

OLLAMA = "Ollama 0.35.1 (llama.cpp, q4_k_m)"


@pytest.fixture(scope="module")
def rows():
    return {(r["engine"], r["model"], r["users"]): r for r in main()}


def test_calibration_rows_are_reproduced(rows):
    for users in (1, 2):
        r = rows[(OLLAMA, "0.5b", users)]
        assert not r["held_out"]
        assert abs(r["err_tpot_ms_pct"]) < 10


def test_held_out_batching_on_calibrated_model(rows):
    r = rows[(OLLAMA, "0.5b", 4)]
    assert r["held_out"]
    assert abs(r["err_tpot_ms_pct"]) < 10
    assert abs(r["err_ttft_ms_pct"]) < 15


@pytest.mark.parametrize("model", ["1.5b", "3b"])
def test_held_out_single_user_decode_on_larger_models(rows, model):
    r = rows[(OLLAMA, model, 1)]
    assert r["held_out"]
    assert abs(r["err_tpot_ms_pct"]) < 15
