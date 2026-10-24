"""Trace loader for replaying real-world traces (e.g. LMSYS, ShareGPT, custom)."""

import json
from pathlib import Path

from sched_sim.core.request import Request


def load_trace_file(file_path: str | Path) -> list[Request]:
    """Load an external trace file into Request objects.

    Expected JSON formats:
    1. List of dicts: [{"id": "...", "timestamp": 0.1, "prompt_tokens": 120, "output_tokens": 40}, ...]
    2. LMSYS Chatbot Arena formatted conversation traces.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    requests: list[Request] = []
    if isinstance(data, list):
        for idx, item in enumerate(data):
            req_id = item.get("id", item.get("request_id", f"trace_{idx:04d}"))
            arrival = float(
                item.get(
                    "timestamp",
                    item.get("arrived_at", item.get("arrival_time", idx * 0.1)),
                )
            )
            prompt_len = int(item.get("prompt_tokens", item.get("input_tokens", 128)))
            output_len = int(item.get("output_tokens", 64))
            slo_ttft = item.get("slo_ttft", 0.5)
            slo_tpot = item.get("slo_tpot", 0.05)

            requests.append(
                Request(
                    request_id=req_id,
                    arrived_at=round(arrival, 4),
                    prompt_tokens=prompt_len,
                    output_tokens=output_len,
                    slo_ttft=slo_ttft,
                    slo_tpot=slo_tpot,
                )
            )

    return sorted(requests, key=lambda r: r.arrived_at)
