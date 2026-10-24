"""Workload distribution presets representing realistic serving domains."""

import random

from pydantic import BaseModel


class DistributionPreset(BaseModel):
    name: str
    prompt_min: int
    prompt_max: int
    prompt_mean: int
    output_min: int
    output_max: int
    output_mean: int
    slo_ttft_s: float
    slo_tpot_s: float

    def sample_lengths(self) -> tuple[int, int]:
        """Sample (prompt_len, output_len) using triangular / bounded distributions."""
        p = int(random.triangular(self.prompt_min, self.prompt_max, self.prompt_mean))
        o = int(random.triangular(self.output_min, self.output_max, self.output_mean))
        return max(1, p), max(1, o)


DISTRIBUTIONS: dict[str, DistributionPreset] = {
    "chatbot": DistributionPreset(
        name="Interactive Chatbot",
        prompt_min=30,
        prompt_max=400,
        prompt_mean=120,
        output_min=20,
        output_max=300,
        output_mean=90,
        slo_ttft_s=0.25,
        slo_tpot_s=0.04,
    ),
    "rag": DistributionPreset(
        name="Long-Context RAG",
        prompt_min=1000,
        prompt_max=6000,
        prompt_mean=2500,
        output_min=30,
        output_max=250,
        output_mean=80,
        slo_ttft_s=0.80,
        slo_tpot_s=0.05,
    ),
    "codegen": DistributionPreset(
        name="Code Generation / Autonomous Agent",
        prompt_min=400,
        prompt_max=2500,
        prompt_mean=1000,
        output_min=100,
        output_max=800,
        output_mean=350,
        slo_ttft_s=0.40,
        slo_tpot_s=0.035,
    ),
}
