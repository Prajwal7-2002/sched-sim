---
title: sched-sim - LLM Serving Scheduler & Gantt Studio
emoji: ⏱️
colorFrom: purple
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: Discrete-event LLM scheduler simulator & Gantt player.
tags:
  - llm-serving
  - continuous-batching
  - scheduler
  - sarathi
  - chunked-prefill
  - head-of-line-blocking
  - discrete-event-simulation
  - gantt
datasets:
  - prajwalpujari16/llm-serving-traces
---

# sched-sim ⏱️ LLM Serving Scheduler & Gantt Studio

Interactive discrete-event simulation of continuous batching, chunked prefill (Sarathi C=512), and Head-of-Line prefill blocking.

* **Interactive Features:** Animated Gantt player demonstrating Head-of-Line prefill blocking vs. Sarathi Chunked Prefill ($C=512$), SJF, and Disaggregated serving.
* **GitHub Repository:** [Prajwal7-2002/sched-sim](https://github.com/Prajwal7-2002/sched-sim)
* **Dataset:** [prajwalpujari16/llm-serving-traces](https://huggingface.co/datasets/prajwalpujari16/llm-serving-traces)
* **PyPI:** [`pip install sched-sim`](https://pypi.org/project/sched-sim/)
* **Author:** Prajwal Pujari ([@Prajwal7-2002](https://github.com/Prajwal7-2002))

---

### Citation & Attribution

If you use sched-sim, its discrete-event scheduling engine, or serving arrival traces in your research, systems design, or publications, please cite:

```bibtex
@misc{pujari2024schedsim,
  author = {Pujari, Prajwal},
  title = {sched-sim: Discrete-Event Simulator for LLM Serving Schedulers and Batching Policies},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/spaces/prajwalpujari16/sched-sim}},
  url = {https://github.com/Prajwal7-2002/sched-sim}
}
```

