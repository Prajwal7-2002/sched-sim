## Description
<!-- Briefly describe what this PR does, why it is needed, and link any related issues. -->

Fixes #(issue)

## Type of Change
<!-- Please check all options that apply -->
- [ ] 🐛 Bug fix (non-breaking change fixing an issue)
- [ ] ✨ New feature (new scheduler, latency model, or trace exporter)
- [ ] ⚡ Performance optimization (simulation execution speedup)
- [ ] 📚 Documentation update (guides, docstrings, README)
- [ ] 🧪 Tests (coverage improvement or empirical validation tests)
- [ ] 💥 Breaking change (fix or feature that would break existing CLI or API contracts)

## Simulation & Calibration Validation
<!-- If this PR introduces or modifies a scheduler or roofline model, describe the validation. -->
* **Hardware Profile**: (e.g. A100-80GB / H100-SXM / RTX 3050)
* **Simulation Speed**: `____ events/sec` | `____ sec total run time`
* **Validation Target**: (e.g. matched against empirical servebench traces / theoretical roofline bounds)

## Architectural Invariants Checklist
<!-- You MUST verify and check every item below before submitting -->
- [ ] **100% CPU-First**: Zero physical GPU or CUDA compiler requirements. Simulations execute deterministically on standard CPU.
- [ ] **First-Principles Roofline**: Latencies are mathematically calculated from hardware bandwidth and TFLOPS bounds, without magic delays.
- [ ] **Paged Virtual Memory**: Correctly models 16-token physical blocks, block tables, and reference-counted LRU eviction.
- [ ] **Unit Tests**: Ran `uv run pytest` (100% pass rate) and maintained coverage (`uv run pytest --cov=sched_sim`).
- [ ] **Formatting & Linting**: Ran `uv run ruff check .` and `uv run ruff format --check .` with zero errors.
- [ ] **Documentation**: Updated `README.md` or `docs/` if modifying CLI flags or user-facing behavior, and verified with `uv run mkdocs build --strict`.
- [ ] **Clean Commits**: Follows conventional commit format (`feat:`, `fix:`, `perf:`, `docs:`, `test:`).
