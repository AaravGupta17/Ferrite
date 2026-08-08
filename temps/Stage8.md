# Stage 8 — Benchmarking

## Bottom Line Up Front

Benchmarking turns claims into numbers. The README's speedup story must be reproducible: same hardware, same build flags, same commands. **Done when:** a benchmark suite covers naive vs AVX2 vs INT8 across model-relevant sizes, and the numbers are verified against a reference implementation.

Ferrite already has `bench_matmul`, `bench_matmul_avx2`, and `bench_avx2` covering bare 256×256 matmul. This stage widens scope to full models and external comparisons.

## Deliverables

- Benchmark framework
- Timing utilities
- Compare with ONNX Runtime
- Compare with tinygrad
- Compare with TFLite
- Memory benchmarks
- Latency benchmarks
- Throughput benchmarks

## How to Proceed

1. **A benchmark framework is a repeatable harness, not a binary.** Shared tools: warm-up runs, best-of-N measurement, GFLOPS and ms/run reporting, and a printed environment header (CPU, build flags, date). Every benchmark is one documented command.
2. **Timing utilities already exist.** `fe_profiler_now_ns()` uses `CLOCK_MONOTONIC`. Add a loop-timer helper (warm-up, N iterations, report min/mean) on top of it, and reuse it everywhere.
3. **Naive is the floor.** Every optimized number is printed next to the naive number with the speedup ratio, and the result is verified against the naive kernel before it is reported. This is already `bench_avx2`'s contract.
4. **Benchmark what a model actually does.** A single 256×256 matmul is a preview. Add the demo model's shapes and op mix: the latency number that matters is end-to-end inference, not one kernel.
5. **External comparisons are tooling, not runtime dependencies.** ONNX Runtime, tinygrad, and TFLite are Python/reference tools on this machine. Compare the same input and tolerance, report the versions, and keep the reference out of the shipped runtime.
6. **Three axes, three reports.** Latency (ms per inference, best-of-N), throughput (inferences per second at fixed batch), and memory (peak arena usage from `fe_arena_peak`, model size, activation footprint).
7. **Memory benchmarks use the planner.** Once Stage 9/13 land, report planned activation bytes and the savings vs. naive allocation. `fe_plan_print` already reports this.
8. **Write results down.** Every benchmark run records its numbers in a table in the README or `temps/` with the environment header. A number that cannot be reproduced is not a number.

**Verify.** Re-run the suite on a clean checkout after a machine change; the README table stays current. Any reported speedup must reproduce within noise.

**Do not** optimize to win a benchmark. Measure first, then decide — and keep the honest limitations section updated.
