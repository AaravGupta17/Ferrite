# Stage 22 — Research

## Bottom Line Up Front

Research turns Ferrite's claims into analysis: complexity, memory, cache, and SIMD behavior, written down with numbers. It is how the project explains *why* it is fast, not just *that* it is. **Done when:** each performance claim in the docs has an analysis behind it, and the results are reproducible.

This stage is analysis on top of measured code. It must not start until the benchmark suite (Stage 8) exists — analysis of unmeasured performance is fiction.

## Deliverables

- Complexity analysis
- Memory analysis
- Cache analysis
- SIMD analysis
- Benchmark report
- Technical paper
- Whitepaper

## How to Proceed

1. **Complexity analysis is per-op and per-pass, written down.** Big-O for each kernel (matmul `O(MNK)`, im2col `O(batch·C·K·L_out)`), for topo sort `O(N+E)`, for the planner `O(n²)`, for each optimization pass. One table in `explain.md`. This is cheap and permanently useful.
2. **Memory analysis uses the planner's own numbers.** Peak activation memory, weight footprint, per-tensor lifetimes, reuse savings — `fe_plan_print` already computes these. The analysis is interpreting them: which tensors dominate, where reuse helped, where it could not.
3. **Cache analysis is measured, then explained.** For the tiled AVX2 matmul, compute working-set sizes from MC/KC tiles vs. L1/L2 and explain the 13.4× in cache terms. Use `perf` counters where available. The tiling numbers (MC=64, KC=256) must be justified, not just stated.
4. **SIMD analysis covers the vectorization ceiling.** Theoretical peak (8-wide FMA, 2 FMA/cycle/core) vs. measured GFLOPS, and the efficiency ratio. Report where the gap is: memory-bound vs. compute-bound. This is the honest version of a speedup claim.
5. **The benchmark report is the public artifact.** A single document: environment, methodology (warm-up, best-of-N), tables per size and op, conclusions. This is the Stage 8 output, written as analysis rather than a log.
6. **A technical paper is a milestone artifact, not a routine.** A ~10-page writeup of the architecture, the memory system, and the SIMD path with measured results. Write it once the demo and benchmarks are stable; it is the project's strongest form of documentation.
7. **Whitepaper is the shorter public version.** The pitch: what Ferrite is, how it achieves zero-dependency performance, the numbers. It feeds the website (Stage 21).
8. **Everything reproducible.** Every number in every analysis links to the benchmark or test that produced it. An analysis without a reproducible number is an opinion, not research.

**Verify.** Re-run the analysis suite after a code change; the claims update or the analysis is wrong. The `explain.md` complexity table is checked against the actual implementations.

**Do not** publish a number without a run behind it. This stage's entire value is that Ferrite's claims are real.
