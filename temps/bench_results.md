# Ferrite Benchmark Results

Recorded benchmark numbers. Every number below is reproducible with the
command shown. Build modes matter: the **Debug** build compiles the ops
library at `-O0`, so `fe_matmul` is a true scalar reference; the **Release**
build lets GCC auto-vectorize the naive loops, which changes the
naive-vs-SIMD ratios. State the mode whenever you compare.

Environment: AMD Ryzen 5 5600, MinGW GCC 16.2.0 (scoop), single thread.

---

## Model-level (default Debug build)

`cmake --build build --target bench_model && ./build/bench_model`

| Metric | Value |
|---|---|
| Model | `tests/acousticleaknet.onnx` (42 MatMul / 63 Conv / 21 softmax…) |
| Load | 23.06 ms |
| Inference | 5.893 ms/run (mean of 20) |
| Throughput | 169.7 inf/s |
| Useful FLOPs | 34.83 MFLOP/run (5.9 GFLOPS dispatch) |
| Weights | 32.04 MB |
| Planned activations | 896.00 KB |
| Naive activations | 1153.69 KB |
| Memory savings | 22.3% |
| Peak used (incl. metadata) | 897.22 KB |

Per-op share: MatMul 75.2%, Conv 24.2%, Relu 0.5%, rest ≈ 0.

### MatMul naive vs AVX2 at model shapes (Debug — scalar reference path)

| M x K x N | naive ms | avx2 ms | speedup |
|---|---|---|---|
| 1 x 65536 x 128 | 28.529 | 4.769 | 6.0x |
| 1 x 128 x 3 | 0.001 | 0.000 | 3.5x |

The M=1 GEMM is memory-bound: 128 KB of A per output row dominates the AVX2
FMAs.

---

## Model-level (Release build)

`cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release &&
cmake --build build-release --target bench_model && ./build-release/bench_model`

| M x K x N | naive ms | avx2 ms | speedup |
|---|---|---|---|
| 1 x 65536 x 128 | 1.980 | 4.780 | 0.4x |

GCC auto-vectorizes the "naive" loop with 8-wide SIMD at `-O3`, so the
modular AVX2 tiled kernel no longer wins on the memory-bound M=1 shape. The
AVX2 advantage survives only where the scalar fallback is genuinely scalar
(GDebug mode) or where compute-bound dimensions apply.

---

## Micro-benchmarks (`bench_avx2`, Release)

`cmake --build build-release --target bench_avx2 && ./build-release/bench_avx2`

| Benchmark | naive | AVX2 | ratio |
|---|---|---|---|
| MatMul 256x256x256 | 1.25 ms (26.9 GFLOPS) | 1.32 ms (25.4 GFLOPS) | 0.9x |
| Elementwise add, 1M floats | 0.03 ms | 0.17 ms (6.3 GFLOP/s) | 0.2x |
| GEMM base (AVX2) vs transB (scalar) | 19.92 ms (scalar) | 1.30 ms (25.8 GFLOPS) | 15.3x |

- At N=256 both matmul paths are bandwidth-limited and within 10% of each
  other; GFLOPS numbers are memory-bound, not peak-FLOPS.
- The GEMM row shows the real AVX2-vs-scalar gap (15.3x) because the
  transB path (column access on B) cannot be auto-vectorized and stays a true
  scalar fallback.
- Elementwise naive at `-O3` is a fully auto-vectorized loop; the AVX2
  kernel's scalar-tail overhead dominates at these sizes.

Note: these micro-benchmarks are not correctness checks — `test_simd.c` is.

---

## Quantization (Stage 15)

Numbers asserted by tests (not a latency benchmark yet):

| Metric | Value |
|---|---|
| MLP float vs INT8 max abs error | 0.00126 (`test_quantized_engine_mlp`) |
| tiny_mlp weights | 268 B float → 100 B INT8, max err 0.00000 (`test_quantized_onnx_end_to_end`) |

Latency/throughput of INT8 inference is not yet measured; the engine branch
(`ex_matmul`/`ex_linear` on weight dtype) is covered for correctness only.

---

## Parser / runtime robustness (no bench)

`test_parser_fuzz`: every truncated prefix (596) plus 400 random byte flips
load cleanly or fail loudly — no crash, no hang.
`test_chain_stress`: 256-node MatMul chain (258 tensors) runs through the
planner + static plan with an exact identity oracle.
`test_linear_chain_remainder`: 12 Linear+ReLU pairs, K=N=19 (N%8!=0),
exercises the scalar remainder path end-to-end.