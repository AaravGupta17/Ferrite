# Ferrite

A **zero-dependency neural-network inference runtime in C11**, built from
first principles. It parses ONNX protobufs with its own wire parser (no
protobuf library), plans activation memory with its own lifetime-aware
allocator, and runs hand-written kernels (no BLAS, no SIMD wrapper). AVX2 is
runtime-detected; scalar fallback always exists.

Everything ships with one subsystem per commit and one `test_*` binary per
subsystem, all green under ASan/UBSan, and an 8-leg CI pipeline that is green
on `main`.

## Quickstart

```sh
cmake -S . -B build -G Ninja        # GCC/Clang; -D_POSIX_C_SOURCE=199309L C11
cmake --build build                 # all test binaries
ctest --test-dir build              # the full suite (19 binaries)
```

```sh
cmake --build build --target bench_avx2     # perf target (-O3 -mavx2 -mfma)
./build/bench_avx2                          # scalar vs AVX2 timings
```

Build flags are `-Wall -Wextra -fsanitize=address,undefined` by default; the
sanitizer presence is probed and the mandatory `sanitize` CI leg refuses to
build if they get silently disabled.

> Windows/MinGW note: binaries land as `*.exe` in `build/` and copy
> `libwinpthread-1.dll` next to themselves at build time — run tests from
> `build/` or through CTest, not by moving exes elsewhere.

## What works end-to-end

- **`.onnx` → inference.** `fe_onnx_load` parses the file, validates the
  graph, runs the optimizer pass pipeline (shape-infer → simplify →
  constant-fold → Conv+BN fusion → CSE → dead-elim), then the engine executes
  a flat **static execution plan** — every node resolved to a kernel at
  build time, cached memory offsets re-applied unchanged on every
  same-shape run. Re-analysis only ever happens on a rare input-shape change.
- **Correctness vs ONNX Runtime.** `tools/golden_compare.py` runs a seeded
  float32 model zoo through both ONNX Runtime and Ferrite and asserts
  tolerance match — observed `max_abs ≈ 1e-8`, with the MLP bit-exact.
- **Quantization wired into the engine.** Per-channel INT8 repacks 2-D
  MatMul/Linear weights in place; dynamic-int8/int16 activation paths and
  FP16/BF16 storage with upconvert-on-read dispatch. A calibration pipeline
  collects per-tensor activation ranges for future static scales.
- **AVX2 attached where it counts.** `fe_matmul`/`fe_linear` route to the
  tiled AVX2 kernel via CPUID (scalar fallback on failure); relu/add/mul and
  GEMM's base case do the same via a SIMD backend table.
- **Optional subsystems, tested independently.** A thread pool + row-parallel
  GEMM + ready-set DAG scheduler (`parallel/`), a linear IR compiler with
  dead-elim and kernel resolution (`compiler/`), and an FP16/BF16 storage
  layer (`core/fp16.c`).

## Structural map

| Layer | Contents |
|---|---|
| `core/` | strided tensors, arena allocator, serialization, logging, FP16/BF16, platform seam |
| `graph/` | computation-graph IR, tensor registry, Kahn topo sort |
| `ops/` | kernels: matmul/linear, elementwise, activations, math, gemm, conv1d/conv2d (im2col), norms, pooling, sequence, reduce, rand |
| `simd/` | AVX2 tiled matmul + elementwise, runtime CPUID, scalar fallback |
| `planner/` | tensor-lifetime analysis + greedy buffer reuse |
| `parallel/` | thread pool, row-parallel GEMM, ready-set DAG scheduler |
| `compiler/` | linear IR (lower → dead-elim → kernel table → schedule) |
| `optim/` | shape inference (per-op rule table), folding, fusion, CSE, DCE |
| `runtime/` | `FeRuntime` engine + static execution plan |
| `importer/` | hand-rolled ONNX protobuf wire parser + graph loader |
| `quantization/` | symmetric INT8 (engine-wired) + INT16 dynamic quant |
| `tools/` | profiler, benchmark harnesses, FEMD compiler, perf gate checker |
| `tests/` | one test binary per subsystem |

Dependencies point downward only; `tests/` and `tools/` sit on top.

## Performance

Hand-written kernels, measured honestly (`test_simd.c` validates them;
`bench_avx2 --json` feeds the CI gate against a committed baseline). The
scalar side of the microbench is pinned down with volatile stores so `-O3`
can neither auto-vectorize nor fold it into another AVX2 run.

| Bench (Linux Release runner, committed baseline) | naive | AVX2 | ratio |
|---|---|---|---|
| MatMul 256³ | 6.92 ms | 1.45 ms | **4.78×** |
| Elementwise add, 1M floats | 0.75 ms | 0.19 ms | **3.92×** |
| GEMM base vs scalar transB | 22.09 ms | 1.41 ms | **15.65×** |

Model-level (Debug, `tests/acousticleaknet.onnx`): fc1 `[1×65536×128]` 28.5 →
4.77 ms (**6.0×**); the lifetime planner reuses activation memory down to
896 KB from a naive 1153.7 KB (**−22.3%**). INT8: MLP float vs INT8 max abs
error 1.3e-3; `tiny_mlp` weights 268 B → 100 B integer-exact.

Full numbers: `temps/bench_results.md`.

## Ports

- **Raspberry Pi Zero W** — ARMv6/hard-float cross preset, full suite runs
  under QEMU in CI (`docs/pi-zero-w-port.md`).
- **ESP32** — ESP-IDF component compiling the *device runtime* (scalar
  kernels, planner, engine, quant; host subsystems compiled out, FLOAT64
  rejected at load), plus an `idf/demo` hardware smoke project that runs two
  inferences to prove static-plan reuse. Compile-gated green in CI
  (`docs/esp32-port.md`, `idf/README.md`).

## CI

Eight legs, green on `main`: build+test (Debug), **mandatory ASan/UBSan**,
libFuzzer against the ONNX parser (persistent corpus), perf gate (Release
`bench_avx2` vs `temps/bench_baseline.json`, 30% threshold, baseline
re-seeded from the runner), coverage (`--fail-under-line 70`), Pi Zero W
cross+QEMU, ESP32 IDF compile gate, and golden-vs-ONNX-Runtime. Nothing is
silently skipped: a leg that can't do its check fails.

## Reproduce the golden comparisons

```sh
pip install onnxruntime onnx
python3 tools/golden_gen.py                     # build the four-model float32 zoo
python3 tools/golden_compare.py --run-model build/run_model
```

## Where to start reading

- `AGENTS.md` — this repo's operating guide
- `temps/explain.md` — how every subsystem works
- `temps/roadmap.md` — where it is and what is next
- `tests/test_engine.c` — best end-to-end example

## Honest limitations

- Fixed-but-configurable capacities (512 nodes / 1024 tensors / 8 dims by
  default; threadable per target via CMake).
- AVX2 only; the vectorized prefix requires `N % 8 == 0` with a scalar tail
  for the rest. NEON waits for a Pi Zero 2 W+ target.
- The engine hot path is deliberately single-threaded (parallel subsystem
  exists and is tested but is opt-in).
- ONNX surface is the 20+ mapped ops — anything else fails loudly with
  `FE_ERR_SHAPE`, never silently.
- INT16/FP16/BF16 are API-complete and engine-wired; static activation
  scales are collected but not yet fed back to the engine.