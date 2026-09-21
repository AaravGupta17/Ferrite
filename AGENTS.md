# Ferrite — Agent Guide

## Bottom Line Up Front

Ferrite is a zero-dependency neural-network inference runtime in C11, built from first principles. It loads ONNX models and runs them with hand-written kernels. No BLAS. No protobuf library. No SIMD wrapper. The git history shows one subsystem per commit, each with its own test.

This file tells an agent or contributor how to work in this repo: what the project is, how to build and test it, the conventions to follow, and the gaps to respect. The full technical guide is `docs/guide.md`; the plan is `docs/roadmap.md`.

---

## Quickstart

```sh
cmake -S . -B build -G Ninja   # configure (GCC/Clang required)
cmake --build build            # builds all test binaries
cmake --build build --target bench_avx2   # performance target (-O3 -mavx2 -mfma)
```

Run the suite or one binary to verify a change:

```sh
ctest --test-dir build          # all tests, from anywhere
./build/test_tensor             # single binary
./build/test_engine
```

**Platform caveat.** Binaries land in `build/` as `*.exe` on Windows. On Windows/MinGW without sanitizers, each executable copies `libwinpthread-1.dll` next to itself at build time — run tests through CTest or from `build/`, not by copying exes elsewhere.

---

## Repository Layout

| Directory | Role | Key files |
|---|---|---|
| `core/` | Universal data structures: strided tensors, arena allocator, serialization, logging, FP16 storage, shared types | `types.h`, `tensor.c/h`, `tensor_ser.c/h`, `allocator.c/h`, `log.c/h`, `fp16.c/h` |
| `graph/` | Computation-graph IR, tensor registry, Kahn topological sort | `graph.c/h` |
| `ops/` | Operator kernels: matmul/linear, elementwise, activations, math (exp/ln/pow), gemm/transpose, conv1d/conv2d (im2col), norms, pooling, sequence, reduce, rand, stability | `ops.h`, `matmul.c`, `activations.c`, `conv1d.c`, `conv2d.c`, `elementwise.c`, `gemm.c`, `math.c`, `norm.c`, `pool.c`, `sequence.c`, `reduce.c`, `rand.c`, `stability.c` |
| `simd/` | AVX2 tiled matmul with runtime CPUID detection | `matmul_avx2.c/h`, `elementwise_avx2.c/h` |
| `planner/` | Tensor-lifetime analysis + greedy buffer reuse | `memory_planner.c/h` |
| `parallel/` | Thread pool, row-parallel GEMM, ready-set DAG scheduler | `threadpool.c/h`, `parallel_gemm.c/h`, `scheduler.c/h` |
| `compiler/` | Linear IR (lower → dead-elim → kernel table → schedule, `fe_ir_run`) | `ir.h`, `lower.c`, `codegen.c`, `schedule.c` |
| `optim/` | Graph optimization passes: shape inference (per-op rule table), folding, fusion, CSE, DCE | `shape_infer.c/h`, `optim.c/h` |
| `runtime/` | Execution engine: `FeRuntime` ties graph + arenas + kernels; static exec plan | `engine.c/h`, `exec_plan.c/h` |
| `importer/` | Hand-rolled ONNX protobuf wire parser and graph loader | `onnx.c/h` |
| `quantization/` | Symmetric INT8 (engine-wired) + INT16 dynamic-quant paths | `quant.c/h` |
| `tools/` | Profiler, benchmark harnesses, FEMD compiler, perf gate checker | `profiler.c/h`, `bench.c/h`, `bench_model.c`, `bench_avx2.c`, `ferrite_compile.c`, `check_perf.py` |
| `tests/` | One test binary per subsystem | `test_*.c`, `tiny_mlp.onnx` |
| `fuzz/` | ONNX-parser fuzz harness (libFuzzer + standalone replayer) | `fuzz_onnx.c/h`, `fuzz_main.c`, `corpus/` |
| `cmake/` | Cross toolchain presets (Pi Zero W, ESP32) | `raspberry-pi-zero-w.toolchain.cmake`, `esp32.toolchain.cmake` |
| `idf/` | ESP-IDF integration: component + demo project | `ferrite/` (compiled via `ferrite_device` sources), `demo/`, `README.md` |
| `docs/` | Port docs | `pi-zero-w-port.md`, `esp32-port.md` |

Layer dependencies point downward only: `importer/` → `optim/` + `graph/` → `compiler/` (on graph/ops/core) → `planner/` → `parallel/` → `runtime/` → `ops/` + `simd/` + `core/`. `tools/` and `tests/` sit on top.

---

## Data Flow (One Inference)

1. **Load** (`importer/`): `fe_onnx_load` parses the `.onnx` protobuf, builds the `FeGraph` (nodes + tensor registry), runs topo sort, copies weights into the weight arena.
2. **Plan** (`planner/`): lifetime analysis + greedy reuse produce per-tensor offsets into one activation buffer.
3. **Init** (`runtime/`): `fe_runtime_init` binds caller buffers to both arenas; `fe_runtime_alloc_weights` allocates weights once.
4. **Run** (`runtime/`): first run (or an input-shape change) re-seeds the input, forgets produced shapes, re-infers, re-plans memory, and resolves every node to a kernel fn via the `k_fns[]` table (`fe_exec_build`); then reset activation arena → `fe_plan_apply` re-applies the cached byte offsets (data region reserved first, `FeTensor` metadata appended after it) → bind the graph input tensor to the caller's buffer → walk the flat `FeExecStep` array (no dispatch switch) → copy the output.

Note: ONNX-loaded graphs have **no** `FE_OP_INPUT`/`FE_OP_OUTPUT` nodes (the hand-built graphs do). `find_graph_input()` (`runtime/engine.c`) binds the input as the first non-weight, consumed-but-unproduced tensor; the output is read from the last node.
5. **Optional**: quantize weights (`quantization/`), profile per-op (`tools/`).

---

## Code Conventions

- **Language.** C11. Build flags: `-std=c11 -D_POSIX_C_SOURCE=199309L -Wall -Wextra -fsanitize=address,undefined -g`.
- **Build.** CMake is canonical (`cmake -S . -B build && cmake --build build`; `ctest --test-dir build` runs the suite). Requires GCC/Clang — MSVC lacks the POSIX clock the profiler uses.
- **Naming.** Public API prefix `fe_`. Header guards `FERRITE_X_H`. Files `module.c`/`module.h`.
- **Returns.** Every public function returns `FeStatus` (`FE_OK`, `FE_ERR_NULL/SHAPE/DTYPE/NOMEM/BOUNDS/IO`) unless it is a constructor.
- **Configuration.** Constants live in headers (`FERRITE_MAX_DIMS`, `FERRITE_LOG_LEVEL`). No config machinery exists or is wanted until a real runtime knob appears.
- **Logging.** `core/log.h` for load/init-time diagnostics only; kernels never log (hot-path rule).
- **Kernel contract** (`ops/ops.h`): inputs read-only, output caller-allocated with correct shape, no allocation inside kernels, validate pointers/dtypes/shapes before touching data.
- **Tensors** reference graph entries by **index**, never by pointer. `FeTensor.data` may be `NULL` until memory is assigned.
- **Docs with code.** A change is not done until its doc is done. Update header comments in the same commit.
- **Commits.** One subsystem per commit, scoped: `feat(scope): short summary`. Say why, not what.

---

## Invariants to Not Break

- **No allocation in hot paths.** Kernels never `malloc`. `fe_conv1d` is the sole exception; it frees its scratch after use.
- **Views over copies.** `fe_tensor_transpose` and `fe_tensor_reshape` never copy data.
- **Arenas over per-tensor malloc.** Weights live forever in the weight arena; activations reset every inference.
- **Correctness first.** The naive `fe_matmul` is the reference that validates `fe_matmul_avx2`.
- **Ownership.** `owns_data == true` tensors free their buffer; views never free the parent's.
- **Strides** are row-major in elements: `strides[last] = 1`, `strides[i] = strides[i+1] * shape[i+1]`.

---

## Current State and Known Gaps

- **The run loop is a static execution plan** (`runtime/exec_plan.c`, Stage 13). `fe_exec_build` resolves each `topo_order` node to a kernel wrapper via the `k_fns[]` table indexed by `FeOpType` (every op exports one; no entry → loud `FE_ERR_SHAPE` at build, nothing silently skipped). `fe_runtime_run` re-applies the cached memory plan and walks the flat `FeExecStep` array — the only per-run analysis is an input-shape comparison triggering a rare rebuild. `FeRuntime` embeds the `FeExecPlan`.
- **AVX2 is wired into dispatch.** `fe_matmul` (`ops/matmul.c`) already routes to `fe_matmul_avx2` behind `fe_cpu_has_avx2()`, with scalar fallback — this is what the engine calls for every `MATMUL`/`LINEAR` node. `fe_relu`/`fe_add`/`fe_mul` route to `simd/elementwise_avx2.c` the same way; `fe_gemm` routes its base case (no transpose, `alpha=1`, `beta=0`). `tests/test_simd.c` is the regression check (naive vs. AVX2, vectorized and remainder paths, elementwise equivalence); `bench_avx2` times real scalar vs. vector paths (scalar side is pinned down — volatile stores so the -O3 compiler can't auto-vectorize or fold the reference kernels, which would otherwise silently re-measure AVX2) and prints a naive-vs-AVX2 checksum agreement bar purely to keep the timing honest. `bench_model` is timing only.
- **Alignment is guaranteed and tested.** Planner data offsets and arena allocations are aligned to `FERRITE_PLANNER_ALIGN` — 64 bytes on SIMD-enabled targets (`tests/test_planner.c` `test_alignment`), so SIMD kernels never hit misaligned loads/stores; scalar-only device ports lower it (Section 4.3).
- **Planner is wired into the runtime.** `fe_runtime_run` executes the static plan: on the first run (or an input shape change) it re-infers shapes and calls `fe_plan_memory`, caching the `FePlan`; per run it resets the arena and calls `fe_plan_apply` to re-assign the cached byte offsets (`runtime/exec_plan.c`). `fe_plan_apply` reserves the planned data region in the arena before arena-allocating the `FeTensor` metadata structs, so metadata never collides with tensor data. `fe_runtime_init` validates the graph (`fe_graph_validate`) right after `fe_graph_topo_sort`. `tests/test_engine.c` asserts the planned buffer is smaller than the naive no-reuse allocation, running off one planned activation buffer, plus an ONNX load→run test (`test_onnx_load_and_run`) and a static-plan test (`test_exec_plan_static` — same-shape reruns do zero re-analysis).
- **Graph optimization is a wired-in pass pipeline.** `fe_optimize` (`optim/optim.c`) runs shape-infer → simplify (identity transpose/flatten) → constant-fold → Conv+BN fusion → CSE → dead-elim, then re-sorts and re-validates. `fe_onnx_load` runs it after `fe_graph_validate`, so loaded models arrive optimized (e.g. the demo model's Conv+BN pairs fuse away at load — no `BatchNorm` rows left in the profile). Pass math and both refactors are in `tests/test_opt.c`, which also proves output equivalence pre/post optimize through the engine. Conventions: passes rewrite dead producers into `INPUT` feeds (op + zero inputs, outputs kept) and let dead-elim compact; `fe_runtime_alloc_weights` skips already-backed (`e->tensor != NULL`) weights so folded/fused constants survive runtime init.
- **The importer fails loudly on unsupported ONNX ops** (`FE_ERR_SHAPE` + stderr message) — nothing is silently dropped. It maps 20+ op strings, reads per-op attributes (`alpha`, `epsilon`, `num_groups`, `kernel_shape`, `strides`, `pads`, `num_heads`, `axis`), applies ONNX defaults, and requires critical inputs/attrs before accepting a node. `Gemm` maps to `Linear` and warns (never guesses) when non-default `transA`/`transB`/`alpha`/`beta` would change the result. `Softmax`'s `axis` is recorded during parsing and judged by `check_softmax_axes()` *after* shape inference, when output ranks are final — checking it at parse time flagged valid models (axis=1 on a rank-2 output is the last axis) as unsupported.
- **Quantization is per-tensor and per-channel AND wired into the engine** (`quantization/quant.c` — `fe_quantize`/`fe_matmul_int8` per-tensor, `fe_quantize_per_channel`/`fe_matmul_int8_per_channel` per-channel; `fe_quantize_model` repacks 2-D MATMUL/LINEAR weights to per-channel INT8 in place, scales in the weight arena; `ex_matmul`/`ex_linear` branch on the weight dtype at run time to `fe_linear_int8`/`fe_matmul_int8_dyn` with dynamic activations). A calibration pipeline exists (`fe_runtime_calibrate`, `runtime/engine.c`) — it runs a sample set through `fe_runtime_run` and records per-tensor activation `max(|x|)` ranges. `fe_quantize_model_static` turns those ranges into static activation scales recorded on the registry, and the engine routes MATMUL/LINEAR nodes whose activation input has a scale to `fe_matmul_int8_static`/`fe_linear_int8_static` (INT16 analogs alongside); unsampled tensors stay dynamic. Percentile/stat smoothing of the ranges is not built yet.
- **INT16, FP16 and BF16 are engine-wired.** `fe_quantize_int16`/`fe_matmul_int16_dyn`/`fe_linear_int16` mirror the INT8 engine kernels (dynamic activations, per-channel weights); `core/fp16.c/h` adds `DTYPE_FLOAT16` and `DTYPE_BFLOAT16` storage with upconvert-on-read kernel dispatch in `runtime/kernels.c` (half-precision weights upconverted to FP32 at run time).
- **Parallel subsystem exists and is tested, but the engine hot path stays single-threaded.** `parallel/` ships a thread pool, a chunked row-parallel GEMM (`fe_matmul_parallel` — deterministic chunking, no shared mutable accumulator), and a ready-set DAG scheduler. `tests/test_parallel.c` covers pool partition, GEMM-vs-oracle equivalence, and scheduler causality.
- **Compiler IR (Stage 14) exists and is tested.** `compiler/` lowers `FeGraph` → linear `FeIrProgram` (`fe_ir_lower`), dead-eliminates, resolves a static kernel table (`fe_ir_codegen`), and schedules producers-first (`fe_ir_schedule`); `fe_ir_run` executes the stream. `tests/test_compiler.c` proves IR output matches `fe_runtime_run` (≤1e-5) and the diamond schedule is causal.
- **CI, fuzz, and perf gates are authored artifacts.** `.github/workflows/ci.yml`
  has eleven jobs: the eight core legs (test — matrixed over gcc and clang,
  sanitize-ASan/UBSan-mandatory, fuzz, perf, coverage, Pi QEMU cross, ESP32
  IDF compile gate, golden-vs-ONNX-Runtime) plus a Windows/MinGW leg and a
  Release leg built with `-Werror`;
  `fuzz/` ships a libFuzzer entry (bridged via `fuzz_onnx_libfuzzer.c`) plus a
  standalone replayer; `tools/check_perf.py` diffs `bench_avx2 --json`
  against a per-CPU baseline under `bench/` (**model-independent** — the
  acoustic `bench_model` is deliberately not wired into automation). All jobs
  are green on `main`; the perf baseline is seeded from the Linux Release
  runner (matmul 4.78x / elementwise 3.92x / gemm 15.65x, 30% threshold, the
  elementwise 200-pass window averaged against shared-runner jitter); the
  ESP32 leg is a compile gate only (flashing is procedural in `docs/`); the
  **golden leg is locally verifiable** on the Windows dev host via
  `tools/golden_compare.py --run-model build/run_model` (onnxruntime + onnx
  pip packages).
- **Golden correctness vs ONNX Runtime exists (Phase D).** `tools/golden_gen.py`
  builds a four-model float32 zoo (MLP, mini-CNN, Conv+BN-fusion, LayerNorm
  MLP) using only importer-mapped ops with ONNX-default Gemm attributes;
  `tools/golden_compare.py` runs the same seeded input through ONNX Runtime
  and Ferrite (`tools/run_model.c`, loading `.onnx` through the importer) and
  asserts tolerance match — observed `max_abs ≈ 1e-8` with `mlp` bit-exact.
- **Ports: Pi Zero W and ESP32.** CMake toolchains + ESP-IDF component +
  docs (`docs/pi-zero-w-port.md`, `docs/esp32-port.md`, `idf/README.md`).
  Both compile out the host subsystems and x86 SIMD; ESP32 additionally drops
  `DTYPE_FLOAT64` (load-time `FE_ERR_DTYPE`), sizes ceilings via a
  `config/config.h` include override, and lowers `FERRITE_PLANNER_ALIGN` to
  scalar natural alignment.
- **AVX2 vectorized path needs `N % 8 == 0`** on the SIMD units; the kernels
  split into a vectorized prefix + scalar tail so remainder shapes still win.
- **Fixed-but-configurable capacities:** 512 nodes / 1024 tensors / 8 dims by
  default, threadable per target via CMake cache vars (`FE_MAX_NODES`,
  `FE_MAX_TENSORS`, `FERRITE_MAX_DIMS`, `FE_MAX_ALLOCS`,
  `FERRITE_PLANNER_ALIGN`). AVX2-on-non-x86 is a CMake `FATAL_ERROR`, not a
  silent downgrade.

---

## Working Here

- **Start** with `tests/` and header comments — each header documents its contract in one paragraph.
- **Verify** a change by building and running the matching `test_*` binary under ASan/UBSan. `tests/test_engine.c` is the best end-to-end example.
- **Keep docs in sync:** code changes update `docs/guide.md`; scope/status changes update `docs/roadmap.md`.
- **Do not** add new root-level Markdown files without a reason. `AGENTS.md` and `README.md` are the two root docs.
- **Do not** hide unsupported behavior behind silence. Fail loudly (`FE_ERR_SHAPE`), as the engine does today.

---

## Reading Order for New Context

1. `docs/guide.md` — how every subsystem works and connects.
2. `docs/roadmap.md` — where the project is going and immediate next steps.
3. `docs/documentation-standards.md` — the documentation standards (BLUF, honesty over polish).
