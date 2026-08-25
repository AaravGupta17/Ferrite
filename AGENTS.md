# Ferrite — Agent Guide

## Bottom Line Up Front

Ferrite is a zero-dependency neural-network inference runtime in C11, built from first principles. It loads ONNX models and runs them with hand-written kernels. No BLAS. No protobuf library. No SIMD wrapper. The git history shows one subsystem per commit, each with its own test.

This file tells an agent or contributor how to work in this repo: what the project is, how to build and test it, the conventions to follow, and the gaps to respect. The full technical guide is `temps/explain.md`; the plan is `temps/roadmap.md`.

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
| `core/` | Universal data structures: strided tensors, arena allocator, serialization, logging, shared types | `types.h`, `tensor.c/h`, `tensor_ser.c/h`, `allocator.c/h`, `log.c/h` |
| `graph/` | Computation-graph IR, tensor registry, Kahn topological sort | `graph.c/h` |
| `ops/` | Operator kernels: matmul, linear, relu, softmax, bias_add, conv1d (im2col) | `ops.h`, `matmul.c`, `activations.c`, `conv1d.c` |
| `simd/` | AVX2 tiled matmul with runtime CPUID detection | `matmul_avx2.c/h` |
| `planner/` | Tensor-lifetime analysis + greedy buffer reuse | `memory_planner.c/h` |
| `runtime/` | Execution engine: `FeRuntime` ties graph + arenas + kernels | `engine.c/h` |
| `importer/` | Hand-rolled ONNX protobuf wire parser and graph loader | `onnx.c/h` |
| `quantization/` | Symmetric per-tensor INT8 quantization | `quant.c/h` |
| `tools/` | Profiler, benchmark harnesses | `profiler.c/h`, `bench_matmul.c`, `bench_avx2.c` |
| `tests/` | One test binary per subsystem | `test_*.c`, `tiny_mlp.onnx` |
| `temps/` | Working docs: guide, roadmap, doc standards, staged plan | `explain.md`, `roadmap.md`, `documentation.md`, `Stage*.md` |

Layer dependencies point downward only: `importer/` → `graph/` → `planner/` → `runtime/` → `ops/` + `simd/` + `core/`. `tools/` and `tests/` sit on top.

---

## Data Flow (One Inference)

1. **Load** (`importer/`): `fe_onnx_load` parses the `.onnx` protobuf, builds the `FeGraph` (nodes + tensor registry), runs topo sort, copies weights into the weight arena.
2. **Plan** (`planner/`): lifetime analysis + greedy reuse produce per-tensor offsets into one activation buffer.
3. **Init** (`runtime/`): `fe_runtime_init` binds caller buffers to both arenas; `fe_runtime_alloc_weights` allocates weights once.
4. **Run** (`runtime/`): reset activation arena → bind input → bump-allocate activations → walk `topo_order` dispatching each node to a kernel → copy the output.
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

- **Engine dispatches only:** `INPUT`, `OUTPUT`, `MATMUL`, `LINEAR`, `RELU`, `SOFTMAX`, `ADD`, `FLATTEN` (`runtime/engine.c:18-50`). `CONV1D` and `BATCHNORM` hit `FE_ERR_SHAPE` ("unimplemented op").
- **Planner is standalone.** `fe_plan_apply` exists (`planner/memory_planner.h:67`) but nothing calls it; the runtime uses the arena bump allocator directly.
- **AVX2 not in dispatch.** The engine always uses naive `fe_matmul`; only `bench_avx2` exercises `fe_matmul_avx2`.
- **Importer skips shape inference.** ONNX `value_info` is ignored; shapes must be known ahead of time.
- **Unsupported ONNX ops are silently skipped.** Only models built from supported ops load correctly.
- **Quantization is per-tensor, not per-channel.**
- **AVX2 vectorized path needs `N % 8 == 0`**; remainders fall back to scalar.
- **Fixed capacities:** 512 nodes, 1024 tensors, 8 dims.
- **Single-threaded** execution.

---

## Working Here

- **Start** with `tests/` and header comments — each header documents its contract in one paragraph.
- **Verify** a change by building and running the matching `test_*` binary under ASan/UBSan. `tests/test_engine.c` is the best end-to-end example.
- **Keep docs in sync:** code changes update `temps/explain.md`; scope/status changes update `temps/roadmap.md`; the staged plan lives in `temps/Stage*.md`.
- **Do not** add new root-level Markdown files without a reason. `AGENTS.md` and `README.md` are the two root docs.
- **Do not** hide unsupported behavior behind silence. Fail loudly (`FE_ERR_SHAPE`), as the engine does today.

---

## Reading Order for New Context

1. `temps/explain.md` — how every subsystem works and connects.
2. `temps/roadmap.md` — where the project is going and immediate next steps.
3. `temps/documentation.md` — the documentation standards (BLUF, honesty over polish).
4. `temps/Stage*.md` — the staged long-term plan (Stage 0 = current).
