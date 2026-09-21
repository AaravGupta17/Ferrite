# CLAUDE.md — Ferrite Operating Guide
DO NOT PUSH WITH THE CLAUDE TAG ONLY PUSH WITH THE GITHUB ACCOUNT ON THIS PC WHICH IS ogshrug
## Bottom Line Up Front

Ferrite is a **zero-dependency neural-network inference runtime in C11**. It parses
ONNX protobufs with its own wire parser, plans activation memory with its own
lifetime-aware allocator, and runs hand-written kernels. No BLAS. No protobuf library.
No SIMD wrapper. AVX2 is runtime-detected via CPUID; a scalar path always exists.

Zero dependencies is the point of the project, not an accident. Do not add one.

This file is self-contained: build, conventions, invariants, and traps. `AGENTS.md`
carries the same operating guidance; `docs/guide.md` and `docs/deep-dive.md` go deeper
per subsystem. See **Doc map** at the bottom for who owns what.

---

## 1. Build and Verify

```sh
cmake -S . -B build -G Ninja                  # configure (GCC/Clang only)
cmake --build build                           # 18 test binaries + ferrite_compile + fuzz_runner
ctest --test-dir build --output-on-failure    # 19 entries
./build/test_engine                           # one binary (add .exe on Windows)
```

**GCC or Clang is mandatory.** CMake hard-fails on any other compiler: the profiler uses
the POSIX monotonic clock (`clock_gettime` via `-D_POSIX_C_SOURCE=199309L`), which MSVC
does not provide. Global flags are `-Wall -Wextra -g`; `CMAKE_BUILD_TYPE` defaults to
`Debug`.

**Opt-in targets are `EXCLUDE_FROM_ALL`** — name them explicitly:

| Target | Purpose |
|---|---|
| `bench_avx2` | scalar vs AVX2 microbench; `--json` feeds the CI perf gate (`-O3 -mavx2 -mfma`) |
| `bench_matmul` / `bench_matmul_avx2` | same source at `-O2` vs `-O3 -mavx2 -mfma` |
| `bench_model` | end-to-end model timing (deliberately **not** wired into CI) |
| `demo` | loads `tests/acousticleaknet.onnx`, prints per-op latency + probabilities |
| `run_model` | `run_model <model.onnx> <input.bin> <output.bin> <out_dims.cfg>`; the golden harness drives it |
| `fuzz_onnx_fuzzer` | libFuzzer entry, Clang-only, gated on a `check_c_source_compiles` probe |

`ferrite_compile` (ONNX → FEMD artifact) and `fuzz_runner` (standalone replayer) **are**
built by default.

---

## 2. Repository Layout

| Directory | Role | Key files |
|---|---|---|
| `core/` | strided tensors, arena allocator, serialization, logging, FP16/BF16, platform seam | `types.h`, `tensor.c/h`, `allocator.c/h`, `fp16.c/h`, `platform.c` |
| `graph/` | computation-graph IR, tensor registry, Kahn topo sort | `graph.c/h` |
| `ops/` | kernels: matmul/linear, elementwise, activations, math, gemm/transpose, conv1d/conv2d (im2col), norms, pooling, sequence, reduce, rand, stability | `ops.h`, `matmul.c`, `activations.c`, … |
| `simd/` | backend dispatch table + scalar module; AVX2 tiled matmul and elementwise behind CPUID | `backend.c`, `scalar.c`, `matmul_avx2.c/h`, `elementwise_avx2.c/h` |
| `planner/` | tensor-lifetime analysis + greedy buffer reuse | `memory_planner.c/h` |
| `parallel/` | thread pool, row-parallel GEMM, ready-set DAG scheduler | `threadpool.c/h`, `parallel_gemm.c/h`, `scheduler.c/h` |
| `compiler/` | linear IR: lower → dead-elim → kernel table → schedule, `fe_ir_run` | `ir.h`, `lower.c`, `codegen.c`, `schedule.c` |
| `optim/` | shape inference (per-op rule table), folding, fusion, CSE, DCE | `shape_infer.c/h`, `optim.c/h` |
| `runtime/` | `FeRuntime` engine + static execution plan + device kernel table | `engine.c/h`, `exec_plan.c/h`, `kernels.c`, `model_ser.c` |
| `importer/` | hand-rolled ONNX protobuf wire parser + graph loader | `onnx.c/h` |
| `quantization/` | symmetric INT8/INT16, per-tensor and per-channel, dynamic and static | `quant.c/h` |
| `tools/` | profiler, benchmarks, FEMD compiler, perf gate, golden harness | `profiler.c/h`, `bench.c/h`, `check_perf.py`, `golden_*.py` |
| `tests/` | one test binary per subsystem + ONNX/FEMD fixtures | `test_*.c`, `tiny_mlp.onnx` |
| `fuzz/` | ONNX-parser fuzz harness + corpus | `fuzz_onnx.c/h`, `fuzz_main.c`, `corpus/` |
| `cmake/`, `idf/` | Pi Zero W + ESP32 cross toolchains, ESP-IDF component and demo | toolchain files, `idf/ferrite/` |

**Thirteen static libraries; dependencies point downward only.** `ferrite_core` →
`ferrite_graph` → `ferrite_ops`/`ferrite_simd`/`ferrite_planner` → `ferrite_device` →
`ferrite_runtime`; `ferrite_importer`, `ferrite_optim`, `ferrite_quant`,
`ferrite_tools`, `ferrite_parallel`, and `ferrite_compiler` hang off that spine.
`tools/` and `tests/` sit on top.

**`ferrite_device` is the device-only library** (`runtime/kernels.c` +
`runtime/model_ser.c`, plus quant and profiler when enabled). It has kernels and the
FEMD artifact loader — no ONNX parser, no shape inference. That is what an ESP32 or Pi
build compiles.

---

## 3. One Inference, End to End

1. **Load** (`importer/`) — `fe_onnx_load` parses the protobuf, builds the `FeGraph`,
   topo-sorts, validates, runs `fe_optimize`, copies weights into the weight arena.
2. **Optimize** (`optim/`) — shape-infer → simplify → constant-fold → Conv+BN fusion →
   CSE → dead-elim, then re-sort and re-validate. Models arrive at the engine already
   optimized.
3. **Init** (`runtime/`) — `fe_runtime_init` binds caller buffers to both arenas;
   `fe_runtime_alloc_weights` allocates weights once, skipping already-backed entries so
   folded and fused constants survive.
4. **Run** (`runtime/exec_plan.c`) — the first run (or an input-shape change) re-infers
   shapes, re-plans memory, and resolves every node to a kernel through the `k_fns[]`
   table indexed by `FeOpType`. Then: reset the activation arena → `fe_plan_apply`
   re-applies cached byte offsets (data region reserved first, `FeTensor` metadata
   appended after) → bind the input → walk the flat `FeExecStep` array → copy the output.

**A same-shape rerun does zero re-analysis.** The only per-run check is an input-shape
comparison. `test_exec_plan_static` in `tests/test_engine.c` guards this.

ONNX-loaded graphs have **no** `FE_OP_INPUT`/`FE_OP_OUTPUT` nodes (hand-built ones do).
`find_graph_input()` in `runtime/engine.c` binds the input as the first non-weight,
consumed-but-unproduced tensor; the output is read from the last node.

---

## 4. Code Conventions

- **Language.** C11, `-std=c11`, `C_EXTENSIONS OFF`.
- **Naming.** Public API prefix `fe_`. Header guards `FERRITE_X_H`. Files
  `module.c` / `module.h`.
- **Returns.** Every public function returns `FeStatus` (`FE_OK`,
  `FE_ERR_NULL/SHAPE/DTYPE/NOMEM/BOUNDS/IO`) unless it is a constructor.
- **Configuration.** Constants live in headers. No config machinery until a real runtime
  knob appears.
- **Logging.** `core/log.h` is for load- and init-time diagnostics only. **Kernels never
  log** — that is the hot-path rule.
- **Kernel contract** (`ops/ops.h`): inputs read-only, output caller-allocated with the
  correct shape, no allocation inside kernels, validate pointers, dtypes, and shapes
  before touching data.
- **Tensors reference graph entries by index, never by pointer.** `FeTensor.data` may be
  `NULL` until memory is assigned.

---

## 5. Invariants to Not Break

- **No allocation in hot paths.** Kernels never `malloc`. `fe_conv1d` is the sole
  exception and frees its scratch after use.
- **Views over copies.** `fe_tensor_transpose` and `fe_tensor_reshape` never copy data.
- **Arenas over per-tensor malloc.** Weights live forever in the weight arena;
  activations reset every inference.
- **Correctness first.** The naive `fe_matmul` is the reference that validates
  `fe_matmul_avx2`. Never optimize the reference away.
- **Ownership.** Tensors with `owns_data == true` free their buffer; a view never frees
  the parent's.
- **Strides** are row-major in elements: `strides[last] = 1`,
  `strides[i] = strides[i+1] * shape[i+1]`.
- **Fail loudly.** An unsupported ONNX op returns `FE_ERR_SHAPE` with a stderr message.
  Nothing is ever silently dropped or skipped. A missing `k_fns[]` entry is a loud build
  failure, not a no-op.

---

## 6. Adding a Test

There is **no test framework and no assertion macro.** The pattern:

```c
#include <assert.h>
#include <stdio.h>
#include "../core/tensor.h"          /* headers by relative path */

static void test_thing(void) {
    /* ... */
    assert(cond);
    printf("PASS test_thing\n");
}

int main(void) {
    test_thing();
    printf("\nAll tests passed.\n");
    return 0;
}
```

Fixtures are opened relative to the **source root** (`tests/tiny_mlp.onnx`), which is why
CTest sets `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}`.

Registering `test_foo` takes **three edits in `CMakeLists.txt`**:

1. `add_executable(test_foo tests/test_foo.c)` — block at `CMakeLists.txt:305`.
2. A `target_link_libraries(test_foo ...)` line — block at `CMakeLists.txt:362`.
3. Append `test_foo` to the `foreach(t ...)` list at `CMakeLists.txt:389`. That loop is
   what supplies `ferrite_exe()` (the Windows DLL copy), `-UNDEBUG`, and `add_test`.

Miss step 3 and the binary builds but never runs. For an AVX2-only test, mirror the
`test_simd` pattern inside the `if(FERRITE_ENABLE_AVX2)` guard instead.

`-UNDEBUG` is load-bearing: tests rely on bare `assert()`, so `NDEBUG` must never win.
That is what lets the Release CI leg still catch assertion failures.

**The 18 binaries:** `test_tensor`, `test_allocator`, `test_ops`, `test_ops3`,
`test_graph`, `test_engine`, `test_onnx`, `test_planner`, `test_conv1d`,
`test_profiler`, `test_quant`, `test_tensor_ser`, `test_opt`, `test_parallel`,
`test_compiler`, `test_artifact`, `test_error_contract`, and `test_simd` (AVX2 builds
only). Plus `fuzz_runner` as a CTest entry = **19**.

`tests/test_engine.c` is the best end-to-end example.

---

## 7. CMake Knobs

| Option | Default | Effect |
|---|---|---|
| `FERRITE_SANITIZE` | ON | ASan+UBSan, **probed** — silently skipped if the toolchain lacks the runtimes |
| `FERRITE_COVERAGE` | OFF | `--coverage` on compile and link |
| `FERRITE_ENABLE_AVX2` | ON | AVX2 kernels; `FATAL_ERROR` on aarch64/arm/riscv/mips/ppc |
| `FERRITE_ENABLE_NEON` | **OFF** | not implemented; needs a Pi Zero 2 W+ target |
| `FERRITE_ENABLE_{IMPORTER,OPTIM,COMPILER,PARALLEL,QUANT,PROFILER,FLOAT64}` | ON | each OFF defines `FERRITE_NO_<SUBSYS>` and drops the library |

Capacity ceilings (cache STRINGs, substituted into `core/config.h.in` to produce a
generated `config.h` that precedes the committed `core/config.h` on the include path):

| Var | Default | Meaning |
|---|---|---|
| `FERRITE_MAX_DIMS` | 8 | tensor rank ceiling |
| `FE_MAX_NODES` | 512 | graph node ceiling |
| `FE_MAX_TENSORS` | 1024 | tensor registry ceiling |
| `FE_MAX_ALLOCS` | 1024 | planner allocation ceiling |
| `FERRITE_PLANNER_ALIGN` | 64 | activation-buffer alignment (device ports lower it to 4) |

Two configure-time `FATAL_ERROR`s exist on purpose, because each would otherwise be
silent corruption or a silent downgrade:

- `FE_MAX_ALLOCS < FE_MAX_TENSORS` — planner tables are indexed by tensor index. Mirrored
  by a `_Static_assert` in `planner/memory_planner.h`.
- AVX2 requested on a non-x86 `CMAKE_SYSTEM_PROCESSOR`.

---

## 8. CI

`.github/workflows/ci.yml` runs eleven jobs on push to `main` and on every pull request —
the eight named legs plus a Windows and a Release job.

| Job | What it does |
|---|---|
| `test` | Debug build + ctest, matrix over **gcc and clang** |
| `windows` | MSYS2 MINGW64 Debug build + ctest |
| `sanitize` | **Mandatory** ASan/UBSan. Greps the configure log for `building without` and fails if sanitizers were silently disabled |
| `fuzz` | Clang libFuzzer, `-max_total_time=120`, seeded from `fuzz/corpus` |
| `perf` | Release `bench_avx2 --json` vs a **per-CPU** baseline; `tools/check_perf.py --threshold 0.30`; an unknown CPU warns and skips |
| `golden` | ONNX Runtime comparison (below) |
| `coverage` | `gcovr -f core/ -f graph/ -f ops/ --fail-under-line 70` |
| `pi` | ARMv6 hard-float cross-build + full ctest under QEMU |
| `esp` | ESP-IDF container compile gate; also asserts `core/platform_esp32.c` **fails** to compile on the host |
| `release-test` | Release build with `-Werror` + ctest |

Reproduce the golden leg locally (pip packages, not runtime dependencies):

```sh
pip install onnxruntime onnx numpy
cmake --build build --target run_model
python3 tools/golden_gen.py
python3 tools/golden_compare.py --run-model build/run_model
```

Re-seed the perf baseline after an intentional performance change:

```sh
./build-release/bench_avx2 --json > /tmp/cur.json
python3 tools/check_perf.py --current /tmp/cur.json --seed bench/baseline.json --label linux-ci
```

---

## 9. Definition of Done

A change is not done until all of these hold:

1. `ctest --test-dir build --output-on-failure` is green, under ASan/UBSan where the
   toolchain supports them.
2. The matching `test_*` binary covers the new behavior. A new subsystem gets a new test
   binary.
3. Header comments updated **in the same commit** — one paragraph per public function or
   type, stating what it does, what it returns, and any constraint (alignment,
   contiguity, ownership).
4. Docs synced in the same commit: behavior or structure changes `docs/guide.md`; status,
   scope, or timeline changes `docs/roadmap.md`; build, usage, or benchmarks change
   `README.md`.
5. Every number written into a doc is real and reproducible by a command stated next to
   it.

---

## 10. Commits

One subsystem per commit, scoped: `feat(scope): short summary`. Say **why**, not what.
Recent history has drifted toward a capitalized-topic form (`CI: ...`, `Golden: ...`);
the scoped form above is the one to follow.

**No AI attribution trailers.** Do not end commit messages with
`Co-Authored-By: Claude ...` or any equivalent tag — see the rule at the top of this
file. Commits are authored by the repo owner alone.

Commit and push as `ArmaanGuha <armaanguha@gmail.com>`, which is the GitHub account
**ogshrug**. GitHub attributes by email, so that address is what makes a commit show up
under the right account; the display name differing from the handle is expected. Note
that this repo's older commits use a different identity (`Aarav
<aaravgupta170909@gmail.com>`) and the remote is owned by `AaravGupta17` — do not copy
either when making new commits.

---

## 11. Traps

Things that quietly waste a session here:

- **Windows/MinGW DLL.** Every executable gets `libwinpthread-1.dll` copied next to it at
  build time. Run through CTest or from `build/` — never by moving an exe elsewhere.
- **Sanitizers can be silently absent.** MinGW builds (CLion's bundled, Scoop's basic)
  often lack the runtimes; configure prints `ASan/UBSan requested but unavailable` and
  proceeds without them. A change that passes locally can still fail CI's `sanitize` leg.
  MSYS2 GCC and Linux GCC have them.
- **The `Makefile` is stale and not canonical.** Its `all:` target is missing
  `test_ops3`, `test_opt`, `test_simd`, `test_parallel`, and `test_compiler`. Use CMake.
- **AVX2 needs `N % 8 == 0`** for the vectorized prefix; a scalar tail handles the
  remainder. `M == 1` (GEMV) routes to scalar on purpose — tiling has no reuse to
  amortize packing there.
- **`tests/acousticleaknet.onnx` is 32 MB.** Tools that load it need buffers sized for it.
- **Debug and Release benchmarks are not comparable.** Debug is `-O0`, so the scalar
  reference is artificially slow; Release lets the compiler auto-vectorize it. The
  `bench_avx2` scalar side is pinned with volatile stores so `-O3` cannot fold it into a
  second AVX2 run. Keep it pinned.
- **The engine hot path is single-threaded by design.** `parallel/` exists, is tested, and
  is opt-in. Do not wire it into `fe_runtime_run` without a measured reason.

---

## 12. Current State

Working end to end: `.onnx` → optimize → static execution plan → inference, validated
against ONNX Runtime (`max_abs ≈ 1e-8`, MLP bit-exact). AVX2 is wired into
`fe_matmul`/`fe_linear`, relu/add/mul, and GEMM's base case. The planner drives
activation memory. Quantization is engine-wired: per-channel INT8 and INT16, dynamic
**and** calibrated-static activation scales, plus FP16/BF16 storage with
upconvert-on-read dispatch. The Pi Zero W and ESP32 ports compile and are CI-gated.

**Honest gaps:**

- The ONNX surface is the 20+ mapped ops; anything else fails loudly.
- AVX2 only. No NEON, no AVX-512.
- Fixed-but-configurable capacities (Section 7).
- Calibration collects `max(|x|)` ranges; percentile and stat smoothing are not built.
- Parallel execution and the compiler IR are tested but are not in the engine hot path.
- ESP32 hardware validation is a documented manual step; CI is a compile gate only.

---

## 13. Doc Map

| Doc | Owns |
|---|---|
| `CLAUDE.md`, `AGENTS.md` | how to work here — build, conventions, invariants |
| `README.md` | public face: what it is, quickstart, headline numbers |
| `docs/guide.md` | how every subsystem works and connects |
| `docs/deep-dive.md` | long-form internals, byte layouts |
| `docs/roadmap.md` | direction, per-stage status table |
| `docs/benchmarks.md` | recorded numbers plus the command that reproduces each |
| `docs/getting-started.md`, `install.md`, `requirements.md` | setup and running |
| `docs/documentation-standards.md` | BLUF, docs ship with code, honesty over polish |
| `docs/pi-zero-w-port.md`, `docs/esp32-port.md`, `idf/README.md` | ports |

**Verify before trusting a doc.** These docs have drifted from the code before — a
subsystem gets wired in and a "not yet wired" line survives in three files. When a doc
states a gap or a count, check it against the code or `CMakeLists.txt` before repeating
it, and fix it in the same commit as the work that proves it wrong.

**Do not add new root-level Markdown files.** `CLAUDE.md`, `AGENTS.md`, `README.md`, and
`CHANGELOG.md` are the root docs. Everything else lives in `docs/` as `kebab-case.md`.
