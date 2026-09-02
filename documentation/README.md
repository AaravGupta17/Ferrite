# Running Ferrite

## Bottom Line Up Front

Ferrite is a zero-dependency C11 neural-network inference runtime. It parses ONNX files by hand, loads them into a computation graph, and runs them with hand-written kernels. This document tells you exactly how to build it and run it. Everything is done from the repository root.

> **New here?** See [`INSTALL.md`](INSTALL.md) to set up the toolchain, and [`REQUIREMENTS.md`](REQUIREMENTS.md) for what Ferrite needs — including which compilers support the ASan/UBSan memory checks.

## Requirements

- **CMake 3.16+**
- **A C11 compiler — GCC or Clang.** Ferrite needs the POSIX monotonic clock (`clock_gettime`, pulled in via `_POSIX_C_SOURCE=199309L`) used by the profiler, and MSVC does not provide it.
- **Ninja** (recommended generator; you can substitute `-G "Unix Makefiles"` on Linux/WSL).

On Windows, use the MinGW from **MSYS2** or a MinGW toolchain (e.g. installed via Scoop). The compiler is probed at configure time and the build refuses to proceed otherwise.

> **Sanitizers note.** Configure enables ASan+UBSan **only if** the toolchain ships the sanitizer runtimes. CMake probes for this automatically. MSYS2 GCC and Linux GCC have them; some MinGW builds (e.g. CLion's bundled, or Scoop's basic MinGW) do **not** — in that case the build proceeds without sanitizers and prints `ASan/UBSan requested but unavailable`. The tests still run; you just lose the memory/UB checks.

---

## 1. Configure

```sh
cmake -S . -B build -G Ninja
```

This creates the `build/` directory with all test binaries as the default target. An unconfigured `build/` dir gets re-run automatically the first time you build.

## 2. Build

```sh
cmake --build build
```

This builds all **12 test binaries** and the static subsystem libraries. Benchmarks and the demo are opt-in and **excluded** from the default build (see below).

## 3. Run the test suite

```sh
ctest --test-dir build
```

Runs all 12 tests. Add `--output-on-failure` to see failing output inline. For one binary directly (from `build/`):

```sh
./build/test_engine.exe      # Windows
./build/test_engine          # Linux/macOS/WSL
```

The full list of test binaries: `test_tensor`, `test_allocator`, `test_ops`, `test_ops3`, `test_graph`, `test_engine`, `test_onnx`, `test_planner`, `test_conv1d`, `test_profiler`, `test_quant`, `test_tensor_ser`.

**End-to-end test.** `test_engine.exe` runs two full inferences through the engine (`fe_runtime_run`): a 2-layer MLP and a Stage 3 graph (exp → sigmoid → layernorm → gemm → softmax). It is the best single sign that the runtime works.

## 4. Run the demo

The demo loads a real trained model (`tests/acousticleaknet.onnx`, a 1D-CNN for acoustic pipe-leak detection) and prints per-op latency, class probabilities, and memory stats.

```sh
cmake --build build --target demo
./build/demo.exe              # Windows
./build/demo                  # Linux/macOS/WSL
```

`demo.exe` runs from the repository root by default (it looks for `tests/acousticleaknet.onnx`). To point it at another model:

```sh
./build/demo.exe /path/to/model.onnx
```

## 5. Run the benchmarks

Three benchmark binaries compare the naive matmul, the AVX2 matmul, and performance builds. Each is excluded from the default build, so pass the target explicitly:

```sh
cmake --build build --target bench_matmul        # naive, -O2
cmake --build build --target bench_matmul_avx2   # same source, -O3 -mavx2 -mfma
cmake --build build --target bench_avx2          # AVX2 tiled matmul with runtime CPUID
./build/bench_avx2.exe
```

The AVX2 target requires a CPU with AVX2/FMA; the kernel checks CPUID at runtime and falls back to scalar code if unsupported.

---

## Platform notes

- **Windows DLL.** Executables use `clock_gettime` from `libwinpthread-1.dll`. The build copies that DLL next to every executable automatically, and the Windows loader looks there first. Run tests/executables from `build/` (or via CTest) — don't copy an exe elsewhere and expect it to run.
- **Working directory for tests.** CTest runs each test from the source root because some tests reference fixtures by relative path (e.g. `tests/tiny_mlp.onnx`). Run binary tests from a directory where those relative paths resolve.

## Layout at a glance

| Path | Purpose |
|---|---|
| `core/` | Strided tensors, arena allocator, serialization, logging |
| `graph/` | Computation-graph IR + topo sort |
| `ops/` | Operator kernels (matmul, activations, conv, norm, sequence, ...) |
| `simd/` | AVX2 tiled matmul with runtime CPUID |
| `planner/` | Tensor-lifetime + buffer reuse analysis |
| `runtime/` | Execution engine |
| `importer/` | Hand-rolled ONNX protobuf parser + graph loader |
| `quantization/` | INT8 quantization |
| `tools/` | Profiler, benchmark harnesses, demo |
| `tests/` | One test binary per subsystem + ONNX fixtures |
