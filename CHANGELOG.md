# Changelog

All notable changes to Ferrite's public C API are documented here, per the
[semantic versioning][semver] rules below. The public API surface is every
`fe_*` function, type, enum, and macro declared in a public `*.h` in `core/`,
`graph/`, `ops/`, `simd/`, `planner/`, `parallel/`, `compiler/`, `optim/`,
`runtime/`, `importer/`, and `quantization/`.

[semver]: https://semver.org/

## Versioning policy

- **Breaking change** (MAJOR bump): a change that breaks a downstream caller
  at compile time. Concretely: `FeTensor`/`FeNode`/`FeGraph`/`FePlan`/`FeRuntime`
  struct layout changes, `FeOpType`/`FeDtype`/`FeStatus` enum value
  re-ordering or renumbering, function signature changes, removal of any
  public symbol, or a change to a `FERRITE_*`/`FE_*` API macro's meaning.
- **Non-breaking change** (MINOR bump): new public functions, types, or
  entries appended to existing enums (appended at the end only — never
  inserted or renumbered), new compile-time options, or documentation changes.
- **Patch** (PATCH bump): internal fixes that do not touch the public API.

## [Unreleased]

This is the baseline: the first production-oriented release. Everything
below landed in one series of scoped changes; the API is now declared stable
under the versioning policy above.

### Added

- **CI pipeline** (`.github/workflows/ci.yml`): test (Debug), sanitize (ASan/
  UBSan mandatory — the job fails if the toolchain silently disables them),
  fuzz (libFuzzer, time-boxed, persistent corpus), perf (Release `bench_avx2`
  vs committed baseline), and coverage (gcovr, `--fail-under-line 70` on
  core/graph/ops) legs.
- **Continuous-fuzz harness** (`fuzz/fuzz_onnx.c`, `fuzz/fuzz_main.c`,
  `fuzz/corpus/`, `fuzz/fuzz_onnx.h`): the ONNX parser entry point wrapped
  for libFuzzer (`LLVMFuzzerTestOneInput`) with a standalone corpus-replay
  driver for non-Clang toolchains; the `fuzz_runner` CTest replays the seed
  corpus. Complements the existing fixed-corpus `test_parser_fuzz`.
- **Performance regression gate** (`tools/check_perf.py`, `bench_avx2
  --json`, `temps/bench_baseline.json`): model-independent AVX2 microbench
  numbers compared against a committed baseline with a default 20% threshold
  and a `--seed` mode for regenerating the baseline. (Deliberately not wired
  to the acoustic `bench_model`; that stays a manual tool.)
- **Error-contract audit** (`tests/test_error_contract.c`): a checklist test
  that deliberately triggers every documented `FE_ERR_*` path across the
  public API and asserts outputs are untouched on failure (no partial
  mutation, no caller-memory freeing on error). Sweeps tensor, arena, ops,
  graph, planner, serialization, runtime, FEMD, and ONNX entry points.
- **Raspberry Pi Zero W port (Section 3)**: `cmake/raspberry-pi-zero-w.
  toolchain.cmake` cross preset (ARMv6, hard-float, device gates) with a
  QEMU cross-emulation hook so CTest runs the full suite unmodified; CI
  `pi` leg builds and runs it under QEMU. `-mavx2 -mfma` is now emitted only
  with the AVX2 backend on, and requesting AVX2 on a non-x86 target is a
  CMake-time `FATAL_ERROR` instead of a silent scalar downgrade. Doc:
  `docs/pi-zero-w-port.md`.
- **ESP32 port (Section 4)**: `idf/ferrite/` ESP-IDF component (device
  runtime only) + `config/config.h` device ceilings override +
  `core/platform_esp32.c` (esp_timer clock, ROM printf, threads
  unsupported) + `cmake/esp32.toolchain.cmake` plain-CMake preset +
  `idf/demo` hardware smoke project; CI `esp` leg compile-gates the
  component in the `espressif/idf` container. Docs: `idf/README.md`,
  `docs/esp32-port.md`.
- **Device-target compile gate for FEMD artifacts**: `ferrite-compile
  --target esp32` rejects FLOAT64 models and enforces the ESP32
  node/tensor ceilings at build time.
- **`FERRITE_ENABLE_FLOAT64`** compile-time option (default ON): when off,
  `fe_model_load` rejects `DTYPE_FLOAT64` entries with `FE_ERR_DTYPE`,
  keeping software-emulated doubles off devices that can't do them.
- **`FERRITE_PLANNER_ALIGN`** build-time macro (generated `core/config.h`):
  the planner's activation-buffer alignment was a hardcoded 64; it is now
  per-target (64 for SIMD backends, 4 for scalar-only devices), threading
  through `core/config.h.in` and the committed Makefile copy.
- **CHANGELOG.md and semver policy** (this file): establishes what counts as
  a breaking change for the public C API.
- **Static activation scales** (`fe_runtime_calibrate_pct`, `act_scale`,
  static INT8/INT16 kernels): per-run dynamic activation quantization can now
  be replaced with calibrated, percentile-smoothed static scales set at
  model-quantization time.
- **Engine-wired INT16 quantization**: `fe_quantize_int16` /
  `fe_matmul_int16_dyn` / `fe_linear_int16` reachable from the exec-plan
  dispatch; the INT16 activation scale now uses `max/32767` instead of the
  INT8-style `max/127`.
- **Engine-wired FP16/BF16 storage**: `core/fp16.[ch]` conversions backed by
  a model-level weight-repack step and upconvert-on-read kernel dispatch
  (half-precision weights stored as FLOAT16/BFLOAT16, upconverted to FP32 at
  run time).
- **AVX2 vectorized-prefix/scalar-tail matmul**: shapes with `N % 8 != 0`
  keep most of the SIMD win instead of falling back to scalar wholesale;
  `tests/test_simd.c` covers shapes crossing the KC=256 tile boundary.
- **Host/device toolchain split (Section 2)**:
  - `ferrite_device` library (`runtime/kernels.c` + `runtime/model_ser.c`):
    kernel dispatch table + FEMD artifact parse/run. The only library an
    embedded target compiles; optionally linked by the host runtime too.
  - `ferrite_compile` host tool (`tools/ferrite_compile.c`): ONNX →
    optimize → plan → bind weights → serialize to a self-contained FEMD
    artifact, with `--in-shape` and `--emit-header` (C byte-array output for
    filesystem-less targets).
  - FEMD artifact format (magic `"FEMD"`, version 1): strict little-endian
    fixed-width reader/writer, per-tensor registry entries with inline weight
    payloads, node attrs, resolved memory plan including lifetimes, per-node
    kernel resolution at load, and a trailing-EOF check.
  - `tests/test_artifact.c`: save→load→run equivalence with the engine plus
    corruption rejection (bad magic/version/truncation/trailing garbage).
- **CMake subsystem gating** (`FERRITE_ENABLE_IMPORTER/OPTIM/COMPILER/
  PARALLEL/QUANT/PROFILER/AVX2/NEON`), defaulting all ON. Verified
  `ferrite-runtime`-only builds (host subsystems OFF, QUANT either way)
  compile the device library with no host references.
- **Configurable capacity ceilings**: `FE_MAX_NODES`, `FE_MAX_TENSORS`,
  `FERRITE_MAX_DIMS`, `FE_MAX_ALLOCS` are now CMake cache variables threaded
  through a generated `core/config.h`. Defaults unchanged.
- **Generalized SIMD dispatch** (`simd/backend.[ch]`, `simd/scalar.c`):
  `fe_matmul`/`fe_relu`/`fe_add`/`fe_mul`/`fe_gemm` route through a
  backend table (`fe_simd_ops`) instead of per-op CPUID branches; NEON or a
  scalar-only backend slot in without touching `ops/`.
- **Platform abstraction seam** (`core/platform.[ch]`): monotonic-time,
  logging-sink, and threading-primitive hooks (`fe_platform_*`) backing the
  profiler, `core/log.c`, and the thread pool, so a non-POSIX target swaps
  in `esp_timer_get_time()` / `ESP_LOGx` / FreeRTOS without `#ifdef` litter.
- `fe_graph_find_input` / `fe_graph_find_output` public graph helpers
  (`graph/graph.c`), extracted from the runtime's private helper.

### Fixed

- **INT16 activation scale**: `max/32767` (was reusing the INT8 `max/127`).
- **QUANT=OFF device build**: half-precision upconversion paths in
  `runtime/kernels.c` previously triggered an implicit-declaration error;
  they now fail loudly with `FE_ERR_DTYPE` + log when FLOAT16/BFLOAT16
  storage is requested on a build without quantization.
- **FEMD plan lifetimes**: the artifact serializes plan lifetimes; omitting
  them made `fe_plan_apply` fail to bind activation pointers after load
  (`fe_model_run` reported `FE_ERR_NULL` on every node).

### Notes

- MSVC is still rejected at configure time (profiler/threadpool POSIX clock
  plumbing predates the platform seam); lifting the rejection is tracked as
  a follow-up once winpthread plumbing lands.
- The legacy `Makefile` is synced for the new targets but is not CI-verified;
  CMake is canonical.
- Ports and CI legs are authored artifacts, per "Everything, artifacts for
  unverifiable": the sanitize/fuzz/perf/coverage/pi/esp legs are CI-only
  (no Linux runner, libFuzzer, ARM toolchain, or ESP-IDF on the dev host);
  the Pi leg additionally runs the full suite under QEMU. Hardware-only
  steps (ESP32 flashing) are procedural in `docs/*.md`.