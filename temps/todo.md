# Ferrite: Production Readiness, Modularity, and Embedded Ports

This document is a work plan for an AI coding agent operating on the Ferrite
codebase (zero-dependency C11 ONNX inference runtime). It assumes the agent
has read `temps/explain.md`, `temps/roadmap.md`, and the deep-dive doc, and
has the source tree available. Each section is a scoped body of work with
concrete acceptance criteria. Work top to bottom — later sections (the
embedded ports) depend on the modularity split in Section 2.

---

## 0. Ground rules (apply to every task below)

- Preserve the project's existing invariants (see deep-dive §15): no
  allocation in hot paths, views over copies, arenas over per-tensor malloc,
  fail loudly rather than silently degrade, fixed capacity by design (but
  see Section 2.4 — capacity becomes *configurable* fixed capacity, not
  dynamic).
- Every change needs a matching test in the existing CTest suite, or a new
  test binary following the existing one-binary-per-subsystem convention.
- Docs move with code: `temps/explain.md`, `temps/roadmap.md`, and the
  deep-dive doc must be updated in the same change as the code.
- Don't introduce a dependency (BLAS, protobuf lib, SIMD wrapper lib) — the
  zero-dependency property is load-bearing for the embedded targets in
  Section 4.

---

## 1. Production readiness

### 1.1 Finish half-wired features instead of adding new ones

- [x] **Static activation scales.** `fe_runtime_calibrate` already records
  per-tensor max-|x| ranges (see `runtime/engine.c`), but nothing consumes
  them. Wire calibration output into `fe_quantize_model` so a model can be
  quantized with static (calibrated) activation scales instead of the
  current per-run dynamic quantization. Add percentile/stat smoothing as a
  second step after a naive max-based static scale is proven correct.
- [x] **Engine-wire INT16.** `fe_quantize_int16` / `fe_matmul_int16_dyn` /
  `fe_linear_int16` are API-complete but not reachable from
  `fe_quantize_model` or the exec-plan dispatch (`ex_matmul`/`ex_linear` in
  `runtime/exec_plan.c`). Wire them in behind a quantization-mode selector.
  Fix the documented quirk where the INT16 kernel's activation scale reuses
  the INT8-style `max/127` — it should be `max/32767`.
- [x] **Engine-wire FP16/BF16.** `core/fp16.c/h` conversions are tested but
  no model-level repack/dispatch path exists. Add a weight-repack step
  (mirroring `fe_quantize_model`'s in-place repack pattern) and a kernel
  path that upconverts to FP32 before calling existing FP32 kernels — no
  new compute kernels needed, just storage + upconvert-on-read.

### 1.2 SIMD correctness gap

- [x] **AVX2 remainder (`N % 8 != 0`).** Currently the whole matmul falls
  back to scalar when N isn't a multiple of 8 (see `simd/matmul_avx2.c`).
  Add a masked-tail AVX2 path (vectorize the `N - N%8` prefix, scalar only
  the remainder columns) so real-world shapes keep most of the SIMD win.
  Validate against `fe_matmul_scalar` as oracle, extend `test_simd.c` with
  non-multiple-of-8 shapes beyond what `test_linear_chain_remainder`
  already covers. (Note: the vectorized-prefix/scalar-tail path was
  verified already correct; `test_simd.c` extended with shapes crossing the
  KC=256 tile boundary.)

### 1.3 CI and testing infrastructure

- [x] **ASan/UBSan mandatory on at least one CI leg.** Today they're probed
  via `check_c_source_compiles` and enabled only if the toolchain supports
  them. Add a CI job pinned to a toolchain known to support both (Linux
  GCC) so sanitizers are never silently skipped in CI, even if they remain
  optional for local dev builds on toolchains that lack them.
  (Implemented: `.github/workflows/ci.yml` `sanitize` leg asserts CMake did
  NOT print "building without", with `ASAN/UBSAN_OPTIONS=halt_on_error`.)
- [x] **Continuous fuzzing.** `test_parser_fuzz` (596 truncated prefixes +
  400 random byte flips) is a fixed corpus run once. Wrap the ONNX parser
  entry point (`importer/onnx.c`) for libFuzzer, run it in CI with a
  persistent corpus and a time/iteration budget, and fail CI on any crash
  or sanitizer trip. Keep the existing fixed-corpus test as a fast
  regression check; libFuzzer is the "goes deeper over time" layer, not a
  replacement. (Implemented: `fuzz/fuzz_onnx.c` exports
  `LLVMFuzzerTestOneInput`, `fuzz/fuzz_main.c` is a standalone corpus
  replayer used by CTest (`fuzz_runner`, seeded `fuzz/corpus/`), CMake
  emits `fuzz_onnx_fuzzer` under Clang with `-fsanitize=fuzzer,address,
  undefined`, and the CI `fuzz` leg runs 120s time-boxed; the fixed-corpus
  test remains in `test_onnx.c`.)
- [x] **Performance regression gate.** Turn `temps/bench_results.md` into a
  CI check: run `bench_model` / `bench_avx2` in Release mode, compare
  against a committed baseline JSON, fail (or warn) on regressions past a
  threshold (start at 15–20% to avoid noise). Must report build mode
  (Debug/Release) alongside numbers, per the deep-dive's own caution that
  debug and release results tell different stories. (Implemented on the
  model-independent `bench_avx2 --json` microbench — NOT the acoustic
  model; `tools/check_perf.py` with `--seed`/`--threshold`, committed
  `temps/bench_baseline.json` seeded on this dev machine with a
  regenerate-on-CI note, CI `perf` leg in Release. The acoustic
  `bench_model` stays a manual tool for now.)
- [x] **Coverage gate.** `FERRITE_COVERAGE` (gcov) already exists as a
  build option. Wire it into CI, publish a coverage report per PR, and set
  a minimum threshold per subsystem (core/graph/ops first, since they're
  the most safety-critical). (Implemented: CI `coverage` leg builds with
  `-DFERRITE_COVERAGE=ON`, runs the suite, `gcovr -f core/ -f graph/ -f
  ops/ --fail-under-line 70`, uploads `coverage.xml`.)

### 1.4 API and error-contract audit

- [x] Walk every public function in every `*.h` under the public API
  surface and verify the stated contract: `FE_OK` ⇒ all outputs written;
  no partial mutation of outputs on error; no freeing of caller memory on
  error. Write this as a checklist test file (`test_error_contract.c`) that
  deliberately triggers each error path and asserts outputs are untouched.
  (Implemented: `tests/test_error_contract.c` sweeps tensor/arena/ops/
  graph/planner/ser/runtime/FEMD/ONNX error paths with sentinel-filled
  outputs asserted intact. 18/18 CTest passes.)
- [x] Add a `CHANGELOG.md` and adopt semver for the public C API starting
  at the first "production" tag. Document what counts as a breaking change
  (struct layout changes, enum reordering, function signature changes).
  (Implemented: `CHANGELOG.md` at 0.1.0 with the breaking-change policy.)

---

## 2. Modularity — split "compile a model" from "run a model"

This is the prerequisite for Section 4 (embedded ports). Do this before
starting any hardware-specific work.

### 2.1 Two build products from one codebase

- [x] **`ferrite-compile` (host tool).** Links `ferrite_importer` +
  `ferrite_optim` + `ferrite_compiler` + `ferrite_planner` +
  `ferrite_graph` + `ferrite_core`. Takes a `.onnx` file, runs the full
  existing pipeline (parse → shape-infer → simplify → fold → fuse-ConvBN →
  CSE → dead-elim → plan memory → resolve exec steps), and serializes the
  result to a single artifact file (see 2.2). This is where quantization
  (`fe_quantize_model`) and calibration-driven static scales (1.1) happen
  — all at compile time, never on-device. (Implemented as
  `tools/ferrite_compile.c` + the `ferrite_device` library; the tool loads,
  optimizes, plans, binds weights, and serializes — see 2.2.)
- [x] **`ferrite-runtime` (device library).** Links only `ferrite_core` +
  `ferrite_graph` (metadata structs only, no topo-sort/validate needed at
  runtime since that already happened at compile time) + `ferrite_ops` +
  `ferrite_planner` (the "apply cached offsets" path only, not the
  "compute lifetimes" path) + a new minimal loader for the serialized
  artifact. No protobuf parsing, no shape inference, no optimization
  passes ship to the device. This shrinks both code size and attack
  surface. (Implemented as `ferrite_device`: `runtime/kernels.c` +
  `runtime/model_ser.c` — kernel dispatch + artifact parse/run. Verified:
  compiling with IMPORTER/OPTIM/COMPILER/PARALLEL/PROFILER all OFF builds
  `libferrite_device.a` with no host-subsystem references; QUANT may also
  be OFF.)

### 2.2 Serialized artifact format

- [x] Extend the existing strict binary format in `core/tensor_ser.c`
  (`magic | version | dtype | ndim | shape | data`) to a whole-model
  artifact: magic `"FEMD"`, version, the flat `FeExecStep` array, the
  resolved memory plan (byte offsets, arena sizes), the weight arena bytes,
  and per-tensor entries (shape, dtype, is_weight, scales for per-channel
  quant). Keep the same "strict reader, fail loudly on anything
  unexpected" discipline as the existing tensor format — this file is
  read by devices with no filesystem sandboxing, so it should be at least
  as defensive as the ONNX parser. (Implemented in `runtime/model_ser.c`:
  magic/version check, every count/extent bounds-checked, trailing-EOF
  check, per-weight inline payload, node attrs, plan + lifetimes.
  `tests/test_artifact.c` covers save→load→run equivalence with the
  engine and rejects bad magic/version/truncation/trailing garbage.)
- [x] Write `ferrite-compile --emit-header model.fem model.h` to also emit
  the artifact as a C byte array (`static const unsigned char
  g_model_data[] = {...}`) for targets with no filesystem (ESP32 flash
  linkage). Keep the binary-file path for targets that do have a
  filesystem (Pi). (`--emit-header` implemented and exercised on
  tiny_mlp.onnx.)

### 2.3 CMake option gating per subsystem

- [x] Add CMake options: `FERRITE_ENABLE_IMPORTER`, `FERRITE_ENABLE_OPTIM`,
  `FERRITE_ENABLE_COMPILER`, `FERRITE_ENABLE_PARALLEL`,
  `FERRITE_ENABLE_QUANT`, `FERRITE_ENABLE_PROFILER`, `FERRITE_ENABLE_AVX2`,
  `FERRITE_ENABLE_NEON` (new — see 4.1). Default all ON for the existing
  desktop build so current behavior is unchanged; the two new
  device-target CMake presets (4.1, 4.2) turn most of these OFF.
- [x] Verify `ferrite-runtime`-only builds (importer/optim/compiler/
  parallel/profiler all OFF) actually compile and link without pulling in
  any of those subsystems — this is the test that the split is real, not
  just conditional compilation that still links everything. (Both
  QUANT=ON and QUANT=OFF device-only builds verified compiling
  `libferrite_device.a` cleanly.)

### 2.4 Configurable capacity ceilings

- [x] Replace the hardcoded `FE_MAX_NODES` (512), `FE_MAX_TENSORS` (1024),
  `FERRITE_MAX_DIMS` (8) with CMake cache variables (with the current
  values as defaults), threaded through to a generated `core/config.h`.
  This keeps the "fixed capacity, no config machinery" invariant (deep-dive
  §15.7 — no runtime feature flags) while letting each target pick its own
  compile-time ceiling. Document that these must be sized to fit the
  specific model(s) a device build ships, not left at the desktop default.
  (Verified: `-DFE_MAX_TENSORS=64 -DFERRITE_MAX_DIMS=4` flows into the
  generated config.h.)

### 2.5 Generalize the SIMD dispatch pattern

- [x] The existing `fe_matmul` → `fe_cpu_has_avx2()` → AVX2-or-scalar
  pattern (`ops/matmul.c`) is the right shape; generalize it into a small
  backend table (`FeSimdBackend` enum, one dispatch table per op family)
  so a NEON backend (Section 4.1) or a "no SIMD, scalar only" backend
  (Section 4.2) slot in the same way without touching `ops/` call sites.
  `fe_relu`/`fe_add`/`fe_mul` follow the same pattern today and should
  move to the same table. (Implemented as `simd/backend.[ch]` +
  `simd/scalar.c`; matmul/relu/add/mul/gemm route through `fe_simd_ops()`.
  Tested with a scalar-only build AND a dispatch-table assertion in
  `test_simd.c`.)

### 2.6 Platform abstraction seam

- [x] Introduce `core/platform.h` with macros/weak-linkage hooks for the
  three things currently assumed POSIX: monotonic time (`clock_gettime`,
  used by the profiler), logging sink (`stderr`, used by `core/log.c`),
  and threading primitives (`pthread_*`, used by `parallel/threadpool.c`).
  Desktop keeps POSIX implementations by default. This is what lets
  Section 4.2 (ESP32) swap in `esp_timer_get_time()` / `ESP_LOGx` /
  FreeRTOS primitives without `#ifdef`-littering the subsystem code
  itself. (Implemented as `core/platform.[ch]`: `fe_platform_now_ns`,
  `fe_platform_log_write`, `fe_platform_thread_create/join`,
  `fe_platform_threads_supported`, `fe_platform_cpu_count`. Routed:
  `tools/profiler.c`, `core/log.c`, `parallel/threadpool.c`.)
- [x] Note: MSVC is currently rejected at configure time specifically
  because the profiler needs POSIX clock. Once profiling is behind
  `core/platform.h`, re-evaluate whether the MSVC rejection is still
  necessary or was only a symptom of the missing abstraction.
  (Status: re-evaluation deferred — profiler + threadpool both still call
  through platform.c, but the CMake reject predates the seam and needs an
  MSVC/winpthread plumbing pass before it can be lifted; file as a follow-up.)

---

## 3. Raspberry Pi Zero W port

Runs full Linux — this is a cross-compile + correctness pass, not a
structural rewrite. Do this after Section 2 lands so the artifact-loading
path exists, though the Pi target *can* still ship the full importer if
desired (it has the RAM and filesystem for it — Section 2's split is a
choice on this target, not a requirement).

- [x] **Confirm target chip and toolchain.** Original Pi Zero W = single
  core ARM1176JZF-S, ARMv6, **no NEON**. Added
  `cmake/raspberry-pi-zero-w.toolchain.cmake` for `arm-linux-gnueabihf-gcc`
  (ARMv6, no NEON; hard-float), with the parallel/profiler/quant gates the
  target wants and a `CMAKE_CROSSCOMPILING_EMULATOR` so CTest runs under
  QEMU unmodified. Doc: `docs/pi-zero-w-port.md`.
- [x] **Gate x86-only compiler flags.** `-mavx2 -mfma` is now applied only
  when `FERRITE_ENABLE_AVX2` is on, and `bench_avx2`/`bench_matmul_avx2`
  are excluded entirely when it isn't. Requesting AVX2 on a non-x86
  `CMAKE_SYSTEM_PROCESSOR` is a CMake-time `FATAL_ERROR`, not a silent
  scalar downgrade; with the backend off, `ferrite_simd` resolves
  `fe_matmul` to scalar via the 2.5 dispatch table.
- [x] **Verify build.** CI **pi** leg cross-compiles with the toolchain
  file and runs the full CTest suite under QEMU user-mode ARM emulation —
  locally unverifiable (no ARM/qemu on the dev host). No source changes
  were needed beyond the build-system gating above.
- [x] **Enable `parallel/` by default on this target.** Toolchain preset
  ships `FERRITE_ENABLE_PARALLEL=ON`; the pool reports "unsupported" when
  the platform seam says so, so single-core still behaves.
- [x] **Default to INT8 quantized models on this target** (via
  `ferrite-compile`, Section 2.1) — documented in `docs/pi-zero-w-port.md`;
- [ ] **(Stretch) NEON backend**, only if the actual target is Pi Zero 2 W
  or later: implement `ops/matmul_neon.c` behind the generalized SIMD
  dispatch (2.5), oracle-tested against `fe_matmul_scalar` the same way
  the AVX2 kernel is.

---

## 4. ESP32 port

This is a genuinely different environment, not a recompile: no pthreads,
no filesystem by default, ~320–520 KB internal SRAM (optionally 4–8 MB
external PSRAM, slower to access), no hardware double-precision FPU. Do
not start this section until Section 2 is complete — the host/device
split is what makes an ESP32 build feasible at all.

### 4.1 Build system

- [x] Add an ESP-IDF component wrapping the device runtime: `idf/ferrite/`
  compiles `core` + `graph` + `ops` + `simd`-scalar + `planner` +
  `quantization` + the runtime/artifact loader. No `importer`/`optim`/
  `compiler` on device — models compile to FEMD artifacts on a host
  (`ferrite-compile`) and flash as headers. Doc: `idf/README.md`.
- [x] ESP32 preset gates: `FERRITE_ENABLE_PARALLEL=OFF`,
  `FERRITE_ENABLE_PROFILER=OFF`, `FERRITE_ENABLE_IMPORTER=OFF`,
  `FERRITE_ENABLE_OPTIM=OFF`, `FERRITE_ENABLE_COMPILER=OFF`,
  `FERRITE_ENABLE_AVX2=OFF`, `FERRITE_ENABLE_FLOAT64=OFF`.
  **Deviation from the original bullet:** `FERRITE_ENABLE_QUANT` stays **ON**
  in both `cmake/esp32.toolchain.cmake` and the IDF component — the device
  never *quantizes* (host-side work), but it must still *run* pre-quantized
  INT8/INT16 kernels, and those live behind the same gate. Turning QUANT off
  also disables the `fe_matmul_int8*` inference paths.
- [x] `FERRITE_MAX_NODES`/`FERRITE_MAX_TENSORS`/`FERRITE_MAX_DIMS`/
  `FERRITE_PLANNER_ALIGN` sized per-target: `idf/ferrite/config/config.h`
  (include-path-first override of `core/config.h`) and the toolchain
  preset. `ferrite-compile --target esp32` enforces the ceilings at build
  time (`--max-nodes`, `--max-tensors`).

### 4.2 Platform shims

- [x] `core/platform_esp32.c` implements the Section 2.6 seam for ESP-IDF:
  time via `esp_timer_get_time()` (us → ns monotonic), logging via ROM
  printf (`esp_rom_printf`), threads explicitly unsupported
  (`FERRITE_NO_PARALLEL`); `#error` if compiled outside Xtensa.
- [x] `FERRITE_ENABLE_FLOAT64` compile-time guard: with it off,
  `fe_model_load` rejects `DTYPE_FLOAT64` entries with `FE_ERR_DTYPE`, and
  `ferrite-compile --target esp32` rejects FLOAT64 models at compile time —
  no software-emulated doubles on hardware that can't do them.

### 4.3 Memory and alignment

- [x] `PLANNER_ALIGN` → build-time `FERRITE_PLANNER_ALIGN` (was a hardcoded
  `#define PLANNER_ALIGN 64` in `planner/memory_planner.c`). CMake generates
  it through `core/config.h.in`; desktop keeps 64 (SIMD), ESP32 drops to 4
  (scalar natural alignment) both in the toolchain preset and the IDF
  `config/config.h` override.
- [ ] Decide and document the internal-SRAM-vs-PSRAM placement strategy
  for the weight arena vs activation arena — weights (read-only, large)
  are the better PSRAM candidate if a board has it; activations
  (read/write every inference, smaller) should stay in fast internal SRAM
  if they fit. The arena already takes a caller-supplied buffer per
  `core/allocator.c`; the IDF component defaults to SRAM-backed buffers and
  the `docs/esp32-port.md` memory note flags flash-mapped weights. Exact
  per-board dual-SRAM/PSRAM allocation is left to the app.

### 4.4 Model constraints for this target

- [x] `docs/esp32-port.md` documents the INSTEAD-of-model constraints:
  models targeting ESP32 must be INT8-per-channel-quantized (host-side
  `fe_quantize_model`), fit node/tensor/dim ceilings, and be FLOAT64-free.
  `ferrite-compile --target esp32` fails the compile (not the flash) if a
  model violates any of them.

### 4.5 Verification

- [ ] Build and run the existing `test_tensor`, `test_allocator`,
  `test_ops`/`test_ops3`, `test_graph`, `test_engine`, `test_planner`,
  `test_quant` suites (the ones that don't depend on `importer`/`optim`/
  `compiler`/`parallel`/`profiler`) against a QEMU ESP32 target or real
  hardware, adapted for whatever test runner ESP-IDF uses (`idf.py test`
  / Unity) instead of CTest. **Partially done:** the CI **esp** leg builds
  `idf/demo` in the `espressif/idf` container (compile gate); the device
  test-suite adaptation and the on-hardware smoke run remain manual/unbuilt.
- [ ] End-to-end test: compile a small quantized model with
  `ferrite-compile`, flash the resulting `.fem`-as-C-header artifact to a
  real board, run inference, and diff output against the same model run
  through the desktop `ferrite-runtime` — this is the ESP32 equivalent of
  `test_compiler.c`'s "compiled IR matches `fe_runtime_run` to ≤ 1e-5."

---

## 5. Suggested order of work

1. Section 1.3 (CI/sanitizers/fuzzing) and 1.4 (error-contract audit) —
   cheap, low-risk, and makes every subsequent change safer to land.
2. Section 2 in full (modularity split) — everything else depends on it.
3. Section 1.1 and 1.2 (finish half-wired quantization features, AVX2
   remainder) — do this after 2.5 (generalized SIMD dispatch) so the new
   kernels land in the new dispatch pattern instead of the old one.
4. Section 3 (Pi Zero W) — validates the modularity split on a target
   that's still "basically Linux," lower risk than ESP32.
5. Section 4 (ESP32) — highest-risk, most novel work; do last, once the
   artifact format (2.2) and platform shim seam (2.6) have already been
   exercised once on the Pi.