# ESP32 port (Section 4)

Target: ESP32-S3 (dual-core Xtensa LX7) with SRAM + PSRAM. Models arrive as
FEMD artifacts (`ferrite-compile`), never as `.onnx`.

## What ships

- `idf/ferrite/` — an ESP-IDF component compiling the **device runtime**
  (scalar kernels + graph + planner + quant + engine + `model_ser`) with the
  host subsystems and FLOAT64 compiled out. Includes `config/config.h`, an
  include-path-first override of `core/config.h` with device ceilings
  (48 nodes / 96 tensors / rank 4 / natural alignment).
- `core/platform_esp32.c` — the Section 2.6 seam for Xtensa: monotonic time
  via `esp_timer_get_time()`, ROM printf logging, threads declared
  unsupported (single-task runtime; pool calls fall back to in-thread work).
  Fails with `#error` if compiled on a non-Xtensa host.
- `cmake/esp32.toolchain.cmake` — plain-CMake device-only build preset that
  mirrors the component's `FERRITE_ENABLE_*` gating and ceilings, so CI can
  compile-check the same surface without the IDF build system when desired.
- `idf/demo/` — buildable project (relu smoke test) that compiles the
  component; run it on hardware to validate the runtime.
- `ferrite-compile --target esp32` — load-time model gate: rejects FLOAT64
  tensors (device kernels are FP32/FP16/BF16/INT8/INT16) and enforces the
  device ceilings (`--max-nodes`, `--max-tensors`).

## Deviations from the host build (all fail loudly)

| Host | ESP32 | Failure mode |
|---|---|---|
| FP32 graph, FLOAT64 allowed | FLOAT64 rejected | `fe_model_load` returns `FE_ERR_DTYPE` (`FERRITE_NO_FLOAT64`); compile-time gate for new artifacts |
| graph up to 512 nodes / rank 8 | 48 nodes / rank 4 | load-time `FE_ERR_SHAPE`; `ferrite-compile --target esp32` rejects at build |
| importer / optimizer / IR | compiled out | build-time `FERRITE_NO_*` |
| thread pool, row-parallel GEMM | absent | kernel falls back to scalar; pool "unsupported" |
| CLOCK_MONOTONIC + stderr | esp_timer + ROM printk | n/a |
| profiler | compiled out | n/a |

## Build

```sh
cd idf/demo
idf.py set-target esp32s3
idf.py build flash monitor
```

Standalone device-only library (no IDF):

```sh
cmake -S . -B build-esp32 -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=cmake/esp32.toolchain.cmake
cmake --build build-esp32 --target ferrite_device
```

## Memory notes

- `FERRITE_PLANNER_ALIGN=4`: scalar-only, so the planner stops padding every
  tensor to 64 bytes. The SIMD-backed desktop default stays 64.
- Weights are flash-mapped: `--emit-header` output can be `const`
  (XIP/flash-cached) and the weight arena given that memory; activations stay
  in SRAM/PSRAM.

## Verification status

- Locally unverifiable (no ESP-IDF toolchain on the dev host).
- CI: `.github/workflows/ci.yml` **esp** leg builds `idf/demo` inside the
  `espressif/idf:5.4` container — green on `main`. It is a compile gate,
  not a hardware run: it builds every object the component ships (complete
  device op set, `optim/shape_infer.c`, `runtime/exec_plan.c`) and links the
  app against `libferrite.a`. The `idf/demo` relu smoke test plus a
  quantized FEMD inference are manual on-device steps.