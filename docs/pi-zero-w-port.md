# Raspberry Pi Zero W port (Section 3)

Target: Raspberry Pi Zero W / Pi Zero 2 W running Raspberry Pi OS Lite (64-bit
kernel, 32-bit userspace ARMv6 hard-float build).

## What ships

- `cmake/raspberry-pi-zero-w.toolchain.cmake` — cross toolchain preset using
  `arm-linux-gnueabihf-gcc`, tuned to the ARM1176 (Armv6): AVX2/NEON off,
  parallel/profiler/quant on, device-sized ceilings, and a
  `CMAKE_CROSSCOMPILING_EMULATOR` entry so `ctest` runs the whole suite under
  QEMU user-mode emulation unmodified.
- `-mavx2 -mfma` flags: now only emitted for x86 builds; `bench_avx2`/
  `bench_matmul_avx2` are excluded entirely when the AVX2 backend is off. An
  x86-only-SIMD request on a non-x86 target is a CMake-time `FATAL_ERROR`,
  not a silent scalar downgrade.

## Build

```sh
sudo apt-get install gcc-arm-linux-gnueabihf qemu-user-static ninja-build
cmake -S . -B build-pi -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=cmake/raspberry-pi-zero-w.toolchain.cmake
cmake --build build-pi
ctest --test-dir build-pi        # under QEMU, nothing to flash
```

To cross the same build to a board (or run tests there), override
`CMAKE_CROSSCOMPILING_EMULATOR`.

To include the full host stack on-device (the Pi has RAM + a filesystem, so
the importer/optimizer are viable), add
`-DFERRITE_ENABLE_IMPORTER=ON -DFERRITE_ENABLE_OPTIM=ON`. The model family
that these ceilings fit is in `temps/roadmap.md`; raise `FE_MAX_NODES` etc.
for bigger graphs.

## Runtime choice

The Pi is a full-Linux target and single-core, so the scalar **and** the
row-parallel scheduler both matter less than cache behavior. Run-time
recommendations for the shipped model class:

- prefer INT8 (per-channel) quantized weights via `fe_quantize_model` —
  halves memory-bandwidth pressure, the dominant cost on ARM1176;
- keep `FERRITE_ENABLE_PARALLEL=ON` in the toolchain preset — the chunked
  row-parallel GEMM (`fe_gemm_parallel`) spreads one core's cycles across the
  system scheduler only when it wins; pool calls report "unsupported"
  otherwise.

## Stretch (not done)

- NEON backend (`FERRITE_ENABLE_NEON`): the original Pi Zero's ARM1176 has no
  NEON; a `fe_matmul_neon` backend belongs on Pi Zero 2 W+ (Cortex-A53+). The
  simd backend-selection seam (`simd/backend.h`) already supports the slot.

## Verification status

- Locally unverifiable (no ARM toolchain/QEMU on the dev host).
- CI: `.github/workflows/ci.yml` **pi** leg cross-builds, then runs the whole
  suite under QEMU — authoritative unless something CI can't spot (e.g. FPU
  ABI quirks) shows up on hardware.