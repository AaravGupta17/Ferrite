# cmake/esp32.toolchain.cmake
#
# Plain-CMake device build for ESP32 (Section 4), OUTSIDE the ESP-IDF build
# system. This exercises the FERRITE_ENABLE_* gating + per-target config
# ceilings the same way the Pi toolchain file does, on any host with the
# xtensa-esp32s3-elf-gcc toolchain on PATH. The production integration is the
# ESP-IDF component in idf/ferrite (which has its own build system via
# idf.py); this file exists so the device-only library can be configured and
# CI-verified without the IDF toolchain, and the two stay in sync by sharing
# the same FERRITE_ENABLE_* knobs.
#
# Usage:
#   cmake -S . -B build-esp32 -G Ninja \
#       -DCMAKE_TOOLCHAIN_FILE=cmake/esp32.toolchain.cmake
#   cmake --build build-esp32 --target ferrite_device
#
# No tests run under QEMU here — ESP32 has no user-space Linux mode; run the
# end-to-end tests on hardware (docs/esp32-port.md).

set(CMAKE_SYSTEM_NAME       Generic)
set(CMAKE_SYSTEM_PROCESSOR  xtensa)

set(CMAKE_C_COMPILER        xtensa-esp32s3-elf-gcc)

# Device-only build: no importer/optimizer/compiler/IR on the part — models
# arrive compiled to FEMD headers. Quantization stays (INT8/INT16 kernels
# are the ESP32 default path). Threads vanish (no POSIX threads).
set(FERRITE_ENABLE_IMPORTER OFF CACHE BOOL "Host: ONNX protobuf loading"   FORCE)
set(FERRITE_ENABLE_OPTIM    OFF CACHE BOOL "Host: optimization passes"     FORCE)
set(FERRITE_ENABLE_COMPILER OFF CACHE BOOL "Host: stage-14 linear IR"      FORCE)
set(FERRITE_ENABLE_PARALLEL OFF CACHE BOOL "Host: thread pool + DAG"       FORCE)
set(FERRITE_ENABLE_PROFILER OFF CACHE BOOL "Timing profiler (POSIX clock)" FORCE)
set(FERRITE_ENABLE_AVX2     OFF CACHE BOOL "x86 AVX2 SIMD backend"         FORCE)
set(FERRITE_ENABLE_NEON     OFF CACHE BOOL "ARM NEON SIMD backend"         FORCE)
set(FERRITE_ENABLE_FLOAT64  OFF CACHE BOOL "Double-precision tensor support" FORCE)
set(FERRITE_ENABLE_QUANT    ON  CACHE BOOL "Quantization kernels + engine" FORCE)

# Scalar-only: natural alignment, not the SIMD 64; every wasted byte costs
# SRAM/PSRAM on this target.
set(FERRITE_PLANNER_ALIGN   4   CACHE STRING "Activation-buffer alignment" FORCE)

# Ceilings sized for the FEMD-artifact model class (Section 2.4 / 4.4).
set(FE_MAX_NODES            48  CACHE STRING "Graph node ceiling" FORCE)
set(FE_MAX_TENSORS          96  CACHE STRING "Tensor registry ceiling" FORCE)
set(FERRITE_MAX_DIMS        4   CACHE STRING "Tensor rank ceiling" FORCE)