# cmake/raspberry-pi-zero-w.toolchain.cmake
#
# Cross-compile for the original Pi Zero W: ARM1176JZF-S, ARMv6, single core,
# **no NEON**. This is a full-Linux target, so the device runtime is optional
# here too — the Pi CAN ship the importer/optimizer (it has RAM + filesystem)
# — but the default preset below keeps it a lean device build. To include the
# host toolchain, pass -DFERRITE_ENABLE_IMPORTER=ON -DFERRITE_ENABLE_OPTIM=ON.
#
# Usage:
#   cmake -S . -B build-pi -G Ninja \
#       -DCMAKE_TOOLCHAIN_FILE=cmake/raspberry-pi-zero-w.toolchain.cmake
#   cmake --build build-pi
#   # run the suite under QEMU user-mode ARM emulation:
#   ctest --test-dir build-pi
#
# Requires the cross GCC + qemu-user-static on the host. The CI `pi` leg in
# .github/workflows/ci.yml does exactly this.

set(CMAKE_SYSTEM_NAME       Linux)
set(CMAKE_SYSTEM_PROCESSOR  armv6)

set(CMAKE_C_COMPILER        arm-linux-gnueabihf-gcc)

# Run cross-built test binaries under QEMU user-mode emulation so CTest works
# unmodified (or on hardware, point this at the device's SSH runner).
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-arm-static" "-L"
    "/usr/arm-linux-gnueabihf")

# No NEON on ARM1176: the only SIMD backend that exists today is x86 AVX2,
# and turning it on would make every matmul silently fall back to scalar.
set(FERRITE_ENABLE_AVX2     OFF CACHE BOOL "x86 AVX2 SIMD backend" FORCE)
set(FERRITE_ENABLE_NEON     OFF CACHE BOOL "ARM NEON SIMD backend" FORCE)

# ARM1176 FPU is slow: the thread pool and row-parallel GEMM are a real win
# on this target (Section 3), so unlike the ESP32 build the parallel
# subsystem stays ON.
set(FERRITE_ENABLE_PARALLEL ON  CACHE BOOL "Host: thread pool + DAG" FORCE)
set(FERRITE_ENABLE_PROFILER ON  CACHE BOOL "Timing profiler (POSIX clock)" FORCE)

# Embedded Linux has a real clock (clock_gettime) so the profiler stays on.
# Quantization ships pre-quantized INT8/INT16 models on this target — see the
# port doc — keep the kernel-side paths available.
set(FERRITE_ENABLE_QUANT    ON  CACHE BOOL "Quantization kernels + engine" FORCE)

# Capacity ceilings sized for the demo-model class on this target
# (Section 2.4): tiny MLPs and small 1-D CNNs, not 512-node graph monsters.
set(FE_MAX_NODES            64  CACHE STRING "Graph node ceiling" FORCE)
set(FE_MAX_TENSORS          128 CACHE STRING "Tensor registry ceiling" FORCE)
set(FERRITE_MAX_DIMS        4   CACHE STRING "Tensor rank ceiling" FORCE)