# Ferrite — Requirements

## Bottom Line Up Front

Ferrite is a zero-dependency C11 inference runtime. Building it needs a POSIX-compatible C compiler (GCC or Clang) and CMake. Everything else — ONNX parsing, kernels, memory — is hand-written and needs no third-party libraries at build or run time.

## Software requirements

| Requirement | Version | Notes |
|---|---|---|
| C11 compiler — **GCC or Clang** | any recent | **MSVC does not work.** The profiler uses the POSIX monotonic clock (`clock_gettime`, enabled by `_POSIX_C_SOURCE=199309L`) which MSVC lacks. |
| CMake | 3.16+ | The canonical build system. |
| Ninja | any | Recommended generator. Linux/WSL users may substitute `-G "Unix Makefiles"`. |
| Git | any | Not strictly required to build, but needed to clone the repo and for the one-command-per-subsystem commit workflow. |

There are **no runtime or build-time third-party libraries** — no BLAS, no protobuf, no SIMD wrapper, no math library. `libm` is linked on Unix and corresponds to `msvcrt` equivalents on Windows; neither is an external dependency.

## Toolchain vs. sanitizers

ASan+UBSan are enabled **only if** the compiler actually ships the sanitizer runtimes. CMake probes for this automatically:

| Toolchain | ASan/UBSan |
|---|---|
| Linux GCC / Clang | Available — enabled by default |
| MSYS2 MinGW GCC | Available — enabled by default |
| Scoop / CLion-bundled MinGW | Usually **missing** — build proceeds without, prints `ASan/UBSan requested but unavailable` |

Sanitizers are a strict subset of the requirement: you don't *need* them to build or test Ferrite, but they are required to catch memory/UB bugs. Prefer a toolchain that has them.

## Hardware requirements

- **AVX2 + FMA** — needed only for the `bench_avx2`/`bench_matmul_avx2` targets. The AVX2 matmul checks CPUID at runtime and falls back to scalar code if unsupported, so Ferrite **runs on any x86-64 CPU**; it's just slower without AVX2.
- **Memory** — the AcousticLeakNet demo reserves a 64 MB weight arena and a 32 MB activation arena at startup, so plan for ~100 MB. The core tests need far less (< 10 MB).

## Operating systems

| OS | Status |
|---|---|
| Linux / WSL | Fully supported (GCC or Clang + Ninja or Makefiles) |
| Windows | Supported via MinGW (MSYS2 preferred; Scoop MinGW may lack sanitizers) |
| macOS | Supported via Clang (Xcode Command Line Tools) |

> **Windows note.** Ferrite is not a native MSVC build. Use MSYS2 or Scoop MinGW. Windows binaries also need `libwinpthread-1.dll` (for `clock_gettime`); the build copies it next to every executable automatically.

## Verifying your environment

After installing the toolchain (see `INSTALL.md`), confirm:

```sh
gcc --version    # or: clang --version
cmake --version
ninja --version
```

All three should print version numbers. If you plan to use sanitizers, a quick check that they link:

```sh
printf 'int main(void){return 0;}\n' | gcc -x c -fsanitize=address,undefined -o /tmp/probe -
```

If this compiles and links cleanly, your toolchain supports ASan/UBSan.
