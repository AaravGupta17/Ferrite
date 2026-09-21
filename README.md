# Ferrite

[![CI](https://github.com/AaravGupta17/Ferrite/actions/workflows/ci.yml/badge.svg)](https://github.com/AaravGupta17/Ferrite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Website](https://img.shields.io/badge/site-ferrite--c11.github.io-a51c30)](https://ferrite-c11.github.io/FerriteWebsite/)

**A neural-network inference runtime in 222 KB of C11, with zero dependencies.**

It parses ONNX protobufs with its own wire parser, plans activation memory with its
own lifetime-aware allocator, and runs hand-written kernels. No BLAS. No protobuf
library. No SIMD wrapper. Nothing but libc.

|  | Ferrite | ONNX Runtime |
|---|---|---|
| Binary | **222 KB** stripped | 18.4 MB (`onnxruntime.dll`) |
| Dependencies | **none** | many |
| MNIST accuracy | **97.52%** (9752/10000) | same model, same weights |
| Latency, 1 thread | 2.42 ms | 1.06 ms |
| Agreement | `max_abs ≈ 1e-8`, 3 of 7 models **bit-exact** | reference |

**83× smaller. 2.3× slower. Numerically identical.** Those numbers are measured, not
estimated, and every one of them has a command below that reproduces it.

---

## See it work

```sh
cmake -S . -B build -G Ninja
cmake --build build --target demo_mnist
./build/demo_mnist
```

```
                        ++@@##..
                      ..@@@@@@==
                    ..##@@@@@@==
                  ==##@@@@@@@@**==::
                  @@@@@@@@@@@@@@@@%%::
                **@@@@@@@@@@%%%%@@@@@@
              ::%%@@@@@@@@++--::++@@@@==
            ..%%@@@@@@%%++      ..%%@@@@==
            ..@@@@@@##..          ==@@@@@@..
            ..@@@@##..            ..##@@@@..

    Ferrite says: 0   (100.0% confident)   truth: 0  OK
    0.286 ms

---------------------------------------------
  10/10 correct on the embedded samples
  0.286 ms/inference (mean of 10)

  model load    0.6 ms
  weights       398 KB
  activations   1.5 KB peak
  dependencies  none
```

That model was trained in PyTorch, exported to ONNX, and is executed here by C code
that shares nothing with PyTorch or ONNX Runtime. **1.5 KB of activation memory** for
the whole forward pass.

Reproduce it end to end — train, export, run, and score the full test set:

```sh
pip install torch                    # torchvision not required
python tools/train_mnist.py          # trains, reports 97.52%, writes tests/mnist.onnx
cmake --build build --target mnist_eval
./build/mnist_eval tests/mnist.onnx build/mnist-data/test_images.bin build/mnist-data/test_labels.bin
```

```
Ferrite MNIST accuracy: 97.52% (9752/10000)
  0.303 ms/image over the full test set
  3300 images/s single-threaded
  activations 1.5 KB peak
```

Ferrite's accuracy matches PyTorch's to the image. That is the claim, and it is
Ferrite's own measurement over all 10,000 test images — not a number copied from the
training script.

---

## Build and test

```sh
cmake -S . -B build -G Ninja                  # GCC or Clang; C11
cmake --build build                           # 18 test binaries
ctest --test-dir build --output-on-failure    # 19 entries
```

Default flags are `-Wall -Wextra -fsanitize=address,undefined`. Sanitizer availability
is probed, and a dedicated CI leg fails the build if they are ever silently dropped.

> **Windows/MinGW:** binaries land as `*.exe` in `build/` and copy
> `libwinpthread-1.dll` beside themselves. Run through CTest or from `build/`.

---

## How it works

An `.onnx` file goes in; a flat array of kernel calls comes out.

1. **Parse** — a hand-rolled protobuf wire reader builds the graph. Unsupported ops
   fail loudly; a truncated or over-rank tensor is refused, never quietly reshaped.
2. **Optimize** — shape inference → simplify → constant-fold → Conv+BN fusion → CSE →
   dead-code elimination.
3. **Plan** — lifetime analysis assigns every activation an offset into one buffer.
   AcousticLeakNet's 32 MB of weights run in **896 KB** of activations, 22% below a
   naive allocation.
4. **Execute** — every node is resolved to a kernel pointer once. A same-shape rerun
   does *zero* re-analysis: no dispatch switch, no re-planning, just a walk down a
   flat array.

| Layer | Contents |
|---|---|
| `core/` | strided tensors, arena allocator, serialization, FP16/BF16, platform seam |
| `graph/` | computation-graph IR, tensor registry, Kahn topological sort |
| `ops/` | matmul/linear, elementwise, activations, GEMM, conv1d/conv2d (im2col), norms, pooling, attention |
| `simd/` | AVX2 tiled matmul + elementwise behind runtime CPUID, scalar fallback |
| `planner/` | tensor-lifetime analysis + greedy buffer reuse |
| `optim/` | shape inference, folding, fusion, CSE, DCE |
| `runtime/` | the engine and its static execution plan |
| `importer/` | the ONNX protobuf parser |
| `quantization/` | symmetric INT8/INT16, per-tensor and per-channel, dynamic and static |
| `compiler/`, `parallel/` | linear IR; thread pool + row-parallel GEMM (both opt-in) |

Dependencies point downward only.

---

## Correctness

Being small is worthless if the numbers are wrong. Ferrite is checked against ONNX
Runtime on every push:

```sh
pip install onnxruntime onnx numpy
python tools/golden_gen.py
python tools/golden_compare.py --run-model build/run_model
```

Seven models — MLP, mini-CNN, Conv+BN fusion, LayerNorm MLP, MNIST, and a real 32 MB
1D-CNN — run through both runtimes with identical seeded input. Worst observed
disagreement is `max_abs ≈ 1e-8`; three models are **bit-exact**.

Also guarding the code: mandatory ASan/UBSan, libFuzzer against the ONNX parser with a
persistent corpus, a coverage floor, and a perf gate diffing against a per-CPU
baseline.

---

## Performance, honestly

```sh
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DFERRITE_SANITIZE=OFF
cmake --build build-release --target bench_model
./build-release/bench_model
python tools/bench_onnxruntime.py     # the same model, same input, same timing
```

On AcousticLeakNet (AMD Ryzen 5 5600, single-threaded):

| | Ferrite | ONNX Runtime |
|---|---|---|
| Latency | 2.42 ms | **1.06 ms** |
| Binary | **222 KB** | 18.4 MB |

**ONNX Runtime is faster, and that is the expected result** — it is a mature runtime
with years of kernel tuning behind it. Ferrite's claim is that you can get within
~2.3× of it, numerically identical, in 1/83rd the binary and with no dependencies at
all.

Where the hand-written SIMD does pay off, against Ferrite's own scalar reference:

| Kernel | naive | AVX2 | ratio |
|---|---|---|---|
| MatMul 256³ | 6.92 ms | 1.45 ms | **4.78×** |
| Elementwise add, 1M floats | 0.75 ms | 0.19 ms | **3.92×** |
| GEMM base vs scalar transB | 22.09 ms | 1.41 ms | **15.65×** |

And where it does not: on the `[1 × 65536 × 128]` GEMV in AcousticLeakNet, the tiled
AVX2 kernel is **0.4×** — *slower* than scalar, because a single output row gives the
packing cost nothing to amortize against. The dispatcher routes `M == 1` to scalar for
exactly this reason. Full numbers and methodology: [`docs/benchmarks.md`](docs/benchmarks.md).

---

## It runs on hardware you have lying around

- **Raspberry Pi Zero W** — ARMv6 hard-float cross build; the full test suite runs
  under QEMU in CI. See [`docs/pi-zero-w-port.md`](docs/pi-zero-w-port.md).
- **ESP32** — an ESP-IDF component compiling the *device runtime* only: scalar kernels,
  planner, engine, quantization. The ONNX parser is compiled out; models are loaded
  from a pre-compiled FEMD artifact. Compile-gated in CI.
  See [`docs/esp32-port.md`](docs/esp32-port.md).

---

## Honest limitations

- **ONNX Runtime is ~2.3× faster.** Ferrite trades speed for size and simplicity.
- The ONNX surface is the 20+ mapped ops. Anything else fails loudly with
  `FE_ERR_SHAPE` — never silently.
- AVX2 only. No NEON, no AVX-512. NEON waits for a Pi Zero 2 W+ target.
- Capacities are fixed but configurable: 512 nodes / 1024 tensors / 8 dims by default,
  raised with `-DFE_MAX_NODES=`, `-DFE_MAX_TENSORS=`, `-DFERRITE_MAX_DIMS=`. Exceeding
  one is a loud load-time error naming the knob, never a truncated shape.
- The engine hot path is deliberately single-threaded. The thread pool and the linear
  IR compiler exist and are tested, but are opt-in.
- Calibration collects raw `max(|x|)` activation ranges; percentile smoothing is not
  built.
- ESP32 hardware validation is a documented manual step; CI is a compile gate.

---

## Where to start reading

- [`docs/guide.md`](docs/guide.md) — how every subsystem works and connects
- [`docs/deep-dive.md`](docs/deep-dive.md) — long-form internals, down to byte layouts
- [`tests/test_engine.c`](tests/test_engine.c) — the best end-to-end example
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — build, test, and commit conventions
- [ferrite-c11.github.io/FerriteWebsite](https://ferrite-c11.github.io/FerriteWebsite/) — the same
  material as a page, if you would rather scroll than read source

## License

MIT — see [LICENSE](LICENSE).
