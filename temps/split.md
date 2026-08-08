# Ferrite — Two-Person Split Plan

## Bottom Line Up Front

This is how two people split every stage of the roadmap. The rule that makes it work: **agree on the seam first, then work in parallel.** Every stage has a seam — a header, a struct, a function signature, a file format — that both sides compile against before they start. Each person owns one half, tests their own half, and integration is a separate, explicit step.

**Roles are roles, not people.** A and B are roles; swap who plays which per stage so neither becomes a bottleneck or a single point of failure.

## Collaboration Rules

1. **Write the seam header first, together.** The interface is the contract. Freeze it before either side writes implementation. Changing it mid-stage costs more than writing it right.
2. **One reference path stays whole.** The naive engine and naive kernels are the correctness oracle. Only one person owns them at a time; the other builds against them.
3. **Integrate after every stage, not every task.** Both sides complete their half, then one shared session merges and runs the full test suite under ASan/UBSan. Green is the stage's done-when.
4. **One subsystem per commit, even with two people.** Both commit to the same repo with the same `feat(scope):` style. The git history does not know there are two people, and should not.
5. **Docs ship with code.** Whichever half lands updates its header comments and the relevant `temps/explain.md` section in the same commit.

---

## Stage 0 — Foundation

| Person | Owns |
|---|---|
| **A** | Project vision + architecture, folder structure, coding standards, documentation skeleton |
| **B** | CMake build system, logging, error handling framework, configuration system |

**Seam.** The folder structure and the `CMakeLists.txt` targets. A writes the layout and `AGENTS.md` conventions; B's CMake must build every directory A names. Agree on the directory list and binary names first.

**Integration.** `make`/CMake produces all test binaries from a clean clone; both run them. No code to merge yet, but the build is the contract.

---

## Stage 1 — Core Tensor Library

| Person | Owns |
|---|---|
| **A** | `FeTensor` struct, metadata, dtype table, memory ownership, indexing, strides, printing |
| **B** | Reshape, transpose, slicing, broadcasting, serialization, comparison |

**Seam.** `core/tensor.h` — the struct layout, `fe_dtype_size`, and the ownership rule (`owns_data`). Both sides code against the frozen header.

**Integration.** B's shape ops are tested against A's contiguous/ownership helpers. One test binary (`test_tensor`) covers both halves.

---

## Stage 2 — Math Backend

| Person | Owns |
|---|---|
| **A** | Scalar + vector operations, matrix multiplication, dot product |
| **B** | Reduction operators, elementwise operations, random generation, numerical stability helpers |

**Seam.** `ops/ops.h` kernel contract (read-only inputs, caller-allocated outputs, `FeStatus` returns). A's naive matmul is the reference; B's reductions reuse A's shape validation helpers.

**Integration.** A's matmul drives B's reductions in tests (row-sum via matmul with ones). `test_ops` is green for both halves.

---

## Stage 3 — Operator Library

| Person | Owns |
|---|---|
| **A** | Basic ops (Add, Sub, Mul, Div, Neg, Exp, Log, Pow) + Neural Network activations (ReLU, Sigmoid, Tanh, GELU, Softmax, Leaky ReLU, ELU, Swish) |
| **B** | Linear algebra (MatMul, GEMM, Transpose) + CNN (Conv2D, MaxPool, AvgPool, BatchNorm, LayerNorm, GroupNorm) + Sequence (Attention, MultiHead, Embedding, Positional Encoding) |

**Seam.** The `FeOpType` enum and `FeNode.attrs` union in `graph/graph.h`. A and B each extend the enum for their own ops; the attributes layout is agreed first.

**Integration.** B's GEMM is the consumer of A's matmul — B's GEMM output is tested against A's `fe_matmul`. Each side dispatches its own ops into `runtime/engine.c`; both cases must coexist.

---

## Stage 4 — Graph System

| Person | Owns |
|---|---|
| **A** | Graph class, node class, tensor registry, graph validation |
| **B** | Edge class, dependency tracking, topological sort, cycle detection, graph visualization |

**Seam.** The `FeGraph` struct. A owns the arrays (`nodes[]`, `tensors[]`) and the add functions; B owns `topo_order[]` and the producer/consumer maps. The struct layout is the contract.

**Integration.** A's validation consumes B's `topo_order` (every node input produced before use). `test_graph` covers both.

---

## Stage 5 — ONNX

| Person | Owns |
|---|---|
| **A** | Protobuf wire primitives, tensor parsing, weight parsing (initializer → weight arena) |
| **B** | Node + attribute parsing, graph build, graph verification, opset support |

**Seam.** `importer/onnx.h` — the `FePbReader` primitives are A's public API; B's parsers call them. Weights written by A are consumed by B's graph build.

**Integration.** `test_onnx` loads `tiny_mlp.onnx` end-to-end; A verifies the weight bytes, B verifies the node structure.

---

## Stage 6 — Execution Engine

| Person | Owns |
|---|---|
| **A** | Runtime context (`FeRuntime`), activation arena management, input/output binding, error propagation |
| **B** | Executor, operator dispatcher, graph execution, batch execution |

**Seam.** `runtime/engine.h` and the dispatch function signature `FeStatus dispatch(FeRuntime*, const FeNode*)`. B's dispatcher calls A's arena helpers.

**Integration.** `test_engine` runs the MLP end-to-end. B's dispatcher cases only pass if A's arena lifecycle is correct, and vice versa.

---

## Stage 7 — Testing

| Person | Owns |
|---|---|
| **A** | Tensor tests, operator tests, graph tests |
| **B** | Parser tests, runtime tests, regression tests, stress tests, fuzz testing |

**Seam.** The shared test conventions (assertion style, ASan/UBSan build) and the test-target names in the Makefile.

**Integration.** A's tests exercise B's subsystems at the boundaries and vice versa (A's tensor tests cover B's shape ops). The fuzz harness B writes feeds A's parser.

---

## Stage 8 — Benchmarking

| Person | Owns |
|---|---|
| **A** | Benchmark framework, timing utilities, latency benchmarks, throughput benchmarks |
| **B** | Comparisons (ONNX Runtime, tinygrad, TFLite), memory benchmarks, benchmark reports |

**Seam.** The timing helper (`fe_profiler_now_ns` based loop-timer) and the report format (environment header + table). B's comparison harness runs the same inputs A's framework times.

**Integration.** One suite, one README table. A's latency numbers and B's external comparisons use identical methodology.

---

## Stage 9 — Memory System

| Person | Owns |
|---|---|
| **A** | Arena allocator, memory pool, memory alignment |
| **B** | Tensor reuse, buffer planner (lifetime analysis + greedy assignment), cache-friendly layouts |

**Seam.** `core/allocator.h` — B's planner allocates through A's arena and returns offsets A's alignment rules must accept. The 64-byte alignment rule is part of the contract.

**Integration.** `test_planner` asserts the runtime footprint equals `plan->total_activation_bytes`, then A's arena peak confirms it.

---

## Stage 10 — SIMD

| Person | Owns |
|---|---|
| **A** | AVX2 vector class, SIMD Add, SIMD Mul, FMA optimization |
| **B** | SIMD GEMM, SIMD Conv, SIMD Activation |

**Seam.** `simd/matmul_avx2.h` and `fe_cpu_has_avx2()`. B's GEMM uses A's vector class; both follow the ops contract.

**Integration.** The naive kernels (owned per Stage 2) validate every SIMD kernel. B's GEMM is tested against A's SIMD building blocks and the scalar reference.

---

## Stage 11 — Parallel Runtime

| Person | Owns |
|---|---|
| **A** | Thread pool, task scheduler |
| **B** | Work stealing, parallel operators, NUMA awareness |

**Seam.** The thread-pool API (submit, wait, join). B's parallel kernels and work stealing run on A's pool.

**Integration.** ThreadSanitizer runs over the whole suite. B's parallel GEMM must match the sequential reference exactly.

---

## Stage 12 — Graph Optimization

| Person | Owns |
|---|---|
| **A** | Shape inference, constant folding, constant propagation |
| **B** | Dead node/tensor elimination, operator fusion, graph simplification, CSE |

**Seam.** The pass interface (`FeGraph *pass(FeGraph *g)` — in, out, no mutation) and the equivalence-test harness. A's shape inference must run before B's fusion passes.

**Integration.** The fused conv-batchnorm graph B produces is validated against A's folded/shape-inferred baseline. Every pass keeps output identical within tolerance.

---

## Stage 13 — Runtime Optimization

| Person | Owns |
|---|---|
| **A** | Memory planner integration (`fe_plan_apply`), static execution plan |
| **B** | Lazy execution, kernel caching, dynamic execution plan, prefetching |

**Seam.** The static-plan struct (flat array of {kernel fn, input ptrs, output ptr}). B's dynamic plan reuses A's static plan on shape change.

**Integration.** Static vs. non-static execution produces identical results; the profiler shows dispatch overhead gone.

---

## Stage 14 — Compiler Features

| Person | Owns |
|---|---|
| **A** | Intermediate Representation (IR), lowering passes |
| **B** | Optimization passes (IR-level), kernel generation, instruction scheduling |

**Seam.** The IR instruction format. A lowers the graph into it; B transforms and schedules it. The format is frozen first — the hardest decision in the stage.

**Integration.** Compiled output equals the reference engine's output (Stage 6 oracle) on the demo model.

---

## Stage 15 — Quantization

| Person | Owns |
|---|---|
| **A** | INT8, INT16, quantized kernels (int8→int32 GEMM) |
| **B** | FP16, BF16, calibration |

**Seam.** `FeQuantParams` and the scale policy. A's kernels consume the scales B's calibration produces; B's FP16 storage converts into buffers A's kernels read.

**Integration.** Float vs. INT8 vs. FP16 all run the demo model; accuracy delta, size, and speedup are all reported.

---

## Stage 16 — GPU

| Person | Owns |
|---|---|
| **A** | CUDA backend (init/run_op/sync), stream execution, memory transfer optimization |
| **B** | CUDA kernels, cuBLAS integration |

**Seam.** The backend interface (Stage 6 style) and the transfer policy (upload weights once, run on device, download once). A's backend calls B's kernels and cuBLAS.

**Integration.** CPU vs. GPU output match within tolerance on every test. The GPU path is an optional build target; CPU stays default.

---

## Stage 17 — Plugin System

| Person | Owns |
|---|---|
| **A** | Plugin API, compile-time registration, dynamic loading |
| **B** | Third-party kernels, backend plugins |

**Seam.** The registration API (`fe_register_op(name, validate, run)`), the API version constant, and the kernel contract. B's plugin ops implement A's interface.

**Integration.** A registered custom op runs end-to-end through the engine and survives the ONNX round-trip by name.

---

## Stage 18 — Developer Tools

| Person | Owns |
|---|---|
| **A** | CLI, model inspector, profiler |
| **B** | Python bindings, C API stabilization, graph visualizer, debugger |

**Seam.** The stable C API (the frozen public headers) and the version macro. B freezes the API; A's CLI is the first consumer of it; B's Python bindings wrap the same surface.

**Integration.** `ferrite run model.onnx` and the Python call produce identical output on the demo model.

---

## Stage 19 — Documentation

| Person | Owns |
|---|---|
| **A** | Professional README, architecture guide, API docs |
| **B** | Tutorials, examples, FAQ, contributing guide, coding guidelines |

**Seam.** The doc standards in `temps/documentation.md` (BLUF, docs ship with code, honesty over polish) and the shared voice. Both write to the same rules.

**Integration.** The cold-read test: someone builds and runs the demo in under a minute using only A's README + B's tutorial.

---

## Stage 20 — Infrastructure

| Person | Owns |
|---|---|
| **A** | GitHub Actions, unit-testing CI, code coverage |
| **B** | Formatting, clang-tidy, cppcheck, releases, versioning |

**Seam.** The build system and the CI matrix (sanitizer + AVX2 configurations). A's CI runs B's tools; B's version macro feeds A's release pipeline.

**Integration.** A PR with a test failure, format violation, or coverage drop is blocked. A `v0.1.0` tag produces artifacts from a clean runner.

---

## Stage 21 — Website

| Person | Owns |
|---|---|
| **A** | Landing page, documentation site |
| **B** | Benchmark dashboard, roadmap page, blog |

**Seam.** The static-site generator and the content pipeline from the repo's Markdown. A's site shell hosts B's dashboard and roadmap.

**Integration.** The benchmark numbers on B's dashboard match the committed Stage 8 results; the roadmap page renders `temps/roadmap.md`.

---

## Stage 22 — Research

| Person | Owns |
|---|---|
| **A** | Complexity analysis, memory analysis |
| **B** | Cache analysis, SIMD analysis, benchmark report, technical paper, whitepaper |

**Seam.** The benchmark data (Stage 8) and the `explain.md` complexity table. Both analyze the same measured runs.

**Integration.** Every claim in both halves links to a reproducible run. A's memory numbers come from the planner; B's SIMD numbers from the same benchmark suite.

---

## Stage 23 — Community

| Person | Owns |
|---|---|
| **A** | Issue templates, discussions, Good First Issues |
| **B** | Contribution templates, release notes, Discord |

**Seam.** The contributing guide (Stage 19) — both halves's process documents must agree with it and with `AGENTS.md`.

**Integration.** A newcomer opens a Good First Issue, follows B's PR template, and merges without hand-holding.

---

## Stage 24 — Future Runtime

| Person | Owns |
|---|---|
| **A** | Dynamic shapes, auto scheduler, auto tuning |
| **B** | JIT compilation, distributed inference, tensor/pipeline parallelism, model sharding |

**Seam.** The static execution plan (Stage 13). A's re-plan path and B's JIT/distributed paths all consume and extend it.

**Integration.** Pick one item per integration cycle. Single-node output is the reference for every parallel/distributed variant.

---

## Stage 25 — Ecosystem

| Person | Owns |
|---|---|
| **A** | Model Zoo, pretrained models, cloud inference |
| **B** | Training support, custom compiler tool, IDE/VSCode integration, package manager |

**Seam.** The release channel (Stage 20) and the demo model. A's zoo entries must run through B's packaging; B's compiler tool must load A's models.

**Integration.** Install → load a zoo model → run → correct output matching the golden tensor.

---

## Stage 26 — Dream Features

| Person | Owns |
|---|---|
| **A** | ARM NEON backend, automatic kernel generation |
| **B** | MLIR/LLVM integration, WebGPU/Metal/FPGA/RISC-V backends |

**Seam.** The backend interface (Stage 16) and the CPU reference. Every backend validates against the reference engine with a measured speedup.

**Integration.** One feature at a time, behind the optional-build rule. Each ships with its equivalence test and benchmark, or a documented rejection.

---

## How the Split Scales

- **Never split a seam.** The interface is the one thing both people share; it gets one owner at a time and changes are agreed, not authored.
- **Swap roles per stage.** Stage 1 A becomes Stage 5 B. Both people learn both halves of the stack and no one is irreplaceable.
- **Pair on the hard parts.** Planner integration (Stage 9), the IR format (Stage 14), and the backend interface (Stage 16) are pair-programmed sessions even though the implementation splits.
- **The reference path is the tie-breaker.** When the two halves disagree on correctness, the naive reference wins. That rule is non-negotiable.
