# Ferrite — Two-Person Split Plan

## Bottom Line Up Front

This is how two people split every stage of the roadmap. The split follows the ownership boundary already present in the code: **Person A owns the numerical/systems core** — `core/`, `ops/`, `simd/`, `quantization/` — every byte of tensor memory layout, every kernel, and every hardware-aware speed decision. **Person B owns the graph/IO side** — `importer/`, `graph/`, `runtime/`, `planner/` — the parsing, the algorithms, and the execution machinery around those kernels.

The rule that makes it work: **agree on the seam first, then work in parallel.** Every stage has a seam — a header, a struct, a function signature, a file format — that both sides compile against before they start. Each person owns one half, tests their own half, and integration is a separate, explicit step.

**Roles are locked, not swapped.** A is the kernel person; B is the systems person. The split works because each half stays inside its domain — A never parses ONNX, B never writes a kernel.

## Collaboration Rules

1. **Write the seam header first, together.** The interface is the contract. Freeze it before either side writes implementation. Changing it mid-stage costs more than writing it right.
2. **The naive kernels are A's oracle; the naive engine is B's.** Each reference path has one owner. The other person builds against it and never "fixes" it to match their own code.
3. **Integrate after every stage, not every task.** Both sides complete their half, then one shared session merges and runs the full test suite under ASan/UBSan. Green is the stage's done-when.
4. **One subsystem per commit, even with two people.** Both commit to the same repo with the same `feat(scope):` style. The git history does not know there are two people, and should not.
5. **Docs ship with code.** Whichever half lands updates its header comments and the relevant `temps/explain.md` section in the same commit.

---

## Stage 0 — Foundation

| Person | Owns |
|---|---|
| **A** | Vision for the numerical core: dtype set, tensor layout, alignment rules, coding standards for kernels |
| **B** | CMake build system, logging, error handling framework, configuration system, documentation skeleton |

**Seam.** The folder structure and the `CMakeLists.txt` targets. A names the directory layout and the kernel coding standards; B's CMake must build every directory A names. Agree on the directory list and binary names first.

**Integration.** `make`/CMake produces all test binaries from a clean clone; both run them. No code to merge yet, but the build is the contract.

---

## Stage 1 — Core Tensor Library

| Person | Owns |
|---|---|
| **A** | `FeTensor` struct, shape + metadata, dtype table, memory ownership, indexing, slicing, reshaping, broadcasting, printing, comparison |
| **B** | Tensor serialization (binary format, read/write round-trip, versioning) |

**Seam.** `core/tensor.h` — the struct layout, `fe_dtype_size`, and the ownership rule (`owns_data`). B serializes whatever A's struct holds, so the layout is frozen first.

**Integration.** `test_tensor` covers A's layout/ownership rules; B's serialization round-trips a tensor A built and reconstructs it byte-identical.

---

## Stage 2 — Math Backend

| Person | Owns |
|---|---|
| **A** | Scalar + vector operations, matrix multiplication, dot product, reduction operators, elementwise operations, numerical stability helpers |
| **B** | Random tensor generation (seeded, deterministic — the test-input generator both sides depend on) |

**Seam.** `ops/ops.h` kernel contract (read-only inputs, caller-allocated outputs, `FeStatus` returns). B's generator produces the tensors A's kernels consume; A's naive matmul is the reference.

**Integration.** `test_ops` is green for A's full math set. B's seeded generator feeds A's tests so failures reproduce exactly.

---

## Stage 3 — Operator Library

| Person | Owns |
|---|---|
| **A** | All kernel implementations: Basic (Add, Sub, Mul, Div, Neg, Exp, Log, Pow), NN activations (ReLU, Sigmoid, Tanh, GELU, Softmax, Leaky ReLU, ELU, Swish), Linear algebra (MatMul, GEMM, Transpose), CNN (Conv2D, MaxPool, AvgPool, BatchNorm, LayerNorm, GroupNorm), Sequence (Attention, MultiHead, Embedding, Positional Encoding) |
| **B** | Op registry in the IR: `FeOpType` enum, `FeNode.attrs` union in `graph/graph.h`, engine dispatch cases in `runtime/engine.c` |

**Seam.** `graph/graph.h` — the op enum and attribute layout B owns; A implements a kernel for every op B registers. The enum is the contract; a registered op without a kernel is a loud error, never a silent gap.

**Integration.** Every op A implements runs end-to-end through B's dispatcher in a graph-level test. `test_conv1d` and the norm/sequence tests pass through the engine.

---

## Stage 4 — Graph System

| Person | Owns |
|---|---|
| **B** | Graph class, node class, edge class, dependency tracking, graph validation, topological sort (Kahn's algorithm), cycle detection, graph visualization |
| **A** | Tensor registry inside the graph: `FeTensorEntry` metadata, shape/dtype consistency checks across edges |

**Seam.** The `FeGraph` struct. B owns `nodes[]`, `topo_order[]`, and the producer/consumer maps; A owns the `tensors[]` metadata each node references by index. The struct layout is the contract.

**Integration.** B's topo sort orders a graph whose tensor metadata A validated. `test_graph` covers both: every edge satisfies `u` before `v`, and every tensor referenced is consistent.

---

## Stage 5 — ONNX

| Person | Owns |
|---|---|
| **B** | Protobuf wire primitives, tensor parsing, node + attribute parsing, graph build, graph verification, opset support |
| **A** | The initializer → weight-arena path: alignment, 64-byte SIMD alignment, and `memcpy` of parsed weights into the weight arena |

**Seam.** `importer/onnx.h` and `core/allocator.h`. B parses and hands A raw weight bytes plus dims; A's arena helpers place them at aligned offsets. The weight-arena contract (allocated once at load, never reset) is frozen first.

**Integration.** `test_onnx` loads `tiny_mlp.onnx`; B verifies node structure, A verifies weight bytes landed aligned in the arena. Negative tests (truncated files, bad wire types) return errors on both sides.

---

## Stage 6 — Execution Engine

| Person | Owns |
|---|---|
| **B** | Executor, operator dispatcher, runtime context (`FeRuntime`), intermediate tensor management, error propagation, graph execution, batch execution |
| **A** | Kernel-call wrappers: shape/dtype validation before each kernel call and the mapping of node attributes to kernel arguments |

**Seam.** The dispatch signature in `runtime/engine.h` and the node→kernel mapping table. B owns the walk over `topo_order`; A owns the per-call validation that guarantees B's dispatcher never feeds a kernel bad shapes.

**Integration.** `test_engine` runs the MLP (Linear → ReLU → Linear → Softmax) end-to-end with exact expected output. B's lifecycle only passes if A's validation is correct, and vice versa.

---

## Stage 7 — Testing

| Person | Owns |
|---|---|
| **A** | Tensor tests, operator tests, kernel equivalence tests (SIMD vs. naive), fuzz targets for kernels (shape/dtype combinations) |
| **B** | Graph tests, parser tests, runtime tests, regression tests, stress tests (capacity limits: 512 nodes / 1024 tensors / 8 dims), fuzz targets for the ONNX parser |

**Seam.** The shared test conventions (assertion style, ASan/UBSan build) and test-target names in the Makefile. Each person's fuzz harness feeds the other's boundary: B fuzzes the parser A's weights flow through; A fuzzes kernels B dispatches.

**Integration.** The full suite is one command. B's stress test of 512 nodes exercises A's kernels at max load; A's dtype-fuzz proves no kernel accepts what B's graph would never produce.

---

## Stage 8 — Benchmarking

| Person | Owns |
|---|---|
| **A** | Benchmark framework, timing utilities, latency benchmarks, throughput benchmarks, GFLOPS reporting |
| **B** | External comparisons (ONNX Runtime, tinygrad, TFLite), memory benchmarks (planner-driven), benchmark reports and the README table |

**Seam.** The timing helper (loop-timer on `fe_profiler_now_ns`) and the report format (environment header + table). A's latency numbers and B's external comparisons use identical methodology and inputs.

**Integration.** One suite, one table. A reports the AVX2 speedup (~13.4× on 256×256) and the INT8 size/speed tradeoff; B's comparison harness confirms the same inputs against reference runtimes.

---

## Stage 9 — Memory System

| Person | Owns |
|---|---|
| **A** | Arena allocator, memory pool, memory alignment (64-byte SIMD rule), cache-friendly layouts |
| **B** | Tensor reuse, buffer planner (lifetime analysis + greedy interval scheduling), `fe_plan_apply` integration into `fe_runtime_run` |

**Seam.** `core/allocator.h` and the `FePlan` offsets. B's planner returns byte offsets into one activation buffer; A's arena must accept and align them. The alignment rule is part of the contract — B's reuse is only safe if A's alignment math is right.

**Integration.** `test_planner` asserts the runtime footprint equals `plan->total_activation_bytes`, and B's greedy reuse shows its savings (~60% vs. naive allocation) in `fe_plan_print`.

---

## Stage 10 — SIMD

| Person | Owns |
|---|---|
| **A** | AVX2 vector class, SIMD Add, SIMD Mul, SIMD GEMM, SIMD Conv, SIMD Activation, FMA optimization, `fe_cpu_has_avx2()` detection |
| **B** | Routing AVX2 into engine dispatch behind CPUID detection, and the SIMD-vs-scalar equivalence test harness at graph level |

**Seam.** `simd/matmul_avx2.h`. A's kernels are the fast path; B's dispatcher chooses them when `fe_cpu_has_avx2()` returns true. A's naive kernel stays the reference that validates every AVX2 result (max-error check).

**Integration.** A reports the ~13.4× speedup; B routes the engine so the demo model uses it automatically. Remainder-column handling (`N % 8 != 0`) is tested on both sides.

---

## Stage 11 — Parallel Runtime

| Person | Owns |
|---|---|
| **B** | Thread pool, task scheduler, work stealing |
| **A** | Parallel operators (row-split GEMM tiles across threads), NUMA awareness |

**Seam.** The thread-pool API (submit, wait, join). A's parallel kernels run on B's pool; A never builds its own scheduler. B's scheduler only hands A work that respects kernel input-read-only rules.

**Integration.** ThreadSanitizer runs over the whole suite. A's parallel GEMM must match the sequential reference exactly; B's scheduler must not change results at any thread count.

---

## Stage 12 — Graph Optimization

| Person | Owns |
|---|---|
| **B** | Shape inference (the pass that closes the demo-blocking gap), dead node/tensor elimination, graph simplification, common subexpression elimination, constant propagation, the pass framework + equivalence-test harness |
| **A** | Constant folding and operator-fusion math: running A's kernels over constant/weight subgraphs (e.g., conv-batchnorm fusion folds into weights) |

**Seam.** The pass interface (`FeGraph *pass(FeGraph *g)` — in, out, no mutation). B's shape inference must run before A's folding, since folding needs shapes. B owns the pipeline order; A owns the math inside the fold.

**Integration.** The fused conv-batchnorm graph produces identical output to the unfused one within tolerance. B's shape inference unblocks the end-to-end demo; A's folding shrinks its weight compute.

---

## Stage 13 — Runtime Optimization

| Person | Owns |
|---|---|
| **B** | Memory planner integration (`fe_plan_apply`), static execution plan, dynamic execution plan, lazy execution |
| **A** | Kernel caching (resolving each node to a concrete kernel function pointer), prefetching (cache hints in hot GEMM loops) |

**Seam.** The static-plan struct (flat array of {kernel fn, input ptrs, output ptr}). B builds the plan and its memory layout; A fills in the kernel function pointers and validates shapes once at plan time.

**Integration.** Static vs. non-static execution produces identical results; the profiler shows dispatch overhead gone. B's plan + A's cached kernels turn the run loop into one array walk.

---

## Stage 14 — Compiler Features

| Person | Owns |
|---|---|
| **B** | Intermediate Representation (IR), lowering passes, IR-level optimization passes, instruction scheduling |
| **A** | Kernel generation: emitting the specialized loop nests (fused op codegen) that lowered instructions map to |

**Seam.** The IR instruction format — frozen first, it is the hardest decision in the stage. B lowers the graph into IR and schedules it; A generates the kernels each instruction invokes.

**Integration.** Compiled output equals the reference engine's output (Stage 6 oracle) on the demo model. A's generated matmul/fused-act kernel is the one the schedule is measured on.

---

## Stage 15 — Quantization

| Person | Owns |
|---|---|
| **A** | INT8, INT16, FP16, BF16 storage + conversion, quantized kernels (int8→int32 accumulation GEMM) |
| **B** | Calibration pipeline: running the float model on a calibration set, collecting activation ranges, deriving scales |

**Seam.** `FeQuantParams` and the scale policy (symmetric, zero-point-free to start). B's calibration produces the scales A's kernels consume. A's kernels dequantize with `scale_A * scale_B`.

**Integration.** Float vs. INT8 vs. FP16 all run the demo model. A reports the ~75% weight-memory reduction (4× size) at measured error (~0.0006); B reports the accuracy delta and speedup end-to-end.

---

## Stage 16 — GPU

| Person | Owns |
|---|---|
| **A** | CUDA kernels, cuBLAS integration, memory transfer optimization (upload weights once, run on device, download once) |
| **B** | CUDA backend lifecycle (init/run_op/sync), backend interface, stream execution |

**Seam.** The backend interface and the transfer policy. A writes the kernels cuBLAS wraps and the transfer buffers; B owns the backend object that calls them. The CPU path stays the default; CUDA is an optional build target.

**Integration.** CPU vs. GPU output match within tolerance on every test; the benchmark shows the transfer-inclusive speedup or the backend is not done.

---

## Stage 17 — Plugin System

| Person | Owns |
|---|---|
| **B** | Plugin API, compile-time registration, dynamic loading, backend plugins |
| **A** | Third-party kernel plugins and kernel-contract enforcement at registration |

**Seam.** The registration API (`fe_register_op(name, validate, run)`) and the API version constant. B owns the registry and dispatch; A's plugin kernels implement B's interface and pass B's contract checks.

**Integration.** A registered custom op runs end-to-end through B's engine and survives the ONNX round-trip by name.

---

## Stage 18 — Developer Tools

| Person | Owns |
|---|---|
| **B** | CLI, model inspector, C API stabilization, graph visualizer, debugger, Python bindings |
| **A** | Profiler (per-kernel timing instrumentation), performance reporting from the CLI |

**Seam.** The stable C API and the profiler API. B's CLI calls A's profiler to print the per-op table; A's timing hooks fire inside B's dispatch loop.

**Integration.** `ferrite run model.onnx` and the Python call produce identical output; `ferrite profile` prints A's per-op timing table through B's CLI.

---

## Stage 19 — Documentation

| Person | Owns |
|---|---|
| **A** | Kernel + API docs (header comments), numerical architecture sections of `explain.md`, coding guidelines |
| **B** | README, tutorials, examples, FAQ, contributing guide |

**Seam.** The doc standards in `temps/documentation.md` (BLUF, docs ship with code, honesty over polish) and the shared voice.

**Integration.** The cold-read test: someone builds and runs the demo in under a minute using B's README + tutorial, then reads A's kernel docs and can trace a matmul through memory.

---

## Stage 20 — Infrastructure

| Person | Owns |
|---|---|
| **B** | GitHub Actions, unit-testing CI, formatting, clang-tidy, cppcheck, code coverage, releases, versioning |
| **A** | Performance-regression gating in CI (benchmarks fail the build on slowdown) and sanitizer profiles for the kernel suite |

**Seam.** The build system and the CI matrix. B's CI runs A's benchmark gates; A's gates use the versions B's release process tags.

**Integration.** A PR with a test failure, format violation, coverage drop, or benchmark regression is blocked. A `v0.1.0` tag produces artifacts from a clean runner.

---

## Stage 21 — Website

| Person | Owns |
|---|---|
| **B** | Landing page, documentation site, roadmap page, blog |
| **A** | Benchmark dashboard data and methodology (the numbers behind the speedup and quant stories) |

**Seam.** The static-site generator and the benchmark data format. B's dashboard renders A's committed benchmark results; the numbers cannot drift from the Stage 8 table.

**Integration.** A fresh visitor follows the site to a working build of the demo. B's roadmap page renders `temps/roadmap.md`; A's dashboard numbers match the committed benchmarks.

---

## Stage 22 — Research

| Person | Owns |
|---|---|
| **A** | Complexity analysis of kernels, cache analysis, SIMD analysis (vectorization ceiling vs. measured GFLOPS), benchmark report |
| **B** | Memory analysis (planner/lifetimes), complexity analysis of graph passes, technical paper, whitepaper |

**Seam.** The benchmark data (Stage 8) and the `explain.md` complexity table. Both analyze the same measured runs and link every claim to a reproducible command.

**Integration.** A's SIMD ceiling explains the 13.4×; B's memory analysis explains the 60% reuse savings. The paper and whitepaper combine both halves.

---

## Stage 23 — Community

| Person | Owns |
|---|---|
| **B** | Issue templates, discussions, contribution templates, release notes, Discord, Good First Issues |
| **A** | Good First Issues for the kernel/perf suite and triage of precision/performance bug reports |

**Seam.** The contributing guide (Stage 19) and `AGENTS.md` — both halves' process documents must agree with them.

**Integration.** A newcomer opens a Good First Issue (A's kernel tasks or B's parser tasks), follows B's PR template, and merges without hand-holding.

---

## Stage 24 — Future Runtime

| Person | Owns |
|---|---|
| **B** | Dynamic shapes, auto scheduler, distributed inference, pipeline parallelism orchestration, model sharding |
| **A** | Auto tuning (tile sizes MC/KC, kernel thresholds), tensor-parallel kernels (sharded GEMM), JIT kernel compilation |

**Seam.** The static execution plan (Stage 13). B's re-plan path and A's tuned kernels both consume and extend it. Single-node output is the reference for every parallel variant.

**Integration.** Pick one item per integration cycle. A's tuned/sharded GEMM matches B's single-node plan output within tolerance.

---

## Stage 25 — Ecosystem

| Person | Owns |
|---|---|
| **B** | Model Zoo, pretrained models, training support, IDE/VSCode integration, package manager, cloud inference |
| **A** | Custom compiler tool (the kernel-generation path from Stage 14) |

**Seam.** The release channel (Stage 20) and the demo model. A's compiler tool must load B's zoo models; B's packaging must ship A's compiled artifacts.

**Integration.** Install → load a zoo model → run → correct output matching the golden tensor, whether run from B's package or A's compiled artifact.

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
- **Roles are domain-locked.** A does not parse ONNX; B does not write kernels. Swapping would defeat the specialization that makes the split work. Balance within a stage happens by re-splitting at the seam, not by role swap.
- **Pair on the hard parts.** The planner integration (B's `fe_plan_apply`) depends on A's arena alignment. The shape-inference bug (B's current focus) needs A's tensor metadata rules. The static plan (B) bridges A's cached kernels. These are pair-programmed sessions even though the implementation splits.
- **The reference path is the tie-breaker.** When the two halves disagree on correctness, the naive reference wins. That rule is non-negotiable.

---

## Contribution Summary

**Person A owns the numerical/systems core** — `core/`, `ops/`, `simd/`, `quantization/`. That means the tensor memory layout, the matrix multiplication and activation kernels, the AVX2-vectorized SIMD code (13.4× measured speedup on 256×256), and INT8 quantization (75% memory reduction at 0.0006 error). It is precision-vs-speed, hardware-aware work — every decision that maps directly onto GPU/HPC developer skills.

**Person B owns the graph/IO side** — `importer/`, `graph/`, `runtime/`, `planner/`. That means the hand-written ONNX protobuf parser, the computation graph with Kahn's-algorithm topological sort and cycle detection, the execution engine that dispatches and runs nodes in order, and the memory planner that reuses buffers across tensors (60% savings via greedy interval scheduling). It is the classical algorithms and systems half of the project, currently focused on closing out the shape-inference bug that blocks the end-to-end demo.

| Stage | Person A (numerical core) | Person B (graph/IO) |
|---|---|---|
| 0 — Foundation | Numerical vision, layout + alignment rules, kernel standards | Build, logging, errors, config, doc skeleton |
| 1 — Tensor Library | Tensor struct, layout, ownership, indexing, slicing, reshape, broadcast | Serialization |
| 2 — Math Backend | Scalar/vector, matmul, dot, reductions, elementwise, stability | Random tensor generation |
| 3 — Operator Library | All kernels: basic, activations, GEMM, CNN, sequence | Op enum, attrs union, engine dispatch cases |
| 4 — Graph System | Tensor registry + shape/dtype consistency | Graph, nodes, topo sort, cycle detection, validation, viz |
| 5 — ONNX | Initializer → weight-arena path (aligned `memcpy`) | Protobuf primitives, tensor/node/attr parsing, graph build, opsets |
| 6 — Engine | Kernel-call wrappers: shape/dtype validation, attr mapping | Executor, dispatcher, runtime context, error propagation, batch |
| 7 — Testing | Tensor/op/kernel tests, kernel fuzz | Graph/parser/runtime tests, regression, stress, parser fuzz |
| 8 — Benchmarking | Framework, timing, latency/throughput, GFLOPS | External comparisons, memory benchmarks, reports |
| 9 — Memory System | Arena, pool, alignment, cache-friendly layouts | Planner, tensor reuse, greedy scheduling, `fe_plan_apply` |
| 10 — SIMD | Vector class, SIMD add/mul/GEMM/conv/act, FMA, CPUID | AVX2 dispatch routing, SIMD-vs-scalar graph tests |
| 11 — Parallel Runtime | Parallel kernels (row-split GEMM), NUMA | Thread pool, scheduler, work stealing |
| 12 — Graph Optimization | Constant folding, fusion math | Shape inference, dead elimination, CSE, propagation, pass pipeline |
| 13 — Runtime Optimization | Kernel caching, prefetching | Static/dynamic plan, lazy execution, planner integration |
| 14 — Compiler | Kernel generation (loop-nest codegen) | IR, lowering, IR optimizations, instruction scheduling |
| 15 — Quantization | INT8/INT16/FP16/BF16, quantized GEMM | Calibration pipeline (scale derivation) |
| 16 — GPU | CUDA kernels, cuBLAS, transfer optimization | Backend interface, backend lifecycle, streams |
| 17 — Plugins | Third-party kernel plugins, contract enforcement | Plugin API, registration, dynamic loading, backend plugins |
| 18 — Dev Tools | Profiler, per-kernel timing | CLI, model inspector, C API, viz, debugger, Python bindings |
| 19 — Documentation | Kernel/API docs, numerical architecture sections | README, tutorials, examples, FAQ, contributing guide |
| 20 — Infrastructure | Benchmark regression gates, kernel sanitizer profiles | CI, formatting, clang-tidy, cppcheck, coverage, releases |
| 21 — Website | Benchmark dashboard data + methodology | Landing page, docs site, roadmap, blog |
| 22 — Research | Kernel complexity, cache/SIMD analysis, benchmark report | Memory analysis, pass complexity, paper, whitepaper |
| 23 — Community | Kernel/perf good-first-issues, precision bug triage | Issues, templates, release notes, Discord, good-first-issues |
| 24 — Future Runtime | Auto tuning, tensor-parallel GEMM, JIT kernels | Dynamic shapes, auto scheduler, distributed inference, sharding |
| 25 — Ecosystem | Custom compiler tool (kernel generation) | Model Zoo, training support, IDE/VSCode, packaging, cloud |
| 26 — Dream Features | ARM NEON, automatic kernel generation | MLIR/LLVM, WebGPU/Metal/FPGA/RISC-V backends |

**The shape of it.** A's work is all precision-vs-speed and hardware awareness — kernels, memory layout, SIMD, quantization — the parts that answer "is it fast, and how fast is it really?" B's work is parsing, algorithms, and execution — the ONNX importer, the graph, the engine, the planner — the parts that answer "does it load real models and run them correctly?" The seams (tensor contract, op registry, dispatch signature, static plan, backend interface) are where the two halves meet, and they are never split.

**Where the halves touch today.** A's arena alignment is the floor B's planner offsets must respect. B's shape-inference bug is the current blocker for the end-to-end demo, and closing it unblocks A's benchmark and quant stories. When either half looks overloaded, re-split at the stage's seam rather than adding a third worker or crossing the domain boundary.
