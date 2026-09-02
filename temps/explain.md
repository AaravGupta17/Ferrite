# Ferrite — Project Guide

## Bottom Line Up Front

Ferrite is a zero-dependency neural network inference runtime in C11, built from first principles. It runs ONNX models using hand-written kernels. No BLAS. No protobuf library. No SIMD wrapper. The git history shows one subsystem per commit, each with its own test.

This guide explains each subsystem, how they connect, and how data flows through the system. Read it in order.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Core Data Structures (`core/`)](#core-data-structures)
3. [Logging and Errors (`core/log.h`, `core/types.h`)](#logging-and-errors)
4. [Computation Graph (`graph/`)](#computation-graph)
5. [Operator Kernels (`ops/`)](#operator-kernels)
6. [SIMD Matmul (`simd/`)](#simd-matmul)
7. [Memory Planner (`planner/`)](#memory-planner)
8. [Execution Engine (`runtime/`)](#execution-engine)
9. [ONNX Importer (`importer/`)](#onnx-importer)
10. [Quantization (`quantization/`)](#quantization)
11. [Profiling and Benchmarks (`tools/`)](#profiling-and-benchmarks)
12. [Tests and Build System](#tests-and-build-system)
13. [End-to-End Data Flow](#end-to-end-data-flow)
14. [Known Limitations](#known-limitations)

---

## Architecture

**Bottom line.** Ferrite is a layered runtime, like a miniature ONNX Runtime or PyTorch. Each layer depends only on the layers below it.

```
┌─────────────────────────────────────────────────────────────┐
│                     importer/  (ONNX)                       │
│            hand-rolled protobuf wire parser                 │
└───────────────────────────┬─────────────────────────────────┘
                            │ builds
┌───────────────────────────▼─────────────────────────────────┐
│                     graph/  (IR)                            │
│          FeGraph: nodes + tensor registry                   │
│          topological sort (Kahn's algorithm)                │
└───────────────────────────┬─────────────────────────────────┘
                            │ analyzes
┌───────────────────────────▼─────────────────────────────────┐
│                    planner/  (memory)                       │
│        lifetime analysis + greedy buffer reuse              │
└───────────────────────────┬─────────────────────────────────┘
                            │ executes
┌───────────────────────────▼─────────────────────────────────┐
│                     runtime/  (engine)                      │
│        dispatches nodes to kernels in topo order            │
└───────────────────────────┬─────────────────────────────────┘
                            │ calls
┌───────────────────────────▼─────────────────────────────────┐
│    ops/  +  simd/          core/  (tensors, arenas)         │
│    kernels                 universal data structures        │
└─────────────────────────────────────────────────────────────┘
```

`tools/` (profiler, benchmarks) and `tests/` sit on top and pull pieces together.

**Design rules.**

- **No allocation in hot paths.** Kernels never `malloc`. Only `fe_conv1d` allocates, and it frees its scratch after use.
- **Views over copies.** Transpose and reshape never copy data.
- **Arenas over per-tensor malloc.** Weights live forever in one arena. Activations reset after every inference.
- **Correctness first, speed second.** The naive matmul validates the AVX2 kernel.
- **One test per subsystem.** Each binary builds with `-fsanitize=address,undefined`.

---

## Core Data Structures

### `core/types.h` — Shared types

The foundation everything uses:

- `FeDtype` — `DTYPE_FLOAT32` (0), `DTYPE_INT8` (1), `DTYPE_INT32` (2).
- `FeStatus` — error codes: `FE_OK` and `FE_ERR_NULL/SHAPE/DTYPE/NOMEM/BOUNDS`.
- `fe_dtype_size()` — inline helper, bytes per element (4/1/4).
- `FERRITE_MAX_DIMS` — 8, the maximum tensor rank.

### `core/tensor.h` / `core/tensor.c` — `FeTensor`

**Bottom line.** A tensor is a strided view into a flat memory buffer.

```c
typedef struct {
    void    *data;
    FeDtype  dtype;
    int      ndim;
    int      shape  [FERRITE_MAX_DIMS];
    int      strides[FERRITE_MAX_DIMS];  /* in elements, not bytes */
    size_t   nbytes;
    bool     owns_data;
} FeTensor;
```

Key concepts:

- **Strides** are row-major (C order): `strides[last] = 1`, `strides[i] = strides[i+1] * shape[i+1]`. An element at index `idx` lives at byte offset `Σ idx[d] * strides[d] * elem_size`.
- **Ownership.** `owns_data == true` means the tensor frees its buffer in `fe_tensor_free()`. Views set `owns_data = false` and never free the parent's data.
- **`fe_tensor_alloc`** mallocs a fresh buffer. **`fe_tensor_from_data`** wraps a caller-owned buffer.
- **`fe_tensor_transpose`** swaps `shape` and `strides` for two axes. O(1), no data movement.
- **`fe_tensor_reshape`** succeeds only when the tensor is contiguous and the element count holds. Returns a view; no copy.
- **`fe_tensor_contiguous`** repacks a non-contiguous tensor element-by-element into a fresh buffer.
- **`fe_tensor_copy`** uses fast `memcpy` when both sides are contiguous; otherwise falls back to strided element-wise copy.
- `fe_tensor_get_f32` / `fe_tensor_set_f32` are debug-only. Not the hot path.

### `core/allocator.h` / `core/allocator.c` — `FeArena`

**Bottom line.** An arena is a bump-pointer allocator. Reset it and all memory frees at once.

```c
typedef struct {
    unsigned char *base;
    size_t         size;
    size_t         offset;  /* current bump pointer */
    size_t         peak;    /* high-water mark      */
} FeArena;
```

- `fe_arena_alloc(size, align)` is O(1): aligns the **actual pointer address** (not just the offset — a subtle fix in git history), checks capacity, bumps the offset.
- `fe_arena_reset()` returns the bump pointer to zero. It preserves `peak` for profiling.
- `fe_arena_alloc_tensor()` allocates both the `FeTensor` metadata and its data buffer, aligned to 64 bytes for SIMD.
- **Two intended arenas.** A *weight arena*, allocated once at load, never reset. An *activation arena*, reset after every inference. This is the classic inference-runtime strategy.

### `core/tensor_ser.h` / `core/tensor_ser.c` — serialization

**Bottom line.** Tensors round-trip to disk as a versioned binary format. What is stored is the *logical contents*, not the struct.

```
magic "FETN" | u32 version | u32 dtype | u32 ndim | int32 shape[ndim]
| u64 data_len | payload bytes (row-major, contiguous)
```

- All integers are **little-endian regardless of host**, written through explicit byte-shuffling helpers — the file is identical on any machine.
- Strides, data pointer, and ownership are never serialized; they are layout details. Saving a non-contiguous tensor repacks it first via `fe_tensor_contiguous()`, so views save their logical contents.
- Loading always produces a fresh **owning, row-major tensor** (`owns_data = true`), freed with `fe_tensor_free()`.
- Readers are strict for v1: wrong magic, newer version, unknown dtype, impossible shape, a `data_len` that disagrees with the computed byte count, truncated payloads, and trailing garbage all fail loudly — `FE_ERR_IO` for structural corruption, `FE_ERR_DTYPE`/`FE_ERR_SHAPE` where the field names the problem. No partial state leaks: the output pointer is untouched on failure.
- Versioning starts at 1 on purpose: readers reject `version > 1`, so v2 can change the layout without old builds misreading it.

---

## Logging and Errors

**Bottom line.** Two small mechanisms cover all diagnostics: `FeStatus` return codes and a level-filtered stderr logger. Neither appears in hot paths.

- **Errors.** Every fallible public function returns `FeStatus`; `FE_OK` means full success with all outputs written. The codes (`core/types.h`) name the failure class: `FE_ERR_NULL/SHAPE/DTYPE/NOMEM/BOUNDS/IO`. Constructors that return pointers signal failure with `NULL`. Callees never free caller memory or partially mutate outputs on error.
- **Logging.** `fe_log_debug/info/warn/error(...)` macros (`core/log.h`) write `[ferrite LEVEL file:line] msg` to stderr. Levels filter at compile time via `FERRITE_LOG_LEVEL` (default WARN) — filtered messages compile to nothing.
- **Policy.** Logging is for load/init-time diagnostics only: model import, runtime init, fatal dispatch errors. Kernels never log and never allocate (hot-path rule).

---

## Computation Graph

### `graph/graph.h` / `graph/graph.c` — `FeGraph`

**Bottom line.** The graph separates structure from data. At build time, tensors are only shape/dtype metadata. The planner or arena assigns actual data pointers later.

```c
typedef struct {
    FeNode        nodes  [FE_MAX_NODES];
    FeTensorEntry tensors[FE_MAX_TENSORS];
    int           n_nodes;
    int           n_tensors;
    int           topo_order[FE_MAX_NODES];  /* execution order */
    int           topo_valid;
} FeGraph;
```

- **Ops** (`FeOpType`): `INPUT`, `OUTPUT`, `MATMUL`, `LINEAR`, `RELU`, `SOFTMAX`, `CONV1D`, `BATCHNORM`, `ADD`, `FLATTEN` plus the Stage 3 families — Exp, Log, Pow, Sub/Mul/Div/Neg, Sigmoid/Tanh/GELU/LeakyReLU/ELU/Swish, GEMM, Transpose, Conv2D, MaxPool, AvgPool, LayerNorm/GroupNorm, Attention/Multi-Head Attention/Embedding/Positional Encoding. The `attrs` union carries op-specific attributes (softmax axis, conv stride/pad, norm eps, gemm trans/alpha/beta).
- **Nodes** reference tensors by **index** into the registry, not by pointer.
- **Tensor entries** (`FeTensorEntry`) hold name, shape, dtype, an `is_weight` flag, and a `FeTensor *tensor` that stays `NULL` until memory is assigned.
- **Capacity limits**: 512 nodes, 1024 tensors, 8 inputs / 4 outputs per node, 64-char names. Fixed capacity — typical of embedded runtimes.

### Topological sort (Kahn's algorithm)

`fe_graph_topo_sort()` computes a valid execution order in O(N + E):

1. Build a `producer` map: which node outputs each tensor.
2. Compute in-degrees: count inputs produced by *another* node. Weights and graph inputs do not count.
3. Queue all in-degree-0 nodes.
4. Pop a node, append to `topo_order`, decrement the in-degree of every consumer, queue newly-zeroed nodes.
5. If fewer than `n_nodes` were ordered, the graph has a cycle → `FE_ERR_SHAPE`.

This order guarantees every node runs only after its inputs are ready.

---

## Operator Kernels

### `ops/ops.h` — The kernel contract

```c
/* - Inputs are read-only
 * - Output tensor is caller-allocated with correct shape
 * - Returns FE_OK on success, error code otherwise
 * - No memory allocation inside kernels */
```

Every kernel validates pointers, dtypes, and shapes before touching data. This makes the engine's dispatch trivial.

### `ops/matmul.c` — `fe_matmul` and `fe_linear`

The naive matmul exists to **validate correctness** before the SIMD kernel replaces it:

```c
for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++) {
        float a_ik = a[i * K + k];
        for (int j = 0; j < N; j++)
            c[i * N + j] += a_ik * b[k * N + j];
    }
```

**The core performance problem.** A is read row-major (cache-friendly). B is read column-major (stride-N, cache-unfriendly). The `simd/` module fixes this with tiling and packing. `fe_linear` is just `fe_matmul` followed by `fe_bias_add`.

### `ops/activations.c` — relu, softmax, bias_add

- **`fe_relu`** — elementwise `max(0, x)`. Supports in-place.
- **`fe_softmax`** — numerically stable along the **last axis**:
  1. Find `max_val` per row to prevent `exp()` overflow.
  2. Compute `exp(x - max_val)` and accumulate the sum.
  3. Normalize by `1/sum`.
  - `rows` is the product of all dims except the last; `cols` is the last dim. Works for any rank ≥ 1.
- **`fe_bias_add`** — `out = in + bias`. The bias matches the last dimension and broadcasts over all rows.

### `ops/conv1d.c` — 1D convolution via im2col + matmul

**Bottom line.** Convolution becomes a matrix multiply so GEMM kernels do the work.

1. **`fe_im2col`**: for each batch element, copy each input patch `[C_in, K]` at every output position into a column. Produces a `[C_in*K, L_out]` matrix. Zero the buffer first, so padding needs no special handling.
2. Reshape the `[C_out, C_in, K]` weight to `[C_out, C_in*K]` — a view, no copy — via `fe_tensor_reshape`.
3. Per batch element: `out_slice = W @ col`, then add bias if provided.

Output length: `L_out = (L - K + 2*pad)/stride + 1`.

---

## SIMD Matmul

### `simd/matmul_avx2.h` / `simd/matmul_avx2.c`

**Bottom line.** The performance matmul uses tiling, packing, and 8-wide FMA to solve the cache problem in the naive kernel.

- **Tile sizes.** `MC = 64` (rows), `KC = 256` (K-slice). The output is processed in `mc × kc` tiles so the working set fits in cache.
- **Packing.** A tiles pack row-major into a contiguous `A_tile[actual_mc][actual_kc]`. B tiles pack into `B_tile[k][8]` so the innermost loop reads 8 contiguous floats.
- **Kernel** (`kernel_mc_nr`): load 8 accumulator rows into `__m256` registers. For each `k`, broadcast `A[i][k]` and do a fused multiply-add (`_mm256_fmadd_ps`) against the 8-wide B row. That computes 8 output elements per instruction.
- **Remainder.** Columns where `N % 8 != 0` fall back to a scalar loop.
- **Detection.** `fe_cpu_has_avx2()` uses CPUID (`__cpuid_count(7, 0, ...)`, EBX bit 5) at runtime.

`tools/bench_avx2.c` reports a ~13.4× speedup over the naive loop for 256×256 matmuls.

---

## Memory Planner

### `planner/memory_planner.h` / `planner/memory_planner.c`

**Bottom line.** The planner computes when each tensor is alive and assigns non-overlapping tensors to the same buffer. This minimizes total activation memory.

**Step 1 — Lifetime analysis** (`compute_lifetimes`):
Walk the graph in topological order. For each step `s`:
- Every output tensor gets `first_use = s`.
- Every input tensor's `last_use` updates to `s`.

Weights are excluded (they live in the weight arena). The graph's model input is excluded too (the caller provides it).

**Step 2 — Greedy buffer assignment** (`fe_plan_memory`):
A greedy interval-graph coloring problem, O(n²):
- Process tensors by `first_use`.
- A *free pool* tracks released regions (`offset`, `size`, `freed_at`).
- For each tensor, find the smallest free slot that (a) freed before this tensor's `first_use` and (b) fits after 64-byte alignment. Reuse it. Otherwise allocate at the high-water mark.
- When a tensor dies (`last_use`), its slot returns to the pool.

**Result.** A `FePlan` with a per-tensor byte `offset` into one activation buffer and a `total_activation_bytes`. `fe_plan_print` reports savings vs. naive allocation.

**Step 3 — Apply** (`fe_plan_apply`):
Write tensor metadata pointing into the activation buffer at the planned offsets.

---

## Execution Engine

### `runtime/engine.h` / `runtime/engine.c`

**Bottom line.** `FeRuntime` ties the graph, arenas, and kernels together.

```c
typedef struct {
    FeGraph    *graph;
    FeArena     weight_arena;
    FeArena     activation_arena;
    FeProfiler *profiler;
} FeRuntime;
```

**Lifecycle.**

1. **`fe_runtime_init`** — stores the graph, initializes both arenas, runs the topological sort if needed.
2. **`fe_runtime_alloc_weights`** — allocates every `is_weight` tensor from the weight arena, once, at load.
3. **`fe_runtime_run`** — one inference:
   - Reset the activation arena.
   - Clear all non-weight tensor pointers.
   - Bind the caller's input tensor to the `FE_OP_INPUT` node's output.
   - `alloc_activations()` allocates every non-weight tensor from the activation arena (bump pointer, so cheap).
   - Walk `topo_order`, dispatching each node to its kernel via a `switch` on `node->op`, using the `IN(i)` / `OUT(i)` macros (tensor index → tensor pointer).
   - Copy the output node's result into the caller's buffer.
4. **`dispatch_node`** wraps every op call with `fe_profiler_now_ns()` timing when a profiler is attached.

**Current state.** The engine uses the arena bump allocator directly for activations. It does **not** consume the planner's `FePlan`. The planner exists as a standalone, tested subsystem. Wiring `fe_plan_apply` into the runtime is an obvious integration task.

**Dispatched ops:** every `FeOpType` in the enum now has a `switch` case in `dispatch_node`. Core ops: `MATMUL`, `LINEAR`, `RELU`, `SOFTMAX`, `ADD`, `FLATTEN`, `CONV1D`, `BATCHNORM`. Stage 3: basic math (`SUB`/`MUL`/`DIV`/`NEG`/`EXP`/`LOG`/`POW`), activations (`SIGMOID`/`TANH`/`GELU`/`LEAKY_RELU`/`ELU`/`SWISH`), linear algebra (`GEMM`, `TRANSPOSE`), CNN (`CONV2D`, `MAXPOOL`, `AVGPOOL`, `LAYERNORM`, `GROUPNORM`), and sequence (`ATTENTION`, `MULTIHEAD_ATTN`, `EMBEDDING`, `POSITIONAL_ENCOD`). No op is silently skipped.

---

## ONNX Importer

### `importer/onnx.h` / `importer/onnx.c`

**Bottom line.** Ferrite parses ONNX `.onnx` files with a hand-written protobuf wire-format parser. No protobuf compiler or library.

**Protobuf wire primitives** (exposed for testing):
- `fe_pb_varint` — LEB128 varint decoding.
- `fe_pb_tag` — reads the tag; splits it into `field_number` (tag >> 3) and `wire_type` (tag & 7).
- `fe_pb_skip` — skips a field by wire type (0=varint, 1=64-bit, 2=length-delimited, 5=32-bit).
- `fe_pb_bytes` — reads a length-delimited blob.

**ONNX structure** (ModelProto → GraphProto → NodeProto/TensorProto):
- Top level: find `field 7` (graph); skip everything else.
- `parse_graph`: `field 1` = node (repeated NodeProto), `field 5` = initializer (repeated TensorProto), `field 11` = input ValueInfo (skipped — the test harness infers shapes instead).
- `parse_node`: reads input names (`field 1`), output names (`field 2`), node name (`field 3`), and `op_type` (`field 4`). Maps the op string to an `FeOpType` via `op_type_from_string`. **Unsupported ops are logged and skipped**, keeping the load resilient.
- `parse_initializer`: reads dims (`field 1`), data_type (`field 2`, assumed float32), name (`field 8`), and `raw_data` (`field 9`, packed float32 bytes). Registers the tensor as a weight and `memcpy`s its data into the weight arena.

`find_or_add_tensor` deduplicates tensors by name, so one graph entry serves every node referencing the same tensor.

After parsing, the graph is topologically sorted and validated before returning `FE_OK`.

---

## Quantization

### `quantization/quant.h` / `quantization/quant.c`

**Bottom line.** Post-training INT8 quantization with **symmetric per-tensor** scaling.

- `FeQuantParams` = `{ float scale; int zero_point; }`. `zero_point` is always 0 (symmetric).
- **`fe_quantize`** — `scale = max(|x|)/127`, then `q = clamp(round(x/scale), -127, 127)`.
- **`fe_dequantize`** — `x = q * scale`.
- **`fe_matmul_int8`** — the quantized GEMM pipeline:
  1. Quantize A and B to INT8.
  2. Multiply with **int32 accumulation** (protects against overflow).
  3. Dequantize the result with the combined scale `pA.scale * pB.scale`.
- **`fe_quantize_weights`** — converts all weight tensors to INT8 and stores their scales.

This is the classic symmetric INT8 scheme used in early quantized ONNX runtimes. The natural improvement is **per-channel scales** for convolutional weights.

---

## Profiling and Benchmarks

### `tools/profiler.h` / `tools/profiler.c`

A per-operator timing profiler:

- `fe_profiler_now_ns()` uses `clock_gettime(CLOCK_MONOTONIC)`.
- `fe_profiler_record(name, ns)` accumulates total time and call count per operator (up to 64 ops).
- `fe_profiler_print()` sorts records by total time (insertion sort), computes each op's share of total time, and prints a formatted table.
- `fe_profiler_reset()` zeroes counters between inference runs.

The engine hooks this around every `dispatch_node` call, so a report looks like:

```
Operator   Calls  Total(ms)   Avg(ms)  % Time
linear0        1      0.423     0.423   78.1%
relu0          1      0.065     0.065   12.0%
...
```

### Benchmark tools

- `tools/bench_matmul.c` — times the naive `fe_matmul` for 256×256, prints ms/run and GFLOPS (using `2·N³` FLOPs).
- `tools/bench_avx2.c` — runs `fe_matmul` and `fe_matmul_avx2`, **verifies the AVX2 result against the naive one** (max error check), then prints both GFLOPS and the speedup ratio.

---

## Tests and Build System

**CMake is canonical.** One command configures and builds everything; CTest runs the suite:

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```

The build produces one static library per subsystem (`libferrite_core`, `libferrite_graph`, …) linked strictly downward, one test binary per subsystem, and opt-in benchmarks (`--target bench_avx2`). Sanitizers are probed at configure time and enabled when the toolchain ships them (`FERRITE_SANITIZE=OFF` to force off). On Windows/MinGW each executable copies the matching `libwinpthread-1.dll` beside itself so binaries do not resolve a stale copy from PATH.

A Unix-oriented `Makefile` with equivalent targets is kept for WSL during the transition.

| Target | Covers |
|---|---|
| `test_tensor` | strided tensor ops, reshape/transpose/contiguous |
| `test_allocator` | arena allocation, alignment |
| `test_ops` | matmul, linear, relu, softmax, bias_add |
| `test_graph` | graph building + topological sort |
| `test_engine` | end-to-end 2-layer MLP inference |
| `test_onnx` | loads `tests/tiny_mlp.onnx` |
| `test_planner` | lifetime analysis + buffer reuse |
| `test_conv1d` | im2col + conv1d |
| `test_profiler` | timing/recording |
| `test_quant` | INT8 quantization round-trip |
| `test_tensor_ser` | serialization round-trips + corruption rejection |
| `bench_matmul` / `bench_matmul_avx2` / `bench_avx2` | performance measurements |

Build flags:

```
-std=c11 -D_POSIX_C_SOURCE=199309L -Wall -Wextra -fsanitize=address,undefined -g
```

Every test runs under AddressSanitizer + UndefinedBehaviorSanitizer (where available). AVX2 targets add `-O3 -mavx2 -mfma`.

`tests/test_engine.c` is a good end-to-end example: it builds a `1→4→8→3` MLP by hand (Linear → ReLU → Linear → Softmax), sets uniform weights (`0.1f`), and verifies the softmax output is exactly `1/3` per class. `tests/tiny_mlp.onnx` exercises the importer against a real ONNX file.

---

## End-to-End Data Flow

Tracing one full inference through every layer:

1. **Load** (`importer/`): `fe_onnx_load` reads the `.onnx`, parses protobuf, builds the `FeGraph` (nodes + tensor registry), topologically sorts it, and copies all weights into the weight arena.
2. **Plan** (`planner/`): lifetime analysis finds each tensor's first/last use. Greedy assignment overlaps non-conflicting tensors in one activation buffer. The plan reports minimum buffer size and savings.
3. **Init runtime** (`runtime/`): arenas bind to caller buffers; weight tensors allocate from the weight arena.
4. **Run** (`runtime/`):
   - Reset the activation arena.
   - Bind the input tensor.
   - Allocate all activation tensors (bump pointer).
   - For each node in topological order, dispatch to the matching kernel in `ops/` (or `simd/` for the fast matmul), timing each call if profiling.
   - Copy the final output into the caller's buffer.
5. **(Optional) Quantize** (`quantization/`): convert weights to INT8 to shrink the model and use int8→int32 GEMMs.
6. **(Optional) Profile** (`tools/`): print per-operator timing to find the hotspot — almost always matmul.

---

## Known Limitations

- **Planner not integrated.** `fe_runtime_run` uses the arena directly. `fe_plan_apply` exists but nothing calls it. Wiring it in would give deterministic, minimal activation memory.
- **Limited opset.** Unsupported ONNX ops are silently skipped. Only models made of supported ops load correctly.
- **No shape inference.** Shapes must exist in the ONNX file or be filled in by the caller. The importer skips value-info shape data.
- **Quantization is per-tensor, not per-channel.** Per-channel scales would improve INT8 accuracy.
- **AVX2 needs `N % 8 == 0`** for the vectorized path. Remainders fall back to scalar code.
- **Fixed capacities** (512 nodes / 1024 tensors / 8 dims). Fine for small models; would need dynamic growth for larger ones.
- **Single-threaded.** No parallel execution across independent subgraphs.
