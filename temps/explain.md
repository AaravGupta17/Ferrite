# Ferrite — In-Depth Project Explanation

Ferrite is a minimal neural-network inference runtime written in C11, built "from first principles". It has **zero external dependencies** — no BLAS, no protobuf library, no SIMD wrapper — everything is hand-implemented. The git history shows it was built incrementally, one subsystem per commit, each with its own test binary.

This document walks through every subsystem, how they connect, and how data flows through the system.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Core Data Structures (`core/`)](#core-data-structures)
3. [Computation Graph (`graph/`)](#computation-graph)
4. [Operator Kernels (`ops/`)](#operator-kernels)
5. [SIMD Matmul (`simd/`)](#simd-matmul)
6. [Memory Planner (`planner/`)](#memory-planner)
7. [Execution Engine (`runtime/`)](#execution-engine)
8. [ONNX Importer (`importer/`)](#onnx-importer)
9. [Quantization (`quantization/`)](#quantization)
10. [Profiling & Benchmarks (`tools/`)](#profiling--benchmarks)
11. [Tests & Build System](#tests--build-system)
12. [End-to-End Data Flow](#end-to-end-data-flow)
13. [Known Limitations & Next Steps](#known-limitations--next-steps)

---

## High-Level Architecture

Ferrite is layered like a miniature version of a production inference runtime (ONNX Runtime, PyTorch, etc.):

```
┌─────────────────────────────────────────────────────────────┐
│                     importer/  (ONNX)                       │
│            hand-rolled protobuf wire parser                 │
└───────────────────────────┬─────────────────────────────────┘
                            │ builds
┌───────────────────────────▼─────────────────────────────────┐
│                     graph/  (IR)                            │
│          FeGraph: nodes + tensor registry                    │
│          topological sort (Kahn's algorithm)                 │
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

Layers above depend only on layers below. The `core/` layer is shared by everything. `tools/` (profiler, benchmarks) and `tests/` sit on top and pull pieces together.

### Design philosophy

- **No allocation in hot paths.** Kernels never `malloc`; outputs are caller-allocated. The only kernel that allocates (`fe_conv1d`) allocates a temporary im2col scratch buffer and frees it.
- **Views over copies.** Tensors are strided views — transpose and reshape never copy data.
- **Arenas instead of per-tensor malloc.** Weights live forever in one arena; activations are reset after every inference.
- **Correctness first, speed second.** The naive matmul exists as a reference implementation to validate the AVX2 kernel against.
- **Every subsystem has a test.** One test binary per module, all built with `-fsanitize=address,undefined`.

---

## Core Data Structures

### `core/types.h` — Shared types

The foundation everything uses:

- `FeDtype` — `DTYPE_FLOAT32` (0), `DTYPE_INT8` (1), `DTYPE_INT32` (2).
- `FeStatus` — error codes: `FE_OK`, and `FE_ERR_NULL/SHAPE/DTYPE/NOMEM/BOUNDS`.
- `fe_dtype_size()` — inline helper returning bytes per element (4/1/4).
- `FERRITE_MAX_DIMS` — 8, the maximum tensor rank supported.

### `core/tensor.h` / `core/tensor.c` — `FeTensor`

The universal data structure. A tensor is a **strided view into a flat memory buffer**:

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

- **Strides** are computed row-major (C order): `strides[last] = 1`, `strides[i] = strides[i+1] * shape[i+1]`. An element at multi-index `idx` lives at byte offset `Σ idx[d] * strides[d] * elem_size`.
- **Ownership**: `owns_data == true` means the tensor frees its buffer in `fe_tensor_free()`. Views (`fe_tensor_transpose`, `fe_tensor_reshape`) set `owns_data = false` — they share the parent's data and must never free it.
- **`fe_tensor_alloc`** mallocs a fresh buffer; **`fe_tensor_from_data`** wraps a caller-owned buffer.
- **`fe_tensor_transpose`** swaps `shape` and `strides` for two axes — O(1), no data movement.
- **`fe_tensor_reshape`** only succeeds if the tensor is contiguous (strides match row-major with no gaps) and the element count is preserved. Returns a view, no copy.
- **`fe_tensor_contiguous`** repacks a non-contiguous tensor element-by-element into a fresh contiguous buffer.
- **`fe_tensor_copy`** does a fast `memcpy` when both sides are contiguous, else falls back to strided element-wise copy.
- `fe_tensor_get_f32` / `fe_tensor_set_f32` are explicitly marked as debug-only — not the hot path.

### `core/allocator.h` / `core/allocator.c` — `FeArena`

A **bump-pointer memory arena**:

```c
typedef struct {
    unsigned char *base;
    size_t         size;
    size_t         offset;  /* current bump pointer */
    size_t         peak;    /* high-water mark      */
} FeArena;
```

- `fe_arena_alloc(size, align)` is O(1): aligns the **actual pointer address** (not just the offset — a subtle bug fix visible in the git history), checks capacity, bumps the offset.
- `fe_arena_reset()` returns the bump pointer to zero — all memory is "freed" at once. It deliberately preserves `peak` for profiling.
- `fe_arena_alloc_tensor()` allocates both the `FeTensor` metadata and its 64-byte-aligned data buffer from the arena (64-byte alignment for SIMD).
- **Two intended arenas**: a *weight arena* (allocated once at model load, never reset) and an *activation arena* (reset after every inference). This is the classic inference-runtime memory strategy.

---

## Computation Graph

### `graph/graph.h` / `graph/graph.c` — `FeGraph`

The graph IR separates **structure** from **data**. At build time, tensors are only shape/dtype metadata; the actual data pointers are assigned later by the planner or the arena allocator.

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

- **Ops** (`FeOpType`): `INPUT`, `OUTPUT`, `MATMUL`, `LINEAR`, `RELU`, `SOFTMAX`, `CONV1D`, `BATCHNORM`, `ADD`, `FLATTEN`. The union `attrs` carries op-specific attributes (softmax axis, conv1d stride/pad).
- **Nodes** reference tensors by **index** into the registry (inputs and outputs arrays), not by pointer.
- **Tensor entries** (`FeTensorEntry`) hold name, shape, dtype, an `is_weight` flag, and a `FeTensor *tensor` that stays `NULL` until memory is assigned.
- **Capacity limits**: 512 nodes, 1024 tensors, 8 inputs / 4 outputs per node, 64-char names — a fixed-capacity design typical of embedded runtimes.

### Topological sort (Kahn's algorithm)

`fe_graph_topo_sort()` computes a valid execution order in O(N + E):

1. Build a `producer` map: for each tensor index, which node outputs it.
2. Compute in-degrees: for each node, count inputs that are produced by *another node* (weights and graph inputs don't count as dependencies).
3. Queue all in-degree-0 nodes.
4. Pop a node, append to `topo_order`, decrement the in-degree of every consumer of its outputs; queue newly-zeroed nodes.
5. If fewer than `n_nodes` were ordered, the graph has a cycle → `FE_ERR_SHAPE`.

Every node execution must wait until all its inputs are ready — this is exactly what the topo order guarantees.

---

## Operator Kernels

### `ops/ops.h` — The kernel contract

```c
/* - Inputs are read-only
 * - Output tensor is caller-allocated with correct shape
 * - Returns FE_OK on success, error code otherwise
 * - No memory allocation inside kernels */
```

Every kernel validates pointers, dtypes, and shapes before touching data. This uniform contract makes the engine's dispatch trivial.

### `ops/matmul.c` — `fe_matmul` and `fe_linear`

The naive matmul is deliberately simple — it exists to **validate correctness** before the SIMD kernel replaces it:

```c
for (int i = 0; i < M; i++)
    for (int k = 0; k < K; k++) {
        float a_ik = a[i * K + k];
        for (int j = 0; j < N; j++)
            c[i * N + j] += a_ik * b[k * N + j];
    }
```

The header documents the core performance problem this project is built around: **A is accessed row-major (cache-friendly) but B is accessed column-major (stride-N, cache-unfriendly)**. The `simd/` module fixes this with tiling and packing. `fe_linear` is just `fe_matmul` followed by `fe_bias_add`.

### `ops/activations.c` — relu, softmax, bias_add

- **`fe_relu`** — elementwise `max(0, x)`, supports in-place.
- **`fe_softmax`** — numerically stable version along the **last axis**:
  1. Find `max_val` per row (to prevent `exp()` overflow).
  2. Compute `exp(x - max_val)` and accumulate the sum.
  3. Normalize by `1/sum`.
  - `rows` is the product of all dims except the last, `cols` is the last dim — so it generalizes to any rank ≥ 1.
- **`fe_bias_add`** — `out = in + bias` where bias matches the last dimension, broadcast over all rows.

### `ops/conv1d.c` — 1D convolution via im2col + matmul

Convolution is converted into a matrix multiplication — the standard trick that lets you reuse GEMM kernels:

1. **`fe_im2col`**: for each batch element, copy each input patch `[C_in, K]` at every output position into a column, producing a `[C_in*K, L_out]` matrix. Out-of-bounds reads (padding) are skipped, and the buffer is zeroed first, so padding is handled implicitly.
2. Reshape the `[C_out, C_in, K]` weight to `[C_out, C_in*K]` — a **view, no copy** — via `fe_tensor_reshape`.
3. Per batch element: `out_slice = W @ col` using `fe_matmul`, then add bias if provided.

Output length is computed as `L_out = (L - K + 2*pad)/stride + 1`.

---

## SIMD Matmul

### `simd/matmul_avx2.h` / `simd/matmul_avx2.c`

The performance-focused matmul. It uses **tiling**, **packing**, and **8-wide FMA** to solve the cache problem identified in the naive kernel:

- **Tile sizes**: `MC = 64` (rows) and `KC = 256` (K-slice). The output is processed in `mc × kc` tiles so that the working set fits in cache.
- **Packing**: `A` tiles are packed row-major into a contiguous `A_tile[actual_mc][actual_kc]` buffer; `B` tiles are packed into `B_tile[k][8]` so the innermost loop reads 8 contiguous floats.
- **Kernel** (`kernel_mc_nr`): loads 8 accumulator rows into `__m256` registers, then for each `k` broadcasts `A[i][k]` and does a fused multiply-add (`_mm256_fmadd_ps`) against the 8-wide `B` row. This computes 8 output elements per instruction.
- **Remainder**: columns where `N` is not a multiple of 8 fall back to a scalar loop.
- **Detection**: `fe_cpu_has_avx2()` uses CPUID (`__cpuid_count(7, 0, ...)`, EBX bit 5) at runtime.

The benchmark tool reports a ~13.4× speedup over the naive loop for 256×256 matmuls.

---

## Memory Planner

### `planner/memory_planner.h` / `planner/memory_planner.c`

The planner performs **static memory reuse**: it computes when each tensor is alive and assigns non-overlapping tensors to the same buffer, minimizing the total activation memory.

**Step 1 — Lifetime analysis** (`compute_lifetimes`):
Walk the graph in topological order. For each step `s`:
- Every output tensor gets `first_use = s`.
- Every input tensor's `last_use` is updated to `s`.

Weights are excluded (they live in the weight arena). The graph's model input is also excluded (the caller provides it).

**Step 2 — Greedy buffer assignment** (`fe_plan_memory`):
This is essentially a greedy **interval graph coloring** problem, O(n²):
- Process tensors by `first_use`.
- A *free pool* tracks released buffer regions (`offset`, `size`, `freed_at`).
- For each tensor, find the smallest free slot that (a) was freed before this tensor's `first_use` and (b) is large enough (after 64-byte alignment). Reuse it. Otherwise allocate a new region at the high-water mark.
- When a tensor dies (`last_use`), its slot returns to the pool.

The result is a `FePlan` with a per-tensor byte `offset` into a single activation buffer and `total_activation_bytes`. `fe_plan_print` reports the savings vs. naive (no-reuse) allocation.

**Step 3 — Apply** (`fe_plan_apply`):
Writes tensor metadata pointing into the activation buffer at the planned offsets.

---

## Execution Engine

### `runtime/engine.h` / `runtime/engine.c`

`FeRuntime` ties the graph, arenas, and kernels together:

```c
typedef struct {
    FeGraph    *graph;
    FeArena     weight_arena;
    FeArena     activation_arena;
    FeProfiler *profiler;
} FeRuntime;
```

Lifecycle:

1. **`fe_runtime_init`** — stores the graph, initializes both arenas, and runs the topological sort if it hasn't been done.
2. **`fe_runtime_alloc_weights`** — allocates every `is_weight` tensor from the weight arena (once, at load).
3. **`fe_runtime_run`** — one inference:
   - Reset the activation arena.
   - Clear all non-weight tensor pointers.
   - Bind the caller's input tensor to the `FE_OP_INPUT` node's output.
   - `alloc_activations()` allocates every non-weight tensor from the activation arena (bump pointer, so cheap).
   - Walk `topo_order`, dispatching each node to its kernel via a `switch` on `node->op` using the `IN(i)` / `OUT(i)` macros (tensor index → tensor pointer).
   - Copy the output node's result into the caller's buffer.
4. **`dispatch_node`** wraps every op call with `fe_profiler_now_ns()` timing when a profiler is attached.

**Note on current state**: the engine uses the arena bump allocator directly for activations, and does not consume the `FePlan` produced by the planner. The planner exists as a standalone, tested subsystem (and a measurement of what the memory footprint *could* be). Wiring `fe_plan_apply` into the runtime is an obvious integration task.

Ops wired in the dispatcher: `MATMUL`, `LINEAR`, `RELU`, `SOFTMAX`, `ADD` (as bias-add), `FLATTEN` (via reshape view + copy). `CONV1D` and `BATCHNORM` are in the op enum but **not yet dispatched** — the dispatch hits `FE_ERR_SHAPE` ("unimplemented op").

---

## ONNX Importer

### `importer/onnx.h` / `importer/onnx.c`

Ferrite parses ONNX `.onnx` files with a **hand-written protobuf wire-format parser** — no protobuf compiler or library.

**Protobuf wire format primitives** (exposed for testing):
- `fe_pb_varint` — LEB128 varint decoding.
- `fe_pb_tag` — reads the tag, splitting it into `field_number` (tag >> 3) and `wire_type` (tag & 7).
- `fe_pb_skip` — skips a field by wire type (0=varint, 1=64-bit, 2=length-delimited, 5=32-bit).
- `fe_pb_bytes` — reads a length-delimited blob.

**ONNX structure** (ModelProto → GraphProto → NodeProto/TensorProto):
- Top level: find `field 7` (graph), skip everything else.
- `parse_graph`: `field 1` = node (repeated NodeProto), `field 5` = initializer (repeated TensorProto), `field 11` = input ValueInfo (skipped — shapes are inferred by the test harness instead).
- `parse_node`: reads input names (`field 1`), output names (`field 2`), node name (`field 3`), and `op_type` string (`field 4`), then maps the op string to an `FeOpType` via `op_type_from_string`. **Unsupported ops are logged and skipped**, keeping the load resilient.
- `parse_initializer`: reads dims (`field 1`), data_type (`field 2`, assumed float32), name (`field 8`), and `raw_data` (`field 9`, packed float32 bytes). The tensor is registered in the graph as a weight and its data is `memcpy`'d into the weight arena.

The `find_or_add_tensor` helper deduplicates tensors by name, so the same tensor referenced by multiple nodes resolves to one graph entry.

After parsing, the graph is topologically sorted and validated before returning `FE_OK`.

---

## Quantization

### `quantization/quant.h` / `quantization/quant.c`

Post-training INT8 quantization using **symmetric per-tensor** scaling:

- `FeQuantParams` = `{ float scale; int zero_point; }`, with `zero_point` always 0 (symmetric).
- **`fe_quantize`** — computes `scale = max(|x|)/127`, then `q = clamp(round(x/scale), -127, 127)`.
- **`fe_dequantize`** — `x = q * scale`.
- **`fe_matmul_int8`** — the quantized GEMM pipeline:
  1. Quantize A and B to INT8.
  2. Multiply with **int32 accumulation** (protects against overflow).
  3. Dequantize the result with the combined scale `pA.scale * pB.scale`.
- **`fe_quantize_weights`** — a graph-level helper to convert all weight tensors to INT8 and store their scale factors.

This is the classic symmetric INT8 scheme (the same one used in early quantized ONNX runtimes); a future improvement would be per-channel scales for better accuracy on convolutional weights.

---

## Profiling & Benchmarks

### `tools/profiler.h` / `tools/profiler.c`

A per-operator timing profiler:
- `fe_profiler_now_ns()` uses `clock_gettime(CLOCK_MONOTONIC)`.
- `fe_profiler_record(name, ns)` accumulates total time and call count per operator name (up to 64 ops).
- `fe_profiler_print()` sorts records by total time (insertion sort), computes each op's % of total time, and prints a formatted table.
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
- `tools/bench_avx2.c` — runs both `fe_matmul` and `fe_matmul_avx2`, **verifies the AVX2 result against the naive one** (max error check), then prints both GFLOPS and the speedup ratio.

---

## Tests & Build System

The `Makefile` builds one test binary per subsystem plus two benchmark binaries:

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
| `bench_matmul` / `bench_matmul_avx2` / `bench_avx2` | performance measurements |

Build flags are notable:
```
-std=c11 -D_POSIX_C_SOURCE=199309L -Wall -Wextra -fsanitize=address,undefined -g
```
Every test runs under **AddressSanitizer + UndefinedBehaviorSanitizer**. The AVX2 targets add `-O3 -mavx2 -mfma`.

`tests/test_engine.c` is a good end-to-end example: it builds a `1→4→8→3` MLP (Linear → ReLU → Linear → Softmax) by hand, sets uniform weights (`0.1f`), and verifies the softmax output is analytically exactly `1/3` per class. `tests/tiny_mlp.onnx` exercises the importer against a real ONNX file.

---

## End-to-End Data Flow

Tracing one full inference through every layer:

1. **Load** (`importer/`): `fe_onnx_load` reads the `.onnx` file, parses protobuf, builds the `FeGraph` (nodes + tensor registry), topologically sorts it, and copies all weights into the weight arena.
2. **Plan** (`planner/`): lifetime analysis finds each tensor's first/last use; greedy assignment overlaps non-conflicting tensors in one activation buffer. The plan reports the minimum buffer size and savings.
3. **Init runtime** (`runtime/`): arenas bound to caller buffers; weight tensors allocated from the weight arena.
4. **Run** (`runtime/`):
   - Reset the activation arena.
   - Bind the input tensor.
   - Allocate all activation tensors (bump pointer).
   - For each node in topological order, dispatch to the matching kernel in `ops/` (or `simd/` for the fast matmul), timing each call if profiling.
   - Copy the final output into the caller's buffer.
5. **(Optional) Quantize** (`quantization/`): convert weights to INT8 to shrink the model and use int8→int32 GEMMs.
6. **(Optional) Profile** (`tools/`): print per-operator timing to find the hotspot (almost always matmul).

---

## Known Limitations & Next Steps

- **Planner not integrated with the engine** — `fe_runtime_run` uses the arena directly; `fe_plan_apply` exists but nothing calls it in the runtime path. Wiring it in would give deterministic, minimal activation memory.
- **`CONV1D` and `BATCHNORM` are not dispatched** in `engine.c` even though the op enum and the conv1d kernel exist — an engine integration gap.
- **Limited opset** — unsupported ONNX ops are silently skipped, so only models made of the supported ops load correctly.
- **No shape inference** — tensor shapes must be present in the ONNX file or filled in by the caller; the importer skips ONNX's value-info shape data.
- **Quantization is per-tensor, not per-channel** — per-channel scales would improve INT8 accuracy.
- **AVX2 requires `N % 8 == 0`** for the vectorized path; the remainder falls back to scalar code.
- **Fixed capacities** (512 nodes / 1024 tensors / 8 dims) — fine for small models, would need dynamic growth for larger ones.
- **Single-threaded** — no parallel execution across independent subgraphs yet.
