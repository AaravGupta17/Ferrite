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
9. [Parallel Execution (`parallel/`)](#parallel-execution)
10. [Compiler IR (`compiler/`)](#compiler-ir)
11. [ONNX Importer (`importer/`)](#onnx-importer)
12. [Quantization (`quantization/`)](#quantization)
13. [Profiling and Benchmarks (`tools/`)](#profiling-and-benchmarks)
14. [Tests and Build System](#tests-and-build-system)
15. [End-to-End Data Flow](#end-to-end-data-flow)
16. [Known Limitations](#known-limitations)

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
                            │ lowers
┌───────────────────────────▼─────────────────────────────────┐
│                  compiler/  (linear IR)                     │
│        lowering, dead-elim, kernel table, scheduling        │
└───────────────────────────┬─────────────────────────────────┘
                            │ analyzes
┌───────────────────────────▼─────────────────────────────────┐
│                    planner/  (memory)                       │
│        lifetime analysis + greedy buffer reuse              │
└───────────────────────────┬─────────────────────────────────┘
                            │ executes
┌───────────────────────────▼─────────────────────────────────┐
│                  parallel/  (pool + scheduler)              │
│        thread pool, row-parallel GEMM, DAG scheduler        │
└───────────────────────────┬─────────────────────────────────┘
                            │ dispatches
┌───────────────────────────▼─────────────────────────────────┐
│                     runtime/  (engine)                      │
│        dispatches nodes to kernels in topo order            │
└───────────────────────────┬─────────────────────────────────┘
                            │ calls
┌───────────────────────────▼─────────────────────────────────┐
│    ops/  +  simd/          core/  (tensors, arenas, fp16)   │
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

- `FeDtype` — `DTYPE_FLOAT32` (0), `DTYPE_INT8` (1), `DTYPE_INT32` (2),
  `DTYPE_FLOAT64` (3), `DTYPE_FLOAT16` (4, **storage only** — compute happens
  after an FP32 upconvert), `DTYPE_INT16` (5, INT16 dynamic-quant path),
  `DTYPE_BFLOAT16` (6, **storage only** — compute happens after an FP32 upconvert).
- `FeStatus` — error codes: `FE_OK` and `FE_ERR_NULL/SHAPE/DTYPE/NOMEM/BOUNDS`.
- `fe_dtype_size()` — inline helper, bytes per element (4/1/4/8/2/2/2).
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

### `core/fp16.h` / `core/fp16.c` — half-precision (FP16 + BF16) storage

**Bottom line.** Half-precision formats are **storage formats**, not compute formats: FP16 (binary16) and BF16 weights halve RAM, and every access upconverts to FP32 before touching a kernel.

- `fe_f32_to_fp16(f)` / `fe_fp16_to_f32(h)` — bit-level conversion with round-to-nearest-even. Exact for the representable range, saturates to ±Inf on overflow, and correct for subnormals and NaN.
- `fe_f32_to_bf16(f)` / `fe_bf16_to_f32(b)` — BF16 keeps the FP32 exponent and the top 8 mantissa bits (RNE on the discarded low bits); overflow stays Inf.
- `fe_f32_to_fp16_buf` / `fe_fp16_to_f32_buf` and `fe_f32_to_bf16_buf` / `fe_bf16_to_f32_buf` — bulk conversions, position matched.
- `DTYPE_FLOAT16` (4) and `DTYPE_BFLOAT16` (6) each size to 2 bytes in `fe_dtype_size()`. `tests/test_tensor.c` pins known bit patterns (FP16: 1.0 → 0x3C00, −2.0 → 0xC000 … plus a ≤2⁻¹⁰ sweep; BF16: 1.0 → 0x3F80, π → 0x4049 … plus a ≤2⁻⁸ sweep).

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
- **Capacity limits**: 512 nodes, 1024 tensors, 8 inputs / 4 outputs per node, 64-char names. Fixed capacity — typical of embedded runtimes. Raise the ceilings at configure time via the `FE_MAX_NODES`, `FE_MAX_TENSORS` and `FERRITE_MAX_DIMS` CMake cache variables; `FE_MAX_ALLOCS` must stay >= `FE_MAX_TENSORS`, which a configure-time `FATAL_ERROR` and a `_Static_assert` in `planner/memory_planner.h` both enforce. Exceeding any ceiling is a **loud** load-time failure naming the knob to raise: the importer refuses a tensor whose rank passes `FERRITE_MAX_DIMS` rather than dropping the surplus extents, and reports registry/node exhaustion instead of returning a bare `FE_ERR_NOMEM` that looks like an undersized arena.

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
typedef struct FeRuntime {
    FeGraph    *graph;
    FeArena     weight_arena;
    FeArena     activation_arena;
    FeProfiler *profiler;
    FeExecPlan  exec;   /* cached kernel resolution + memory plan */
} FeRuntime;
```

**Lifecycle.**

1. **`fe_runtime_init`** — stores the graph, initializes both arenas, runs the topological sort and validation.
2. **`fe_runtime_alloc_weights`** — allocates every unbacked `is_weight` tensor from the weight arena, once, at load. Folded/fused constants allocated during `fe_optimize` are already backed and skipped.
3. **`fe_runtime_run`** — one inference, delegated to the static execution plan (`runtime/exec_plan.c`):
   - **First run (or input-shape change):** re-seed the input (dynamic graphs), forget every produced shape, `fe_infer_shapes`, `fe_plan_memory` (cached `FePlan`), and `fe_exec_build` resolves each `topo_order` node to a wrapper function via the `k_fns[]` table.
   - Reset the activation arena; clear non-weight tensor pointers.
   - `fe_plan_apply` re-assigns the cached byte offsets (data region reserved before `FeTensor` metadata).
   - Bind the caller's input tensor (`find_graph_input` — ONNX graphs have no `FE_OP_INPUT` node).
   - Walk the flat `FeExecStep{fn, node_index}` array — no `switch`, no lifetime analysis.
   - Copy the OUTPUT node's result (or, for ONNX graphs, the last planned node's output) into the caller's buffer.
4. **`fe_exec_*` wrappers** time every op call with `fe_profiler_now_ns()` when a profiler is attached (`EXEC_BEGIN`/`EXEC_END` per step).

**Current state.** `FeRuntime` embeds the `FeExecPlan` (`exec`, hence the `runtime/exec_plan.h` include in `engine.h`). The engine no longer dispatches: steady-state runs execute the cached plan end-to-end, and the only per-run analysis is an input-shape comparison. The memory planner provides the layout the plan re-applies per run.

**Dispatched ops:** every `FeOpType` in the enum has an entry in the `k_fns[]` table (`runtime/exec_plan.c`), mapped to a wrapper that calls the `ops/` kernel (or `simd/` behind `fe_cpu_has_avx2`). Core ops: `MATMUL`, `LINEAR`, `RELU`, `SOFTMAX`, `ADD`, `FLATTEN`, `CONV1D`, `BATCHNORM`. Stage 3: basic math (`SUB`/`MUL`/`DIV`/`NEG`/`EXP`/`LOG`/`POW`), activations (`SIGMOID`/`TANH`/`GELU`/`LEAKY_RELU`/`ELU`/`SWISH`), linear algebra (`GEMM`, `TRANSPOSE`), CNN (`CONV2D`, `MAXPOOL`, `AVGPOOL`, `LAYERNORM`, `GROUPNORM`), and sequence (`ATTENTION`, `MULTIHEAD_ATTN`, `EMBEDDING`, `POSITIONAL_ENCOD`). A node with no table entry fails loudly at plan build — nothing is silently skipped.

---

## Parallel Execution

### `parallel/threadpool.h` / `parallel/threadpool.c`

**Bottom line.** A small mutex+condvar FIFO worker pool with `fe_cpu_count()` (returns 1 when the caller hasn't called `fe_threadpool_init`).

- `fe_threadpool_init` — spawn N workers.
- `fe_threadpool_submit` — enqueue a job `{fn, arg}`; a worker drains FIFOs to completion.
- `fe_threadpool_wait` — block until every submitted job has run.

### `parallel/parallel_gemm.h` / `parallel/parallel_gemm.c`

**Bottom line.** Row-split matmul: `fe_matmul_parallel(A, B, C, pool, nchunks)` partitions the M dimension across workers.

- Chunk boundaries are derived deterministically from the chunk index, so each worker writes its own C rows with **no shared mutable context** — there is no accumulator race.
- With 1 chunk the function degenerates to the plain `fe_matmul`, so `tests/test_parallel.c` compares the parallel path against the sequential oracle (M=37 K=17 N=23, max err ~9.5e-7).

### `parallel/scheduler.h` / `parallel/scheduler.c`

**Bottom line.** A ready-set DAG scheduler: `fe_graph_schedule(g, pool, ...)` walks the topo order honoring producer-before-consumer and dispatches ready work.

- In pool mode the worker that runs a node **resubmits that node's children** directly to the pool, so one `fe_threadpool_wait` covers the whole cascade — no shared ready queue, no spin-wait.
- `tests/test_parallel.c:test_scheduler_causal_order` builds a diamond DAG and asserts every consumer runs strictly after its producers, in both sequential and pool modes.

---

## Compiler IR

### `compiler/ir.h` — the linear IR

**Bottom line.** Stage 14 flattens the graph into a straight-line IR (`FeIrProgram`) decoupled from the engine: lower → optimize → schedule → select kernels → run. The IR is intentionally minimal — one instruction per graph node — and exists as a clean seam for future passes (fusion, allocation, threading).

```c
typedef enum { IR_INPUT, IR_OUTPUT, IR_MATMUL, IR_LINEAR, IR_RELU,
               IR_ADD, IR_SOFTMAX, IR_CONV1D, IR_BATCHNORM, IR_NONE } FeIrOp;

typedef struct {
    FeIrOp  op;
    int     node;        /* source graph node (for attrs: stride, pad, eps) */
    int     dst;
    int     srcs[FE_MAX_NODE_INPUTS];
    int     n_srcs;
    int     dead;        /* dead-elim flag */
} FeIrInstr;
```

### `compiler/lower.c`, `compiler/codegen.c`, `compiler/schedule.c`

- **`fe_ir_lower(g, &p)`** — maps every `FeOpType` to an `FeIrOp`, records io_in/io_out using the engine's binding rules. Fails loudly on an unmapped op.
- **`fe_ir_dead_elim`** — mark-and-sweep: any dst that is neither an instruction source nor the program output is dead and compacted away (guarded against `dst == -1` on `IR_OUTPUT`).
- **`fe_ir_codegen`** — fills a static `FeIrKernel` table (one wrapper per op, `op`-indexed); every entry must resolve or codegen fails. Wrappers pull per-node attributes (conv1d stride/pad, batchnorm eps) from `ctx.g->nodes[node].attrs`.
- **`fe_ir_schedule`** — producers-before-consumers with a tiny defuse heuristic: among ready instructions, prefer the one sharing a source tensor with the previously scheduled instruction (cache/arena locality). Deterministic, quadratic, bounded by `FE_MAX_NODES`.
- **`fe_ir_run`** — executes the instruction stream against a caller-bound `FeIrCtx {tensors, input, output}`, in *lowered* order (the schedule is the testable seam).

`tests/test_compiler.c` proves the pipeline end-to-end: the two-layer MLP compiled through `fe_ir_run` matches `fe_runtime_run` bit-for-bit within `1e-5`, dead-elim removes an unconnected node, and the diamond schedule is causally valid.

---

## ONNX Importer

### `importer/onnx.h` / `importer/onnx.c`

**Bottom line.** Ferrite parses ONNX `.onnx` files with a hand-written protobuf wire-format parser. No protobuf compiler or library.

**Protobuf wire primitives** (exposed for testing):
- `fe_pb_varint` — LEB128 varint decoding.
- `fe_pb_tag` — reads the tag; splits it into `field_number` (tag >> 3) and `wire_type` (tag & 7).
- `fe_pb_skip` — skips a field by wire type (0=varint, 1=64-bit, 2=length-delimited, 5=32-bit).
- `fe_pb_bytes` — reads a length-delimited blob. The declared length is checked
  against the bytes actually remaining **before** the pointer/length view is
  published, and on failure neither output is written. This is load-bearing: a
  caller that ignores the return value must not be handed a view describing
  memory past the end of the buffer. `parse_initializer` additionally caps a
  weight's element count (2^28) so a product of individually-legal dims cannot
  overflow. Both guards turn a malformed file into a loud `FE_ERR_SHAPE`.

**ONNX structure** (ModelProto → GraphProto → NodeProto/TensorProto):
- Top level: find `field 7` (graph); skip everything else.
- `parse_graph`: `field 1` = node (repeated NodeProto), `field 5` = initializer (repeated TensorProto), `field 11` = input ValueInfo (shape data feeds static `FeTensorEntry`s via fields 11/12/13; remaining unknowns are filled by the shape-inference pass).
- `parse_node`: reads input names (`field 1`), output names (`field 2`), node name (`field 3`), and `op_type` (`field 4`). Maps the op string to an `FeOpType` via `op_type_from_string`. `Gemm` maps to `FE_OP_LINEAR` with a warning when non-default `transA`/`transB`/`alpha`/`beta` would change the result; **unsupported ops fail loudly** rather than being skipped.
- `check_softmax_axes`: runs after `fe_optimize`, once shape inference has settled every rank, and warns for any `Softmax` whose axis is provably not the last one (the engine always reduces over the last axis). Attribute parsing only *records* the axis. Judging it during parsing read a rank that was not yet inferred, so `tiny_mlp.onnx` — axis=1 on a rank-2 output, i.e. the last axis — was reported as unsupported on every load. An unresolved rank is left alone rather than guessed at.
- `parse_initializer`: reads dims (`field 1`), data_type (`field 2`, assumed float32), name (`field 8`), and `raw_data` (`field 9`, packed float32 bytes). Registers the tensor as a weight and `memcpy`s its data into the weight arena.

`find_or_add_tensor` deduplicates tensors by name, so one graph entry serves every node referencing the same tensor.

After parsing, the graph is topologically sorted and validated before returning `FE_OK`.

---

## Quantization

### `quantization/quant.h` / `quantization/quant.c`

**Bottom line.** Post-training INT8 quantization with **symmetric per-tensor or per-channel** scaling, **integrated into the engine**.

- `FeQuantParams` = `{ float scale; int zero_point; }`. `zero_point` is always 0 (symmetric).
- **`fe_quantize`** — `scale = max(|x|)/127`, then `q = clamp(round(x/scale), -127, 127)`.
- **`fe_dequantize`** — `x = q * scale`.
- **`fe_matmul_int8`** — the quantized GEMM pipeline:
  1. Quantize A and B to INT8.
  2. Multiply with **int32 accumulation** (protects against overflow).
  3. Dequantize the result with the combined scale `pA.scale * pB.scale`.
- **`fe_quantize_per_channel` / `fe_matmul_int8_per_channel`** — one scale per output column, which matters for skewed weight distributions (per-channel error 0.00015 vs per-tensor 0.053 on the skewed test).
- **`fe_quantize_weights`** — converts all weight tensors to INT8, repacking **in place** (the INT8 data is ¼ the size, so it fits the float buffer; arena-backed weights are never `fe_tensor_free`d). The bias stays float.
- **`fe_quantize_model(g, arena)`** — the engine hook. For every 2-D float weight consumed as `inputs[1]` of a MATMUL/LINEAR node: quantizes per channel, appends the scales to the weight arena (bumping the bump pointer past the already-loaded weight bytes), repacks the weight buffer in place, and sets the entry's dtype/scales. Weights not feeding a linear op are left float.
- **`fe_matmul_int8_dyn` / `fe_linear_int8`** — run-time INT8 GEMM with dynamic two-pass activation quantization (no scratch buffer, no per-run allocation). Used by the engine.

**Engine integration.** `ex_matmul`/`ex_linear` in `runtime/exec_plan.c` branch on the weight tensor's dtype per node: INT8 → `fe_linear_int8`/`fe_matmul_int8_dyn` (scales read from the graph entry), float → the scalar/AVX2 float path. A model can mix float and quantized linear layers; quantization is a load-time choice (`fe_quantize_model` after load/runtime-init), not a compile-time flag.

This is the classic symmetric INT8 scheme used in early quantized ONNX runtimes; the natural improvements are **calibration-driven scale inference** and more storage dtypes (INT16/FP16/BF16).

**Calibration pipeline.** `fe_runtime_calibrate` (`runtime/engine.c`) runs a caller-provided sample set through `fe_runtime_run` and records, for every non-weight float tensor (inputs and activations), the maximum |x| observed across the samples into a `graph->n_tensors`-entry `ranges` array. It is the data-collection half of calibration: the ranges are what a static-activation-scale build consumes (the engine today quantizes activations dynamically per run). No percentile/stat-smoothing pass over the observed ranges exists yet.

**INT16, FP16 and BF16 paths.** All are engine-wired. `ex_matmul`/`ex_linear` (`runtime/kernels.c`) branch on the weight tensor's dtype at run time: an `DTYPE_INT16` weight routes to `fe_matmul_int16_dyn`/`fe_linear_int16`; a half-precision weight (`DTYPE_FLOAT16`/`DTYPE_BFLOAT16`) is upconverted to an FP32 shadow once and runs the float kernel; INT8 (the classic scheme below) routes to the dynamic/static INT8 kernels. The engine quantizes activations dynamically per run unless a recorded static activation scale (`act_scale > 0`) selects the single-pass static kernel:

- `fe_quantize_int16` — symmetric per-tensor INT16 quantization (`scale = max|x|/32767`).
- `fe_matmul_int16_dyn` / `fe_linear_int16` — dynamic-activation-quant INT16 GEMM/Linear with per-channel weights, mirroring the INT8 engine kernels.
- `fe_f32_to_fp16`/`fe_fp16_to_f32`, `fe_f32_to_bf16`/`fe_bf16_to_f32`, and `DTYPE_FLOAT16`/`DTYPE_BFLOAT16` — storage-only half precision (see the `core/` section). An INT16/half model-level repack (`fe_quantize_model` analog) is the natural next step.

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
| `test_parallel` | thread-pool partition, parallel GEMM vs oracle, scheduler causal order |
| `test_compiler` | IR lowering, dead-elim, kernel selection, IR output == engine output, schedule validity |
| `bench_matmul` / `bench_matmul_avx2` / `bench_avx2` / `bench_model` | performance measurements (Debug and Release; see `docs/benchmarks.md`) |

Build flags:

```
-std=c11 -D_POSIX_C_SOURCE=199309L -Wall -Wextra -fsanitize=address,undefined -g
```

Every test runs under AddressSanitizer + UndefinedBehaviorSanitizer (where available). AVX2 targets add `-O3 -mavx2 -mfma`.

`tests/test_engine.c` is a good end-to-end example: it builds a `1→4→8→3` MLP by hand (Linear → ReLU → Linear → Softmax), sets uniform weights (`0.1f`), and verifies the softmax output is exactly `1/3` per class. `tests/tiny_mlp.onnx` exercises the importer against a real ONNX file.

---

## End-to-End Data Flow

Tracing one full inference through every layer:

1. **Load** (`importer/`): `fe_onnx_load` reads the `.onnx`, parses protobuf, builds the `FeGraph` (nodes + tensor registry), topologically sorts it, validates it, runs `fe_optimize`, and copies all weights into the weight arena.
2. **Optimize** (`optim/`): `fe_optimize` runs shape-infer → simplify → constant-fold → Conv+BN fusion → CSE → dead-elim over the freshly-built graph (weights already resident in the arena). Folded/fused constants are allocated into the same arena; the engine's `fe_runtime_alloc_weights` skips already-backed weights so they survive runtime init.
3. **Plan** (`planner/`): lifetime analysis finds each tensor's first/last use. Greedy assignment overlaps non-conflicting tensors in one activation buffer. The plan reports minimum buffer size and savings.
4. **Init runtime** (`runtime/`): arenas bind to caller buffers; weight tensors allocate from the weight arena; the graph is re-validated.
5. **Run** (`runtime/`):
   - First run (or any run whose input shape differs from the last planned one): re-seed the input shape, forget every produced shape, `fe_infer_shapes` re-propagates, `fe_plan_memory` computes a fresh `FePlan`, and `fe_exec_build` resolves each `topo_order` node to a kernel function pointer via the `k_fns[]` table.
   - Reset the activation arena, `fe_plan_apply` re-assigns the cached byte offsets (data region reserved before `FeTensor` metadata), bind the input tensor (`find_graph_input` — ONNX graphs have no `FE_OP_INPUT` node), allocate a producer-less OUTPUT-node tensor if needed.
   - Execute the flat `FeExecStep` array — a straight walk of pre-resolved kernel wrappers, no dispatch switch, no lifetime analysis. Each step records its own profiler timing. Nothing is silently skipped — an op with no table entry returns `FE_ERR_SHAPE` loudly at build time.
   - Copy the final output into the caller's buffer.
6. **(Optional) Quantize** (`quantization/`): `fe_quantize_model` converts MATMUL/LINEAR weights to per-channel INT8 in place; subsequent runs route those steps through `fe_linear_int8`/`fe_matmul_int8_dyn` with dynamic INT8 activations. INT8 weights need no re-planning — the exec steps were already resolved, and the wrappers branch on the weight dtype at run time.
7. **(Optional) Profile** (`tools/`): print per-operator timing to find the hotspot — almost always matmul.

---

## Known Limitations

- **Planner integrated.** `fe_runtime_run` plans memory and runs off one planned activation buffer; `fe_plan_apply` reserves the data region before arena-allocating `FeTensor` metadata so the two never collide.
- **Limited opset.** Unsupported ONNX ops fail loudly (`FE_ERR_SHAPE`), never silently drop. The importer maps 20+ op strings (math, activations, pool, norms, conv2d) and reads per-op attributes.
- **Static execution plan.** `fe_runtime_run` executes a precomputed flat array of kernel function pointers over a cached memory plan. The only per-run analysis is an input-shape comparison; a mismatch triggers a rebuild (re-infer → re-plan → re-resolve steps). Dynamic-batch graphs re-seed their input on every rebuild because shape inference only fills unknown shapes and never shrinks. Steady-state runs do zero re-analysis (`n_plan_builds` stays 1).
- **Optimization runs at load, once.** `fe_optimize` runs in `fe_onnx_load` and again nowhere; interop with dynamic input shapes is limited to shape inference (dead-elim's live set ignores shapes).
- **Quantization is per-tensor and per-channel, wired into the engine.** `fe_quantize_model` converts MATMUL/LINEAR weights to INT8 (per-channel scales) and the exec-plan wrappers dispatch dynamically. Scales still come from a single tensor's max, not a calibration set; no calibration pipeline. INT16 and FP16 are API-complete (`fe_quantize_int16`/`fe_matmul_int16_dyn`/`fe_linear_int16`, `core/fp16.h`) but not wired into engine dispatch. Activations are quantized dynamically per run — there is no precomputed activation scale.
- **AVX2 needs `N % 8 == 0`** for the vectorized path. Remainders fall back to scalar code.
- **Fixed capacities** (512 nodes / 1024 tensors / 8 dims), raisable per target through CMake cache variables and refused loudly when exceeded. Fine for small models; avoiding a reconfigure for larger ones would need dynamic growth.
- **Parallelism is opt-in, not in the engine hot path.** The thread pool, row-parallel GEMM, and DAG scheduler live in `parallel/` with their own tests; `fe_runtime_run` remains single-threaded unless a caller routes work through the pool explicitly.
