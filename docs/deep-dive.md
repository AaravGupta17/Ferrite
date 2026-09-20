# Ferrite — A Deep Dive

A zero-dependency neural-network inference runtime in C11. It loads ONNX models and runs them with hand-written kernels — no BLAS, no protobuf library, no SIMD wrapper, no malloc on the inference path. This document walks through the entire project, subsystem by subsystem, from the byte layout of a tensor to the static execution plan that runs a model.

The shorter bridge document is `docs/guide.md` (how the pieces connect); the plan is `docs/roadmap.md`; this file is the deep dive into *how* each piece works.

---

## 1. Bottom Line Up Front

**What it is.** Ferrite parses a `.onnx` protobuf with a hand-rolled wire-format reader, builds a computation-graph IR, optimizes it (shape inference, constant folding, Conv+BN fusion, CSE, dead-code elimination), plans one shared activation buffer using tensor lifetimes, resolves every node to a kernel function pointer at build time, and executes a flat array of `FeExecStep`s per inference. No per-run switch, no per-tensor allocation, one malloc-free hot path.

**Who owns what.** Every public function returns `FeStatus` (or `NULL` if it is a constructor). Kernels never allocate and never log. Tensors reference graph entries by index, never by pointer. Weights live forever in a weight arena; activations reset every inference.

**Birds-eye dependency picture** (each layer may only use the layers below):

```
      tests/  tools/
        |   |
        v   v
   runtime/  compiler/
   importer/ optim/  parallel/
        -----------------
   planner/        (ops + simd + core)
   graph/  ops/  simd/  core/
```

| Subsystem | Role | Key files |
|---|---|---|
| `core/` | tensors, arenas, serialization, logging, half-precision storage | `types.h`, `tensor.c/h`, `allocator.c/h`, `tensor_ser.c/h`, `log.c/h`, `fp16.c/h` |
| `graph/` | computation-graph IR, tensor registry, Kahn topo sort, validation | `graph.c/h` |
| `ops/` | operator kernels: math, activations, GEMM/linear, conv via im2col, norms, pooling, sequence ops | `ops.h`, `matmul.c`, `activations.c`, `conv1d.c`, `conv2d.c`, `elementwise.c`, `norm.c`, `pool.c`, `sequence.c`, … |
| `simd/` | AVX2 kernels with runtime CPUID detection | `matmul_avx2.c/h`, `elementwise_avx2.c/h` |
| `planner/` | tensor-lifetime analysis + greedy buffer reuse | `memory_planner.c/h` |
| `parallel/` | thread pool, row-parallel GEMM, ready-set DAG scheduler | `threadpool.c/h`, `parallel_gemm.c/h`, `scheduler.c/h` |
| `optim/` | graph passes: shape infer, simplify, fold, fuse, CSE, DCE | `shape_infer.c/h`, `optim.c/h` |
| `compiler/` | linear IR: lower, dead-elim, kernel table, schedule, `fe_ir_run` | `ir.h`, `lower.c`, `codegen.c`, `schedule.c` |
| `runtime/` | `FeRuntime`, static execution plan | `engine.c/h`, `exec_plan.c/h` |
| `importer/` | hand-rolled ONNX protobuf parser + graph loader | `onnx.c/h` |
| `quantization/` | symmetric INT8 (engine-wired), INT16 dynamic-quant, calibration | `quant.c/h` |
| `tools/` | profiler, benchmark harness | `profiler.c/h`, `bench.c/h`, `bench_model.c`, `bench_avx2.c` |

---

## 2. Core Data Model

### 2.1 `FeTensor` — a strided view, not a buffer

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

A tensor is a **view** into a flat memory buffer (`core/tensor.h`). The same buffer can be seen as different shapes (reshape), different orders (transpose), or a subset (slice) — all with **zero data movement**:

- `fe_tensor_transpose(t, ax0, ax1)` — swaps two axes of the stride array. O(1), purely metadata.
- `fe_tensor_reshape(t, ndim, shape)` — returns a new view **only if the input is contiguous**; non-contiguous reshapes fail with `NULL`. This is the contract that makes `fe_tensor_contiguous` the go/no-go query for kernels: kernels require contiguous rows, and the engine guarantees contiguity by construction (planner offsets + stable activations).
- `fe_tensor_slice(t, axis, start, len)` — a non-owning row/plane/block view; adjusts the base data pointer and the leading stride.
- `fe_tensor_broadcast_to(t, ndim, target)` — a view with stride 0 on the broadcast axes.

**Ownership.** `owns_data == true` tensors free their buffer in `fe_tensor_free`; views never free the parent's. Arena-backed tensors must never be passed to `fe_tensor_free` — the arena owns them.

**Strides** are row-major in elements: `strides[last] = 1`, `strides[i] = strides[i+1] * shape[i+1]`. `fe_tensor_is_contiguous` checks that the stride array equals that formula. This convention (strides in *elements*, last stride = 1) is the one that threads through the planner, the kernels, and the SIMD code, so computing a flat index is always `Σ idx[d] * strides[d]`.

### 2.2 Dtypes

`FeDtype` (`core/types.h`):

| Value | Dtype | Bytes | Note |
|---|---|---|---|
| 0 | `DTYPE_FLOAT32` | 4 | compute format |
| 1 | `DTYPE_INT8` | 1 | quantized weights/activations |
| 2 | `DTYPE_INT32` | 4 | quant accumulation, index tensors |
| 3 | `DTYPE_FLOAT64` | 8 | full-precision passthrough |
| 4 | `DTYPE_FLOAT16` | 2 | **storage only** |
| 5 | `DTYPE_INT16` | 2 | INT16 dynamic-quant path |
| 6 | `DTYPE_BFLOAT16` | 2 | **storage only** |

`fe_dtype_size()` is a single inline switch — the one place that knows element width.

### 2.3 Half-precision storage (`core/fp16.c/h`)

FP16 and BF16 are **storage formats**, not compute formats: low-precision weights halve RAM, and every access upconverts to FP32 before a kernel sees the data.

- `fe_f32_to_fp16`/`fe_fp16_to_f32` — bit-level IEEE binary16 conversion with **round-to-nearest-even**: exact for representable values, saturates to ±Inf on overflow, correct for subnormals and NaN.
- `fe_f32_to_bf16`/`fe_bf16_to_f32` — BF16 is the high 16 bits of FP32; conversion adds `0x7FFF + lsb` (RNE) to the low word before truncation. Keeps the FP32 exponent, 8 mantissa bits → ≤ 2⁻⁸ relative error.
- Bulk variants (`fe_f32_to_fp16_buf`, …) are position-matched and in-place safe.

Bit patterns are pinned in `tests/test_tensor.c` (`test_fp16_roundtrip`, `test_bf16_roundtrip`), including π → 0x4049 for BF16.

### 2.4 Arena allocator (`core/allocator.c/h`)

A bump-pointer arena over a caller-supplied buffer:

```c
typedef struct {
    unsigned char *base;
    size_t         size;
    size_t         offset;  /* bump pointer */
    size_t         peak;
} FeArena;
```

- `fe_arena_alloc` returns `size` bytes aligned to a power-of-two alignment. Per-submission check: `test_allocator` asserts **64-byte alignment** for data, which is what lets SIMD kernels run without misaligned loads/stores.
- `fe_arena_reset` rewinds the bump pointer — all-or-nothing free in O(1).
- `fe_arena_alloc_tensor` arena-allocates the `FeTensor` metadata *and* its data buffer in one call.

The runtime owns two arenas: a **weight arena** (allocated at load, never reset) and an **activation arena** (reset every inference). Ferrite never heap-allocates in the hot path.

### 2.5 Serialization (`core/tensor_ser.c/h`)

A strict little-endian binary blob format for single tensors:

```
magic "FETN" | u32 version | u32 dtype | u32 ndim | int32 shape[ndim] | <data>
```

Readers are strict: bad magic, newer version, unknown dtype, impossible shape, or a `data_len` that overruns the file all fail with `FE_ERR_IO`. Nothing is guessed.

### 2.6 Errors and logging

- **Errors.** `FeStatus` names the failure class (`FE_ERR_NULL/SHAPE/DTYPE/NOMEM/BOUNDS/IO`). Contract: `FE_OK` means all outputs written; callees never free caller memory on error and never partially mutate outputs.
- **Logging.** `fe_log_debug/info/warn/error` (`core/log.h`) write `[ferrite LEVEL file:line] msg` to stderr, filtered at compile time by `FERRITE_LOG_LEVEL` (default `FE_LOG_WARN`). Policy: load/init diagnostics only — kernels never log (hot-path rule).

---

## 3. The Computation Graph (`graph/`)

The graph separates **structure** from **data**: at build time tensors are shape/dtype metadata only; the planner or arena assigns data pointers later.

```c
typedef struct {
    FeNode        nodes  [FE_MAX_NODES];      /* 512 nodes */
    FeTensorEntry tensors[FE_MAX_TENSORS];    /* 1024 tensors */
    int           n_nodes, n_tensors;
    int           topo_order[FE_MAX_NODES];
    int           topo_valid;
} FeGraph;
```

**Tensor entries** (`FeTensorEntry`) carry name, shape, dtype, an `is_weight` flag, the `FeTensor *tensor` (NULL until memory is assigned), and — for per-channel INT8 weights — `scales`/`n_scales` (arena-allocated with the weight they describe).

**Nodes** (`FeNode`) carry `op` (one of 34 `FeOpType` values covering math, activations, GEMM, conv, pooling, norms, and sequence ops), up to 8 inputs / 4 outputs by *tensor index*, and a `union` of op-specific attributes:

```c
union {
    struct { int axis; }                 softmax;
    struct { int stride; int pad; }      conv1d;
    struct { int transA, transB; float alpha, beta; } gemm;
    struct { int groups; float eps; }    groupnorm;
    struct { int num_heads; }            multihead;
    /* ... leaky_relu, elu, conv2d, pool, layernorm, batchnorm */
} attrs;
```

The attributes union keeps kernel parameters on the node — the driver never needs a side table, and the importer stores per-op ONNX attributes here.

**Topological sort** (`fe_graph_topo_sort`) is Kahn's algorithm over the producer relation: build a producer map, count in-degrees (weights and graph inputs don't count), peel in-degree-0 nodes, O(N + E). A cycle → `FE_ERR_SHAPE`.

**Validation** (`fe_graph_validate`) runs after building and after any mutation: every entry has a recognized dtype and a bounded non-negative shape; every node's tensor indices land in-bounds; and **no tensor has more than one producer** (a producer conflict is a build bug, not a runtime issue).

---

## 4. Operator Kernels (`ops/`)

### 4.1 The contract

Every kernel, from `fe_matmul` to `fe_groupnorm`, obeys `ops/ops.h`:

- inputs are read-only,
- the output tensor is **caller-allocated with the correct shape**,
- returns `FE_OK` or an error code,
- **no allocation inside kernels**,
- validate pointers, dtypes, and shapes before touching data.

The one exception is `fe_conv1d` (and conv2d), which allocates an im2col scratch matrix and frees it after use — documented and deliberate: the im2col column matrix *is* the convolution, and its size is bounded by the geometry.

### 4.2 MatMul and Linear

`fe_matmul_scalar` is the **reference implementation**. It exists to validate correctness before SIMD replaces it and as the fallback on non-AVX2 hardware. Its loop order is `i-k-j` so `B` is walked column-wise in the innermost loop — deliberately cache-hostile, which is what makes the AVX2 tiled kernel's win measurable.

`fe_matmul` dispatches at runtime:

```c
FeStatus fe_matmul(const FeTensor *A, const FeTensor *B, FeTensor *C) {
    if (fe_cpu_has_avx2()) {
        FeStatus s = fe_matmul_avx2(A, B, C);
        if (s == FE_OK) return s;       /* fall through on ANY failure */
    }
    return fe_matmul_scalar(A, B, C);
}
```

`fe_linear` is `fe_matmul` followed by a broadcast bias add (`fe_bias_add`, in-place safe).

### 4.3 The op family (Stage 3)

A board of kernels, all validated, all in `ops/`:

- **Elementwise / math:** add, sub, mul, div, pow (+ scalar variants), neg, exp, log; reduce sum/max/min; dot, log-sum-exp.
- **Activations:** relu (→ AVX2), sigmoid, tanh, GELU, leaky ReLU (negative_slope attr), ELU (alpha attr), swish.
- **CNN:** `fe_conv1d`/`fe_conv2d` via **im2col → GEMM**, max/avg pooling.
- **Norms:** batch-norm (inference fold), layer-norm, group-norm — all with the standard `1/sqrt(var + eps)` form.
- **Sequence (transformer-family):** scaled dot-product attention, multi-head attention, embedding lookup, sinusoidal positional encoding.
- **Deterministic random fill:** `fe_rand_uniform`/`fe_rand_normal` with a fixed seedable PRNG — a given seed always yields byte-identical output. Stress tests and reproducible benchmarks depend on this; no system RNG is consulted.

### 4.4 Stability

`fe_softmax` subtracts the row max before `exp`; `fe_logsumexp` and `fe_stable_exp_normalize` exist for numerically fragile paths. Nothing in the kernels silently swallows a NaN or clamps an activation without the op's semantics saying so.

---

## 5. SIMD (`simd/`)

The SIMD library is compiled with `-O3 -mavx2 -mfma` unconditionally (`CMakeLists.txt`); **runtime CPUID decides whether the code ever executes**. `fe_cpu_has_avx2()` gates dispatch, so the same binary runs on any machine.

### 5.1 `fe_matmul_avx2` — tiled FMA GEMM

- 8-wide FMA computes 8 output elements per instruction.
- The **K dimension is tiled** to keep A's row slice resident in registers/L1 while sweeping N, fixing the cache-hostile column access on B that the reference loop suffers.
- **Requirement: `N % 8 == 0`** for the vectorized path; remainder columns fall back to scalar. (N here means the inner dimension of B.) This is exercised end-to-end by `test_simd.c` (vectorized and remainder paths) and `test_linear_chain_remainder` (K=N=19 through the engine).

### 5.2 Elementwise AVX2

`fe_relu_avx2`, `fe_add_avx2`, `fe_mul_avx2` — 8-wide with a scalar tail, so they work for *any* element count. Wired into `fe_relu`/`fe_add`/`fe_mul` behind the same CPUID gate. **Sigmoid is deliberately not vectorized**: an accurate AVX2 `exp()` needs AVX-512ER or a fast-exp approximation, and Ferrite refuses approximations that change numerics. Scalar `expf` remains the sigmoid path.

### 5.3 Alignment

SIMD correctness rests on alignment: planner offsets and arena allocations are **FERRITE_PLANNER_ALIGN-aligned** (default 64 — `FERRITE_PLANNER_ALIGN` in `core/config.h`, threaded via `config.h.in`; scalar-only device ports drop it to 4), asserted by `test_planner.c:test_alignment`, so AVX2 units never see misaligned loads/stores.

The naive scalar matmul is the oracle for the AVX2 kernel — `tests/test_simd.c` compares them element-wise.

---

## 6. The Memory Planner (`planner/`)

### 6.1 Lifetime analysis

`fe_plan_memory(g, plan)` walks the topo order and builds a lifetime table: for each produced (non-weight) tensor, `first_use` = the step that produces it, `last_use` = the step that last consumes it; `size_bytes` from dtype × shape. Consumed-but-unproduced tensors (graph inputs, output nodes' inputs that no node produces) get **no** lifetime — they are provided/owned by the caller or by an earlier layer.

### 6.2 Greedy reuse

Tensors are assigned in order of first use. For each, the planner scans a free-slot pool for the smallest slot whose owning tensor's `last_use < first_use` (its memory is dead) and whose size fits. If none, the slot is carved at the high-water mark. This is interval graph coloring — greedy, not optimal, but O(n²) and provably ≥ naive reuse:

```
Memory savings on the demo model: 1153.69 KB naive → 896.00 KB planned (22.3%)
```

### 6.3 Applying the plan

`fe_plan_apply` reserves the planned **data region first** (`total_activation_bytes`, 64-byte aligned) from the start of the activation arena, *then* arena-allocates the `FeTensor` metadata structs after it. This separation is why offset 0 (the graph-input tensor's data when it has no lifetime) never collides with metadata. Each tensor entry gets a non-owning `FeTensor` pointing at `base + offsets[i]`, with dtype/shape/strides reconstructed from the registry entry. Because metadata is rebuilt per plan apply and the data offsets are pure bytes, the whole plan is **re-applicable on every inference** without touching offsets.

---

## 7. ONNX Import (`importer/`)

`fe_onnx_load(graph, weight_arena, path)` is a from-scratch protobuf pipeline: **no protobuf library**.

### 7.1 The wire reader

`FePbReader` (`onnx.h`) is a cursor over the raw bytes exposing the primitives needed to walk protobuf encoding:

- `fe_pb_varint` — varint with a 10-byte runaway guard,
- `fe_pb_fixed32` — little-endian fixed32 (wire type 5; the FP32 `float_data` fields),
- `fe_pb_tag` — field number + wire type,
- `fe_pb_bytes` — length-delimited payloads,
- `fe_pb_skip` — type-directed skipping for unknown fields (schema drift tolerance).

### 7.2 The loader

The parser walks the `ModelProto` → `GraphProto` (nodes, initializers, inputs, outputs, value_info, `field numbers 11/12/13`) structure. It:

1. registers **initializers** as weight tensors and copies their bytes into the weight arena,
2. registers graph inputs/outputs and value_info as non-weight tensor metadata (unknown dims parse as `0`, a "to be inferred" marker),
3. maps every node's `op_type` string through `op_type_from_string` (~30 supported strings), applies **ONNX defaults** for missing attributes (e.g. `LeakyRelu.alpha=0.01`, conv stride/pad `1`/`0`), stores them into the node `attrs` union,
4. requires critical inputs/attrs before accepting a node, and
5. on success runs `fe_graph_validate` → `fe_optimize` → re-validate, so **loaded models arrive already optimized**.

### 7.3 Fail-loud policy

- Unsupported `op_type` → `(FeOpType)-1` → the loader returns `FE_ERR_SHAPE` **with a stderr message**. Nothing is silently dropped.
- `Gemm` maps to `Linear` and **warns, never guesses**, when non-default `transA`/`transB`/`alpha`/`beta` would change the result.
- ONNX `Conv` is N-D: the importer upgrades `FE_OP_CONV1D` → `FE_OP_CONV2D` when `kernel_shape` has rank ≥ 2.
- `Softmax` with an axis other than the engine's last-axis implementation is rejected explicitly.
- `Gemm` and `BatchNormalization` require their mandatory bias/norm inputs, or the load fails.

Robustness is tested with `test_parser_fuzz`: every truncated prefix (596) plus 400 random byte flips load cleanly or fail loudly — never crash, never hang.

---

## 8. Graph Optimization (`optim/`)

`fe_optimize(g, weight_arena)` runs a wired-in pass pipeline (Stage 12):

```
graph → shape-infer → simplify → fold-constants → fuse-ConvBN
     → CSE → dead-elim → re-sort → re-validate
```

### 8.1 Shape inference

`fe_infer_shapes` iterates the topo order to a fixpoint, one rule per op:

- **same-shape** ops (activations, norm, attention) copy input shape to output,
- **broadcast** ops (add/sub/mul/div/pow) broadcast-merge the two input shapes,
- MATMUL/LINEAR/GEMM merge leading dims and rank-pad scalars, Transpose permutes the shape, CONV1D/2D compute output length from kernel/stride/pad, pooling likewise, Flatten merges dims, Embedding/pos-enc build their output shapes.

Unknown dims stay **0** rather than erroring; geometry contradictions (e.g. matmul's inner dims disagreeing) fail loudly. `fe_infer_shapes` is a *general pass*, wired into both the importer and the engine's rebuild path, closing the value_info gap.

### 8.2 The passes

- **Simplify** — removes identity/flatten no-ops (identity transpose, degenerate flatten).
- **Constant fold** — evaluates constant-only subgraphs at load time, writing new constants into the weight arena.
- **Conv+BN fusion** — folds batch-norm scale/shift into the preceding conv's weights and biases when the BN's inputs are constants. The demo model's Conv+BN pairs fuse away at load — no `BatchNorm` rows remain in its profile.
- **CSE** — common-subexpression elimination.
- **Dead elim** — compacts the node/tensor arrays to remove tensors written but never read.

### 8.3 Conventions

Optimization passes **rewrite dead producers into INPUT feeds** (an op with zero inputs whose outputs are kept) rather than deleting nodes outright, so downstream consumer indices stay valid; dead-elim then compacts the arrays to drop real garbage. `fe_runtime_alloc_weights` skips already-backed weights (`e->tensor != NULL`), so folded/fused constants survive runtime init.

Output equivalence pre/post optimize is proven end-to-end in `tests/test_opt.c` through the engine.

---

## 9. The Runtime (`runtime/`)

### 9.1 `FeRuntime`

```c
struct FeRuntime {
    FeGraph    *graph;
    FeArena     weight_arena;      /* weights: allocated once, never reset */
    FeArena     activation_arena;  /* activations: reset every inference   */
    FeProfiler *profiler;
    FeExecPlan  exec;              /* cached kernel resolution + memory plan */
};
```

`fe_runtime_init` validates the graph (topo sort + validate) and binds caller buffers to both arenas. `fe_runtime_alloc_weights` allocates the weight tensors once.

### 9.2 The static execution plan (`exec_plan.c`, Stage 13)

The engine's dispatch table is a **compile-time constant array indexed by `FeOpType`**:

```c
static FeExecFn const k_fns[] = {
    [FE_OP_INPUT]   = ex_noop,   [FE_OP_MATMUL] = ex_matmul,
    [FE_OP_RELU]    = ex_relu,   [FE_OP_CONV1D] = ex_conv1d,
    [FE_OP_LINEAR]  = ex_linear, [FE_OP_GEMM]   = ex_gemm,
    /* ... every op; unlisted values map to NULL */
};
```

Every `FeExecFn` is a thin wrapper (`EXEC_BEGIN`/`EXEC_END` around the underlying `fe_*` kernel, with optional profiler timing) that hoists its node's inputs/outputs from the registry into an `IN(n)`/`OUT(n)` alias. Wrappers can **branch on the weight dtype** — `ex_matmul`/`ex_linear` route INT8 weights to `fe_matmul_int8_dyn`/`fe_linear_int8`, float weights to the AVX2/scalar path — so a model can mix float and quantized layers.

`fe_exec_build` resolves each node in topo order to its `k_fns` entry (NULL → loud `FE_ERR_SHAPE`, nothing skipped silently) and produces the flat step array:

```c
typedef struct { FeExecFn fn; int node_index; } FeExecStep;  /* FeExecPlan */
```

### 9.3 One inference, in full

1. **First run** (or a run whose input shape differs from the cached `bound_input_shape`): the plan is not valid.
2. **Rebuild path** (`shape_changed`): `reset_produced_shapes` zeroes every non-weight tensor's shape *after saving the graph input's own shape*, then `infer_shapes` re-seeds the input (if it is dynamic/`{0,...}`) from the caller's tensor and re-infers every produced shape, then `fe_plan_memory` recomputes offsets. `n_plan_builds` increments — the counter tests use to assert steady-state runs do *zero* re-analysis.
3. **Every run**: `fe_arena_reset(activation)` → clear `e->tensor` on all non-weight entries → `fe_plan_apply` re-applies the **cached byte offsets** (data region reserved first, `FeTensor` metadata arena-allocated after it) → bind the graph input tensor (found by `find_graph_input`: an explicit `FE_OP_INPUT` node's output, or else the first consumed-but-unproduced non-weight tensor) to the caller's buffer → allocate the OUTPUT node's un-produced input tensor if present → **walk the flat `FeExecStep` array** (a function-pointer call per node; no dispatch switch) → `copy_output` copies the OUTPUT-node input, or else the last planned step's output, into the caller's output tensor.

That is the entire hot path: arena reset, rebinding, a table-walk. The only per-run analysis is the `memcmp` on the input shape.

### 9.4 Batch and calibration

- `fe_runtime_run_batch(rt, inputs[], outputs[], n)` — n samples of the same shape; samples 2..n skip all re-analysis (tests assert `n_plan_builds == 1` after 3 runs).
- `fe_runtime_calibrate(rt, inputs[], n, output, ranges)` — the data-collection half of the quantization calibration pipeline: runs each calibration sample and records, per non-weight float tensor, the **max |x| observed across the sample set**. The ranges are what a static-activation-scale build would consume; the engine today quantizes activations dynamically per run.

---

## 10. Quantization (`quantization/`)

### 10.1 The scheme

Post-training symmetric quantization. `FeQuantParams = { scale, zero_point }` with `zero_point ≡ 0`:

- per-tensor: `scale = max|x| / 127`, `q = clamp(round(x/scale), -127, 127)`;
- per-channel: one scale **per output column** of a `[K, N]` weight matrix, `scales[j] = max_k|x[k][j]| / 127`. Per-channel matters because one global scale is dominated by the loudest channel — on the skewed-weights test, per-channel error is `0.00015` vs per-tensor `0.053`.

`fe_matmul_int8`'s pipeline: quantize A and B → multiply with **int32 accumulation** → dequantize with `scale_A * scale_B[j]`.

### 10.2 Engine integration (`fe_quantize_model`)

Every 2-D float weight consumed as `inputs[1]` of a MATMUL/LINEAR node is per-channel quantized **in place**:

1. the value bytes are repacked into the *existing* buffer (INT8 data is ¼ the size, so it always fits — arena-backed weights are never freed),
2. the scale array is appended to the **weight arena** (advancing the bump pointer past the already-loaded bytes),
3. `e->dtype = DTYPE_INT8`, `e->scales`/`e->n_scales` are recorded on the tensor entry.

Activations stay float; the engine path (`fe_matmul_int8_dyn` / `fe_linear_int8`) quantizes them **dynamically per run** with a two-pass loop (max, then product) and never materializes the quantized activation — no scratch, no allocation, safe for the hot path.

Quantization is a **load-time choice** (`fe_quantize_model` after load/runtime-init), not a compile-time flag.

### 10.3 INT16, FP16, BF16, calibration

- `fe_quantize_int16` / `fe_matmul_int16_dyn` / `fe_linear_int16` mirror the INT8 kernels (per-channel weights, int32 accumulate, dynamic activations) but are **API-complete, not engine-wired**. Note: the INT16 kernel's activation scale currently reuses `max/127` (INT8-style) — a documented quirk, not a silent guarantee.
- FP16/BF16 storage is `core/fp16` (section 2.3), also not engine-wired.
- Calibration (section 9.4) collects activation ranges; static activation scales and percentile smoothing are the not-yet-built consumption side.

The test-suite numbers: MLP float-vs-INT8 max abs error `0.00126`; tiny_mlp weights 268 B → 100 B with `0.00000` max error.

---

## 11. Parallel Execution (`parallel/`)

Stage 11 ships three tested pieces (thread pool, parallel GEMM, DAG scheduler), but the engine hot path stays single-threaded by design:

- **`FeThreadPool`** — fixed workers, FIFO job queue, one mutex + one condvar pair guarding the queue and the outstanding counter. `submit`/`wait` are trivially correct; jobs partition work by integer index. Single-file, zero-dependency apart from pthreads (winpthread on Windows). `fe_cpu_count()` reports usable hardware threads.
- **`fe_matmul_parallel`** — row-parallel GEMM: output rows split into **disjoint** ranges across workers, each worker runs the scalar kernel on an `A`-row slice (`fe_tensor_slice`, an O(1) view). No shared mutable accumulator, so there is no synchronization beyond the join barrier. `pool == NULL` runs the sequential reference path — the same code parallelism is validated against.
- **`FeScheduler`** — a ready-set DAG scheduler: a node becomes executable the moment every input is produced (weights and caller-seeded tensors count as pre-produced); ready nodes run on the pool, a node may discover and resubmit children, and the pool's outstanding counter doubles as the join barrier. The correctness contract is that `exec_order[]` records every node exactly once, after all its producers.

`tests/test_parallel.c` covers pool partition, GEMM-vs-oracle equivalence, and scheduler causality.

---

## 12. The Compiler IR (`compiler/`, Stage 14)

A linear IR **below** the graph: `FeGraph` is lowered to a flat instruction stream where every instruction binds one node to a kernel call.

```c
typedef enum { IR_NONE, IR_INPUT, IR_MATMUL, IR_LINEAR, IR_RELU,
               IR_ADD, IR_SOFTMAX, IR_CONV1D, IR_BATCHNORM, IR_OUTPUT,
               IR_OP_COUNT } FeIrOp;

typedef struct {
    FeIrOp op;
    int    node;          /* source FeGraph node index (for attrs) */
    int    dst;           /* produced tensor index (IR_OUTPUT: none) */
    int    srcs[FE_MAX_NODE_INPUTS];
    int    n_srcs;
    int    dead;          /* set by passes */
} FeIrInstr;
```

- **`fe_ir_lower`** maps each topo-ordered node to an `FeIrInstr` (`op_map` is a 1:1 mirror of the engine's `FeOpType` table — an unmapped op fails loudly at `FE_ERR_SHAPE`), resolves `io_in`/`io_out` with the **same rules as the engine's** `find_graph_input` / last-node-output, and carries `attrs` through `in->node`.
- **`fe_ir_dead_elim`** — mark/sweep: instructions whose produced tensor is never consumed and is not the output are marked dead (with the `dst < 0` guard for `IR_OUTPUT`).
- **`fe_ir_codegen`** — a static kernel table `kfns[IR_OP_COUNT]` (every op has a kernel like `k_matmul`, `k_conv1d` reading stride/pad from the source graph node, `k_batchnorm` reading eps, `k_input`, `k_output`); codegen errors if any op lacks an entry, so a program is either fully runnable or rejected at build time.
- **`fe_ir_schedule`** — producer-before-consumer ordering with a locality heuristic (prefer instructions that reuse the previous instruction's tensors, for cache + arena reuse). The diamond test asserts the causally correct order `0 1 2 3 4`.
- **`fe_ir_run`** executes the stream against an `FeIrCtx` (a live `FeTensor**` array + caller input/output bindings).

`tests/test_compiler.c` proves the compiled IR output matches `fe_runtime_run` to ≤ 1e-5 (two-layer MLP, 1/3 each after softmax) and that the dead-list + diamond schedules are correct. It is the clean seam for future passes (fusion, allocation, threading).

---

## 13. Profiling and Benchmarking (`tools/`)

- **Profiler** (`profiler.c/h`): per-operator timing via `clock_gettime(CLOCK_MONOTONIC)`, accumulated `total_ns`/`call_count` per op (up to 64 ops), sorted report with % time. The engine hooks it around every step when enabled.
- **Bench harness** (`bench.h`): `fe_bench_ms`, `fe_bench_mean` (warm-up then mean over `runs`; use best-of-N for sub-ms kernels), `fe_bench_gflops`, header printing.
- **`bench_model`**: end-to-end latency + per-op profile + memory plan vs naive + a naive-vs-AVX2 table derived from the loaded model's matmul shapes.
- **`bench_avx2`**: micro benches — naive vs AVX2 matmul (with correctness verification), elementwise add, GEMM base vs transB.

### Recorded numbers (`docs/benchmarks.md`)

Environment: AMD Ryzen 5 5600, MinGW GCC 16.2.0, single thread. The demo model (`tests/acousticleaknet.onnx`: 42 MatMul / 63 Conv / 21 softmax) does 5.893 ms/run (169.7 inf/s), 34.83 MFLOP/run, 32.04 MB weights; the planner saves **22.3%** of activation memory. fc1 `[1×65536×128]`: naive 28.5 → AVX2 4.77 ms = **6.0×** (Debug, scalar reference). Release tells a different, honest story: at `-O3` GCC auto-vectorizes the "naive" loop, so on the memory-bound M=1 shape the modular AVX2 kernel no longer wins (0.4×); the real AVX2-vs-scalar gap shows where scalar cannot be auto-vectorized — GEMM transB is 15.3× faster in AVX2. The debug/release split is why numbers must always be reported with the build mode.

---

## 14. Build System and Test Suite

**CMake is canonical** (`CMakeLists.txt`); a Makefile is kept for WSL during the transition. Requirements: GCC or Clang (`-std=c11 -D_POSIX_C_SOURCE=199309L -Wall -Wextra -g`); MSVC is rejected at configure time because the profiler needs the POSIX clock. ASan/UBSan are **probed** at configure time (`check_c_source_compiles`) and enabled only when the toolchain ships them — CLion's bundled MinGW doesn't, MSYS2 and Linux do. `FERRITE_COVERAGE` adds gcov. On Windows every executable gets `libwinpthread-1.dll` copied beside it at build time.

**Libraries, one per subsystem, pointing downward only:**

```
ferrite_core ← ferrite_graph ← ferrite_planner
ferrite_core ← ferrite_graph ← ferrite_parallel
ferrite_core ← ferrite_graph ← ferrite_quant
ferrite_core+graph+ops ← ferrite_optim
ferrite_core+graph+ops ← ferrite_compiler
ferrite_graph+ops+simd+tools+planner+optim+quant ← ferrite_runtime
ferrite_graph+optim ← ferrite_importer
```

**The suite — 16 CTest entries**, one binary per subsystem, each asserting its own invariants:

| Binary | Covers |
|---|---|
| `test_tensor` | strides, transpose/reshape/slice/broadcast views, ownership, dtypes, FP16+BF16 round-trips |
| `test_allocator` | bump allocation, 64-byte alignment |
| `test_ops` / `test_ops3` | kernel math vs hand-computed oracles |
| `test_simd` | naive-vs-AVX2 matmul equivalence, vectorized + remainder paths |
| `test_graph` | registry, topo sort, validation |
| `test_engine` | hand-built + ONNX graphs end-to-end, batch, static plan, chain stress, N%8 remainder chains |
| `test_onnx` | protobuf parse, op mapping, fuzz robustness |
| `test_planner` | lifetimes, reuse, alignment |
| `test_conv1d` | im2col conv paths |
| `test_profiler` | timing capture |
| `test_quant` | quantize/dequantize, INT8 matmul accuracy, per-channel beat-per-tensor, engine-quantized MLP, ONNX end-to-end via INT8, calibration |
| `test_tensor_ser` | strict serialize/deserialize |
| `test_opt` | pass pipeline, engine equivalence pre/post optimize |
| `test_parallel` | pool partition, parallel GEMM vs oracle, scheduler causality |
| `test_compiler` | lower/codegen, IR matches engine (≤1e-5), dead-elim, diamond schedule |

Tests run from the source root (they reference `tests/*.onnx` fixtures by relative path).

---

## 15. The Deep-Cut Invariants

These are the properties the whole project is built around, and the ones to preserve:

1. **No allocation in hot paths.** Kernels never `malloc` (conv's im2col scratch is the sole documented exception, freed after use).
2. **Views over copies.** transpose/reshape/slice never copy data.
3. **Arenas over per-tensor malloc.** Weights live forever; activations reset per inference.
4. **Correctness first.** The naive scalar matmul is the reference that validates the AVX2 path; the engine's naive topo-walk is the reference the static plan and the compiler IR are validated against.
5. **Ownership is explicit.** `owns_data` tensors free; views don't; arena tensors are never freed by hand.
6. **Fail loudly, never silently skip.** Unsupported ONNX ops, unmapped IR ops, NULL `k_fns` entries, un-consumable shapes → `FE_ERR_SHAPE` + stderr. A model either runs fully or refuses to load.
7. **Fixed capacity, no config machinery.** 512 nodes / 1024 tensors / 8 dims; constants in headers; a boolean can't become a feature flag by accident.
8. **Docs move with code.** A change is not done until its doc is done (`explain.md`, `roadmap.md`, this file).

---

## 16. Known Gaps and Next Steps

Tracked honestly in `docs/roadmap.md` and `AGENTS.md`:

- **Half-precision/INT16 are seeded, not shipped.** INT8 is the engine path; INT16/FP16/BF16 are API-complete and tested, but the model-level repack and dispatch wiring for them are not built. The INT16 kernel's activation scale is still INT8-styled (`max/127`).
- **Calibration collects; nothing consumes.** `fe_runtime_calibrate` records per-tensor max-|x| ranges; static activation scales, percentile/stat smoothing, and engine wiring for static scales are the missing second half.
- **Parallel + compiler are optional.** The thread pool, parallel GEMM, and scheduler are tested but the engine hot path is single-threaded; the compiler IR is a validated seam, not the default runner.
- **AVX2 remainder.** `N % 8 != 0` shapes fall back to scalar.
- **Capacity ceilings** (512/1024) and no DMAs — by design for the embedded framing.