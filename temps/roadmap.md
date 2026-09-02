# Ferrite — Roadmap

## Bottom Line Up Front

**Goal.** Load a genuinely trained model (a small MNIST classifier) exported to ONNX, run it end-to-end in zero-dependency C, and publish real accuracy, latency, and speedup numbers.

**Timeline.** Six weeks, part-time.

**First moves.** Wire the memory planner into the runtime and route the engine's matmul/linear through the AVX2 kernel. Then ship a runnable demo. (Stages 0-3 — tensor, math backend, and the full operator library — are complete; the ops dispatch gap is closed.)

**Guiding rule.** A correct 10-op runtime with a working demo beats a 40-op runtime where half the ops silently fail. Depth over breadth.

---

## 1. Current State

Ferrite is a zero-dependency C11 neural-network inference runtime built from first principles. Each subsystem has a test binary built under `-fsanitize=address,undefined`. The table shows where each piece stands.

| Subsystem | Files | Status | Gap |
|---|---|---|---|
| Tensor (strided views) | `core/tensor.c/h` | Done | — |
| Arena allocator | `core/allocator.c/h` | Done | — |
| Graph IR + topo sort | `graph/graph.c/h` | Done | — |
| Ops: matmul, linear, relu, softmax, bias_add | `ops/matmul.c`, `ops/activations.c` | Done | — |
| Conv1d (im2col + matmul) | `ops/conv1d.c` | Done | — |
| Batchnorm | `ops/norm.c` | Done | — |
| Stage 3 op library (Exp/Log/Pow, activations, GEMM/Transpose, Conv2D/Pool, LayerNorm/GroupNorm, Attention/MHA/Embedding/PosEnc) | `ops/math.c`, `ops/gemm.c`, `ops/pool.c`, `ops/conv2d.c`, `ops/sequence.c`, `ops/norm.c`, `ops/rand.c` | Done | — |
| SIMD AVX2 matmul | `simd/matmul_avx2.c/h` | Done (~13.4×) | Not wired into engine dispatch |
| Memory planner | `planner/memory_planner.c/h` | Standalone | **`fe_plan_apply` never called** |
| Execution engine | `runtime/engine.c/h` | Partial | Dispatches every op in the enum; planner + AVX2 not wired in |
| ONNX importer | `importer/onnx.c/h` | Partial | Skips shape inference; limited op map; silently drops unsupported ops |
| INT8 quantization | `quantization/quant.c/h` | Per-tensor | No per-channel scales; not integrated into a real model run |
| Profiler + benchmarks | `tools/profiler.c/h`, `tools/bench_*.c` | Done | Benchmarks cover bare matmul only |
| Build | `Makefile` | Linux-only | Windows + CLion machine; `.idea/` and stray files untracked |

**Confirmed gaps, by location:**

- `runtime/engine.c` — every `FeOpType` has a dispatch case (Stage 3 adds math, activation, GEMM/transpose, conv2d/pool, norm, and sequence ops). Errors propagate, nothing is silently skipped. Remaining gap: the engine still calls the naive `fe_matmul`, not the AVX2 kernel.
- `fe_plan_apply` (`planner/memory_planner.h`) is never called from `fe_runtime_run`. Activations come from arena bump-alloc instead.
- The engine always uses the naive `fe_matmul`. Only `bench_avx2` exercises the AVX2 kernel.
- The importer skips ONNX `value_info` (shape data). Shapes must be known ahead of time.
- Repo hygiene: `.idea/` untracked; two empty stray files (`float32`, `int8`) in the root.

---

## 2. Goal and Acceptance Criteria

**North star.** Ferrite loads a real trained ONNX model through its own importer, runs it end-to-end, and produces correct predictions with real accuracy numbers. Nothing depends on anything but C and CPU SIMD.

The proof is a demo, not a unit test. Today the strongest evidence is `test_engine.c`'s hand-built 1→4→8→3 MLP with uniform weights — a test, not a demo. The goal is a `demo/` (or `examples/`) directory whose README says, in effect:

> "Trained in PyTorch → exported to ONNX → 97% accuracy in a ~300KB zero-dependency C runtime."

**Acceptance criteria.**

1. **Correctness.** A real `.onnx` model loads with shapes inferred from the file. Float32 output matches ONNX Runtime (or PyTorch) reference output within tolerance on a batch of inputs. Test-set accuracy on MNIST is reported and close to the training number.
2. **Integration gaps closed.** `fe_plan_apply` drives activation memory in `fe_runtime_run`, and the AVX2 matmul is wired into the engine so it actually powers matmul/linear. Both are covered by graph-level tests through the engine.
3. **Performance.** The AVX2 matmul (runtime-detected) powers matmul and linear. The README carries a benchmark table — naive vs AVX2 vs INT8 — with a speedup story, not one 256×256 number.
4. **One deep feature.** Pick per-channel INT8 quantization, threaded matmul tiles, or dynamic capacities. Do it properly and measure it.
5. **No new dependencies.** Everything stays hand-implemented. That is the point.

**Scope guardrails.**

- Depth over breadth. No new ops or layer types before the gaps and demo are done. Conv2D, LayerNorm, and attention are diminishing returns.
- The demo must run on existing ops. A small MLP needs only matmul/linear, relu, softmax — all dispatched. A CNN would pull in Conv2D and pooling scope; defer it.
- Production-grade breadth — threading, AVX-512, every ONNX op, CLI — is out of scope for this horizon.

---

## 3. Priorities

### P1. Close the two integration gaps — highest priority, low effort

The hard parts are built and tested. Wiring them in is plumbing, not algorithm work. It turns "ten well-tested modules" into "a runtime that works end-to-end."

1. **Wire `fe_plan_apply` into `fe_runtime_run`.** The planner is the most sophisticated subsystem. Not calling it undercuts its credibility. Activations should come from one planned, reused buffer. The arena stays as backing store; the plan fixes offsets.
2. **Dispatch `CONV1D` and `BATCHNORM`.** The conv1d kernel and im2col logic exist. Write the missing batchnorm kernel (`y = (x − mean) · inv_std · γ + β`), add dispatch cases, and test through the engine.

### P2. Ship one real, runnable end-to-end demo

1. Add shape inference to the importer — parse ONNX `value_info` and propagate shapes through each supported op.
2. Broaden the op map only as far as the demo needs: `Gemm`, `Reshape`, `Softmax`, `Relu`, `Add`, `Flatten`.
3. Train a small MNIST classifier, export it to ONNX, load it through Ferrite, and show correct predictions with real accuracy.
4. Create `demo/` with a README: trained → exported → run → accuracy and latency numbers.

### P3. Turn the AVX2 speedup into a benchmark story

1. Route engine dispatch to `fe_matmul_avx2` behind `fe_cpu_has_avx2()`. Prove the remainder path for arbitrary `N`.
2. Expand `tools/bench_*` into a suite: naive vs AVX2 vs (after P4) INT8, across several sizes. Compare against NumPy or PyTorch for the same op to give scale.
3. Put the table and a simple bar chart in the README.

### P4. Pick one "hard" feature and go deep

All three candidates are already seeded in the code. Pick one.

- **Per-channel INT8 quantization** *(recommended)* — already a known limitation. Finishing it makes the quant story credible; per-tensor INT8 is a toy version of what real runtimes do.
- **Threaded matmul tiles** — the AVX2 kernel's tiling (MC/KC + packing) makes a pthread-per-tile-row split natural. Ship it with a measured speedup.
- **Dynamic capacities** — replace the fixed 512-node / 1024-tensor / 8-dim arrays with growable ones. Removes the "toy" ceiling and loads larger real models.

### P5. Documentation as a feature, not an afterthought

1. Make this roadmap or the explainer — tightened — the `README.md`.
2. Add a single quickstart: clone → build → run one command → see output.
3. Render the ASCII architecture diagram as an image.
4. Inline the P3 benchmark table.

---

## 4. Timeline (6 Weeks, Part-Time)

Each week means a few focused sessions. Every phase ends with passing tests under ASan/UBSan. Keep the one-subsystem-per-commit rhythm.

### Phase 0 — Baseline and tooling (days 1–2)

- Build everything (`make all` plus bench, quant, conv targets). Confirm all tests pass.
- Settle the build story for this Windows/CLion machine: add a CMakeLists.txt (recommended, CLion-native) **or** standardize on WSL + Makefile. Pick one; do not flip-flop.
- Hygiene: gitignore `.idea/`; remove stray empty `float32` and `int8`; delete root binaries.
- Verify `test_onnx` on `tests/tiny_mlp.onnx` before changing anything.

**Done when:** clean clone → one documented command → all tests green.

### Phase 1 — P1: integration gaps (week 1)

1. Write `ops/batchnorm.c` + test. Add any needed attributes to the graph IR.
2. Add `CONV1D` and `BATCHNORM` cases to `runtime/engine.c`.
3. Wire `fe_plan_apply` into `fe_runtime_run`. Add a test asserting activation footprint equals the planner's `total_activation_bytes`.
4. Add graph-level engine tests for conv1d, batchnorm, and planner-driven memory.

**Done when:** the engine runs conv1d and batchnorm graphs; the runtime uses the planner and reports reuse savings.

### Phase 2 — P2: the end-to-end demo (week 2)

1. Parse ONNX `value_info`; add a shape-inference pass.
2. Broaden the op map to exactly what the demo needs.
3. Train and export a small MNIST classifier to ONNX. Load it through Ferrite. Verify predictions against reference outputs.
4. Stand up `demo/` with a README and the accuracy number.

**Done when:** a real `.onnx` loads with inferred shapes and produces correct predictions end-to-end.

### Phase 3 — P3: the benchmark story (week 3)

1. Route matmul/linear to `fe_matmul_avx2` behind `fe_cpu_has_avx2()`. Validate arbitrary-`N` shapes.
2. Build the benchmark suite across sizes. Produce full-model naive-vs-AVX2 numbers; compare against a reference where possible.
3. Draft the README benchmark section and bar chart.

**Done when:** the README has a reproducible benchmark table with a speedup story.

### Phase 4 — P4: one hard feature, deep (weeks 4–5)

Commit to exactly one candidate. Recommended: **per-channel INT8 quantization**.

1. Extend `quant.c` with symmetric per-channel weight scales and an INT8 conv/matmul path.
2. Run the quantized demo model. Measure accuracy delta vs float32 and size reduction.
3. Keep float32 as the default; make INT8 a run option.

If threading or dynamic capacities are chosen, the shape is the same: build → test → measure → document.

**Done when:** the chosen feature is complete, tested, measured, and written up — not half-implemented.

### Phase 5 — P5: docs and final validation (week 6)

1. Build a golden-tensor harness: generate reference outputs with ONNX Runtime or PyTorch; assert Ferrite matches within tolerance at every stage.
2. Final acceptance run: accuracy, latency, memory footprint, quant delta. Record all of it.
3. Rewrite `README.md`: quickstart, architecture diagram image, benchmark table, demo link, honest limitations.
4. Clean the repo: remove build artifacts, finalize `.gitignore`, tidy `temps/`.

**Done when:** every acceptance criterion in Section 2 passes and is documented.

---

## 5. Immediate Next Steps (Ordered)

1. Hygiene: gitignore `.idea/`; delete stray empty `float32`/`int8`; remove root binaries. *(10 minutes; unblocks everything.)*
2. Build story: add `CMakeLists.txt` for CLion/Windows, or document the WSL path. Get a documented one-command build with all tests green.
3. Write `ops/batchnorm.c` + test; add `CONV1D`/`BATCHNORM` dispatch in `runtime/engine.c`.
4. Call `fe_plan_apply` from `fe_runtime_run`; assert footprint equals `total_activation_bytes`.
5. Parse `value_info` in `importer/onnx.c`; add shape inference; export a small trained MNIST model.
6. Route dispatch to `fe_matmul_avx2` behind `fe_cpu_has_avx2()`; build the benchmark suite.
7. Decide one hard feature (recommend per-channel quant); implement, test, measure.
8. Build the golden-tensor harness; rewrite the README.

---

## 6. Risks and Decisions

- **Reference tooling.** Goldens and model export need Python + PyTorch or ONNX Runtime on this machine. That is a tooling dependency, not a runtime one. Acceptable — but confirm it exists before Phase 2.
- **Demo model.** An MLP MNIST classifier needs zero new ops and satisfies P2 cleanly. A CNN (LeNet) forces Conv2D + pooling scope, violating depth-over-breadth. Defer unless the chosen P4 feature makes it natural.
- **Build platform.** Windows + CLion vs the existing Makefile. Decide in Phase 0; do not flip-flop.
- **P4 pick.** Per-channel quant (recommended) vs threaded tiles vs dynamic capacities. Decide at the start of Phase 4. The last two do not conflict with quant work and can follow later.
- **AVX2 portability.** Keep the scalar path as the default. SIMD is a runtime-detected optimization only.

**Do not spend time on (yet):**

- Opset breadth or extra layer types before P1 and P2 are done.
- Conv2D, LayerNorm, attention, RNNs, or "one more op."
- Silently skipping unsupported ONNX ops. Fail loudly if a model uses something unsupported.

**Out of scope (future work):**

- Full ONNX opset coverage and dynamic shapes.
- AVX-512 and ARM NEON kernels.
- A command-line inference tool.
- 2D conv and pooling (only if a CNN demo becomes a priority).
