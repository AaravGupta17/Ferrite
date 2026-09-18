# Ferrite — Roadmap

## Bottom Line Up Front

**Goal.** Load a genuinely trained model (a small MNIST classifier) exported to ONNX, run it end-to-end in zero-dependency C, and publish real accuracy, latency, and speedup numbers.

**Timeline.** Six weeks, part-time.

"First moves. Wire the memory planner into the runtime. (The AVX2 routing described in earlier drafts of this doc is already done — fe_matmul dispatches to fe_matmul_avx2 via CPUID detection — so the planner is the sole remaining first move.)"

**Guiding rule.** A correct 10-op runtime with a working demo beats a 40-op runtime where half the ops silently fail. Depth over breadth.

---

## 1. Current State

Ferrite is a zero-dependency C11 neural-network inference runtime built from first principles. Each subsystem has a test binary built under `-fsanitize=address,undefined`. The table shows where each piece stands.

| Subsystem | Files | Status | Gap                                                                   |
|---|---|---|-----------------------------------------------------------------------|
| Tensor (strided views) | `core/tensor.c/h` | Done | —                                                                     |
| Arena allocator | `core/allocator.c/h` | Done | —                                                                     |
| Graph IR + topo sort | `graph/graph.c/h` | Done | —                                                                     |
| Ops: matmul, linear, relu, softmax, bias_add | `ops/matmul.c`, `ops/activations.c` | Done | —                                                                     |
| Conv1d (im2col + matmul) | `ops/conv1d.c` | Done | —                                                                     |
| Batchnorm | `ops/norm.c` | Done | —                                                                     |
| Stage 3 op library (Exp/Log/Pow, activations, GEMM/Transpose, Conv2D/Pool, LayerNorm/GroupNorm, Attention/MHA/Embedding/PosEnc) | `ops/math.c`, `ops/gemm.c`, `ops/pool.c`, `ops/conv2d.c`, `ops/sequence.c`, `ops/norm.c`, `ops/rand.c` | Done | —                                                                     |
| SIMD AVX2 matmul | `simd/matmul_avx2.c/h` | Done (~13.4×) | — (wired via fe_matmul; regression-tested in tests/test_simd.c        |
| SIMD elementwise + GEMM | `simd/elementwise_avx2.c/h` | Done | relu/add/mul 8-wide + scalar tail; base-case GEMM dispatch; sigmoid deferred (no accurate AVX2 `exp`)                         |
| Memory planner | `planner/memory_planner.c/h` | Done | Wired into `fe_runtime_run` (plan + apply); reuse proven by footprint test |
| Execution engine | `runtime/engine.c/h` | Done | Dispatches every op in the enum; AVX2 wired; planner-driven activation buffer; ONNX-load→run works end-to-end (demo model: 13 nodes, 0.88 MB activation peak). Hardcoded infer_shapes replaced by `fe_infer_shapes` (optim/) |
| Graph optimization | `optim/*.c/h` | Done — full pass pipeline | `fe_optimize(g, arena)`: shape-infer → simplify → constant-fold → Conv+BN fusion → CSE → dead-elim, wired into load; engine shapes delegated to `fe_infer_shapes`. `tests/test_opt.c` covers every pass + Conv+BN output equivalence |
| ONNX importer | `importer/onnx.c/h` | Partial | Parses `value_info` (GraphProto fields 11/12/13); maps 20+ op strings (math, activations, pool, norms, conv2d); per-op attributes; unsupported ops fail loudly; Gemm→Linear warns on non-default attrs. Runs `fe_infer_shapes` after validate |
| INT8 quantization | `quantization/quant.c/h` | Per-tensor + per-channel, wired into engine | `fe_quantize_model` converts MATMUL/LINEAR 2-D weights to per-channel INT8 (in-place repack, scales in weight arena); exec-plan wrappers route INT8 weights to `fe_linear_int8`/`fe_matmul_int8_dyn`; calibration = `fe_runtime_calibrate` (per-tensor activation-range collection) but no percentile/stat smoothing yet; verified by `test_quantized_engine_mlp` (err 0.0013) + `test_quantized_onnx_end_to_end` (tiny_mlp 268→100 B) |
| INT16 dynamic quant | `quantization/quant.c/h` | API-complete, tested | `fe_quantize_int16`, `fe_matmul_int16_dyn`, `fe_linear_int16` (mirror the INT8 engine kernels); not wired into model repack/dispatch — next step is an `fe_quantize_model` analog |
| FP16 storage | `core/fp16.c/h` | Done | `DTYPE_FLOAT16` (2 bytes), `fe_f32_to_fp16`/`fe_fp16_to_f32` + bulk bufs, bit-exact round-trip test; storage-only, compute happens after FP32 upconvert |
| BF16 storage | `core/fp16.c/h` | Done | `DTYPE_BFLOAT16` (6th dtype, 2 bytes), `fe_f32_to_bf16`/`fe_bf16_to_f32` + bulk bufs (RNE on low 16 bits); pinned round-trip test |
| Calibration | `runtime/engine.c` | DONE (collection half) | `fe_runtime_calibrate` runs a sample set through `fe_runtime_run`, records per non-weight float tensor the observed max |x| across samples; the ranges a static-activation-scale build consumes. Percentile/stat smoothing + static-scale engine wiring remain |
| Parallel execution | `parallel/threadpool.c/h`, `parallel/parallel_gemm.c/h`, `parallel/scheduler.c/h` | Done | Thread pool (mutex+condvar FIFO), row-parallel GEMM (deterministic chunking, no shared mutable ctx), ready-set DAG scheduler (pool resubmits children). `test_parallel.c` covers pool partition, GEMM-vs-oracle, causal order. Opt-in — engine hot path stays single-threaded |
| Compiler IR | `compiler/ir.h`, `lower.c`, `codegen.c`, `schedule.c` | Done | Lower `FeGraph` → linear `FeIrProgram`, dead-elim (mark-and-sweep), kernel table (`fe_ir_codegen`), producer-first schedule. `test_compiler.c` proves IR output matches the engine (≤1e-5) and the diamond schedule is causal |
| Profiler + benchmarks | `tools/profiler.c/h`, `tools/bench.h/c`, `tools/bench_model.c`, `tools/bench_avx2.c` | Done | Shared bench framework; model-relevant bench; Debug + Release numbers recorded in `temps/bench_results.md` (fc1 `[1x65536x128]` naive 28.5→AVX2 4.77 ms = 6.0× Debug; planner saves 22.3% activations). `bench_avx2 --json` feeds the CI perf gate |
| Build | `CMakeLists.txt` (canonical), `Makefile` | Windows/CLion + WSL | `cmake -S . -B build -G Ninja`; sanitizers auto-probed (ASan unavailable in MinGW — built without); `FERRITE_COVERAGE` option; `ferrite_parallel` (Threads::Threads) + `ferrite_compiler` libs wired; 19 CTest entries, all green |
| CI | `.github/workflows/ci.yml` | Authored, mostly CI-only; golden locally verifiable | test / sanitize (mandatory ASan+UBSan) / fuzz (libFuzzer, corpus-seeded) / perf (`bench_avx2` vs `temps/bench_baseline.json`, 20% threshold; leg now uploads the measured JSON for baseline re-seeding from the ubuntu runner) / coverage (gcovr `--fail-under-line 70`) / pi (ARMv6 cross + QEMU ctest) / esp (IDF container compile gate) / golden (`tools/golden_gen.py` → `tools/golden_compare.py` runs the zoo through ONNX Runtime and Ferrite, tolerance-matched; runnable on the dev host with `onnxruntime`+`onnx` pip packages). Baseline regeneration + observation on the Ubuntu runner still pending first push |
| Pi Zero W port | `cmake/raspberry-pi-zero-w.toolchain.cmake`, `docs/pi-zero-w-port.md` | Done (build system; CI-verified under QEMU) | ARMv6/hard-float preset; x86-only `-mavx2 -mfma` gated + FATAL_ERROR on non-x86 AVX2 requests; parallel stays ON; INT8-quantized model default; NEON backend only on Pi Zero 2 W+ (stretch) |
| ESP32 port | `idf/ferrite/` (component), `core/platform_esp32.c`, `cmake/esp32.toolchain.cmake`, `ferrite_compile --target esp32`, `docs/esp32-port.md`, `idf/demo` | Done (component + CI compile gate) | device-runtime-only; FLOAT64 rejected at FEMD load; ceilings via `idf/ferrite/config/config.h`; `FERRITE_PLANNER_ALIGN=4` for scalar; run on hardware (manual): relu smoke + quantized-FEMD diff vs desktop |

**Confirmed gaps, by location:**

- `runtime/engine.c` — dispatches every `FeOpType`; planner-driven activations; graph validation runs inside `fe_onnx_load` and `fe_runtime_init`. AVX2 remainder path closed (vectorized prefix + scalar tail, `tests/test_simd.c`).
- **`infer_shapes` is now a general pass.** `optim/shape_infer.c` propagates shapes through the topo order with one rule per `FeOpType` (stage-3 ops included), to a fixpoint; unknown dims stay unresolved rather than erroring, and geometry contradictions fail loudly. Wired into both `fe_onnx_load` and `fe_runtime_run`, closing the importer's value_info gap.
- Unsupported ONNX ops return `FE_ERR_SHAPE` loudly — nothing is silently dropped.
- **Phase B hardening.** `fe_graph_validate` accepts the full engine dtype set (validated FLOAT16 `DTYPE_FLOAT16`, `DTYPE_INT16`, `DTYPE_BFLOAT16` alongside FLOAT32/INT8/INT32/FLOAT64); the ESP32 IDF component compiles the complete op-source set (`conv2d/elementwise/gemm/math/norm/pool/sequence` added — the device `k_fns[]` table needs them all); `FE_MAX_ALLOCS >= FE_MAX_TENSORS` is now a compile-time invariant (`memory_planner.h` `_Static_assert` + CMake configure-time `FATAL_ERROR`).
- Dynamic ONNX dimensions (`dim_param`) parse as 0 and rely on runtime shape inference.
- INT16/FP16/BF16 are API-complete but not yet wired into the engine dispatch or model repack. Calibration collects per-tensor activation ranges; static activation scales and percentile smoothing are not yet consumed by the engine.
- Parallel kernels and the compiler IR exist and are tested but are not yet integrated into `fe_runtime_run` — the engine hot path remains single-threaded.
- Ports: no NEON matmul (needs a Pi Zero 2 W+ target); ESP32 hardware validation is a documented manual step; CI legs and the baseline in `temps/bench_baseline.json` are authored, not observed.
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
2. Build story: add `CMakeLists.txt` for CLion/Windows. *(done — `cmake -S . -B build -G Ninja`)*
3. Wire `fe_plan_apply` into `fe_runtime_run`; assert footprint equals `total_activation_bytes`. *(done — Phase 1)*
4. Parse `value_info` in `importer/onnx.c`; broaden the op map; fail loudly on unsupported ops. *(done — Phase 3)*
5. ~~**Full shape-inference pass**~~ (Stage 12): one shape table keyed by op, propagating from inputs through every supported op. Closes the importer's value_info reliance on static shapes. **Done** — and with it the whole Stage 12 optimizer (folding, Conv+BN fusion, CSE, simplification, dead elimination) ships as `fe_optimize`, wired into `fe_onnx_load`.
6. ~~**Static execution plan**~~ (Stage 13): dispatch is dead — `fe_runtime_run` walks a flat array of kernel function pointers resolved at plan-build time, executes a cached memory plan, and re-plans only when the input shape changes. **Done** — `runtime/exec_plan.c/h`, embedded in `FeRuntime`; `test_exec_plan_static` proves steady-state runs do zero re-analysis. Prefetching (the one deferred Stage 13 item) waits for a benchmark that shows it paying for itself.
7. ~~**Parallel runtime**~~ (Stage 11): thread pool, row-parallel GEMM, and a ready-set DAG scheduler in `parallel/`; `fe_matmul_parallel` chunking is deterministic (no shared mutable context) so a chunked GEMM matches the scalar oracle; `test_parallel.c` covers pool partition, GEMM equivalence, and scheduler causality. **Done** — `ferrite_parallel` links `Threads::Threads`; engine hot path remains single-threaded by design.
8. ~~**Compiler IR**~~ (Stage 14): `fe_ir_lower` flattens `FeGraph` into a linear `FeIrProgram`; dead-elim, a static kernel table (`fe_ir_codegen`), and a producers-first scheduler (`fe_ir_schedule`) round out the pipeline; `test_compiler.c` proves compiled output matches the engine. **Done** — clean seam for future passes (fusion, allocation, threading).
9. ~~**Persistent/tiling improvements**~~ (Stage 8): `bench_avx2` extended with elementwise + GEMM transposes; Debug and Release numbers recorded in `temps/bench_results.md`. **Done** — remainders still fall back to scalar (documented).
10. ~~**FP16 + INT16**~~ (Stage 15 extras): `core/fp16.h/c` (bit-exact FP16 **and** BF16 storage) and `fe_quantize_int16`/`fe_matmul_int16_dyn`/`fe_linear_int16`. **Done** (API-complete); calibration now collects per-tensor activation ranges (`fe_runtime_calibrate`), and engine dispatch wiring + static activation scales are the remaining follow-ups.
11. Route dispatch to `fe_matmul_avx2` behind `fe_cpu_has_avx2()`; build the benchmark suite. *(dispatch already wired; see `temps/bench_results.md`)*
12. Decide one hard feature (recommend per-channel quant); implement, test, measure. *(per-channel quant done + measured; the natural next pick is engine-wired INT16 or threaded engine dispatch)*
13. Build the golden-tensor harness; rewrite the README.

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
