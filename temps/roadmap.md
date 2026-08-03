# Ferrite — Roadmap

How we go forward, what the final goal is, and the timeline to get there.

The guiding principle for this roadmap:

> An unfinished-but-correct 10-op runtime with a working demo beats a 40-op runtime where half the ops are silently unimplemented.

---

## 1. Current State

Ferrite is a zero-dependency C11 neural-network inference runtime built "from first principles". Every subsystem has a test binary built under `-fsanitize=address,undefined`. Here is exactly where each piece stands today.

| Subsystem | Files | Status | Gap |
|---|---|---|---|
| Tensor (strided views) | `core/tensor.c/h` | ✅ Done | — |
| Arena allocator | `core/allocator.c/h` | ✅ Done | — |
| Graph IR + topo sort | `graph/graph.c/h` | ✅ Done | — |
| Ops: matmul, linear, relu, softmax, bias_add | `ops/matmul.c`, `ops/activations.c` | ✅ Done | — |
| Conv1d (im2col + matmul) | `ops/conv1d.c` | ✅ Kernel only | **Not dispatched in the engine** |
| Batchnorm | — | ❌ Missing | No kernel exists at all |
| SIMD AVX2 matmul | `simd/matmul_avx2.c/h` | ✅ Done (~13.4×) | Not wired into engine dispatch |
| Memory planner | `planner/memory_planner.c/h` | ✅ Standalone | **`fe_plan_apply` never called** |
| Execution engine | `runtime/engine.c/h` | ✅ Partial | Dispatches 6 ops; CONV1D/BATCHNORM hit `FE_ERR_SHAPE` |
| ONNX importer | `importer/onnx.c/h` | ✅ Partial | Skips shape inference; limited op map; unsupported ops silently dropped |
| INT8 quantization | `quantization/quant.c/h` | ✅ Per-tensor | No per-channel scales; not integrated into a real model run |
| Profiler + benchmarks | `tools/profiler.c/h`, `tools/bench_*.c` | ✅ Done | Benchmarks only cover bare matmul |
| Build | `Makefile` | ⚠️ Linux-only | Windows dev machine + CLion; `.idea/` and stray files untracked |

### Confirmed integration gaps (from reading the code)

- `runtime/engine.c:17` — the `dispatch_node` switch handles `MATMUL`, `LINEAR`, `RELU`, `SOFTMAX`, `ADD`, `FLATTEN`; `CONV1D` and `BATCHNORM` fall through to `FE_ERR_SHAPE`.
- `ops/` has no `batchnorm.c`. The op enum (`graph/graph.h:7`) and the importer may map to it, but there is no kernel.
- `fe_plan_apply` (`planner/memory_planner.h`) is never called from `fe_runtime_run`; activations are bump-allocated from the arena instead.
- The engine always uses the naive `fe_matmul`; the AVX2 kernel is only exercised by `bench_avx2`.
- The importer skips ONNX `value_info` (shape data), so shapes must be known ahead of time.
- Repo hygiene: `.idea/` untracked, two empty stray files (`float32`, `int8`) in the root.

---

## 2. Final Goal (North Star)

> **Ferrite loads a genuinely trained model — a small MNIST classifier exported to ONNX — through its own importer, runs it end-to-end, and produces correct predictions with real accuracy numbers. Nothing depends on anything but C and CPU SIMD.**

The demo — not a unit test — is the proof. Right now the strongest evidence is `test_engine.c`'s hand-built 1→4→8→3 MLP with uniform weights; that's a test, not a demo. The goal is a `demo/` (or `examples/`) directory whose README says, in effect:

> "Trained in PyTorch → exported to ONNX → 97% accuracy in a ~300KB zero-dependency C runtime."

That is the concrete, screenshot-able result this project should exist for.

### Acceptance criteria

1. **Correctness**: a real `.onnx` model loads with shapes inferred from the file, and the float32 output matches ONNX Runtime (or PyTorch) reference outputs within float tolerance on a batch of test inputs. Test-set accuracy on MNIST is reported and close to the model's training number.
2. **Integration gaps closed**: `fe_plan_apply` drives activation memory in `fe_runtime_run`; `CONV1D` and `BATCHNORM` are dispatched, with graph-level tests through the engine (not just bare kernels).
3. **Performance**: the AVX2 matmul (runtime-detected) is the kernel used by matmul/linear; the README carries a benchmark table (naive vs AVX2 vs INT8) with a speedup story, not just one 256×256 number.
4. **One deep feature** (pick one): per-channel INT8 quantization *or* threaded matmul tiles *or* dynamic capacities — done properly and measured, not half-added.
5. **No new dependencies**: everything stays hand-implemented — that is the point of the project.

### Scope guardrails

- **Depth over breadth.** Do not add ops or layer types before the integration gaps and demo are done. Adding Conv2D/LayerNorm/attention is diminishing returns for a portfolio piece.
- **The demo must be achievable with existing ops.** A small MLP MNIST classifier needs only matmul/linear, relu, softmax — already dispatched. A CNN would pull in Conv2D/pooling scope; treat that as a later decision, not a Phase-1 requirement.
- "Production-grade" breadth (threading, AVX-512, every ONNX op, CLI) is explicitly **out of scope** for the 6-week horizon, noted as future work.

---

## 3. The Five Priorities

This roadmap is organized as five priorities, in this order.

### P1. Close the two integration gaps — highest priority, low effort

The hard parts are built and tested; wiring them in is plumbing, not algorithm work, and it turns "here are 10 well-tested modules" into "here is a runtime that actually does the thing end-to-end."

1. **Wire `fe_plan_apply` into `fe_runtime_run`.** The planner is the most sophisticated subsystem; not calling it undercuts the whole planner's credibility. Activations should come from a single planned, reused buffer (arena stays as backing store; the plan fixes offsets).
2. **Dispatch `CONV1D` and `BATCHNORM` in the engine switch.** The kernel and im2col logic exist; add the missing batchnorm kernel (`y = (x − mean) · inv_std · γ + β`), the dispatch cases, and graph-level tests through the engine.

### P2. Ship one real, runnable end-to-end demo

1. Get shape inference working in the importer (parse ONNX `value_info`, add a small shape-propagation pass for the supported ops) so a real model loads without hand-filling shapes.
2. Broaden the op map only as far as the demo needs (`Gemm`, `Reshape`, `Softmax`, `Relu`, `Add`, `Flatten`).
3. Train a small MNIST classifier (PyTorch export to ONNX, or a scripted `.onnx`), load it through Ferrite, and show correct predictions with real accuracy.
4. Create `demo/` with a README: trained → exported → run → accuracy/latency numbers.

### P3. Turn the AVX2 speedup into a benchmark story

1. Route engine dispatch to `fe_matmul_avx2` behind `fe_cpu_has_avx2()`; prove the remainder path for arbitrary `N`.
2. Expand `tools/bench_*` into a small benchmark suite: naive vs AVX2 vs (once P4 lands) INT8, across several sizes, and if possible against NumPy/PyTorch for the same op to give scale.
3. Put the table + a simple bar chart in the README.

### P4. Pick one "hard" feature to go deep on, rather than adding breadth

All three are already seeded in the codebase. Pick exactly one and do it properly:

- **Per-channel INT8 quantization** *(recommended — already flagged as a known limitation; finishing it makes the quant story credible, since per-tensor INT8 is a toy version of what real runtimes do)*.
- **Multi-thread the matmul tiles** — the AVX2 kernel's tiling (MC/KC + packing) makes a simple pthread-per-tile-row split natural; ship it with a measured speedup.
- **Dynamic capacities** — replace the fixed 512-node / 1024-tensor / 8-dim arrays with growable ones to remove the "toy" ceiling and load larger real ONNX models.

### P5. Documentation as a feature, not an afterthought

1. Make the roadmap/explainer (or a tightened version) the `README.md`.
2. Add a single **quickstart**: clone → build → run one command → see output.
3. Render the ASCII architecture diagram as an image.
4. Inline the benchmark table from P3.

---

## 4. Timeline (6 weeks, part-time)

Each week = a few focused sessions. Every phase ends with a passing test under ASan/UBSan. Keep the one-subsystem-per-commit rhythm.

### Phase 0 — Baseline & tooling (days 1–2)

- Build everything (`make all` plus the bench/quant/conv targets), confirm all tests pass.
- Decide and document the build story for this Windows/CLion machine: add a CMakeLists.txt (recommended, CLion-native) **or** standardize on WSL + Makefile. Don't leave it ambiguous.
- Repo hygiene: gitignore `.idea/`, remove stray empty `float32`/`int8` files, delete root binaries.
- Verify `test_onnx` output on `tests/tiny_mlp.onnx` is correct before changing anything.

**Done when:** clean clone → one documented command → all tests green.

### Phase 1 — P1: integration gaps (week 1)

1. Write `ops/batchnorm.c` (`fe_batchnorm`) + test; add any needed attrs to the graph IR.
2. Add the `CONV1D` and `BATCHNORM` cases to `runtime/engine.c`.
3. Wire `fe_plan_apply` into `fe_runtime_run`; add a test asserting the runtime's activation footprint equals the planner's `total_activation_bytes`.
4. Graph-level engine tests for conv1d, batchnorm, and planner-driven memory.

**Done when:** `test_engine` runs graphs containing conv1d and batchnorm; the runtime uses the planner and reports reuse savings.

### Phase 2 — P2: the end-to-end demo (week 2)

1. Parse ONNX `value_info` and add a shape-inference pass over the graph.
2. Broaden the op map to exactly what the demo needs.
3. Train/export a small MNIST classifier to ONNX; load it through Ferrite; verify predictions against reference outputs.
4. Stand up `demo/` with a README and the accuracy number.

**Done when:** a real `.onnx` loads with inferred shapes and produces correct predictions end-to-end.

### Phase 3 — P3: the benchmark story (week 3)

1. Route matmul/linear to `fe_matmul_avx2` behind `fe_cpu_has_avx2()`; validate arbitrary-`N` remainder shapes.
2. Build the benchmark suite across sizes; full-model naive-vs-AVX2 numbers; optional reference comparison.
3. Draft the README benchmark section + bar chart.

**Done when:** README has a reproducible benchmark table with a speedup story.

### Phase 4 — P4: one hard feature, deep (weeks 4–5)

Commit to exactly one of the three candidates. Recommended: **per-channel INT8 quantization** —

1. Extend `quant.c` with symmetric per-channel weight scales + an INT8 conv/matmul path.
2. Run the quantized demo model; measure accuracy delta vs float32 and size reduction.
3. Keep float32 as the default; INT8 becomes a run option.

(If threading or dynamic capacities are chosen instead, the shape of the phase is the same: build → test → measure → document.)

**Done when:** the chosen feature is complete, tested, measured, and written up — not half-implemented.

### Phase 5 — P5: docs as a feature + final validation (week 6)

1. Golden-tensor harness: generate reference outputs with ONNX Runtime/PyTorch, assert Ferrite matches within tolerance at every stage.
2. Final acceptance run: accuracy, latency, memory footprint, quant delta — all recorded.
3. `README.md` rewrite: quickstart, architecture diagram image, benchmark table, demo link, honest limitations.
4. Repo cleanup: remove build artifacts, finalize `.gitignore`, tidy `temps/`.

**Done when:** the north-star acceptance criteria in §2 all pass and are documented.

---

## 5. Immediate Next Steps (ordered)

1. Phase 0: add `.gitignore` entries for `.idea/`, delete stray empty `float32`/`int8`, remove root binaries. *(10 min, unblocks everything)*
2. Phase 0: add a `CMakeLists.txt` for CLion/Windows (or document the WSL path). Get a documented one-command build with all tests green.
3. Phase 1: write `ops/batchnorm.c` + test, then add the `CONV1D`/`BATCHNORM` dispatch cases in `runtime/engine.c`.
4. Phase 1: call `fe_plan_apply` from `fe_runtime_run`; assert footprint == `total_activation_bytes`.
5. Phase 2: parse `value_info` in `importer/onnx.c` + shape-inference pass; export a small trained MNIST model to `.onnx`.
6. Phase 3: route dispatch to `fe_matmul_avx2` behind `fe_cpu_has_avx2()`; build the benchmark suite.
7. Phase 4: decide and commit to one hard feature (recommend per-channel quant); implement, test, measure.
8. Phase 5: golden-tensor harness; README rewrite with quickstart + architecture diagram + benchmark table.

---

## 6. Risks & Decisions to Make Along the Way

- **Reference tooling**: generating golden outputs and exporting a real model needs Python + PyTorch or ONNX Runtime on this machine. That is a *tooling* dependency, not a runtime dependency — acceptable, but confirm it's available before Phase 2.
- **Demo model choice**: a small MLP MNIST classifier needs zero new ops and satisfies P2 cleanly. A CNN (LeNet) would force Conv2D + pooling scope, which violates "depth over breadth" — defer it unless the chosen P4 feature makes it natural.
- **Build platform**: Windows + CLion vs the existing Makefile. Decide CMake-vs-WSL in Phase 0 and don't flip-flop.
- **P4 feature pick**: per-channel quant (recommended) vs threaded tiles vs dynamic capacities. Decide at the start of Phase 4; each changes what "done" looks like. Dynamic capacities and threaded tiles don't conflict with the quant work, so they can follow later.
- **AVX2 portability**: the engine must keep the scalar path as the default so the project stays buildable anywhere; SIMD is a runtime-detected optimization only.

### What not to spend time on (yet)

- Don't chase opset breadth or extra layer types before P1 and P2 are done.
- No Conv2D, LayerNorm, attention, RNNs, or "one more op" — diminishing returns for a portfolio piece.
- Don't silently-skip unsupported ONNX ops in a way that hides breakage; the demo must fail loudly if a model uses something we don't support.

### Out of scope for this roadmap (future work)

- Full ONNX opset coverage / dynamic shapes.
- AVX-512 / ARM NEON kernels.
- A command-line inference tool / CLI.
- 2D conv + pooling (only if a CNN demo becomes a priority).
