# Stage 26 — Dream Features

## Bottom Line Up Front

Dream features are the far horizon: backend compilers, kernel auto-generation, and hardware targets. They are each a project-sized commitment that changes Ferrite's identity from "hand-written runtime" to "compiler-backed runtime." **Done when:** one chosen feature reaches a working, tested, measured demo — not a prototype slide.

None of these are on the critical path. They are listed so the plan is honest about what exists beyond it. Revisit only after Stages 12–14 show the current performance ceiling.

## Deliverables

- MLIR backend
- LLVM integration
- Automatic kernel generation
- TVM-style optimization
- XLA-style compiler
- WebGPU backend
- FPGA backend
- ARM NEON backend
- Apple Metal backend
- RISC-V backend

## How to Proceed

1. **ARM NEON is the only one that changes the shipping product.** The CPU story is incomplete without ARM: most inference runs on phones. A NEON matmul is the AVX2 work (Stage 10) repeated with different intrinsics behind the same `fe_cpu_has_*` detection. Do this first if any dream feature is pursued.
2. **Automatic kernel generation is the natural continuation of Stage 14.** Given a fused op (shape, layout, simd width), emit the loop nest and tune tile sizes automatically. TVM-style and XLA-style are the two flavors of this same idea; study them as prior art, not as dependencies.
3. **MLIR/LLVM integration abandons "zero-dependency" deliberately.** Both pull large toolchains into the build. Treat them like CUDA (Stage 16): optional build targets that link only when requested. MLIR gives you the pass infrastructure for free; LLVM gives you codegen. The cost is complexity and build time.
4. **Backends (WebGPU, Metal, FPGA, RISC-V) all follow the same rule as CUDA.** They are optional backends behind the Stage 16 backend interface, validated against the CPU reference, with measured speedups. No backend ships without its equivalence test and benchmark.
5. **FPGA is a different category.** It is hardware engineering: kernels become HDL, and the CPU→FPGA transfer story dominates. Only pursue with a hardware partner or a serious reason.
6. **Every dream feature is timeboxed and scored.** Before starting one, write down: what problem it solves, what it replaces, what it depends on, and what "done" looks like. Kill it if the demo does not move the benchmark.
7. **Keep the reference path alive.** Whatever backend or compiler lands, the naive graph-walk engine (Stage 6) remains the correctness oracle. That invariant is what makes new backends safe to try.
8. **Document the dead ends.** A dream feature that was evaluated and rejected is a valuable writeup (Stage 22) — it saves the next person from re-deriving the conclusion.

**Verify.** The chosen feature: same graph on CPU-reference and the new backend/compiler, outputs within tolerance, and a benchmark that justifies the complexity — or a documented rejection.

**Do not** start any of these before the roadmap's P1–P4 work is done. The project's rule is depth over breadth; dream features are breadth at its most expensive.
