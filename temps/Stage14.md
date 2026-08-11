# Stage 14 — Compiler Features

## Bottom Line Up Front

Compiler features turn the graph into an executable program: an IR, lowering and optimization passes, kernel selection, and scheduling. This is where Ferrite stops being "a runtime that reads a graph" and becomes "a compiler that emits a run." **Done when:** a graph lowers through a pass pipeline to a scheduled kernel program that runs with correct results.

This stage is aspirational for a zero-dependency C11 runtime. It builds on the optimized graph (Stage 12) and the static execution plan (Stage 13). Go deep on one piece rather than shallow on all five.

## Deliverables

- Intermediate Representation (IR)
- Lowering passes
- Optimization passes
- Kernel generation
- Instruction scheduling

## How to Proceed

1. **The graph is already a high-level IR.** `FeGraph` (nodes + edges + attributes) is your starting IR. The compiler question is how far below it to go. Do not build a second graph; define an IR only for what the graph cannot express.
2. **Lowering is one direction, one step.** High-level graph → linear IR of typed instructions (load, matmul, fused-act, store). Keep it boring: one lowering pass with a table per op, unit-tested instruction by instruction.
3. **Reuse Stage 12's passes as the optimization IR.** Constant folding, dead elimination, fusion all operate on the graph. Run them before lowering. Add IR-level optimizations (e.g., common subexpression in the linear IR) only when the graph-level ones are exhausted.
4. **Kernel generation is selection, not synthesis at first.** Map each lowered instruction to an existing kernel (naive or AVX2) via a codegen table. Real *generation* — emitting loop nests for a fused op — comes later and only for the one op that dominates the profile (matmul).
5. **Instruction scheduling is about memory, not pipelines.** Order instructions to maximize arena reuse and cache locality: schedule readers of a tensor near its writer so lifetimes stay short (this feeds the Stage 9 planner). CPU-level scheduling is premature.
6. **Keep a reference path.** The naive graph-walk engine (Stage 6) stays the reference. Every compiled output is validated against it with the Stage 12 equivalence test.
7. **Ship the pipeline as passes.** `compile(graph) → ir → opt_passes(ir) → schedule → runnable`. Each pass is a function, each has a test, each preserves semantics. Same discipline as Stage 12.
8. **Measure before building.** If the static plan (Stage 13) already gives 95% of the performance, the compiler is research, not shipping. Timebox it; the benchmark table decides whether it earns its complexity.

**Verify.** The compiled program produces identical results to the reference engine on the demo model, with the profiler showing scheduling/dispatch overhead lower than the Stage 13 static plan.

**Do not** build a general compiler. Build the minimum that makes the hottest op and the runtime faster, and prove it with Stage 8 numbers.
