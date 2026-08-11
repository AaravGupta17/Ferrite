# Stage 24 — Future Runtime

## Bottom Line Up Front

The future runtime handles shapes and scale the current design does not: dynamic shapes, automatic scheduling and tuning, JIT, and distributed inference across multiple machines. Each item is a research-sized project on its own. **Done when:** the chosen item (pick one at a time) works, is measured, and is written up — not half-implemented.

This stage is the horizon. It assumes Stages 13 and 14 (static plans, compiler features) are the runtime's foundation. Depth over breadth applies harder here than anywhere else.

## Deliverables

- Dynamic shapes
- Auto scheduler
- Auto tuning
- JIT compilation
- Distributed inference
- Tensor parallelism
- Pipeline parallelism
- Model sharding

## How to Proceed

1. **Dynamic shapes is the precondition for the rest.** Static plans break when shapes change per input. The design already hints at it (Stage 13): detect a shape change, re-plan memory, re-resolve kernels, reuse the plan across same-shape runs. `FeTensorEntry` already stores shape; add a `dirty` marker and a re-plan path.
2. **Auto scheduler learns from the profiler.** Collect per-op timing (the profiler already does) and pick the kernel variant (naive vs. AVX2 vs. fused) and execution order that minimizes cost. This is a small search over the static plan, not an AI system. Measure the win before believing it.
3. **Auto tuning is offline and one-shot.** Tune tile sizes (MC/KC), packing, and threshold for AVX2-vs-scalar on the target machine at install/build time, store the result, use it. Never tune at run time. The Stage 22 cache analysis is the guide to which knobs matter.
4. **JIT compiles the hot path.** For the fused kernel that dominates (matmul/fused-act), emit and compile a specialized loop at load time. This reuses the compiler infrastructure from Stage 14. Only pursue when the static plan + SIMD ceiling is reached.
5. **Distributed inference is for models that do not fit one machine.** Split by tensor parallelism (layers are too big for one device) or pipeline parallelism (layers run on different devices in sequence). Both need a messaging layer Ferrite does not have — this is effectively a new project.
6. **Tensor parallelism needs sharded GEMM.** Split K/N of the matmul across workers, partial-sum reduce, all-reduce between layers. The AVX2 tiling structure is the natural shard boundary. Correctness first: single-node output is the reference.
7. **Pipeline parallelism needs batching.** Layer N on device A, N+1 on device B, with microbatches overlapping stages (GPipe-style). Start from the batch-execution loop in Stage 6.
8. **Model sharding is the packaging side.** Split weights across devices at load and reconstruct the logical graph in memory. It is storage/communication engineering, not math.

**Verify.** The chosen item has a test that matches single-node output within tolerance, and a benchmark that reports the scale-up: dynamic-shape overhead, scheduler win, tuning speedup, or distributed throughput per worker.

**Do not** attempt all eight. This stage is a menu, not a checklist. Each item is a separate project with its own "done when."
