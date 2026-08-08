# Stage 13 — Runtime Optimization

## Bottom Line Up Front

Runtime optimization makes the *execution path* fast — static plans, kernel reuse, and memory planning — as opposed to the kernels themselves. The engine stops deciding and starts executing a prepared plan. **Done when:** a static execution plan runs an inference with no per-run analysis and predictable memory.

Ferrite already has the memory planner (Stage 9's core) and a deterministic topo order. This stage turns "walk topo_order and dispatch" into "execute a precomputed plan."

## Deliverables

- Lazy execution
- Kernel caching
- Memory planner
- Static execution plan
- Dynamic execution plan
- Prefetching

## How to Proceed

1. **Lazy execution is a policy, not a feature.** Decide at load time which subgraphs are computed on demand vs. eagerly. For inference, eager is usually right; lazy pays off for branchy conditional graphs. Measure before adopting it.
2. **Kernel caching avoids re-dispatch.** At plan build time, resolve each node to a concrete function pointer and validate its shapes once. At run time the loop is a plain array of calls, not a `switch`. This removes dispatch cost without changing semantics.
3. **Memory planner is the anchor.** `fe_plan_memory` → `fe_plan_apply` gives one activation buffer at `total_activation_bytes` with reuse. This is the roadmap P1 item and the prerequisite for a static plan. Do this before anything else in this stage.
4. **A static execution plan is built once.** At load: topo sort → validate → assign memory offsets → resolve kernels → record input/output bindings. The result is a flat array of {kernel fn, input ptrs, output ptr}. Run time is one loop. This is the natural endpoint of the current engine design.
5. **Dynamic plans are for dynamic shapes.** When shapes change per input (Stage 24), a static plan breaks. Keep a rebuild path that re-plans memory and re-resolves kernels on shape change, and make it rare by reusing the plan across same-shape runs.
6. **Prefetching only after profiling.** Prefetching next-tensor data while the current kernel runs helps memory-bound kernels. Use `prefetch` hints in the hot GEMM loops only if Stage 8 shows a memory-bound bottleneck — it is not free on all CPUs.
7. **Keep the arena reset.** Even with a static plan, activation memory is reset per inference; the plan just fixes *where* activations live. The two compose, they do not compete.
8. **Report the savings.** A runtime that uses the planner reports `total_activation_bytes` and reuse savings vs. naive. This number belongs in the benchmark table (Stage 8).

**Verify.** An engine test asserts: (a) execution results are identical with and without the static plan, (b) activation footprint equals the plan's total, (c) the dispatch loop does no per-run analysis. The profiler should show the dispatch overhead gone.

**Do not** build a static plan before the memory planner is wired in. The plan's memory layout is the hard part; the flat dispatch loop is the easy part.
