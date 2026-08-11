# Stage 11 — Parallel Runtime

## Bottom Line Up Front

Parallelism makes the runtime use all cores by splitting independent work — parallel operators, then whole subgraphs. It is an optimization layer on top of a correct sequential engine, and the sequential result is always the reference. **Done when:** independent nodes or GEMM tiles run in parallel with a measured speedup and identical results.

Ferrite is single-threaded today. This is the roadmap's optional P4 candidate. Start after the engine, planner, and SIMD work are stable.

## Deliverables

- Thread pool
- Task scheduler
- Work stealing
- Parallel operators
- NUMA awareness

## How to Proceed

1. **Sequential first, parallel second.** The engine already produces correct results single-threaded. Parallelism must never change results — only speed. Every parallel path is validated against the sequential path.
2. **A thread pool is the foundation.** Create workers once at runtime init; reuse them for every inference. Spawning threads per call is the classic mistake — a pool of fixed workers with a work queue is the right shape.
3. **Start with parallel operators, not parallel graphs.** The easiest win is inside the big kernels: split GEMM output rows across threads (the AVX2 MC-tile structure makes this natural). A matmul split into 4 row-tiles needs no synchronization beyond a join.
4. **Task scheduler for graph parallelism.** Two passes of work: (a) ready-set tracking over `topo_order` (a node runs when all its inputs are produced), (b) a work queue of ready nodes. Barrier between dependency levels if the scheduler is level-based.
5. **Work stealing only if profiling demands it.** Static level-based scheduling is simpler and often enough for small graphs. Stealing helps load imbalance — add it when measurements show idle workers, not before.
6. **NUMA awareness is the last item, and hardware-specific.** Prefer allocating weight and activation buffers with `alloc` affinity at init and pinning worker threads. Only pursue this on machines where the benchmark shows a NUMA effect.
7. **Data races are the enemy.** All kernel inputs stay read-only (the contract already guarantees this). Shared activation regions across parallel writers must not overlap in lifetime — the planner (Stage 9) already guarantees disjoint lifetimes.
8. **Benchmark the win.** Stage 8 numbers decide: speedup vs. thread count, per-kernel and end-to-end. If parallelism does not help, keep the sequential path and say so.

**Verify.** A multithreaded run matches the sequential run exactly (same tolerance) on a multi-op graph, under ASan/TSan. ThreadSanitizer catches the races that ASan misses — use it for this stage.

**Do not** make the scheduler clever before the kernels are parallel. The GEMM is where the seconds are; graph scheduling is where the microseconds are.
