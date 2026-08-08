# Stage 9 — Memory System

## Bottom Line Up Front

The memory system decides where tensor bytes live and reuses them aggressively. Weights live forever in one arena; activations are recycled across an inference. **Done when:** peak activation memory is bounded and planned, not bump-allocated, and cache-friendly layouts are a measured choice.

Ferrite already has the arena allocator (`core/`) and a standalone memory planner (`planner/`). The gap is integration: `fe_plan_apply` is never called, and the runtime still bump-allocates activations.

## Deliverables

- Arena allocator
- Memory pool
- Tensor reuse
- Buffer planner
- Memory alignment
- Cache-friendly layouts

## How to Proceed

1. **Arenas stay the backbone.** Bump-pointer arenas are O(1) and already correct. Do not replace them with a general pool; use the pool for the one thing arenas cannot do — reuse across tensors with disjoint lifetimes.
2. **Wire the planner in.** Call `fe_plan_apply` from the runtime so activations come from one planned buffer at `plan->total_activation_bytes`. This is the roadmap's P1 item and the single highest-value integration in the project. Assert the runtime footprint equals the plan's total.
3. **Lifetime analysis is the input to reuse.** A tensor is alive from its first use to its last use in topo order. Non-overlapping lifetimes share a region. The greedy free-pool in `fe_plan_memory` already implements this — extend it, do not rewrite it.
4. **Tensor reuse is an ownership swap, not a memcpy.** When tensor B reuses tensor A's region, no data moves; A's bytes are simply considered dead. Only the planner's offset bookkeeping changes.
5. **Alignment is a rule, not an option.** Tensors allocated from the arena align to 64 bytes for SIMD. The arena already aligns the actual pointer address, not the offset — preserve that fix. Alignment must be part of every size computation, or reuse breaks.
6. **Cache-friendly layouts are measured, not assumed.** Row-major is the default. Choose a layout only when a benchmark (Stage 8) justifies it — e.g., the AVX2 kernel's packed `B_tile[k][8]` for the SIMD path. Never restructure a tensor "for cache" without a number.
7. **Report the savings.** `fe_plan_print` already shows reuse savings vs. naive allocation. Keep that visible; it is the proof the memory system works.

**Verify.** `tests/test_planner.c` covers lifetime analysis and reuse. Add an engine-level test asserting the runtime's activation footprint equals the planner's `total_activation_bytes`, plus an ASan run proving no tensor outlives its region.

**Do not** let a tensor outlive its planned region. The whole system depends on lifetimes being correct — stress-test the interval math before trusting it.
