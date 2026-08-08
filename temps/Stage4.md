# Stage 4 — Graph System

## Bottom Line Up Front

The graph separates structure from data. Nodes and edges are metadata only; buffers are assigned later by the planner or arena. A valid topological order is the contract the engine and planner both consume. **Done when:** graph construction, cycle detection, and topo sort are tested, and every tensor is referenced by index — never by pointer.

Ferrite's `graph/` already implements most of this stage: `FeGraph`, `FeNode`, the tensor registry, and Kahn's topological sort with cycle detection. Gaps: a first-class edge representation, dependency tracking, explicit graph validation, and visualization.

## Deliverables

- Graph class
- Node class
- Edge class
- Dependency tracking
- Graph validation
- Topological sort
- Execution ordering
- Cycle detection
- Graph visualization

## How to Proceed

1. **Nodes reference tensors by index.** This is already the rule — keep it. Pointers into the registry invalidate when the graph grows; indices survive.
2. **Edges are derived, not stored.** With indices and a `producer` map you get edges for free. Only add an explicit edge class if an analysis needs per-edge attributes (e.g., broadcast flags). Derive before duplicating.
3. **Dependency tracking.** Maintain the producer/consumer maps during build, not in a separate pass. Kahn's algorithm already needs them (`graph/graph.c`).
4. **Validation is a function, not a hope.** Add `fe_graph_validate`: every node input/output index is in range, every non-input tensor has exactly one producer, dtypes are consistent across edges. Run it after load and after any mutation.
5. **Topological sort is the single source of execution order.** One function, one `topo_order[]`. The engine walks it; the planner walks it; nobody computes their own ordering.
6. **Cycle detection is topo sort.** If Kahn's algorithm orders fewer than `n_nodes`, there is a cycle — return `FE_ERR_SHAPE` and name the remaining node. This is already the behavior; test it explicitly.
7. **Visualization last.** A graphviz/dot export of nodes + edges is enough. It is a debug tool, not a feature — do it after validation and ordering are solid.

**Verify.** `tests/test_graph.c` covers: build, add, duplicate-node rejection, a hand-built cycle that must fail, and a DAG whose `topo_order` satisfies every edge (`u` before `v` for all `u → v`).

**Do not** let the graph own data buffers. Nodes carry metadata; arenas carry bytes. That separation is what makes planning and reuse possible.
