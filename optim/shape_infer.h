// optim/shape_infer.h
#ifndef FERRITE_SHAPE_INFER_H
#define FERRITE_SHAPE_INFER_H

#include "graph.h"

/*
 * General shape-inference pass (Stage 12).
 *
 * Propagates tensor shapes forward through the graph, keyed by operator, in
 * a single per-op rule table. Runs in place on the graph's tensor registry:
 * every tensor that already has a concrete (all-dims-positive) shape is left
 * untouched, `{...}` unknowns are filled from the producer's rule once all
 * its inputs are concrete, and the walk repeats to a fixpoint so ordering
 * never matters.
 *
 * Only genuine geometry contradictions (e.g. a MatMul whose inner dims don't
 * line up) return FE_ERR_SHAPE. Unknown dims are never an error — a dim that
 * cannot be resolved yet simply stays unresolved, and the runtime's own
 * dispatch will fail loudly later if a kernel actually needs it.
 */
FeStatus fe_infer_shapes(FeGraph *g);

#endif // FERRITE_SHAPE_INFER_H