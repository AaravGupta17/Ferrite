/* optim/simplify.c — graph simplification (Stage 12.8). */
#include "optim.h"

/*
 * Gentle algebraic cleanup:
 *   - Transpose(Transpose(x)) with no other use of the middle tensor is the
 *     identity — relink x straight through and let dead-elim clean up.
 *   - Flatten of an already-2D tensor is the identity — same treatment.
 * Runs until a fixpoint since simplifications can chain.
 */

static void relink_consumers(FeGraph *g, int src_out, int dst_in) {
    /* Every consumer of tensor src_out now reads tensor dst_in instead. */
    for (int i = 0; i < g->n_nodes; i++) {
        FeNode *n = &g->nodes[i];
        for (int k = 0; k < n->n_inputs; k++)
            if (n->inputs[k] == src_out) n->inputs[k] = dst_in;
    }
}

static int count_consumers(const FeGraph *g, int tidx) {
    int c = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        const FeNode *n = &g->nodes[i];
        for (int k = 0; k < n->n_inputs; k++)
            if (n->inputs[k] == tidx) c++;
    }
    return c;
}

FeStatus fe_pass_simplify(FeGraph *g) {
    if (!g) return FE_ERR_NULL;
    if (!g->topo_valid) {
        FeStatus s = fe_graph_topo_sort(g);
        if (s != FE_OK) return s;
    }

    int changed = 1;
    int passes = 0;
    while (changed && passes++ < g->n_nodes) {
        changed = 0;
        for (int i = 0; i < g->n_nodes; i++) {
            FeNode *node = &g->nodes[i];
            if (node->op != FE_OP_TRANSPOSE && node->op != FE_OP_FLATTEN)
                continue;
            if (node->n_inputs == 0 || node->n_outputs == 0) continue;

            int in0  = node->inputs[0];
            int out0 = node->outputs[0];

            if (node->op == FE_OP_TRANSPOSE) {
                /* Identity iff the producer is also a transpose. */
                for (int p = 0; p < g->n_nodes; p++) {
                    FeNode *prod = &g->nodes[p];
                    if (prod->op != FE_OP_TRANSPOSE) continue;
                    for (int o = 0; o < prod->n_outputs; o++)
                        if (prod->outputs[o] == in0 &&
                            count_consumers(g, in0) == 1) {
                            /* node(prod(x)) == x */
                            if (count_consumers(g, out0) != 0)
                                relink_consumers(g, out0, prod->inputs[0]);
                            node->op = FE_OP_INPUT;
                            node->n_inputs = 0;
                            changed = 1;
                            break;
                        }
                    if (changed) break;
                }
            } else { /* FLATTEN: identity when the input is already 2D */
                if (g->tensors[in0].ndim == 2) {
                    if (count_consumers(g, out0) != 0)
                        relink_consumers(g, out0, in0);
                    node->op = FE_OP_INPUT;
                    node->n_inputs = 0;
                    changed = 1;
                }
            }
        }
        /* Give the fixpoint a fresh topo order after rewriting inputs. */
        if (changed) fe_graph_topo_sort(g);
    }
    return FE_OK;
}