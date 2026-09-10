/* optim/dead_elim.c — dead node and dead tensor elimination (Stage 12.3/4). */
#include "optim.h"
#include <string.h>

/*
 * Liveness fixpoint from the OUTPUT nodes: a node is live when a live node
 * consumes one of its outputs. INPUT feeds are live only if their output is
 * genuinely read — this is what reclaims the phantom feeds that folding,
 * CSE, simplification and fusion leave behind. Removable garbage is then
 * compacted out of the node and tensor registries (indices stay dense), so
 * the planner schedules a smaller buffer and dispatch iterates fewer nodes.
 *
 * Tensors are kept only while some live node reads them, they are weights
 * read by a live node, or they are the graph output (named by an OUTPUT
 * node, or — for ONNX graphs — the last topo node's output, which the
 * engine reads directly). Everything else is reclaimed.
 */
FeStatus fe_pass_dead_elim(FeGraph *g) {
    if (!g) return FE_ERR_NULL;
    if (!g->topo_valid) {
        FeStatus s = fe_graph_topo_sort(g);
        if (s != FE_OK) return s;
    }

    /* Graph output tensor: an explicit OUTPUT node names it; ONNX graphs have
     * no such node, so it is the output of the last node in topo order, which
     * the engine reads directly. Its producer is seeded live regardless, and
     * the engine and this pass derive the output the same way so they always
     * agree after optimization. Passes zero the inputs of neutered feeds, so
     * such feeds never sort after the real terminal compute node. */
    int output_tidx = -1;
    int has_output_node = 0;
    for (int i = 0; i < g->n_nodes; i++)
        if (g->nodes[i].op == FE_OP_OUTPUT && g->nodes[i].n_inputs > 0) {
            output_tidx = g->nodes[i].inputs[0];
            has_output_node = 1;
            break;
        }
    if (!has_output_node && g->n_nodes > 0) {
        FeNode *last = &g->nodes[g->topo_order[g->n_nodes - 1]];
        if (last->n_outputs > 0) output_tidx = last->outputs[0];
    }

    /* Liveness via fixpoint seeded from OUTPUT nodes and the output
     * producer (they have no live consumer to legitimize them). */
    int live[FE_MAX_NODES] = {0};
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].op == FE_OP_OUTPUT) live[i] = 1;
        for (int o = 0; o < g->nodes[i].n_outputs; o++)
            if (g->nodes[i].outputs[o] == output_tidx) live[i] = 1;
    }

    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < g->n_nodes; i++) {
            if (live[i]) continue;
            for (int o = 0; o < g->nodes[i].n_outputs; o++) {
                int tidx = g->nodes[i].outputs[o];
                for (int c = 0; c < g->n_nodes && !changed; c++) {
                    if (!live[c]) continue;
                    for (int k = 0; k < g->nodes[c].n_inputs; k++)
                        if (g->nodes[c].inputs[k] == tidx) { live[i] = 1; changed = 1; break; }
                }
            }
        }
    }

    /* Tensor kept iff read by a live node, or the graph output. Weights are
     * no exception: an unused weight (e.g. pre-fusion BN coefficients) is
     * reclaimed like any other dead tensor. */
    int keep_tensor[FE_MAX_TENSORS] = {0};
    int live_consumers[FE_MAX_TENSORS] = {0};
    for (int c = 0; c < g->n_nodes; c++) {
        if (!live[c]) continue;
        for (int k = 0; k < g->nodes[c].n_inputs; k++)
            live_consumers[g->nodes[c].inputs[k]]++;
    }
    for (int t = 0; t < g->n_tensors; t++) {
        if (live_consumers[t] > 0) keep_tensor[t] = 1;
        else if (t == output_tidx) keep_tensor[t] = 1;
    }

    /* Compaction: remap live nodes and kept tensors to the front. */
    int node_map[FE_MAX_NODES];
    for (int i = 0; i < FE_MAX_NODES; i++) node_map[i] = -1;
    int new_n_nodes = 0;
    for (int i = 0; i < g->n_nodes; i++)
        if (live[i]) node_map[i] = new_n_nodes++;

    int tensor_map[FE_MAX_TENSORS];
    for (int t = 0; t < FE_MAX_TENSORS; t++) tensor_map[t] = -1;
    int new_n_tensors = 0;
    for (int t = 0; t < g->n_tensors; t++)
        if (keep_tensor[t]) tensor_map[t] = new_n_tensors++;

    if (new_n_nodes == g->n_nodes && new_n_tensors == g->n_tensors)
        return FE_OK;

    FeNode  new_nodes [FE_MAX_NODES];
    FeTensorEntry new_tensors[FE_MAX_TENSORS];

    for (int i = 0; i < g->n_nodes; i++) {
        if (!live[i]) continue;
        FeNode *src = &g->nodes[i];
        FeNode *dst = &new_nodes[node_map[i]];
        memcpy(dst, src, sizeof(FeNode));
        for (int k = 0; k < src->n_inputs; k++)
            dst->inputs[k] = tensor_map[src->inputs[k]];
        for (int o = 0; o < src->n_outputs; o++)
            dst->outputs[o] = tensor_map[src->outputs[o]];
    }
    for (int t = 0; t < g->n_tensors; t++) {
        if (!keep_tensor[t]) continue;
        memcpy(&new_tensors[tensor_map[t]], &g->tensors[t],
               sizeof(FeTensorEntry));
    }

    memcpy(g->nodes,   new_nodes,   sizeof(g->nodes));
    memcpy(g->tensors, new_tensors, sizeof(g->tensors));
    g->n_nodes   = new_n_nodes;
    g->n_tensors = new_n_tensors;
    g->topo_valid = 0;
    return FE_OK;
}