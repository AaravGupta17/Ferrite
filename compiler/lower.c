// compiler/lower.c
/*
 * Lowering: FeGraph -> FeIrProgram. Walks the graph's topological order and
 * maps each FeOpType to one IR instruction carrying the node's attr index.
 * io_in/io_out follow the engine's binding rules: io_in is the first
 * consumed-but-unproduced non-weight tensor; io_out is the OUTPUT node's
 * input or, for plain graphs, the last instruction's destination.
 */
#include "ir.h"
#include <string.h>

static FeIrOp op_map(FeOpType op) {
    switch (op) {
        case FE_OP_INPUT:    return IR_INPUT;
        case FE_OP_OUTPUT:   return IR_OUTPUT;
        case FE_OP_MATMUL:   return IR_MATMUL;
        case FE_OP_LINEAR:   return IR_LINEAR;
        case FE_OP_RELU:     return IR_RELU;
        case FE_OP_ADD:      return IR_ADD;
        case FE_OP_SOFTMAX:  return IR_SOFTMAX;
        case FE_OP_CONV1D:   return IR_CONV1D;
        case FE_OP_BATCHNORM:return IR_BATCHNORM;
        default:             return IR_NONE;
    }
}

static int produced_count(const FeGraph *g, int tensor) {
    int n = 0;
    for (int i = 0; i < g->n_nodes; i++)
        for (int o = 0; o < g->nodes[i].n_outputs; o++)
            if (g->nodes[i].outputs[o] == tensor) n++;
    return n;
}

FeStatus fe_ir_lower(const FeGraph *g, FeIrProgram *p) {
    if (!g || !p) return FE_ERR_NULL;
    memset(p, 0, sizeof(*p));

    /* io_out: OUTPUT node input, else the last node's destination. */
    p->io_out = -1;
    int last_dst = -1;
    for (int i = 0; i < g->n_nodes; i++) {
        const FeNode *n = &g->nodes[i];
        for (int o = 0; o < n->n_outputs; o++)
            if (op_map(n->op) != IR_NONE) last_dst = n->outputs[o];
        if (n->op == FE_OP_OUTPUT && n->n_inputs > 0) {
            p->io_out = n->inputs[0];
            break;
        }
        if (last_dst >= 0) p->io_out = last_dst;
    }
    if (p->io_out < 0) return FE_ERR_SHAPE;

    /* io_in: first consumed-but-unproduced non-weight tensor. */
    p->io_in = -1;
    for (int i = 0; i < g->n_nodes && p->io_in < 0; i++) {
        const FeNode *n = &g->nodes[i];
        for (int k = 0; k < n->n_inputs; k++) {
            int t = n->inputs[k];
            if (g->tensors[t].is_weight) continue;
            if (produced_count(g, t) == 0) { p->io_in = t; break; }
        }
    }
    if (p->io_in < 0) {
        /* A graph with only weights (fused input) — fall back to tensor 0. */
        if (g->n_tensors > 0) p->io_in = 0;
        else return FE_ERR_SHAPE;
    }

    for (int i = 0; i < g->n_nodes; i++) {
        const FeNode *n = &g->nodes[i];
        FeIrOp ir = op_map(n->op);
        if (ir == IR_NONE) return FE_ERR_SHAPE;

        FeIrInstr *in = &p->instrs[p->n];
        in->op   = ir;
        in->node = i;
        in->n_srcs = n->n_inputs;
        for (int k = 0; k < n->n_inputs; k++) in->srcs[k] = n->inputs[k];
        in->dst = n->n_outputs > 0 ? n->outputs[0] : -1;
        p->n++;
    }
    return FE_OK;
}

void fe_ir_dead_elim(FeIrProgram *p) {
    if (!p) return;

    /* Live producers: any tensor referenced by an instruction that (a) is a
     * consumer src, or (b) is the program output, stays alive. */
    int live[FE_MAX_TENSORS];
    memset(live, 0, sizeof(live));

    if (p->io_out >= 0 && p->io_out < FE_MAX_TENSORS)
        live[p->io_out] = 1;
    for (int i = 0; i < p->n; i++)
        for (int k = 0; k < p->instrs[i].n_srcs; k++) {
            int t = p->instrs[i].srcs[k];
            if (t >= 0 && t < FE_MAX_TENSORS) live[t] = 1;
        }

    for (int i = 0; i < p->n; i++) {
        int t = p->instrs[i].dst;
        if (t < 0) continue;   /* IR_OUTPUT has no dst */
        if (!live[t] && t != p->io_out) p->instrs[i].dead = 1;
    }

    int w = 0;
    for (int i = 0; i < p->n; i++) {
        if (!p->instrs[i].dead) {
            if (w != i) p->instrs[w] = p->instrs[i];
            w++;
        }
    }
    p->n = w;
}