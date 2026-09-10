// compiler/codegen.c
/*
 * Kernel selection: map each IR instruction to a FeIrKernel that binds the
 * op's srcs/dst from the live tensor array and reads attributes from the
 * original node. The table fully covers IR_OP_COUNT, so fe_ir_codegen()
 * either succeeds or is never called with a missing entry.
 */
#include "ir.h"
#include "../core/tensor.h"
#include "../ops/ops.h"
#include <string.h>

static FeStatus k_input(FeIrCtx *c, const FeIrInstr *in) {
    if (in->dst < 0 || !c->input) return FE_ERR_SHAPE;
    c->tensors[in->dst] = c->input;
    return FE_OK;
}

static FeStatus k_output(FeIrCtx *c, const FeIrInstr *in) {
    if (in->n_srcs < 1 || !c->output) return FE_ERR_SHAPE;
    return fe_tensor_copy(c->output, c->tensors[in->srcs[0]]);
}

static FeStatus k_matmul(FeIrCtx *c, const FeIrInstr *in) {
    if (in->n_srcs != 2) return FE_ERR_SHAPE;
    return fe_matmul(c->tensors[in->srcs[0]], c->tensors[in->srcs[1]],
                     c->tensors[in->dst]);
}

static FeStatus k_linear(FeIrCtx *c, const FeIrInstr *in) {
    if (in->n_srcs != 3) return FE_ERR_SHAPE;
    return fe_linear(c->tensors[in->srcs[0]], c->tensors[in->srcs[1]],
                     c->tensors[in->srcs[2]], c->tensors[in->dst]);
}

static FeStatus k_relu(FeIrCtx *c, const FeIrInstr *in) {
    if (in->n_srcs != 1) return FE_ERR_SHAPE;
    return fe_relu(c->tensors[in->srcs[0]], c->tensors[in->dst]);
}

static FeStatus k_add(FeIrCtx *c, const FeIrInstr *in) {
    if (in->n_srcs != 2) return FE_ERR_SHAPE;
    return fe_add(c->tensors[in->srcs[0]], c->tensors[in->srcs[1]],
                  c->tensors[in->dst]);
}

static FeStatus k_softmax(FeIrCtx *c, const FeIrInstr *in) {
    if (in->n_srcs != 1) return FE_ERR_SHAPE;
    return fe_softmax(c->tensors[in->srcs[0]], c->tensors[in->dst]);
}

static FeStatus k_conv1d(FeIrCtx *c, const FeIrInstr *in) {
    if (in->n_srcs < 2) return FE_ERR_SHAPE;
    const FeNode *n = &c->g->nodes[in->node];
    int stride = n->attrs.conv1d.stride > 0 ? n->attrs.conv1d.stride : 1;
    return fe_conv1d(c->tensors[in->srcs[0]], c->tensors[in->srcs[1]],
                     in->n_srcs > 2 ? c->tensors[in->srcs[2]] : NULL,
                     c->tensors[in->dst], stride, n->attrs.conv1d.pad);
}

static FeStatus k_batchnorm(FeIrCtx *c, const FeIrInstr *in) {
    if (in->n_srcs != 5) return FE_ERR_SHAPE;
    const FeNode *n = &c->g->nodes[in->node];
    return fe_batchnorm(c->tensors[in->srcs[0]], c->tensors[in->srcs[1]],
                        c->tensors[in->srcs[2]], c->tensors[in->srcs[3]],
                        c->tensors[in->srcs[4]], c->tensors[in->dst],
                        n->attrs.batchnorm.eps);
}

static FeIrKernel kfns[IR_OP_COUNT];

FeStatus fe_ir_codegen(void) {
    memset(kfns, 0, sizeof(kfns));
    kfns[IR_INPUT]     = k_input;
    kfns[IR_OUTPUT]    = k_output;
    kfns[IR_MATMUL]    = k_matmul;
    kfns[IR_LINEAR]    = k_linear;
    kfns[IR_RELU]      = k_relu;
    kfns[IR_ADD]       = k_add;
    kfns[IR_SOFTMAX]   = k_softmax;
    kfns[IR_CONV1D]    = k_conv1d;
    kfns[IR_BATCHNORM] = k_batchnorm;

    for (int i = 1; i < IR_OP_COUNT; i++)
        if (kfns[i] == NULL) return FE_ERR_SHAPE;
    return FE_OK;
}

FeStatus fe_ir_run(const FeIrProgram *p, FeIrCtx *c) {
    if (!p || !c || !c->tensors) return FE_ERR_NULL;
    for (int i = 0; i < p->n; i++) {
        const FeIrInstr *in = &p->instrs[i];
        if (in->dead) continue;
        if (!kfns[in->op]) return FE_ERR_SHAPE;
        FeStatus s = kfns[in->op](c, in);
        if (s != FE_OK) return s;
    }
    return FE_OK;
}