// compiler/ir.h
#ifndef FERRITE_IR_H
#define FERRITE_IR_H

#include "../graph/graph.h"

/* Stage 14 compiler IR: a linear, typed instruction sequence lowered from a
 * FeGraph (which stays the high-level IR). Every instruction binds one node
 * to a flat kernel call: srcs are input tensor indices, dst is the produced
 * tensor index. Arrays of live tensors live in an FeIrCtx at run time.
 *
 * The table below mirrors the engine's FeOpType dispatch 1:1 — every op has
 * a kernel or the codegen stage fails loudly at IR_OP_COUNT lookup — so a
 * compiled IR program is either fully runnable or rejected at build time. */

typedef enum {
    IR_NONE = 0,
    IR_INPUT,        /* binds a caller tensor (dst = graph input) */
    IR_MATMUL,       /* dst = src0 @ src1 */
    IR_LINEAR,       /* dst = src0 @ src1 + src2 (bias) */
    IR_RELU,
    IR_ADD,
    IR_SOFTMAX,      /* last axis */
    IR_CONV1D,       /* src2 optional bias */
    IR_BATCHNORM,    /* src1..4 = gamma/beta/mean/var */
    IR_OUTPUT,       /* copies graph output into a caller tensor */
    IR_OP_COUNT
} FeIrOp;

typedef struct {
    FeIrOp op;
    int    node;        /* source FeGraph node index (for attrs) */
    int    dst;         /* produced tensor index */
    int    srcs[FE_MAX_NODE_INPUTS];
    int    n_srcs;
    int    dead;        /* set by IR optimization passes */
} FeIrInstr;

typedef struct {
    FeIrInstr instrs[FE_MAX_NODES];
    int       n;
    int       io_in;    /* graph tensor index bound at run time */
    int       io_out;
} FeIrProgram;

/* ---- passes ---- */

/* Lower a validated, topo-sorted graph into linear IR (one instruction per
 * node). io_in/io_out resolved with the same rules as the engine's
 * find_graph_input / last-node-output. Returns FE_ERR_SHAPE on an op with
 * no IR mapping. */
FeStatus fe_ir_lower(const FeGraph *g, FeIrProgram *p);

/* IR dead-instruction elimination: removes instructions whose produced
 * tensor is never consumed and is not the program output. */
void fe_ir_dead_elim(FeIrProgram *p);

/* ---- kernel selection + run (codegen) ---- */

typedef struct {
    FeGraph   *g;
    FeTensor **tensors;         /* caller-owned, size = g->n_tensors */
    FeTensor  *input;           /* caller input (IR_INPUT binding) */
    FeTensor  *output;          /* caller output (IR_OUTPUT binding) */
} FeIrCtx;

typedef FeStatus (*FeIrKernel)(FeIrCtx *c, const FeIrInstr *in);

/* Map every IR op to its kernel; error if any op lacks one (should not
 * happen — the table below covers all IR_OP_*). */
FeStatus fe_ir_codegen(void);

/* Execute the lowered program against a live tensor array. */
FeStatus fe_ir_run(const FeIrProgram *p, FeIrCtx *c);

/* ---- scheduling ---- */

/* Order the program's instructions (into order[], length p->n) so every
 * instruction follows all of its producers, preferring instructions that
 * reuse the previous instruction's tensors (producer/consumer adjacency for
 * cache + arena reuse). Inputs/weights count as pre-produced. */
FeStatus fe_ir_schedule(const FeIrProgram *p, int *order);

#endif // FERRITE_IR_H