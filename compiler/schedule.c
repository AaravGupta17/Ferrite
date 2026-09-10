// compiler/schedule.c
/*
 * Instruction scheduling: producers-before-consumers with a defuse heuristic.
 *
 * A tiny EDT-style loop: at each step, among instructions whose every source
 * is either a weight/input or already produced, prefer the one sharing a
 * source tensor with the *previous* instruction (reading what was just
 * written maximizes cache and arena reuse), breaking ties by original order.
 * Stable, quadratic (n <= FE_MAX_NODES), and intentionally simple.
 */
#include "ir.h"
#include <string.h>

static int produced_by(const FeIrProgram *p, int tensor) {
    for (int i = 0; i < p->n; i++)
        if (!p->instrs[i].dead && p->instrs[i].dst == tensor) return i;
    return -1;
}

FeStatus fe_ir_schedule(const FeIrProgram *p, int *order) {
    if (!p || !order) return FE_ERR_NULL;

    /* think[i]: # of producers of i's srcs still unscheduled. */
    int produced[FE_MAX_TENSORS];
    memset(produced, 0, sizeof(produced));
    int done[FE_MAX_NODES];
    memset(done, 0, sizeof(done));
    int n_scheduled = 0;

    while (n_scheduled < p->n) {
        int best = -1, best_score = -1, best_orig = FE_MAX_NODES;

        for (int i = 0; i < p->n; i++) {
            if (done[i] || p->instrs[i].dead) continue;

            int ok = 1;
            for (int k = 0; k < p->instrs[i].n_srcs; k++) {
                int t = p->instrs[i].srcs[k];
                if (t < 0) { ok = 0; break; }
                int prod = produced_by(p, t);
                if (prod >= 0 && !done[prod]) { ok = 0; break; }
                /* weight/input tensors have no producer instr — pre-produced */
            }
            if (!ok) continue;

            /* Score: +1 if a source was produced by the previous instr. */
            int score = 0;
            int prev = n_scheduled > 0 ? order[n_scheduled - 1] : -1;
            if (prev >= 0)
                for (int k = 0; k < p->instrs[i].n_srcs; k++) {
                    if (produced_by(p, p->instrs[i].srcs[k]) == prev) score++;
                }

            if (score > best_score ||
                (score == best_score && i < best_orig)) {
                best = i; best_score = score; best_orig = i;
            }
        }

        if (best < 0) return FE_ERR_SHAPE;   /* unsatisfiable schedule */

        done[best] = 1;
        order[n_scheduled++] = best;
        if (p->instrs[best].dst >= 0) produced[p->instrs[best].dst] = 1;
    }
    return FE_OK;
}