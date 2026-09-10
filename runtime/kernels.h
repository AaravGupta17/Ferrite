// runtime/kernels.h — kernel resolution table for the execution engine.
//
// The per-op kernel wrappers (ex_*) live here instead of in exec_plan.c so
// the device runtime can link the kernels without pulling in shape inference
// or the optimizer: kernels depend on core + ops + quant + graph only, never
// on optim. exec_plan.c (host-only, shape-inference dependent) and the load
// path both resolve nodes through fe_kernels_get().
#ifndef FERRITE_KERNELS_H
#define FERRITE_KERNELS_H

#include "exec_plan.h"

/* Resolve an op to its kernel wrapper. NULL when the op has no kernel —
 * callers must fail loudly on NULL, never silently skip. */
FeExecFn fe_kernels_get(int op);

#endif // FERRITE_KERNELS_H