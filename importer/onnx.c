// importer/onnx.c
#include "onnx.h"
#include "tensor.h"
#include "optim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ------------------------------------------------------------------ */
/* Protobuf wire format parser                                          */
/* ------------------------------------------------------------------ */

void fe_pb_init(FePbReader *r, const unsigned char *data, size_t len) {
    r->data = data;
    r->pos  = 0;
    r->len  = len;
}

int fe_pb_done(const FePbReader *r) {
    return r->pos >= r->len;
}

uint64_t fe_pb_varint(FePbReader *r) {
    uint64_t result = 0;
    int      shift  = 0;
    while (r->pos < r->len) {
        unsigned char b = r->data[r->pos++];
        result |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;   /* MSB clear = last byte */
        shift += 7;
        if (shift >= 63) return 0;   /* > 9 bytes: runaway varint */
    }
    return result;
}

uint32_t fe_pb_fixed32(FePbReader *r) {
    if (r->pos + 4 > r->len) return 0;   /* truncated field */
    uint32_t v = (uint32_t)r->data[r->pos]
               | ((uint32_t)r->data[r->pos + 1] << 8)
               | ((uint32_t)r->data[r->pos + 2] << 16)
               | ((uint32_t)r->data[r->pos + 3] << 24);
    r->pos += 4;
    return v;
}

int fe_pb_tag(FePbReader *r, int *field_number, int *wire_type) {
    if (fe_pb_done(r)) return 0;
    uint64_t tag = fe_pb_varint(r);
    *wire_type    = (int)(tag & 0x07);
    *field_number = (int)(tag >> 3);
    return 1;
}

void fe_pb_skip(FePbReader *r, int wire_type) {
    switch (wire_type) {
        case 0: fe_pb_varint(r); break;                          /* varint */
        case 1: r->pos += 8; break;                             /* 64-bit  */
        case 2: { size_t len = (size_t)fe_pb_varint(r);         /* length-delimited */
                  if (len > r->len - r->pos) r->pos = r->len;   /* clamp: never run past EOF */
                  else r->pos += len;
                  break; }
        case 5: r->pos += 4; break;                             /* 32-bit  */
        default: r->pos = r->len; break;                        /* corrupt */
    }
    if (r->pos > r->len) r->pos = r->len;
}

int fe_pb_bytes(FePbReader *r,
                const unsigned char **out_data, size_t *out_len) {
    if (fe_pb_done(r)) return 0;
    *out_len  = (size_t)fe_pb_varint(r);
    *out_data = r->data + r->pos;
    r->pos   += *out_len;
    return r->pos <= r->len ? 1 : 0;
}

/* Read a length-delimited field into a NUL-terminated string.
 * Returns 1 on success. buf must be at least FE_NAME_LEN bytes. */
static int pb_string(FePbReader *r, char *buf, size_t buflen) {
    const unsigned char *data;
    size_t len;
    if (!fe_pb_bytes(r, &data, &len)) return 0;
    size_t copy = len < buflen - 1 ? len : buflen - 1;
    memcpy(buf, data, copy);
    buf[copy] = '\0';
    return 1;
}

/* ------------------------------------------------------------------ */
/* ONNX op_type string → FeOpType                                      */
/* ------------------------------------------------------------------ */

static FeOpType op_type_from_string(const char *s) {
    if (strcmp(s, "MatMul")             == 0) return FE_OP_MATMUL;
    if (strcmp(s, "Gemm")               == 0) return FE_OP_LINEAR;
    if (strcmp(s, "Relu")               == 0) return FE_OP_RELU;
    if (strcmp(s, "Softmax")            == 0) return FE_OP_SOFTMAX;
    if (strcmp(s, "Conv")               == 0) return FE_OP_CONV1D;   /* upgraded to CONV2D from kernel_shape */
    if (strcmp(s, "BatchNormalization") == 0) return FE_OP_BATCHNORM;
    if (strcmp(s, "Add")                == 0) return FE_OP_ADD;
    if (strcmp(s, "Flatten")            == 0) return FE_OP_FLATTEN;
    /* ---- Basic math ---- */
    if (strcmp(s, "Sub") == 0) return FE_OP_SUB;
    if (strcmp(s, "Mul") == 0) return FE_OP_MUL;
    if (strcmp(s, "Div") == 0) return FE_OP_DIV;
    if (strcmp(s, "Neg") == 0) return FE_OP_NEG;
    if (strcmp(s, "Exp") == 0) return FE_OP_EXP;
    if (strcmp(s, "Log") == 0) return FE_OP_LOG;
    if (strcmp(s, "Pow") == 0) return FE_OP_POW;
    /* ---- Activations ---- */
    if (strcmp(s, "Sigmoid") == 0) return FE_OP_SIGMOID;
    if (strcmp(s, "Tanh")    == 0) return FE_OP_TANH;
    if (strcmp(s, "Gelu")    == 0) return FE_OP_GELU;
    if (strcmp(s, "LeakyRelu") == 0) return FE_OP_LEAKY_RELU;
    if (strcmp(s, "Elu")     == 0) return FE_OP_ELU;
    if (strcmp(s, "Swish")   == 0) return FE_OP_SWISH;
    /* ---- Linear algebra / views ---- */
    if (strcmp(s, "Transpose") == 0) return FE_OP_TRANSPOSE;
    /* ---- CNN ---- */
    if (strcmp(s, "MaxPool")      == 0) return FE_OP_MAXPOOL;
    if (strcmp(s, "AveragePool")  == 0) return FE_OP_AVGPOOL;
    /* ---- Norms ---- */
    if (strcmp(s, "LayerNormalization")       == 0) return FE_OP_LAYERNORM;
    if (strcmp(s, "GroupNormalization")       == 0) return FE_OP_GROUPNORM;
    /* ---- Sequence (ONNX contrib opset 'MultiHeadAttention') ---- */
    if (strcmp(s, "MultiHeadAttention")       == 0) return FE_OP_MULTIHEAD_ATTN;
    return (FeOpType)-1;   /* unsupported */
}

/* ------------------------------------------------------------------ */
/* Tensor name → graph tensor index lookup                             */
/* ------------------------------------------------------------------ */

static int find_or_add_tensor(FeGraph *g, const char *name) {
    /* Search existing tensors by name */
    for (int i = 0; i < g->n_tensors; i++) {
        if (strncmp(g->tensors[i].name, name, FE_NAME_LEN) == 0)
            return i;
    }
    /* Not found — add a placeholder (shape filled in later) */
    int shape[] = {0};
    return fe_graph_add_tensor(g, name, DTYPE_FLOAT32, 1, shape, 0);
}

/* ------------------------------------------------------------------ */
/* Parse TensorProto (initializer / weight)                            */
/* ------------------------------------------------------------------ */

/*
 * TensorProto field numbers:
 *   1 = dims      (repeated int64)
 *   2 = data_type (int32)
 *   7 = float_data (repeated float — used by some exporters for small
 *                   weight tensors instead of packed raw_data)
 *   8 = name      (string)
 *   9 = raw_data  (bytes — float32 values packed)
 */
static FeStatus parse_initializer(FePbReader *r, FeGraph *g,
                                   FeArena *weight_arena) {
    char    name[FE_NAME_LEN] = {0};
    int     dims[FERRITE_MAX_DIMS];
    int     ndim     = 0;
    int     data_type = 0;
    const unsigned char *raw_data = NULL;
    size_t  raw_len  = 0;

    /* float_data (field 7) is repeated float — accumulate in a growing
     * buffer. Sparse path; exporter-generated small weights only. */
    float  *fvals     = NULL;
    size_t  n_fvals   = 0;
    size_t  cap_fvals = 0;

    while (!fe_pb_done(r)) {
        int field, wtype;
        if (!fe_pb_tag(r, &field, &wtype)) break;

        switch (field) {
            case 1: /* dims — repeated int64 */
                if (ndim < FERRITE_MAX_DIMS)
                    dims[ndim++] = (int)fe_pb_varint(r);
                else
                    fe_pb_varint(r);   /* discard */
                break;
            case 2: data_type = (int)fe_pb_varint(r); break;
            case 7: /* float_data — repeated float */
                if (wtype != 5) { fe_pb_skip(r, wtype); break; }
                if (data_type != 0 && data_type != 1) break;  /* not float32 */
                {
                    uint32_t bits = fe_pb_fixed32(r);
                    if (n_fvals == cap_fvals) {
                        size_t nc = cap_fvals ? cap_fvals * 2 : 64;
                        float *nf = (float *)realloc(fvals, nc * sizeof(float));
                        if (!nf) { free(fvals); return FE_ERR_NOMEM; }
                        fvals = nf;
                        cap_fvals = nc;
                    }
                    memcpy(&fvals[n_fvals++], &bits, 4);
                }
                break;
            case 8: pb_string(r, name, FE_NAME_LEN); break;
            case 9: fe_pb_bytes(r, &raw_data, &raw_len); break;
            default: fe_pb_skip(r, wtype); break;
        }
    }

    if (ndim == 0) { free(fvals); return FE_ERR_SHAPE; }

    /* Bounds: reject absurd dims before integer overflow can bite. */
    int64_t numel = 1;
    for (int d = 0; d < ndim; d++) {
        if (dims[d] <= 0 || dims[d] > (1 << 20)) { free(fvals); return FE_ERR_SHAPE; }
        numel *= dims[d];
    }

    /* Find or register this tensor in the graph */
    int tidx = find_or_add_tensor(g, name);
    if (tidx < 0) { free(fvals); return FE_ERR_NOMEM; }
    FeTensorEntry *e = &g->tensors[tidx];
    e->ndim      = ndim;
    e->dtype     = DTYPE_FLOAT32;
    e->is_weight = 1;
    memcpy(e->shape, dims, ndim * sizeof(int));

    /* Allocate and copy weight data into the arena */
    e->tensor = fe_arena_alloc_tensor(weight_arena,
                                       DTYPE_FLOAT32, ndim, dims);
    if (!e->tensor) { free(fvals); return FE_ERR_NOMEM; }

    if (raw_data) {
        if (raw_len != (size_t)(numel * 4)) { free(fvals); return FE_ERR_SHAPE; }
        memcpy(e->tensor->data, raw_data, raw_len);
    } else if (n_fvals > 0) {
        if ((int64_t)n_fvals != numel) { free(fvals); return FE_ERR_SHAPE; }
        memcpy(e->tensor->data, fvals, n_fvals * sizeof(float));
    } else {
        free(fvals);
        return FE_ERR_SHAPE;
    }

    free(fvals);
    return FE_OK;
}

/* ------------------------------------------------------------------ */
/* Node attributes                                                     */
/* ------------------------------------------------------------------ */

#define FE_MAX_ATTRIBUTES 8

/*
 * AttributeProto field numbers:
 *   1 = name    (string)
 *   2 = f       (float)
 *   3 = i       (int64)
 *   7 = floats  (repeated float)
 *   8 = ints    (repeated int64)
 */
typedef struct {
    char name[64];
    int  ints[8];
    int  n_ints;
    int  int_val;
    float float_val;
    int  has_i;
    int  has_f;
} FeAttr;

/* Parse a single AttributeProto sub-message into *a. */
static void parse_attr(FePbReader *r, FeAttr *a) {
    memset(a, 0, sizeof(*a));

    while (!fe_pb_done(r)) {
        int af, awt;
        if (!fe_pb_tag(r, &af, &awt)) break;
        switch (af) {
            case 1: pb_string(r, a->name, sizeof(a->name)); break;
            case 2: { /* f — 32-bit little-endian float. */
                if (r->pos + 4 <= r->len) {
                    const unsigned char *p = r->data + r->pos;
                    unsigned bits = (unsigned)p[0]
                                   | ((unsigned)p[1] << 8)
                                   | ((unsigned)p[2] << 16)
                                   | ((unsigned)p[3] << 24);
                    memcpy(&a->float_val, &bits, 4);
                    a->has_f = 1;
                }
                r->pos += 4;
                break;
            }
            case 3: a->int_val = (int)fe_pb_varint(r); a->has_i = 1; break;
            case 8: /* ints — repeated varint */
                if (a->n_ints < 8)
                    a->ints[a->n_ints++] = (int)fe_pb_varint(r);
                else
                    fe_pb_varint(r);
                break;
            default: fe_pb_skip(r, awt); break;
        }
    }
}

/* Apply one parsed attribute to a node whose op is already known.
 * Unknown attributes on a supported op are a warning, never an error. */
static void apply_attr(FeNode *node, FeGraph *g, FeOpType op,
                       const FeAttr *a) {
    if (strcmp(a->name, "pads") == 0 || strcmp(a->name, "strides") == 0 ||
        strcmp(a->name, "kernel_shape") == 0) {
        switch (op) {
            case FE_OP_CONV1D:
                if (strcmp(a->name, "pads") == 0 && a->n_ints >= 1)
                    node->attrs.conv1d.pad = a->ints[0];
                if (strcmp(a->name, "strides") == 0 && a->n_ints >= 1)
                    node->attrs.conv1d.stride = a->ints[0];
                break;
            case FE_OP_CONV2D:
                if (strcmp(a->name, "pads") == 0 && a->n_ints >= 4) {
                    node->attrs.conv2d.pad_h = a->ints[0];
                    node->attrs.conv2d.pad_w = a->ints[2];
                } else if (strcmp(a->name, "pads") == 0 && a->n_ints >= 2) {
                    node->attrs.conv2d.pad_h = a->ints[0];
                    node->attrs.conv2d.pad_w = a->ints[1];
                }
                if (strcmp(a->name, "strides") == 0 && a->n_ints >= 2) {
                    node->attrs.conv2d.stride_h = a->ints[0];
                    node->attrs.conv2d.stride_w = a->ints[1];
                } else if (strcmp(a->name, "strides") == 0 && a->n_ints >= 1) {
                    node->attrs.conv2d.stride_h = a->ints[0];
                    node->attrs.conv2d.stride_w = a->ints[0];
                }
                break;
            case FE_OP_MAXPOOL:
            case FE_OP_AVGPOOL:
                if (strcmp(a->name, "kernel_shape") == 0 && a->n_ints >= 2) {
                    node->attrs.pool.kh = a->ints[0];
                    node->attrs.pool.kw = a->ints[1];
                } else if (strcmp(a->name, "kernel_shape") == 0 && a->n_ints == 1) {
                    node->attrs.pool.kh = a->ints[0];
                    node->attrs.pool.kw = a->ints[0];
                }
                if (strcmp(a->name, "strides") == 0 && a->n_ints >= 2) {
                    node->attrs.pool.sh = a->ints[0];
                    node->attrs.pool.sw = a->ints[1];
                } else if (strcmp(a->name, "strides") == 0 && a->n_ints == 1) {
                    node->attrs.pool.sh = a->ints[0];
                    node->attrs.pool.sw = a->ints[0];
                }
                if (strcmp(a->name, "pads") == 0 && a->n_ints > 0) {
                    int nonzero = 0;
                    for (int k = 0; k < a->n_ints; k++)
                        if (a->ints[k] != 0) nonzero = 1;
                    if (nonzero)
                        fprintf(stderr,
                                "ONNX: warning: pool 'pads' not supported, "
                                "ignoring\n");
                }
                break;
            default: break;
        }
        return;
    }

    switch (op) {
        case FE_OP_LEAKY_RELU:
            if (strcmp(a->name, "alpha") == 0 && a->has_f)
                node->attrs.leaky_relu.negative_slope = a->float_val;
            break;
        case FE_OP_ELU:
            if (strcmp(a->name, "alpha") == 0 && a->has_f)
                node->attrs.elu.alpha = a->float_val;
            break;
        case FE_OP_BATCHNORM:
            if (strcmp(a->name, "epsilon") == 0 && a->has_f)
                node->attrs.batchnorm.eps = a->float_val;
            break;
        case FE_OP_LAYERNORM:
            if (strcmp(a->name, "epsilon") == 0 && a->has_f)
                node->attrs.layernorm.eps = a->float_val;
            break;
        case FE_OP_GROUPNORM:
            if (strcmp(a->name, "epsilon") == 0 && a->has_f)
                node->attrs.groupnorm.eps = a->float_val;
            if (strcmp(a->name, "num_groups") == 0 && a->has_i)
                node->attrs.groupnorm.groups = a->int_val;
            break;
        case FE_OP_MULTIHEAD_ATTN:
            if (strcmp(a->name, "num_heads") == 0 && a->has_i)
                node->attrs.multihead.num_heads = a->int_val;
            break;
        case FE_OP_SOFTMAX:
            /* Engine computes softmax over the last axis always. ONNX
             * writes axis=-1 for "last", or the explicit last-axis index.
             * Only flag an axis we can positively prove is not the last
             * axis (rank must already be known — never guess). */
            if (strcmp(a->name, "axis") == 0 && a->has_i && a->int_val != -1) {
                int rank = 0;
                if (node->n_outputs > 0)
                    rank = g->tensors[node->outputs[0]].ndim;
                if (rank > 0 && a->int_val != rank - 1) {
                    fprintf(stderr,
                            "ONNX: unsupported Softmax axis=%d (only "
                            "-1/last supported)\n", a->int_val);
                }
            }
            break;
        case FE_OP_LINEAR:
            /* Gemm maps to C = A @ W + b (alpha=1, beta=1, no transpose).
             * Non-default Gemm attributes would change the result — flag
             * them rather than silently computing something else. */
            if (strcmp(a->name, "transA") == 0 && a->has_i && a->int_val != 0)
                fprintf(stderr, "ONNX: unsupported Gemm transA (mapped to "
                        "Linear without transpose)\n");
            if (strcmp(a->name, "transB") == 0 && a->has_i && a->int_val != 0)
                fprintf(stderr, "ONNX: unsupported Gemm transB (mapped to "
                        "Linear without transpose)\n");
            if (strcmp(a->name, "alpha") == 0 && a->has_f &&
                a->float_val != 1.0f)
                fprintf(stderr, "ONNX: unsupported Gemm alpha (mapped to "
                        "Linear with alpha=1)\n");
            if (strcmp(a->name, "beta") == 0 && a->has_f &&
                a->float_val != 1.0f)
                fprintf(stderr, "ONNX: unsupported Gemm beta (mapped to "
                        "Linear with bias included)\n");
            break;
        default:
            break;
    }
}

/* ------------------------------------------------------------------ */
/* Parse NodeProto                                                      */
/* ------------------------------------------------------------------ */

/*
 * NodeProto field numbers:
 *   1 = input     (repeated string)
 *   2 = output    (repeated string)
 *   3 = name      (string)
 *   4 = op_type   (string)
 *   5 = attribute (repeated AttributeProto)
 */
static FeStatus parse_node(FePbReader *r, FeGraph *g) {
    char     node_name[FE_NAME_LEN] = {0};
    char     op_str   [FE_NAME_LEN] = {0};
    int      inputs [FE_MAX_NODE_INPUTS];
    int      outputs[FE_MAX_NODE_OUTPUTS];
    int      n_in = 0, n_out = 0;
    FeAttr   attrs[FE_MAX_ATTRIBUTES];
    int      n_attrs = 0;
    int      kernel_dims = 0;

    while (!fe_pb_done(r)) {
        int field, wtype;
        if (!fe_pb_tag(r, &field, &wtype)) break;

        switch (field) {
            case 1: {
                char tname[FE_NAME_LEN];
                pb_string(r, tname, FE_NAME_LEN);
                if (n_in < FE_MAX_NODE_INPUTS) {
                    int t = find_or_add_tensor(g, tname);
                    if (t < 0) return FE_ERR_NOMEM;   /* registry exhausted */
                    inputs[n_in++] = t;
                }
                break;
            }
            case 2: {
                char tname[FE_NAME_LEN];
                pb_string(r, tname, FE_NAME_LEN);
                if (n_out < FE_MAX_NODE_OUTPUTS) {
                    int t = find_or_add_tensor(g, tname);
                    if (t < 0) return FE_ERR_NOMEM;   /* registry exhausted */
                    outputs[n_out++] = t;
                }
                break;
            }
            case 3: pb_string(r, node_name, FE_NAME_LEN); break;
            case 4: pb_string(r, op_str,    FE_NAME_LEN); break;
            case 5: {
                const unsigned char *attr_data;
                size_t attr_len;
                if (!fe_pb_bytes(r, &attr_data, &attr_len)) break;
                if (n_attrs < FE_MAX_ATTRIBUTES) {
                    FePbReader ar;
                    fe_pb_init(&ar, attr_data, attr_len);
                    parse_attr(&ar, &attrs[n_attrs]);
                    if (strcmp(attrs[n_attrs].name, "kernel_shape") == 0)
                        kernel_dims = attrs[n_attrs].n_ints;
                    n_attrs++;
                }
                break;
            }
            default: fe_pb_skip(r, wtype); break;
        }
    }

    FeOpType op = op_type_from_string(op_str);
    if ((int)op == -1) {
        fprintf(stderr, "ONNX: unsupported op '%s'\n", op_str);
        return FE_ERR_SHAPE;
    }

    /* ONNX "Conv" is N-D; pick 1D or 2D from kernel_shape's rank. */
    if (op == FE_OP_CONV1D && kernel_dims >= 2) op = FE_OP_CONV2D;

    if (node_name[0] == '\0') strncpy(node_name, op_str, FE_NAME_LEN - 1);

    int idx = fe_graph_add_node(g, node_name, op,
                                 inputs, n_in, outputs, n_out);
    if (idx < 0) return FE_ERR_NOMEM;

    FeNode *node = &g->nodes[idx];

    /* Per-op default attribute values (attrs union is zeroed at init). */
    switch (op) {
        case FE_OP_CONV2D:
            node->attrs.conv2d.stride_h = 1;
            node->attrs.conv2d.stride_w = 1;
            break;
        case FE_OP_MAXPOOL:
        case FE_OP_AVGPOOL:
            node->attrs.pool.sh = 1;
            node->attrs.pool.sw = 1;
            break;
        case FE_OP_LEAKY_RELU:
            node->attrs.leaky_relu.negative_slope = 0.01f;
            break;
        case FE_OP_ELU:
            node->attrs.elu.alpha = 1.0f;
            break;
        case FE_OP_BATCHNORM:
            node->attrs.batchnorm.eps = 1e-5f;
            break;
        case FE_OP_LAYERNORM:
            node->attrs.layernorm.eps = 1e-5f;
            break;
        case FE_OP_GROUPNORM:
            node->attrs.groupnorm.eps = 1e-5f;
            break;
        default: break;
    }

    for (int i = 0; i < n_attrs; i++)
        apply_attr(node, g, op, &attrs[i]);

    /* Required attributes and inputs must be present — fail loudly. */
    switch (op) {
        case FE_OP_MAXPOOL:
        case FE_OP_AVGPOOL:
            if (node->attrs.pool.kh == 0 || node->attrs.pool.kw == 0) {
                fprintf(stderr, "ONNX: %s missing required 'kernel_shape'\n",
                        op_str);
                return FE_ERR_SHAPE;
            }
            break;
        case FE_OP_GROUPNORM:
            if (node->attrs.groupnorm.groups <= 0) {
                fprintf(stderr, "ONNX: GroupNormalization missing required "
                        "'num_groups'\n");
                return FE_ERR_SHAPE;
            }
            break;
        default: break;
    }

    /* GEMM/LINEAR and the norm ops have optional extra ONNX inputs
     * (bias / beta) that our kernels require — refuse without them. */
    if (op == FE_OP_LINEAR && n_in < 3) {
        fprintf(stderr, "ONNX: Gemm requires a bias input (C)\n");
        return FE_ERR_SHAPE;
    }
    if ((op == FE_OP_LAYERNORM || op == FE_OP_GROUPNORM) && n_in < 3) {
        fprintf(stderr, "ONNX: %s requires a bias (B) input\n", op_str);
        return FE_ERR_SHAPE;
    }
    if (op == FE_OP_BATCHNORM && n_in != 5) {
        fprintf(stderr, "ONNX: BatchNormalization requires 5 inputs\n");
        return FE_ERR_SHAPE;
    }
    if (op == FE_OP_MULTIHEAD_ATTN && n_in != 5) {
        fprintf(stderr, "ONNX: MultiHeadAttention requires 5 inputs\n");
        return FE_ERR_SHAPE;
    }

    return FE_OK;
}

/* ------------------------------------------------------------------ */
/* Parse ValueInfoProto (graph inputs / outputs / value_info)          */
/* ------------------------------------------------------------------ */

/*
 * GraphProto fields 11 (input), 12 (output) and 13 (value_info) are all
 * repeated ValueInfoProto describing a tensor's name and shape:
 *
 *   ValueInfoProto: name = 1, type = 2 (TypeProto)
 *   TypeProto:      tensor_type = 1 (TypeProto.Tensor)
 *   Tensor:         elem_type = 1, shape = 2 (TensorShapeProto)
 *   TensorShape:    dim = 1 (repeated Dimension)
 *   Dimension:      dim_value = 1 (int64), dim_param = 2 (string)
 *
 * Shapes are filled into the tensor registry. A dim_param (dynamic
 * dimension) parses as 0, preserving the graph's "shape unknown, fill
 * later" contract used by the runtime's infer_shapes.
 */
static FeStatus parse_value_info(FePbReader *r, FeGraph *g) {
    char name[FE_NAME_LEN] = {0};
    int  dims[FERRITE_MAX_DIMS];
    int  ndim = 0;
    int  have_shape = 0;

    while (!fe_pb_done(r)) {
        int field, wtype;
        if (!fe_pb_tag(r, &field, &wtype)) break;

        switch (field) {
            case 1: pb_string(r, name, FE_NAME_LEN); break;
            case 2: {
                /* TypeProto */
                const unsigned char *td; size_t tl;
                if (!fe_pb_bytes(r, &td, &tl)) break;
                FePbReader tr;
                fe_pb_init(&tr, td, tl);
                while (!fe_pb_done(&tr)) {
                    int tf, twt;
                    if (!fe_pb_tag(&tr, &tf, &twt)) break;
                    if (tf == 1) {
                        /* tensor_type */
                        const unsigned char *ts; size_t sl;
                        if (!fe_pb_bytes(&tr, &ts, &sl)) break;
                        FePbReader sr;
                        fe_pb_init(&sr, ts, sl);
                        while (!fe_pb_done(&sr)) {
                            int sf, swt;
                            if (!fe_pb_tag(&sr, &sf, &swt)) break;
                            if (sf == 2) {
                                /* shape */
                                const unsigned char *sh; size_t hl;
                                if (!fe_pb_bytes(&sr, &sh, &hl)) break;
                                FePbReader hr;
                                fe_pb_init(&hr, sh, hl);
                                while (!fe_pb_done(&hr)) {
                                    int hf, hwt;
                                    if (!fe_pb_tag(&hr, &hf, &hwt)) break;
                                    if (hf == 1) {
                                        /* dim */
                                        const unsigned char *dd; size_t dl;
                                        if (!fe_pb_bytes(&hr, &dd, &dl)) break;
                                        FePbReader dr;
                                        fe_pb_init(&dr, dd, dl);
                                        int dim_val = 0;
                                        while (!fe_pb_done(&dr)) {
                                            int df, dwt;
                                            if (!fe_pb_tag(&dr, &df, &dwt)) break;
                                            if (df == 1)
                                                dim_val = (int)fe_pb_varint(&dr);
                                            else
                                                fe_pb_skip(&dr, dwt);
                                        }
                                        if (ndim < FERRITE_MAX_DIMS)
                                            dims[ndim++] = dim_val;
                                    } else {
                                        fe_pb_skip(&hr, hwt);
                                    }
                                }
                                have_shape = 1;
                            } else {
                                fe_pb_skip(&sr, swt);
                            }
                        }
                    } else {
                        fe_pb_skip(&tr, twt);
                    }
                }
                break;
            }
            default: fe_pb_skip(r, wtype); break;
        }
    }

    /* Fill the graph entry for this name. */
    if (name[0] != '\0') {
        int tidx = find_or_add_tensor(g, name);
        if (tidx < 0) return FE_ERR_NOMEM;
        FeTensorEntry *e = &g->tensors[tidx];
        if (have_shape && ndim > 0) {
            e->ndim = ndim;
            memcpy(e->shape, dims, ndim * sizeof(int));
        }
    }

    return FE_OK;
}

static FeStatus parse_graph(FePbReader *r, FeGraph *g,
                              FeArena *weight_arena) {
    while (!fe_pb_done(r)) {
        int field, wtype;
        if (!fe_pb_tag(r, &field, &wtype)) break;

        const unsigned char *sub_data;
        size_t               sub_len;

        switch (field) {
            case 1: { /* node */
                if (!fe_pb_bytes(r, &sub_data, &sub_len)) return FE_ERR_SHAPE;
                FePbReader sub;
                fe_pb_init(&sub, sub_data, sub_len);
                FeStatus s = parse_node(&sub, g);
                if (s != FE_OK) return s;
                break;
            }
            case 5: { /* initializer — weight tensor */
                if (!fe_pb_bytes(r, &sub_data, &sub_len)) return FE_ERR_SHAPE;
                FePbReader sub;
                fe_pb_init(&sub, sub_data, sub_len);
                FeStatus s = parse_initializer(&sub, g, weight_arena);
                if (s != FE_OK) return s;
                break;
            }
            case 11: /* input — ValueInfoProto */
            case 12: /* output — ValueInfoProto */
            case 13: /* value_info — ValueInfoProto */
                if (!fe_pb_bytes(r, &sub_data, &sub_len)) return FE_ERR_SHAPE;
                {
                    FePbReader sub;
                    fe_pb_init(&sub, sub_data, sub_len);
                    FeStatus s = parse_value_info(&sub, g);
                    if (s != FE_OK) return s;
                }
                break;
            default:
                fe_pb_skip(r, wtype);
                break;
        }
    }
    return FE_OK;
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

FeStatus fe_onnx_load(FeGraph *graph, FeArena *weight_arena,
                       const char *path) {
    /* Read entire file into memory */
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ONNX: cannot open '%s'\n", path);
        return FE_ERR_NULL;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    unsigned char *buf = malloc((size_t)fsize);
    if (!buf) { fclose(f); return FE_ERR_NOMEM; }

    if ((long)fread(buf, 1, (size_t)fsize, f) != fsize) {
        free(buf); fclose(f); return FE_ERR_NULL;
    }
    fclose(f);

    fe_graph_init(graph);

    /*
     * ModelProto field numbers:
     *   7 = graph (GraphProto)
     */
    FePbReader r;
    fe_pb_init(&r, buf, (size_t)fsize);

    FeStatus status = FE_OK;
    while (!fe_pb_done(&r)) {
        int field, wtype;
        if (!fe_pb_tag(&r, &field, &wtype)) break;

        if (field == 7) {  /* graph */
            const unsigned char *sub_data;
            size_t               sub_len;
            if (!fe_pb_bytes(&r, &sub_data, &sub_len)) {
                status = FE_ERR_SHAPE;
                break;
            }
            FePbReader sub;
            fe_pb_init(&sub, sub_data, sub_len);
            status = parse_graph(&sub, graph, weight_arena);
            if (status != FE_OK) break;
        } else {
            fe_pb_skip(&r, wtype);
        }
    }

    free(buf);

    if (status == FE_OK)
        status = fe_graph_topo_sort(graph);
    if (status == FE_OK)
        status = fe_graph_validate(graph);
    if (status == FE_OK)
        status = fe_optimize(graph, weight_arena);
    if (status == FE_OK) {
        printf("ONNX: loaded %d nodes, %d tensors\n",
               graph->n_nodes, graph->n_tensors);
        fe_graph_print(graph);
    }

    return status;
}