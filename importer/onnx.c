// importer/onnx.c
#include "onnx.h"
#include "tensor.h"
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
    }
    return result;
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
                  r->pos += len; break; }
        case 5: r->pos += 4; break;                             /* 32-bit  */
        default: r->pos = r->len; break;                        /* corrupt */
    }
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
    if (strcmp(s, "MatMul")   == 0) return FE_OP_MATMUL;
    if (strcmp(s, "Gemm")     == 0) return FE_OP_LINEAR;
    if (strcmp(s, "Relu")     == 0) return FE_OP_RELU;
    if (strcmp(s, "Softmax")  == 0) return FE_OP_SOFTMAX;
    if (strcmp(s, "Conv")     == 0) return FE_OP_CONV1D;
    if (strcmp(s, "BatchNormalization") == 0) return FE_OP_BATCHNORM;
    if (strcmp(s, "Add")      == 0) return FE_OP_ADD;
    if (strcmp(s, "Flatten")  == 0) return FE_OP_FLATTEN;
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
 *   8 = name      (string)
 *   9 = raw_data  (bytes — float32 values packed)
 */
static FeStatus parse_initializer(FePbReader *r, FeGraph *g,
                                   FeArena *weight_arena) {
    char    name[FE_NAME_LEN] = {0};
    int     dims[FERRITE_MAX_DIMS];
    int     ndim     = 0;
    const unsigned char *raw_data = NULL;
    size_t  raw_len  = 0;

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
            case 2: fe_pb_varint(r); break;  /* data_type, assume float32 */
            case 8: pb_string(r, name, FE_NAME_LEN); break;
            case 9: fe_pb_bytes(r, &raw_data, &raw_len); break;
            default: fe_pb_skip(r, wtype); break;
        }
    }

    if (ndim == 0 || !raw_data) return FE_ERR_SHAPE;

    /* Find or register this tensor in the graph */
    int tidx = find_or_add_tensor(g, name);
    FeTensorEntry *e = &g->tensors[tidx];
    e->ndim      = ndim;
    e->dtype     = DTYPE_FLOAT32;
    e->is_weight = 1;
    memcpy(e->shape, dims, ndim * sizeof(int));

    /* Allocate and copy weight data into the arena */
    e->tensor = fe_arena_alloc_tensor(weight_arena,
                                       DTYPE_FLOAT32, ndim, dims);
    if (!e->tensor) return FE_ERR_NOMEM;
    memcpy(e->tensor->data, raw_data, raw_len);

    return FE_OK;
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
 */
static FeStatus parse_node(FePbReader *r, FeGraph *g) {
    char     node_name[FE_NAME_LEN] = {0};
    char     op_str   [FE_NAME_LEN] = {0};
    int      inputs [FE_MAX_NODE_INPUTS];
    int      outputs[FE_MAX_NODE_OUTPUTS];
    int      n_in = 0, n_out = 0;
    int      pads[2]   = {0, 0};
    int      strides[1]= {1};
    int      kernel[1] = {1};

    /* Save position to re-read attributes after we know op_type */
    while (!fe_pb_done(r)) {
        int field, wtype;
        if (!fe_pb_tag(r, &field, &wtype)) break;

        switch (field) {
            case 1: {
                char tname[FE_NAME_LEN];
                pb_string(r, tname, FE_NAME_LEN);
                if (n_in < FE_MAX_NODE_INPUTS)
                    inputs[n_in++] = find_or_add_tensor(g, tname);
                break;
            }
            case 2: {
                char tname[FE_NAME_LEN];
                pb_string(r, tname, FE_NAME_LEN);
                if (n_out < FE_MAX_NODE_OUTPUTS)
                    outputs[n_out++] = find_or_add_tensor(g, tname);
                break;
            }
            case 3: pb_string(r, node_name, FE_NAME_LEN); break;
            case 4: pb_string(r, op_str,    FE_NAME_LEN); break;
            case 5: {
                /* attribute — parse name and value */
                const unsigned char *attr_data;
                size_t attr_len;
                if (!fe_pb_bytes(r, &attr_data, &attr_len)) break;

                FePbReader ar;
                fe_pb_init(&ar, attr_data, attr_len);

                char attr_name[64] = {0};
                int  attr_type = 0;
                int  int_val   = 0;
                int  ints[8]   = {0};
                int  n_ints    = 0;

               while (!fe_pb_done(&ar)) {
                    int af, awt;
                    if (!fe_pb_tag(&ar, &af, &awt)) break;
                    fprintf(stderr, "  attr field=%d wtype=%d\n", af, awt);
                    fe_pb_skip(&ar, awt);
                }
                fprintf(stderr, "ATTR: name='%s'\n", attr_name);
                break;
            }
            default: fe_pb_skip(r, wtype); break;
        }
    }
    FeOpType op = op_type_from_string(op_str);
    if ((int)op == -1) {
        fprintf(stderr, "ONNX: unsupported op '%s', skipping\n", op_str);
        return FE_OK;
    }

    if (node_name[0] == '\0') strncpy(node_name, op_str, FE_NAME_LEN - 1);

    int idx = fe_graph_add_node(g, node_name, op,
                                 inputs, n_in, outputs, n_out);
    if (idx < 0) return FE_ERR_NOMEM;

    /* Store conv attributes on the node */
    if (op == FE_OP_CONV1D) {
        g->nodes[idx].attrs.conv1d.pad    = pads[0];
        g->nodes[idx].attrs.conv1d.stride = strides[0];
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
            case 5: {
                const unsigned char *attr_data;
                size_t attr_len;
                if (!fe_pb_bytes(r, &attr_data, &attr_len)) break;

                FePbReader ar;
                fe_pb_init(&ar, attr_data, attr_len);

                char attr_name[64] = {0};
                int  ints[8] = {0};
                int  n_ints  = 0;

                while (!fe_pb_done(&ar)) {
                    int af, awt;
                    if (!fe_pb_tag(&ar, &af, &awt)) break;
                    switch (af) {
                        case 1: { /* name */
                            const unsigned char *d; size_t l;
                            fe_pb_bytes(&ar, &d, &l);
                            size_t c = l < 63 ? l : 63;
                            memcpy(attr_name, d, c);
                            attr_name[c] = 0;
                            break;
                        }
                        case 7: { /* ints — repeated int64 */
                            if (n_ints < 8)
                                ints[n_ints++] = (int)fe_pb_varint(&ar);
                            else fe_pb_varint(&ar);
                            break;
                        }
                        case 8: { /* i — single int64, also used for repeated */
                            if (n_ints < 8)
                                ints[n_ints++] = (int)fe_pb_varint(&ar);
                            else fe_pb_varint(&ar);
                            break;
                        }
                        default: fe_pb_skip(&ar, awt); break;
                    }
                }

                if (strcmp(attr_name, "pads") == 0 && n_ints >= 2) {
                    pads[0] = ints[0];
                    pads[1] = ints[1];
                } else if (strcmp(attr_name, "strides") == 0 && n_ints >= 1) {
                    strides[0] = ints[0];
                } else if (strcmp(attr_name, "kernel_shape") == 0 && n_ints >= 1) {
                    kernel[0] = ints[0];
                }

                fprintf(stderr, "ATTR: '%s' n_ints=%d [%d,%d]\n",
                        attr_name, n_ints, ints[0], ints[1]);
                break;
            }
            default: fe_pb_skip(r, wtype); break;
        }
    }
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

    if (status == FE_OK) {
        status = fe_graph_topo_sort(graph);
        if (status == FE_OK) {
            printf("ONNX: loaded %d nodes, %d tensors\n",
                   graph->n_nodes, graph->n_tensors);
            fe_graph_print(graph);
        }
    }

    return status;
}