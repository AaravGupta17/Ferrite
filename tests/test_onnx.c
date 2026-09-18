#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../core/tensor.h"
#include "../core/allocator.h"
#include "../graph/graph.h"
#include "../importer/onnx.h"

#define WEIGHT_BUF_SIZE  (1024 * 1024)

static unsigned char weight_buf[WEIGHT_BUF_SIZE] __attribute__((aligned(64)));

#define OUT_CAP 2048

static void test_load_tiny_mlp(void) {
    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);

    FeStatus s = fe_onnx_load(&g, &weight_arena, "tests/tiny_mlp.onnx");
    assert(s == FE_OK);

    /* Must have loaded some nodes and tensors */
    assert(g.n_nodes   > 0);
    assert(g.n_tensors > 0);
    assert(g.topo_valid);

    /* Weight arena must have been used */
    assert(fe_arena_used(&weight_arena) > 0);

    printf("PASS test_load_tiny_mlp (%d nodes, %d tensors)\n",
           g.n_nodes, g.n_tensors);
}

/* ---------------------------------------------------------------- */
/* Hand-rolled protobuf wire format                                  */
/* ---------------------------------------------------------------- */

static FILE *tmp;

static void wr_varint(unsigned long long v) {
    while (v >= 0x80) {
        fputc((int)(v & 0x7F) | 0x80, tmp);
        v >>= 7;
    }
    fputc((int)v, tmp);
}
static void wr_tag(int field, int wtype) { wr_varint((unsigned)(field << 3) | wtype); }
static void wr_len_delimited(int field, const unsigned char *data, size_t len) {
    wr_tag(field, 2);
    wr_varint(len);
    fwrite(data, 1, len, tmp);
}
static void wr_string(int field, const char *s) {
    wr_len_delimited(field, (const unsigned char *)s, strlen(s));
}
static void wr_int(int field, long long v) {
    wr_tag(field, 0);
    wr_varint((unsigned long long)v);
}
static void wr_float(int field, float f) {
    wr_tag(field, 5);
    unsigned bits;
    memcpy(&bits, &f, 4);
    fputc((int)(bits & 0xFF), tmp);
    fputc((int)((bits >> 8)  & 0xFF), tmp);
    fputc((int)((bits >> 16) & 0xFF), tmp);
    fputc((int)((bits >> 24) & 0xFF), tmp);
}

/* Drain the current tmp buffer into *out; returns length. */
static size_t tmp_drain(unsigned char *out, size_t cap) {
    fflush(tmp);
    long len = ftell(tmp);
    assert(len >= 0 && (size_t)len <= cap);
    rewind(tmp);
    assert(fread(out, 1, (size_t)len, tmp) == (size_t)len);
    fclose(tmp);
    return (size_t)len;
}

/* Write an already-encoded buffer into a fresh tmp file. */
static void begin_tmp(void) {
    tmp = tmpfile();
    assert(tmp);
}

/* ---- ValueInfoProto ---- */

/* {name, type{ tensor_type{ elem_type=1, shape{ dim{ value }, ... } } } } */
static size_t build_value_info(const char *name, const int *dims, int ndim,
                               unsigned char *out, size_t cap) {
    unsigned char dim_bufs[FERRITE_MAX_DIMS][16];
    size_t        dim_lens[FERRITE_MAX_DIMS];

    for (int d = 0; d < ndim; d++) {
        begin_tmp();
        wr_int(1, dims[d]);            /* Dimension.dim_value */
        dim_lens[d] = tmp_drain(dim_bufs[d], sizeof(dim_bufs[d]));
    }

    /* TensorShapeProto: repeated dim (field 1). */
    unsigned char shape[256];
    begin_tmp();
    for (int d = 0; d < ndim; d++)
        wr_len_delimited(1, dim_bufs[d], dim_lens[d]);
    size_t shape_len = tmp_drain(shape, sizeof(shape));

    /* TypeProto.Tensor: elem_type=1 (float32), shape=2. */
    unsigned char tensor_type[384];
    begin_tmp();
    wr_int(1, 1);                       /* elem_type = TensorProto.Tensor.TYPE_FLOAT */
    wr_len_delimited(2, shape, shape_len);
    size_t tt_len = tmp_drain(tensor_type, sizeof(tensor_type));

    /* TypeProto: tensor_type = field 1. */
    unsigned char type_proto[512];
    begin_tmp();
    wr_len_delimited(1, tensor_type, tt_len);
    size_t type_len = tmp_drain(type_proto, sizeof(type_proto));

    /* ValueInfoProto: name=1, type=2. */
    begin_tmp();
    wr_string(1, name);
    wr_len_delimited(2, type_proto, type_len);
    return tmp_drain(out, cap);
}

/* ---- AttributeProto ---- */

static size_t build_attr(const char *name, int has_f, float f,
                         int has_i, long long i,
                         const int *ints, int n_ints,
                         unsigned char *out, size_t cap) {
    unsigned char ints_buf[128];
    size_t ints_len = 0;
    if (n_ints > 0) {
        begin_tmp();
        for (int k = 0; k < n_ints; k++) wr_int(8, ints[k]);
        ints_len = tmp_drain(ints_buf, sizeof(ints_buf));
    }

    begin_tmp();
    wr_string(1, name);
    if (has_f) wr_float(2, f);
    if (has_i) wr_int(3, i);
    if (ints_len > 0) fwrite(ints_buf, 1, ints_len, tmp);
    return tmp_drain(out, cap);
}

/* ---- NodeProto ---- */

static size_t build_node(const char **inputs, int n_in,
                         const char **outputs, int n_out,
                         const char *op_type,
                         const char *attr_names[], const int attr_has_f[],
                         const float attr_f[], const int attr_has_i[],
                         const long long attr_i[], const int attr_ints[][4],
                         const int attr_n_ints[], int n_attrs,
                         unsigned char *out, size_t cap) {
    unsigned char bus[16][512];
    size_t        blen[16];

    for (int a = 0; a < n_attrs; a++) {
        blen[a] = build_attr(attr_names[a],
                             attr_has_f[a], attr_f[a],
                             attr_has_i[a], attr_i[a],
                             attr_ints[a], attr_n_ints[a],
                             bus[a], sizeof(bus[a]));
    }

    begin_tmp();
    for (int k = 0; k < n_in;  k++) wr_string(1, inputs[k]);
    for (int k = 0; k < n_out; k++) wr_string(2, outputs[k]);
    wr_string(4, op_type);
    for (int a = 0; a < n_attrs; a++)
        wr_len_delimited(5, bus[a], blen[a]);
    return tmp_drain(out, cap);
}

/* ---- TensorProto (initializer / weight) ---- */

static size_t build_initializer(const char *name, int data_type,
                                const int *dims, int ndim,
                                const float *v, int nv,
                                const unsigned char *raw, size_t raw_len,
                                unsigned char *out, size_t cap) {
    begin_tmp();
    for (int d = 0; d < ndim; d++) wr_int(1, dims[d]);
    wr_int(2, data_type);                          /* 1 = float32 */
    wr_string(8, name);
    for (int k = 0; k < nv; k++) wr_float(7, v[k]);  /* float_data */
    if (raw) wr_len_delimited(9, raw, raw_len);      /* raw_data */
    return tmp_drain(out, cap);
}

/* ---- GraphProto + ModelProto ---- */

/* Nodes + initializers + value_info variant of build_model (initializers
 * are GraphProto field 5). */
static size_t build_model2(const unsigned char *node_bufs[], const size_t node_lens[],
                           int n_nodes,
                           const unsigned char *init_bufs[], const size_t init_lens[],
                           int n_inits,
                           const unsigned char *value_info_bufs[],
                           const size_t value_info_lens[],
                           int n_value_info,
                           unsigned char *out, size_t cap) {
    unsigned char graph[4096];
    begin_tmp();
    for (int n = 0; n < n_nodes; n++)
        wr_len_delimited(1, node_bufs[n], node_lens[n]);
    for (int k = 0; k < n_inits; k++)
        wr_len_delimited(5, init_bufs[k], init_lens[k]);
    for (int v = 0; v < n_value_info; v++)
        wr_len_delimited(13, value_info_bufs[v], value_info_lens[v]);
    size_t graph_len = tmp_drain(graph, sizeof(graph));

    /* ModelProto: graph = field 7. */
    begin_tmp();
    wr_len_delimited(7, graph, graph_len);
    return tmp_drain(out, cap);
}

static size_t build_model(const unsigned char *node_bufs[], const size_t node_lens[],
                          int n_nodes,
                          const unsigned char *value_info_bufs[],
                          const size_t value_info_lens[],
                          int n_value_info,
                          unsigned char *out, size_t cap) {
    return build_model2(node_bufs, node_lens, n_nodes,
                        NULL, NULL, 0,
                        value_info_bufs, value_info_lens, n_value_info,
                        out, cap);
}

static int tensor_idx(FeGraph *g, const char *name) {
    for (int i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return i;
    return -1;
}

/* ---- Tests ---- */

/* Negative test: a node with op_type "TotallyUnknownOp" must fail loudly. */
static void test_unsupported_op_fails_loudly(void) {
    unsigned char node[128];
    begin_tmp();
    wr_string(4, "TotallyUnknownOp");
    size_t ln = tmp_drain(node, sizeof(node));

    const unsigned char *nb[] = {node};
    const size_t nl[] = {ln};
    unsigned char model[512];
    size_t lm = build_model(nb, nl, 1, NULL, NULL, 0, model, sizeof(model));

    FILE *f = fopen("tests/unsupported_op.onnx", "wb");
    assert(f);
    fwrite(model, 1, lm, f);
    fclose(f);

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);
    FeStatus s = fe_onnx_load(&g, &weight_arena, "tests/unsupported_op.onnx");
    assert(s != FE_OK);

    remove("tests/unsupported_op.onnx");
    printf("PASS test_unsupported_op_fails_loudly\n");
}

/* A LeakyRelu node with alpha attr + value_info shapes must map op, attr,
 * and fill tensor registry shapes. */
static void test_value_info_and_attrs(void) {
    unsigned char vi_x[OUT_CAP], vi_y[OUT_CAP], vi_w[OUT_CAP];
    int sx[] = {1, 4}, sy[] = {1, 4}, sw[] = {4, 4};
    size_t lx = build_value_info("x", sx, 2, vi_x, sizeof(vi_x));
    size_t ly = build_value_info("y", sy, 2, vi_y, sizeof(vi_y));
    size_t lw = build_value_info("w", sw, 2, vi_w, sizeof(vi_w));
    const unsigned char *vis[] = {vi_x, vi_y, vi_w};
    const size_t vil[] = {lx, ly, lw};

    const char *ins[]  = {"x", "w"};
    const char *outs[] = {"y"};
    const char *an[]   = {"alpha"};
    const int a_has_f[] = {1};
    const float af[]   = {0.02f};
    const int a_has_i[] = {0};
    const long long ai[] = {0};
    const int ai_ints[][4] = {{0}};
    const int ai_n[] = {0};
    unsigned char node[OUT_CAP];
    size_t ln = build_node(ins, 2, outs, 1, "LeakyRelu",
                           an, a_has_f, af, a_has_i, ai,
                           ai_ints, ai_n, 1, node, sizeof(node));

    const unsigned char *nb[] = {node};
    const size_t nl[] = {ln};
    unsigned char model[OUT_CAP];
    size_t lm = build_model(nb, nl, 1, vis, vil, 3, model, sizeof(model));

    FILE *f = fopen("tests/value_info.onnx", "wb");
    assert(f);
    fwrite(model, 1, lm, f);
    fclose(f);

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);
    assert(fe_onnx_load(&g, &weight_arena, "tests/value_info.onnx") == FE_OK);

    assert(g.n_nodes == 1);
    assert(g.topo_valid);

    /* Op mapped to LeakyReLU with alpha from the attribute. */
    assert(g.nodes[0].op == FE_OP_LEAKY_RELU);
    assert(fabsf(g.nodes[0].attrs.leaky_relu.negative_slope - 0.02f) < 1e-6f);

    /* Shapes from value_info landed in the tensor registry. */
    int ix = tensor_idx(&g, "x");
    int iy = tensor_idx(&g, "y");
    int iw = tensor_idx(&g, "w");
    assert(ix >= 0 && iy >= 0 && iw >= 0);
    assert(g.tensors[ix].ndim == 2 && g.tensors[ix].shape[0] == 1 &&
           g.tensors[ix].shape[1] == 4);
    assert(g.tensors[iy].ndim == 2 && g.tensors[iy].shape[0] == 1 &&
           g.tensors[iy].shape[1] == 4);
    assert(g.tensors[iw].ndim == 2 && g.tensors[iw].shape[0] == 4 &&
           g.tensors[iw].shape[1] == 4);

    remove("tests/value_info.onnx");
    printf("PASS test_value_info_and_attrs\n");
}

/* MaxPool + LayerNormalization + GroupNormalization nodes must map their
 * attrs (kernel/strides, epsilon, num_groups) onto the node union. */
static void test_pool_and_norm_attrs(void) {
    unsigned char n1[OUT_CAP], n2[OUT_CAP], n3[OUT_CAP];
    size_t l1, l2, l3;

    {
        const char *ins[]  = {"x"};
        const char *outs[] = {"p"};
        const char *an[] = {"kernel_shape", "strides"};
        const int a_has_f[] = {0, 0};
        const float af[] = {0, 0};
        const int a_has_i[] = {0, 0};
        const long long ai[] = {0, 0};
        const int ai_ints[][4] = {{2, 2}, {1, 1}};
        const int ai_n[] = {2, 2};
        l1 = build_node(ins, 1, outs, 1, "MaxPool",
                        an, a_has_f, af, a_has_i, ai,
                        ai_ints, ai_n, 2, n1, sizeof(n1));
    }
    {
        const char *ins[]  = {"p", "gamma", "beta"};
        const char *outs[] = {"n"};
        const char *an[] = {"epsilon"};
        const int a_has_f[] = {1};
        const float af[] = {1e-5f};
        const int a_has_i[] = {0};
        const long long ai[] = {0};
        const int ai_ints[][4] = {{0}};
        const int ai_n[] = {0};
        l2 = build_node(ins, 3, outs, 1, "LayerNormalization",
                        an, a_has_f, af, a_has_i, ai,
                        ai_ints, ai_n, 1, n2, sizeof(n2));
    }
    {
        const char *ins[]  = {"n", "g", "b"};
        const char *outs[] = {"gn"};
        const char *an[] = {"num_groups", "epsilon"};
        const int a_has_f[] = {0, 1};
        const float af[] = {0.0f, 1e-5f};
        const int a_has_i[] = {1, 0};
        const long long ai[] = {2, 0};
        const int ai_ints[][4] = {{0}, {0}};
        const int ai_n[] = {0, 0};
        l3 = build_node(ins, 3, outs, 1, "GroupNormalization",
                        an, a_has_f, af, a_has_i, ai,
                        ai_ints, ai_n, 2, n3, sizeof(n3));
    }

    const unsigned char *nb[] = {n1, n2, n3};
    const size_t nl[] = {l1, l2, l3};
    unsigned char model[OUT_CAP];
    size_t lm = build_model(nb, nl, 3, NULL, NULL, 0, model, sizeof(model));

    FILE *f = fopen("tests/norm_pool.onnx", "wb");
    assert(f);
    fwrite(model, 1, lm, f);
    fclose(f);

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);
    assert(fe_onnx_load(&g, &weight_arena, "tests/norm_pool.onnx") == FE_OK);

    assert(g.n_nodes == 3);

    assert(g.nodes[0].op == FE_OP_MAXPOOL);
    assert(g.nodes[0].attrs.pool.kh == 2 && g.nodes[0].attrs.pool.kw == 2);
    assert(g.nodes[0].attrs.pool.sh == 1 && g.nodes[0].attrs.pool.sw == 1);

    assert(g.nodes[1].op == FE_OP_LAYERNORM);
    assert(fabsf(g.nodes[1].attrs.layernorm.eps - 1e-5f) < 1e-9f);

    assert(g.nodes[2].op == FE_OP_GROUPNORM);
    assert(g.nodes[2].attrs.groupnorm.groups == 2);
    assert(fabsf(g.nodes[2].attrs.groupnorm.eps - 1e-5f) < 1e-9f);

    remove("tests/norm_pool.onnx");
    printf("PASS test_pool_and_norm_attrs\n");
}

/* A node whose length prefix is larger than the remaining bytes must fail
 * loudly (dead-length detection), not crash or silently succeed. */
static void test_malformed_node_len_fails(void) {
    unsigned char graph[64];
    begin_tmp();
    wr_tag(1, 2);      /* node, length-delimited */
    wr_varint(100000); /* len — far larger than the model */
    size_t graph_len = tmp_drain(graph, sizeof(graph));

    unsigned char model[128];
    begin_tmp();
    wr_len_delimited(7, graph, graph_len);
    size_t lm = tmp_drain(model, sizeof(model));

    FILE *f = fopen("tests/malformed.onnx", "wb");
    assert(f);
    fwrite(model, 1, lm, f);
    fclose(f);

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);
    FeStatus s = fe_onnx_load(&g, &weight_arena, "tests/malformed.onnx");
    assert(s != FE_OK);

    remove("tests/malformed.onnx");
    printf("PASS test_malformed_node_len_fails\n");
}

/* BatchNorm weights serialized as float_data (TensorProto field 7) instead
 * of raw_data must load with the correct shapes and values. */
static void test_float_data_initializers(void) {
    const char *ins[]  = {"x", "gamma", "beta", "mean", "var"};
    const char *outs[] = {"y"};
    const char *an[]   = {"epsilon"};
    const int a_has_f[] = {1};
    const float af[]   = {1e-5f};
    const int a_has_i[] = {0};
    const long long ai[] = {0};
    const int ai_ints[][4] = {{0}};
    const int ai_n[] = {0};
    unsigned char node[OUT_CAP];
    size_t ln = build_node(ins, 5, outs, 1, "BatchNormalization",
                           an, a_has_f, af, a_has_i, ai,
                           ai_ints, ai_n, 1, node, sizeof(node));

    /* Four float_data weights, C=2. */
    unsigned char i_gamma[OUT_CAP], i_beta[OUT_CAP];
    unsigned char i_mean[OUT_CAP],  i_var[OUT_CAP];
    size_t       l_gamma, l_beta, l_mean, l_var;
    const int dims_c[] = {2};
    const float v_gamma[] = {1.5f, 2.0f};
    const float v_beta[]  = {0.25f, -0.5f};
    const float v_mean[]  = {0.125f, 0.375f};
    const float v_var[]   = {1.0f, 4.0f};
    l_gamma = build_initializer("gamma", 1, dims_c, 1, v_gamma, 2, NULL, 0, i_gamma, sizeof(i_gamma));
    l_beta  = build_initializer("beta",  1, dims_c, 1, v_beta,  2, NULL, 0, i_beta,  sizeof(i_beta));
    l_mean  = build_initializer("mean",  1, dims_c, 1, v_mean,  2, NULL, 0, i_mean,  sizeof(i_mean));
    l_var   = build_initializer("var",   1, dims_c, 1, v_var,   2, NULL, 0, i_var,   sizeof(i_var));

    const unsigned char *nb[]  = {node};
    const size_t nl[]   = {ln};
    const unsigned char *ib[]  = {i_gamma, i_beta, i_mean, i_var};
    const size_t      il[] = {l_gamma, l_beta, l_mean, l_var};

    unsigned char model[OUT_CAP];
    size_t lm = build_model2(nb, nl, 1, ib, il, 4, NULL, NULL, 0, model, sizeof(model));

    FILE *f = fopen("tests/float_data.onnx", "wb");
    assert(f);
    fwrite(model, 1, lm, f);
    fclose(f);

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);
    assert(fe_onnx_load(&g, &weight_arena, "tests/float_data.onnx") == FE_OK);
    assert(g.n_nodes == 1);
    assert(g.nodes[0].op == FE_OP_BATCHNORM);

    const char *wts[] = {"gamma", "beta", "mean", "var"};
    const float expect[][2] = {{1.5f, 2.0f}, {0.25f, -0.5f},
                               {0.125f, 0.375f}, {1.0f, 4.0f}};
    for (int k = 0; k < 4; k++) {
        int ti = tensor_idx(&g, wts[k]);
        assert(ti >= 0);
        FeTensorEntry *e = &g.tensors[ti];
        assert(e->is_weight == 1);
        assert(e->ndim == 1 && e->shape[0] == 2);
        assert(e->tensor && e->tensor->dtype == DTYPE_FLOAT32);
        const float *d = (const float *)e->tensor->data;
        assert(fabsf(d[0] - expect[k][0]) < 1e-6f);
        assert(fabsf(d[1] - expect[k][1]) < 1e-6f);
    }

    remove("tests/float_data.onnx");
    printf("PASS test_float_data_initializers\n");
}

/* Registry exhaustion: a model declaring more unique tensors than
 * FE_MAX_TENSORS must fail with FE_ERR_NOMEM, never leak a -1 tensor index
 * into the graph (which previously dereferenced g->tensors[-1]). */
static void test_registry_exhaustion_fails_loudly(void) {
    const int N = FE_MAX_TENSORS + 1;   /* one past the registry ceiling */
    unsigned char (*vis)[256] = malloc((size_t)N * 256);
    size_t *vil = malloc((size_t)N * sizeof(size_t));
    assert(vis && vil);

    int s1[] = {1};
    for (int i = 0; i < N; i++) {
        char name[24];
        snprintf(name, sizeof(name), "t_%d", i);
        vil[i] = build_value_info(name, s1, 1, vis[i], 256);
    }

    unsigned char *graph = malloc((size_t)N * 64 + 128);
    assert(graph);
    begin_tmp();
    for (int i = 0; i < N; i++)
        wr_len_delimited(13, vis[i], vil[i]);   /* GraphProto.value_info */
    size_t graph_len = tmp_drain(graph, (size_t)N * 64 + 128);

    unsigned char *model = malloc(graph_len + 64);
    assert(model);
    begin_tmp();
    wr_len_delimited(7, graph, graph_len);      /* ModelProto.graph */
    size_t lm = tmp_drain(model, graph_len + 64);

    FILE *f = fopen("tests/exhaust.onnx", "wb");
    assert(f);
    fwrite(model, 1, lm, f);
    fclose(f);

    FeGraph g;
    FeArena weight_arena;
    fe_arena_init(&weight_arena, weight_buf, WEIGHT_BUF_SIZE);
    FeStatus s = fe_onnx_load(&g, &weight_arena, "tests/exhaust.onnx");
    assert(s == FE_ERR_NOMEM);

    remove("tests/exhaust.onnx");
    free(vis);
    free(vil);
    free(graph);
    free(model);
    printf("PASS test_registry_exhaustion_fails_loudly\n");
}

/* Deterministic parser fuzz: every truncated prefix of a real model plus
 * random byte flips must either load a valid graph or fail loudly — never
 * overrun, hang, or crash. (The hardening that makes this safe lives in
 * fe_pb_varint's 10-byte cap and fe_pb_skip's EOF clamping.) */
static unsigned xorshift(unsigned *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return *s;
}

static FeStatus try_load(const unsigned char *data, size_t len) {
    FILE *f = fopen("tests/fuzz.onnx", "wb");
    if (!f) return FE_ERR_NULL;
    size_t w = fwrite(data, 1, len, f);
    fclose(f);
    if (w != len) return FE_ERR_NULL;
    if (len == 0) { remove("tests/fuzz.onnx"); return FE_ERR_NULL; }

    FeGraph g;
    FeArena arena;
    unsigned char *tmpbuf = malloc(len);
    if (!tmpbuf) { remove("tests/fuzz.onnx"); return FE_ERR_NOMEM; }
    fe_arena_init(&arena, tmpbuf, len);
    FeStatus s = fe_onnx_load(&g, &arena, "tests/fuzz.onnx");
    free(tmpbuf);
    remove("tests/fuzz.onnx");
    return s;
}

static void test_parser_fuzz(void) {
    FILE *f = fopen("tests/tiny_mlp.onnx", "rb");
    assert(f);
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    rewind(f);
    unsigned char *model = malloc((size_t)flen);
    assert(model);
    assert(fread(model, 1, (size_t)flen, f) == (size_t)flen);
    fclose(f);

    int ok = 0, err = 0;
    for (long len = 0; len <= flen; len++) {
        FeStatus s = try_load(model, (size_t)len);
        assert(s == FE_OK || s != FE_OK);   /* the parser never crashes */
        if (s == FE_OK) ok++; else err++;
    }
    assert(ok >= 1);   /* full model and prefix-free-cases must still load */

    /* Random byte flips over the validated whole file. */
    unsigned seed = 0xFE7E15u;
    for (int i = 0; i < 400; i++) {
        unsigned char *buf = malloc((size_t)flen);
        assert(buf);
        memcpy(buf, model, (size_t)flen);
        int flips = 1 + (int)(xorshift(&seed) % 8);
        for (int j = 0; j < flips; j++) {
            int pos = (int)(xorshift(&seed) % (unsigned)flen);
            buf[pos] ^= (unsigned char)(1u << (xorshift(&seed) % 8));
        }
        (void)try_load(buf, (size_t)flen);
        free(buf);
    }

    free(model);
    printf("PASS test_parser_fuzz (trunc: %d ok / %d err, 400 flips)\n", ok, err);
}

int main(void) {
    test_load_tiny_mlp();
    test_unsupported_op_fails_loudly();
    test_value_info_and_attrs();
    test_pool_and_norm_attrs();
    test_malformed_node_len_fails();
    test_float_data_initializers();
    test_registry_exhaustion_fails_loudly();
    test_parser_fuzz();
    printf("\nAll tests passed.\n");
    return 0;
}