// core/tensor_ser.c
#include "tensor_ser.h"
#include <stdio.h>
#include <string.h>

/* --- Little-endian fixed-width field I/O (format is LE on all hosts) --- */

static FeStatus write_u32le(FILE *f, uint32_t v) {
    unsigned char b[4] = {
        (unsigned char)(v        & 0xFF),
        (unsigned char)((v >> 8)  & 0xFF),
        (unsigned char)((v >> 16) & 0xFF),
        (unsigned char)((v >> 24) & 0xFF),
    };
    if (fwrite(b, 1, 4, f) != 4) return FE_ERR_IO;
    return FE_OK;
}

static FeStatus write_u64le(FILE *f, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)((v >> (8 * i)) & 0xFF);
    if (fwrite(b, 1, 8, f) != 8) return FE_ERR_IO;
    return FE_OK;
}

static FeStatus read_u32le(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return FE_ERR_IO;
    *v = (uint32_t)b[0]
       | ((uint32_t)b[1] << 8)
       | ((uint32_t)b[2] << 16)
       | ((uint32_t)b[3] << 24);
    return FE_OK;
}

static FeStatus read_u64le(FILE *f, uint64_t *v) {
    unsigned char b[8];
    if (fread(b, 1, 8, f) != 8) return FE_ERR_IO;
    uint64_t out = 0;
    for (int i = 0; i < 8; i++) out |= (uint64_t)b[i] << (8 * i);
    *v = out;
    return FE_OK;
}

/* --- Save --- */

FeStatus fe_tensor_save(const FeTensor *t, const char *path) {
    if (!t || !path || !t->data) return FE_ERR_NULL;

    /* Serialize the canonical form: repack views first. */
    const FeTensor *c = t;
    FeTensor *tmp = NULL;
    if (!fe_tensor_is_contiguous(t)) {
        tmp = fe_tensor_contiguous(t);
        if (!tmp) return FE_ERR_NOMEM;
        c = tmp;
    }

    FILE *f = fopen(path, "wb");
    if (!f) { fe_tensor_free(tmp); return FE_ERR_IO; }

    FeStatus s = FE_OK;
    if (s == FE_OK) s = (fwrite(FERRITE_TENSOR_SER_MAGIC, 1, 4, f) == 4)
                        ? FE_OK : FE_ERR_IO;
    if (s == FE_OK) s = write_u32le(f, FERRITE_TENSOR_SER_VERSION);
    if (s == FE_OK) s = write_u32le(f, (uint32_t)c->dtype);
    if (s == FE_OK) s = write_u32le(f, (uint32_t)c->ndim);
    for (int d = 0; s == FE_OK && d < c->ndim; d++) {
        s = write_u32le(f, (uint32_t)c->shape[d]);
    }
    if (s == FE_OK) s = write_u64le(f, (uint64_t)c->nbytes);
    if (s == FE_OK && c->nbytes > 0 &&
        fwrite(c->data, 1, c->nbytes, f) != c->nbytes) {
        s = FE_ERR_IO;
    }

    fclose(f);
    fe_tensor_free(tmp);
    return s;
}

/* --- Load --- */

FeStatus fe_tensor_load(const char *path, FeTensor **out) {
    if (!path || !out) return FE_ERR_NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return FE_ERR_IO;

    /* Every failure path below frees `t` and closes `f` before returning. */
    FeTensor *t = NULL;
    FeStatus s;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 ||
        memcmp(magic, FERRITE_TENSOR_SER_MAGIC, 4) != 0) {
        fclose(f);
        return FE_ERR_IO;
    }

    uint32_t version, dtype_u32, ndim_u32;
    if ((s = read_u32le(f, &version)) != FE_OK ||
        (s = read_u32le(f, &dtype_u32)) != FE_OK ||
        (s = read_u32le(f, &ndim_u32)) != FE_OK) {
        fclose(f);
        return s;
    }
    if (version != FERRITE_TENSOR_SER_VERSION) { fclose(f); return FE_ERR_IO; }
    if (fe_dtype_size((FeDtype)dtype_u32) == 0) {
        fclose(f);
        return FE_ERR_DTYPE;
    }
    if (ndim_u32 < 1 || ndim_u32 > FERRITE_MAX_DIMS) {
        fclose(f);
        return FE_ERR_SHAPE;
    }

    int shape[FERRITE_MAX_DIMS];
    for (uint32_t d = 0; d < ndim_u32; d++) {
        uint32_t dim;
        if ((s = read_u32le(f, &dim)) != FE_OK) { fclose(f); return s; }
        if (dim < 1 || dim > (uint32_t)-1 / 2) {   /* reject absurd dims */
            fclose(f);
            return FE_ERR_SHAPE;
        }
        shape[d] = (int)dim;
    }

    t = fe_tensor_alloc((FeDtype)dtype_u32, (int)ndim_u32, shape);
    if (!t) { fclose(f); return FE_ERR_NOMEM; }

    uint64_t data_len;
    if ((s = read_u64le(f, &data_len)) != FE_OK ||
        data_len != (uint64_t)t->nbytes) {
        fe_tensor_free(t);
        fclose(f);
        return FE_ERR_IO;
    }

    if (t->nbytes > 0 &&
        fread(t->data, 1, t->nbytes, f) != t->nbytes) {
        fe_tensor_free(t);
        fclose(f);
        return FE_ERR_IO;
    }

    /* Strict v1: nothing may follow the payload. */
    if (fgetc(f) != EOF) {
        fe_tensor_free(t);
        fclose(f);
        return FE_ERR_IO;
    }

    fclose(f);
    *out = t;
    return FE_OK;
}
