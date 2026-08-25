// core/tensor_ser.h
#ifndef FERRITE_TENSOR_SER_H
#define FERRITE_TENSOR_SER_H

#include "tensor.h"

/*
 * Tensor serialization — binary round-trip of a tensor's logical
 * contents to and from a versioned file format.
 *
 * Format v1 (all integers little-endian, fixed-width):
 *
 *   offset  size  field
 *   ------  ----  -------------------------------------------------
 *   0       4     magic: "FETN"
 *   4       4     version: u32, currently 1
 *   8       4     dtype: u32, a FeDtype value
 *   12      4     ndim: u32, 1..FERRITE_MAX_DIMS
 *   16      4*nd  shape: int32 per axis, each >= 1
 *   16+4nd  8     data_len: u64, payload byte count
 *   ...     n     payload: raw element bytes, row-major contiguous
 *
 * What is serialized is the canonical form, not the struct: strides,
 * data pointer, and ownership are layout details and are not stored.
 * Saving a non-contiguous tensor first repacks it via
 * fe_tensor_contiguous(), so any view saves its logical contents.
 * Loading always produces a fresh owning, row-major tensor
 * (owns_data = true) that the caller frees with fe_tensor_free().
 *
 * Readers reject anything not exactly matching v1: wrong magic,
 * newer version, unknown dtype, impossible shape, data_len that does
 * not match the computed byte count, truncated payloads, and trailing
 * garbage all return errors without leaking memory.
 */

#define FERRITE_TENSOR_SER_MAGIC   "FETN"
#define FERRITE_TENSOR_SER_VERSION 1u

/*
 * Write the tensor's logical contents to `path`.
 * Returns FE_ERR_NULL for missing pointers, FE_ERR_IO on any file or
 * encoding failure, FE_ERR_NOMEM if repacking fails.
 */
FeStatus fe_tensor_save(const FeTensor *t, const char *path);

/*
 * Read a tensor file into *out. On success *out holds a fresh owning
 * tensor; on failure *out is untouched. Returns FE_ERR_IO for corrupt,
 * truncated, or unsupported-version files, FE_ERR_DTYPE for unknown
 * dtypes, FE_ERR_SHAPE for impossible dimensions, FE_ERR_NOMEM when
 * allocation fails.
 */
FeStatus fe_tensor_load(const char *path, FeTensor **out);

#endif // FERRITE_TENSOR_SER_H
