# Stage 1 — Core Tensor Library

## Bottom Line Up Front

The tensor is the universal data structure. Everything — kernels, graphs, the engine — moves tensors. Get ownership, strides, and views right here or every later stage pays the cost. **Done when:** a strided `FeTensor` supports shape ops without copying and passes its own test binary under ASan/UBSan.

Ferrite's tensor (`core/tensor.h`) already covers most of this stage: strided views, transpose, reshape, contiguous repack, element access, printing, comparison. The gaps are slicing, broadcasting, serialization, and multi-dtype coverage beyond float32/int8/int32.

## Deliverables

- Tensor class
- Shape class
- Tensor metadata
- Data types (float32, float64, int32...)
- Memory ownership
- Tensor indexing
- Tensor slicing
- Tensor reshaping
- Broadcasting support
- Tensor serialization
- Tensor printing
- Tensor comparison

## How to Proceed

1. **Metadata over behavior.** Keep the tensor a plain struct (data pointer, dtype, ndim, shape, strides, nbytes, ownership flag). Compute nothing; store everything. See the existing `FeTensor` — match its shape.
2. **Data types.** Add a `fe_dtype_size` entry per new dtype (float64 = 8, etc.) and extend the enum in `core/types.h`. Every kernel must validate dtype before touching data.
3. **Memory ownership.** One rule: `owns_data == true` frees the buffer; views never free the parent's. Enforce it in `fe_tensor_free` and never pass arena-allocated tensors to it.
4. **Strides before slicing.** Slicing is just `data + offset` plus adjusted shape/strides. Implement it as a view first; add a materializing copy only when kernels need contiguity.
5. **Reshaping.** Only succeed when the tensor is contiguous and the element count holds. Return a view, never copy — this is already the contract.
6. **Broadcasting.** Implement NumPy-style alignment from the trailing dimension backward, producing a view where broadcast axes have stride 0. Kernels then handle `stride == 0` loops. Do not eagerly materialize broadcast tensors.
7. **Serialization.** Define one simple format (magic, dtype, shape, bytes) and write read/write that round-trips. Keep it versioned from the start.
8. **Printing and comparison.** Debug-only. `fe_tensor_print` and an approximate-equality compare with a tolerance argument. These are test tools, not hot paths.

**Verify.** Each addition gets a case in `tests/test_tensor.c`, built with `-fsanitize=address,undefined`. Contiguity, ownership, and view invariants get explicit tests — these are the rules every other subsystem relies on.

**Do not** add allocation to hot paths. Views and slices are the point of this stage; eager copies are the failure mode.
