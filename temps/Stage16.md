# Stage 16 — GPU

## Bottom Line Up Front

GPU support means an optional backend that offloads whole ops to a GPU, with host↔device memory managed by one transfer policy. It is the largest possible scope jump in this plan and must be strictly gated: **done when** a model runs on GPU with correct results and a benchmarked speedup, while the CPU path stays the default.

This stage assumes CUDA on the dev machine. There is no GPU code in Ferrite today. Do not start this before the CPU runtime is a finished, benchmarked product — GPU is a second backend, not a rescue.

## Deliverables

- CUDA backend
- CUDA kernels
- cuBLAS integration
- Stream execution
- Memory transfer optimization

## How to Proceed

1. **CUDA is a dependency — violate the zero-dependency rule deliberately.** The runtime stays zero-dependency; GPU support is an *optional build target* (`-DFERRITE_CUDA`) that links CUDA only when requested. The core never includes CUDA headers.
2. **Backend abstraction first, CUDA second.** Define a minimal backend interface (`init`, `copy_to_device`, `run_op`, `copy_to_host`, `sync`). Implement a trivial CPU "backend" that is the existing engine, then a CUDA one. The graph and planner never know which backend runs.
3. **cuBLAS for matmul, hand-written kernels for everything else.** GEMM on GPU is a solved problem — use cuBLAS for it. Write small CUDA kernels for elementwise and activation ops. Do not hand-roll GEMM on GPU.
4. **One transfer policy, applied consistently.** The simple correct policy: upload weights once at load, upload the input, run all ops on device, download the output once. Avoid per-op host↔device ping-pong; it is the #1 GPU performance killer.
5. **Streams for overlap, not for complexity.** One compute stream is the baseline. Add a second stream only when you can overlap an upload with a kernel run, and only after measuring the gap.
6. **Memory transfers dominate; optimize them last.** Profiling will show transfers are the bottleneck on small models. Batch uploads, keep weights resident, reuse device buffers. Match this stage's numbers to Stage 8's CPU table.
7. **Validation is against the CPU backend.** Run the same graph on CPU and GPU, compare within tolerance. GPU output that disagrees with CPU output is a GPU bug.
8. **Fail loudly, like everything else.** No CUDA runtime or no GPU → return `FE_ERR` with a clear message. Never silently fall back mid-run.

**Verify.** A GPU build passes the full test suite with outputs matching the CPU run, and a benchmark (matmul-bound model) shows the transfer-inclusive speedup. Without a measured win, the backend is not "done."

**Do not** split the codebase. One engine, two backends behind one interface. GPU-specific code lives in a `gpu/` directory, never in `runtime/`.
