# Stage 17 — Plugin System

## Bottom Line Up Front

The plugin system lets third-party kernels and backends plug into the runtime without touching core code: a registration API, a name-based lookup, and a dispatch that prefers plugins. **Done when:** a custom operator ships as a plugin and runs in a model, with the core unchanged.

Ferrite has no plugin support. It is a natural fit for a layered runtime but must not be built until the core op set is stable — plugins are an extension mechanism, not a way to leave the core unfinished.

## Deliverables

- Dynamic operator loading
- Plugin API
- Third-party kernels
- Backend plugins

## How to Proceed

1. **Start with compile-time registration, not dynamic loading.** A registration table (`fe_register_op(name, fn, ctx)`) is simpler, testable, and sufficient for most needs. Dynamic `.so`/`.dll` loading is a packaging layer on top — add it only when someone needs to ship a kernel without recompiling Ferrite.
2. **The plugin API is the kernel contract, widened.** A plugin op has: a name, a validate function, and a run function, all with `FeTensor*` arguments and `FeStatus` returns. Reuse `ops/ops.h` signatures verbatim; do not invent a second calling convention.
3. **Lookup by name, dispatch by table.** The engine's `switch` stays for built-ins; plugins live in a name→fn map checked before or after the switch by policy. The graph stores op *names* (`FeOpType` for built-ins, strings for plugins) so plugin ops serialize through the ONNX importer unchanged.
4. **Third-party kernels obey the same rules.** Read-only inputs, caller-allocated outputs, no allocation inside, validate before touching data. A plugin that violates the contract is rejected at registration.
5. **Backend plugins mirror the GPU abstraction.** A backend plugin registers `init/run_op/sync` behind the same interface as the CUDA backend (Stage 16). The runtime detects backends at init and picks one — CPU is the default, a plugin overrides.
6. **Version the plugin API.** `FE_PLUGIN_API_VERSION` constant checked at registration. A mismatch is a loud error, never silent undefined behavior.
7. **Sandboxing is the plugin's owner's job.** Loading untrusted code is out of scope; document it. The plugin system provides registration and dispatch, not security.

**Verify.** `tests/test_plugins.c`: register a custom op (e.g., a fused op), build a graph using it, run it through the engine, and assert the plugin dispatch was used. The plugin must also survive the ONNX round-trip by name.

**Do not** build dynamic loading or a sandbox before registration is proven. The interesting engineering is the API and dispatch, not `dlopen`.
