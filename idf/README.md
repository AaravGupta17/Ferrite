# idf/README.md — ESP-IDF integration (Section 4)
#
# The `ferrite` directory is an ESP-IDF component that compiles the Ferrite
# *device runtime* (scalar kernels, planner, engine, quant only) for the
# ESP32 family. Host-only subsystems (importer, optimizer, compiler/IR,
# thread pool, x86 SIMD, FLOAT64) are compiled out of the part.
#
# ## Add to a project
#
# Option A — local component dir (used by `idf/demo`):
#
#   idf.py set-target esp32s3
#   # in your project CMakeLists.txt:
#   set(EXTRA_COMPONENT_DIRS <ferrite-repo>/idf/ferrite)
#
# Option B — managed component: point IDF_COMPONENT_MANAGER at the dir and
# requre `ferrite` in main/idf_component.yml.
#
# Either way, from your app:
#
#   PRIV_REQUIRES ferrite
#   #include "engine.h"   /* FeRuntime */
#   #include "model_ser.h"  /* fe_model_load from a compiled FEMD blob */
#
# ## `idf/demo` — hardware smoke test
#
#   cd idf/demo
#   idf.py set-target esp32s3
#   idf.py build flash monitor
#
# It builds a 1-node relu graph and prints the expected output. Replacing the
# hand-built graph with a compiled FEMD model header is the intended upgrade
# path; models must be quantized INT8/INT16/FP16 (FLOAT64 models fail to load,
# see below) and sized to the idf/ferrite/config/config.h ceilings.
#
# ## Deviations from the host build (all fail loudly, none silently)
# - FLOAT64 tensors: rejected at fe_model_load time (FERRITE_NO_FLOAT64).
# - Models bigger than FE_MAX_NODES / FE_MAX_TENSORS / FERRITE_MAX_DIMS:
#   rejected at load.
# - No importer: `.onnx` never reaches the device; use `ferrite-compile`.
# - No threads: ENGINE_PARALLEL kernels are unavailable; pool calls return
#   "unsupported" from core/platform_esp32.c.
# - Profiler compiled out (esp_timer still available for app code).
#
# ## Verification status
# The `esp` CI job builds this component under the espressif/idf container
# (compile gate). End-to-end execution is hardware-only — docs/esp32-port.md
# lists the on-device smoke test as a manual step.