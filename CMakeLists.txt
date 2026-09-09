# CMakeLists.txt — Ferrite build.
#
# Mirrors the Makefile: same sources, same binaries, same flags.
# Canonical build story (roadmap Phase 0); requires GCC or Clang —
# MSVC lacks clock_gettime/_POSIX_C_SOURCE used by tools/profiler.c.
#
#   cmake -S . -B build -G Ninja
#   cmake --build build              # all tests
#   ctest --test-dir build           # run the suite
#
# Benches and demo are opt-in:  cmake --build build --target bench_avx2

cmake_minimum_required(VERSION 3.16)
project(Ferrite C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
        "Ferrite requires GCC or Clang (POSIX clock + sanitizers). "
        "Use CLion's bundled MinGW, MSYS2, or WSL.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)   # tests run under ASan/UBSan by default
endif()

option(FERRITE_SANITIZE "Build everything with ASan+UBSan" ON)

add_compile_options(-Wall -Wextra -g)
add_compile_definitions(_POSIX_C_SOURCE=199309L)

# Probe whether the toolchain actually ships the sanitizer runtimes
# (CLion's bundled MinGW does not; MSYS2 gcc and Linux gcc do).
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=address,undefined")
check_c_source_compiles("int main(void){return 0;}" FERRITE_HAVE_SANITIZERS)
unset(CMAKE_REQUIRED_FLAGS)

if(FERRITE_SANITIZE AND FERRITE_HAVE_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    set(FERRITE_SANITIZE_ENABLED ON)
elseif(FERRITE_SANITIZE)
    message(STATUS "ASan/UBSan requested but unavailable in this toolchain - building without")
endif()

option(FERRITE_COVERAGE "Build with gcov coverage instrumentation" OFF)
if(FERRITE_COVERAGE)
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

# On Windows, executables need libwinpthread-1.dll (clock_gettime in the
# profiler). This toolchain cannot link its static winpthread against its
# own CRT, so instead ship the correct DLL beside every executable: the
# Windows loader searches the exe directory before PATH, which defeats
# stale copies of the DLL found there.
if(WIN32)
    get_filename_component(FERRITE_MINGW_BIN "${CMAKE_C_COMPILER}" DIRECTORY)
    set(FERRITE_PTHREAD_DLL "${FERRITE_MINGW_BIN}/libwinpthread-1.dll")
endif()

function(ferrite_exe target)
    if(WIN32 AND EXISTS "${FERRITE_PTHREAD_DLL}")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FERRITE_PTHREAD_DLL}" "$<TARGET_FILE_DIR:${target}>")
    endif()
endfunction()

if(UNIX)
    link_libraries(m)
endif()

# Flat include path, matching the Makefile's global -I flags: sources
# include headers by bare name ("tensor.h", "profiler.h").
set(FERRITE_DIRS core graph ops simd planner runtime importer quantization optim tools)
foreach(dir ${FERRITE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${dir})
endforeach()

# --- Libraries (one per subsystem; dependencies point downward only) ---

add_library(ferrite_core STATIC
    core/tensor.c
    core/tensor_ser.c
    core/allocator.c
    core/log.c
    core/fp16.c
)

add_library(ferrite_graph STATIC graph/graph.c)
target_link_libraries(ferrite_graph PUBLIC ferrite_core)

# SIMD kernels are compiled with AVX2+FMA unconditionally; runtime CPUID
# detection (fe_cpu_has_avx2) decides whether they execute.
add_library(ferrite_simd STATIC simd/matmul_avx2.c simd/elementwise_avx2.c)
target_compile_options(ferrite_simd PRIVATE -O3 -mavx2 -mfma)

add_library(ferrite_ops STATIC
    ops/matmul.c
    ops/activations.c
    ops/conv1d.c
    ops/conv2d.c
    ops/elementwise.c
    ops/gemm.c
    ops/math.c
    ops/norm.c
    ops/pool.c
    ops/rand.c
    ops/reduce.c
    ops/sequence.c
    ops/stability.c
)
target_link_libraries(ferrite_ops PUBLIC ferrite_core ferrite_simd)

add_library(ferrite_planner STATIC planner/memory_planner.c)
target_link_libraries(ferrite_planner PUBLIC ferrite_graph)

# Engine calls the profiler (engine.c); Tools here tofix the Makefile's
# old test_engine rule. Exec plan (Stage 13) lives alongside the engine;
# quant.h supplies the engine-path INT8 kernels (Stage 15).
add_library(ferrite_runtime STATIC runtime/engine.c runtime/exec_plan.c)
target_link_libraries(ferrite_runtime PUBLIC ferrite_graph ferrite_ops ferrite_tools ferrite_planner ferrite_optim ferrite_quant)

add_library(ferrite_importer STATIC importer/onnx.c)
target_link_libraries(ferrite_importer PUBLIC ferrite_graph ferrite_optim)

add_library(ferrite_quant STATIC quantization/quant.c)
target_link_libraries(ferrite_quant PUBLIC ferrite_core ferrite_graph)

# Optimization passes (Stage 12): shape infer, folding, fusion, CSE, DCE.
add_library(ferrite_optim STATIC
    optim/shape_infer.c
    optim/constant_fold.c
    optim/dead_elim.c
    optim/simplify.c
    optim/cse.c
    optim/fusion.c
    optim/optim.c
)
target_link_libraries(ferrite_optim PUBLIC ferrite_graph ferrite_ops)

add_library(ferrite_tools STATIC tools/profiler.c tools/bench.c)

# Stage 11 parallel runtime: worker pool, row-parallel GEMM, DAG scheduler.
find_package(Threads REQUIRED)
add_library(ferrite_parallel STATIC
    parallel/threadpool.c
    parallel/parallel_gemm.c
    parallel/scheduler.c
)
target_link_libraries(ferrite_parallel PUBLIC ferrite_core ferrite_graph Threads::Threads)

# Stage 14 compiler: linear IR, lowering, kernel selection, scheduling.
add_library(ferrite_compiler STATIC
    compiler/lower.c
    compiler/codegen.c
    compiler/schedule.c
)
target_link_libraries(ferrite_compiler PUBLIC ferrite_graph ferrite_ops ferrite_core)

# --- Tests (built by default, registered with CTest) ---

enable_testing()

add_executable(test_tensor      tests/test_tensor.c)
add_executable(test_allocator   tests/test_allocator.c)
add_executable(test_ops         tests/test_ops.c)
add_executable(test_ops3        tests/test_ops3.c)
add_executable(test_simd        tests/test_simd.c)
add_executable(test_graph       tests/test_graph.c)
add_executable(test_engine      tests/test_engine.c)
add_executable(test_onnx        tests/test_onnx.c)
add_executable(test_planner     tests/test_planner.c)
add_executable(test_conv1d      tests/test_conv1d.c)
add_executable(test_profiler    tests/test_profiler.c)
add_executable(test_quant       tests/test_quant.c)
add_executable(test_tensor_ser  tests/test_tensor_ser.c)
add_executable(test_opt         tests/test_opt.c)

add_executable(test_parallel   tests/test_parallel.c)
add_executable(test_compiler   tests/test_compiler.c)

target_link_libraries(test_tensor      ferrite_core)
target_link_libraries(test_allocator   ferrite_core)
target_link_libraries(test_ops         ferrite_core ferrite_ops ferrite_simd)
target_link_libraries(test_ops3        ferrite_core ferrite_ops ferrite_simd)
target_link_libraries(test_simd        ferrite_core ferrite_ops ferrite_simd)
target_link_libraries(test_graph       ferrite_core ferrite_graph)
target_link_libraries(test_engine      ferrite_runtime ferrite_importer)
target_link_libraries(test_onnx        ferrite_core ferrite_graph ferrite_importer)
target_link_libraries(test_planner     ferrite_core ferrite_graph ferrite_planner)
target_link_libraries(test_conv1d      ferrite_core ferrite_ops)
target_link_libraries(test_profiler    ferrite_core ferrite_graph ferrite_ops ferrite_simd ferrite_runtime ferrite_tools)
target_link_libraries(test_quant       ferrite_core ferrite_quant ferrite_runtime ferrite_importer)
target_link_libraries(test_tensor_ser  ferrite_core)
target_link_libraries(test_opt         ferrite_core ferrite_graph ferrite_optim ferrite_runtime)
target_link_libraries(test_parallel    ferrite_core ferrite_graph ferrite_ops ferrite_parallel)
target_link_libraries(test_compiler    ferrite_core ferrite_graph ferrite_ops ferrite_compiler ferrite_runtime)

foreach(t test_tensor test_allocator test_ops test_ops3 test_simd test_graph test_engine
        test_onnx test_planner test_conv1d test_profiler test_quant
        test_tensor_ser test_opt test_parallel test_compiler)
    ferrite_exe(${t})
    # Run from the source root: tests reference fixtures by relative
    # path ("tests/tiny_mlp.onnx"), matching the Makefile's cwd.
    add_test(NAME ${t} COMMAND ${t} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

# --- Benchmarks and demo (opt-in, excluded from the default build) ---

add_executable(bench_matmul EXCLUDE_FROM_ALL tools/bench_matmul.c)
target_link_libraries(bench_matmul PRIVATE ferrite_core ferrite_ops ferrite_simd)
target_compile_options(bench_matmul PRIVATE -O2)
ferrite_exe(bench_matmul)

add_executable(bench_matmul_avx2 EXCLUDE_FROM_ALL tools/bench_matmul.c)
target_link_libraries(bench_matmul_avx2 PRIVATE ferrite_core ferrite_ops ferrite_simd)
target_compile_options(bench_matmul_avx2 PRIVATE -O3 -mavx2 -mfma)
ferrite_exe(bench_matmul_avx2)

add_executable(bench_avx2 EXCLUDE_FROM_ALL tools/bench_avx2.c)
target_link_libraries(bench_avx2 PRIVATE ferrite_core ferrite_ops ferrite_simd)
target_compile_options(bench_avx2 PRIVATE -O3 -mavx2 -mfma)
ferrite_exe(bench_avx2)

add_executable(demo EXCLUDE_FROM_ALL tools/demo_acousticleaknet.c)
target_link_libraries(demo PRIVATE
    ferrite_core ferrite_graph ferrite_ops ferrite_simd
    ferrite_runtime ferrite_importer ferrite_tools)
target_compile_options(demo PRIVATE -O2)
ferrite_exe(demo)

# Model-relevant benchmark: end-to-end latency + per-op profile + memory
# + naive-vs-AVX2 table derived from the loaded model's MatMul shapes.
add_executable(bench_model EXCLUDE_FROM_ALL tools/bench_model.c)
target_link_libraries(bench_model PRIVATE
    ferrite_core ferrite_graph ferrite_ops ferrite_simd
    ferrite_planner ferrite_runtime ferrite_importer ferrite_tools)
target_compile_options(bench_model PRIVATE -O2)
ferrite_exe(bench_model)
