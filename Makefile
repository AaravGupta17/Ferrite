CC     = gcc
CFLAGS = -std=c11 -D_POSIX_C_SOURCE=199309L -Wall -Wextra -fsanitize=address,undefined -g -Icore -Igraph -Iops -Iruntime -Iimporter

all: test_tensor test_allocator test_ops test_graph test_engine test_onnx

test_tensor: core/tensor.o tests/test_tensor.o
	$(CC) $(CFLAGS) -o test_tensor core/tensor.o tests/test_tensor.o

core/tensor.o: core/tensor.c core/tensor.h core/types.h
	$(CC) $(CFLAGS) -c core/tensor.c -o core/tensor.o

tests/test_tensor.o: tests/test_tensor.c core/tensor.h core/types.h
	$(CC) $(CFLAGS) -c tests/test_tensor.c -o tests/test_tensor.o

clean:
	rm -f core/*.o tests/*.o test_tensor

test_allocator: core/tensor.o core/allocator.o tests/test_allocator.o
	$(CC) $(CFLAGS) -o test_allocator core/tensor.o core/allocator.o tests/test_allocator.o

core/allocator.o: core/allocator.c core/allocator.h core/tensor.h core/types.h
	$(CC) $(CFLAGS) -c core/allocator.c -o core/allocator.o

tests/test_allocator.o: tests/test_allocator.c core/allocator.h core/tensor.h
	$(CC) $(CFLAGS) -c tests/test_allocator.c -o tests/test_allocator.o

test_ops: core/tensor.o ops/matmul.o ops/activations.o tests/test_ops.o
	$(CC) $(CFLAGS) -o test_ops core/tensor.o ops/matmul.o ops/activations.o tests/test_ops.o -lm

ops/matmul.o: ops/matmul.c ops/ops.h core/tensor.h core/types.h
	$(CC) $(CFLAGS) -c ops/matmul.c -o ops/matmul.o

ops/activations.o: ops/activations.c ops/ops.h core/tensor.h core/types.h
	$(CC) $(CFLAGS) -c ops/activations.c -o ops/activations.o

tests/test_ops.o: tests/test_ops.c ops/ops.h core/tensor.h
	$(CC) $(CFLAGS) -c tests/test_ops.c -o tests/test_ops.o

test_graph: core/tensor.o graph/graph.o tests/test_graph.o
	$(CC) $(CFLAGS) -o test_graph core/tensor.o graph/graph.o tests/test_graph.o

graph/graph.o: graph/graph.c graph/graph.h core/tensor.h core/types.h
	$(CC) $(CFLAGS) -c graph/graph.c -o graph/graph.o

tests/test_graph.o: tests/test_graph.c graph/graph.h core/tensor.h
	$(CC) $(CFLAGS) -c tests/test_graph.c -o tests/test_graph.o

test_engine: core/tensor.o core/allocator.o graph/graph.o \
             ops/matmul.o ops/activations.o \
             runtime/engine.o tests/test_engine.o
	$(CC) $(CFLAGS) -o test_engine core/tensor.o core/allocator.o \
	    graph/graph.o ops/matmul.o ops/activations.o \
	    runtime/engine.o tests/test_engine.o -lm

runtime/engine.o: runtime/engine.c runtime/engine.h \
                  graph/graph.h ops/ops.h core/tensor.h core/types.h
	$(CC) $(CFLAGS) -c runtime/engine.c -o runtime/engine.o

tests/test_engine.o: tests/test_engine.c runtime/engine.h \
                     graph/graph.h core/tensor.h
	$(CC) $(CFLAGS) -c tests/test_engine.c -o tests/test_engine.o

test_onnx: core/tensor.o core/allocator.o graph/graph.o \
           importer/onnx.o tests/test_onnx.o
	$(CC) $(CFLAGS) -o test_onnx core/tensor.o core/allocator.o \
	    graph/graph.o importer/onnx.o tests/test_onnx.o

importer/onnx.o: importer/onnx.c importer/onnx.h \
                 graph/graph.h core/tensor.h core/types.h
	$(CC) $(CFLAGS) -Iimporter -c importer/onnx.c -o importer/onnx.o

tests/test_onnx.o: tests/test_onnx.c importer/onnx.h graph/graph.h
	$(CC) $(CFLAGS) -Iimporter -c tests/test_onnx.c -o tests/test_onnx.o

bench_matmul: core/tensor.o ops/matmul.o ops/activations.o tools/bench_matmul.o
	$(CC) $(CFLAGS) -O2 -o bench_matmul core/tensor.o ops/matmul.o \
	    ops/activations.o tools/bench_matmul.o -lm

tools/bench_matmul.o: tools/bench_matmul.c core/tensor.h ops/ops.h
	$(CC) $(CFLAGS) -O2 -Itools -c tools/bench_matmul.c -o tools/bench_matmul.o

bench_matmul_avx2: core/tensor.o ops/matmul.o ops/activations.o \
                   simd/matmul_avx2.o tools/bench_matmul.o
	$(CC) $(CFLAGS) -O3 -mavx2 -mfma -o bench_matmul_avx2 \
	    core/tensor.o ops/matmul.o ops/activations.o \
	    simd/matmul_avx2.o tools/bench_matmul.o -lm

simd/matmul_avx2.o: simd/matmul_avx2.c simd/matmul_avx2.h \
                    core/tensor.h core/types.h
	$(CC) $(CFLAGS) -O3 -mavx2 -mfma -Isimd \
	    -c simd/matmul_avx2.c -o simd/matmul_avx2.o

bench_avx2: core/tensor.o ops/matmul.o ops/activations.o \
            simd/matmul_avx2.o tools/bench_avx2.o
	$(CC) $(CFLAGS) -O3 -mavx2 -mfma -o bench_avx2 \
	    core/tensor.o ops/matmul.o ops/activations.o \
	    simd/matmul_avx2.o tools/bench_avx2.o -lm

tools/bench_avx2.o: tools/bench_avx2.c core/tensor.h ops/ops.h \
                    simd/matmul_avx2.h
	$(CC) $(CFLAGS) -O3 -mavx2 -mfma -Isimd \
	    -c tools/bench_avx2.c -o tools/bench_avx2.o