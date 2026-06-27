CC     = gcc
CFLAGS = -std=c11 -Wall -Wextra -fsanitize=address,undefined -g -Icore

all: test_tensor

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