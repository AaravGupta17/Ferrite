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
