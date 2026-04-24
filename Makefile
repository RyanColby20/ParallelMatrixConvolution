# Compiler settings
NVCC        = nvcc
CC          = gcc
CFLAGS      = -O2 -Wall
NVCCFLAGS   = -O2 -Xcompiler -Wall


# Default build
all: matrix_convolution


matrix_convolution: cpu_convolution.o gpu_convolution.o main.o
	$(NVCC) -o matrix_convolution cpu_convolution.o gpu_convolution.o main.o -lpthread

# Compile CPU C file
cpu_convolution.o: cpu_convolution.c cpu_convolution.h
	$(CC) $(CFLAGS) -c cpu_convolution.c

# Compile GPU CUDA file
gpu_convolution.o: gpu_convolution.cu gpu_convolution.h
	$(NVCC) $(NVCCFLAGS) -c gpu_convolution.cu

# Compile main (CUDA-aware)
main.o: main.cu cpu_convolution.h gpu_convolution.h
	$(NVCC) $(NVCCFLAGS) -c main.cu

# Cleanup
clean:
	rm -f *.o matrix_convolution *.txt
