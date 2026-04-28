# Compiler settings
NVCC        = nvcc
CC          = gcc
CFLAGS      = -O2 -Wall
NVCCFLAGS   = -O2 -Xcompiler -Wall


# Default build
all: both

both: cpu_convolution.o gpu_convolution.o main.o
	$(NVCC) -o matrix_convolution cpu_convolution.o gpu_convolution.o main.o -lpthread

cpu: cpu_convolution.o main_cpu.o
	$(CC) -o matrix_convolution_cpu cpu_convolution.o main_cpu.o -lpthread
# Compile CPU C file
cpu_convolution.o: cpu_convolution.c cpu_convolution.h
	$(CC) $(CFLAGS) -c cpu_convolution.c

# Compile GPU CUDA file
gpu_convolution.o: gpu_convolution.cu gpu_convolution.h
	$(NVCC) $(NVCCFLAGS) -c gpu_convolution.cu

# Compile main (CUDA-aware)
main.o: main.cu cpu_convolution.h gpu_convolution.h
	$(NVCC) $(NVCCFLAGS) -c main.cu

main_cpu.o: main_cpu.c cpu_convolution.h
	$(CC) $(CFLAGS) -c main_cpu.c

# Cleanup
clean:
	rm -f *.o matrix_convolution *.txt matrix_convolution_cpu
