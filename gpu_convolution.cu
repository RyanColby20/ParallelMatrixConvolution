#include <cuda_runtime.h>
#include <stdio.h>
#include "gpu_convolution.h"

__global__ void convo_kernel(const float *A, const float *K, float *C,
                             int N, int M, int ksize)
{
    // get global x and y coordinates for this thread
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // radius of the kernel (e.g., 1 for a 3x3 kernel)
    int r = ksize / 2;

    // skip the edges
    if (i < r || i >= N - r || j < r || j >= M - r)
        return;

    float sum = 0.0f;

    // apply the kernel
    for (int ki = -r; ki <= r; ki++)
    {
        for (int kj = -r; kj <= r; kj++)
        {
            float a = A[(i + ki) * M + (j + kj)];
            float w = K[(ki + r) * ksize + (kj + r)];
            sum += a * w;
        }
    }
    C[i * M + j] = sum;
}

void run_gpu_convolution(const float *h_A,
                         const float *h_K,
                         float *h_C,
                         int N, int M, int ksize)
{
    size_t bytesA = N * M * sizeof(float);
    size_t bytesC = N * M * sizeof(float);
    size_t bytesK = ksize * ksize * sizeof(float);

    float *d_A, *d_C, *d_K;

    // setup cuda timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate device memory
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_C, bytesC);
    cudaMalloc(&d_K, bytesK);

    cudaMemset(d_C, 0, bytesC);

    // copy data from host to device
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytesK, cudaMemcpyHostToDevice);

    // set up grid and block sizes for 2d matrix
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);

    // start timer and run kernel
    cudaEventRecord(start);

    convo_kernel<<<grid, block>>>(d_A, d_K, d_C, N, M, ksize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calculate time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy results back to host
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    // clean up device memory
    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_K);

    printf("GPU Math-Only time %.4f\n", milliseconds / 1000);
}
