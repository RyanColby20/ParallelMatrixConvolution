#ifndef GPU_CONVOLUTION_H
#define GPU_CONVOLUTION_H

void run_gpu_convolution(const float *h_A,
                         const float *h_K,
                         float *h_C,
                         int N, int M, int ksize);

#endif
