#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cpu_convolution.h"
#include "gpu_convolution.h"

// dynamic pointers for our matrices
double **A;
double **C;

// 3x3 sharpening kernel
double K[3][3] = {
    {0, -1, 0},
    {-1, 5, -1},
    {0, -1, 0}};

// allocate memory for the 2d array
double **allocate_matrix(int size)
{
    // allocate array of row pointers
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
    {
        // allocate memory for each column in the row
        matrix[i] = (double *)malloc(size * sizeof(double));
    }
    return matrix;
}

// check if the system actually has a gpu
int has_gpu()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0)
    {
        return 0;
    }
    return 1;
}

// fill matrix with random numbers
void initMatrices(int size)
{
    srand(1); // seed set to 1 for consistent testing
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            A[i][j] = (double)(rand() % 10);
            C[i][j] = 0;
        }
    }
}

// parse cli inputs
int parse_arguments(int argc, char *argv[], int *n, int *num_threads, char *mode)
{
    // grab inputs if provided
    if (argc >= 2)
        *n = atoi(argv[1]);
    if (argc >= 3)
        *num_threads = atoi(argv[2]);
    if (argc >= 4)
    {
        strncpy(mode, argv[3], 9);
        mode[9] = '\0'; // ensure string ends properly
    }

    // matrix needs to be at least 3x3
    if (*n < 3)
    {
        printf("Matrix size must be at least 3.\n");
        return 1;
    }

    // make sure thread count makes sense
    if (*num_threads < 1)
        *num_threads = 1;
    // clamp threads so we don't exceed valid rows
    if (*num_threads > *n - 2)
        *num_threads = *n - 2;

    // check for valid modes
    if (strcmp(mode, "cpu") != 0 && strcmp(mode, "gpu") != 0 && strcmp(mode, "both") != 0)
    {
        printf("Invalid mode. Please use 'cpu', 'gpu', or 'both'.\n");
        return 1;
    }

    return 0;
}

// helper to print small matrices
void print_matrix(double **mat, int n, const char *title)
{
    printf("%s\n", title);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.0f ", mat[i][j]);
        }
        printf("\n");
    }
}

// save cpu output to file
void save_general_matrix(const char *filename, double **M, int size)
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        perror("fopen");
        return;
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            fprintf(fp, "%.4f ", M[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

// free the memory to prevent leaks
void cleanup_matrices(int n)
{
    for (int i = 0; i < n; i++)
    {
        free(A[i]);
        free(C[i]);
    }
    free(A);
    free(C);
}

// save gpu output to file (needs 1d array)
void save_gpu_matrix(const char *filename, float *M, int size)
{
    {
        FILE *fp = fopen(filename, "w");
        if (!fp)
        {
            perror("fopen");
            return;
        }

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                fprintf(fp, "%.4f ", M[i * size + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);
    }
}

int main(int argc, char *argv[])
{
    int n = 500;
    int num_threads = 5;
    char mode[10] = "cpu"; // iand: why this?

    // parse cli inputs and exit if bad
    if (parse_arguments(argc, argv, &n, &num_threads, mode) != 0)
    {
        return 1;
    }

    // fallback to cpu if no gpu is found
    if (strcmp(mode, "cpu") != 0)
    {
        if (has_gpu())
        {
            printf("GPU available. Mode: %s\n", mode);
        }
        else
        {
            printf("No GPU detected, falling back to CPU only.\n");
            strcpy(mode, "cpu");
        }
    }

    printf("Configuration: %dx%d matrix, %d threads.\n", n, n, num_threads);

    A = allocate_matrix(n);
    C = allocate_matrix(n);
    initMatrices(n);

    // test output (only for small matrices)
    if (n <= 10)
    {
        print_matrix(A, n, "Initial Matrix A:");
    }

    // timer variable
    double start_time;

    // run cpu workload
    if (strcmp(mode, "cpu") == 0 || strcmp(mode, "both") == 0)
    {
        start_time = get_current_time();
        printf("\n--- Running CPU Version ---\n");
        run_convolution(n, num_threads);
        double cpu_time = get_current_time() - start_time;
        printf("CPU Version Elapsed Time: %.4f  \n", cpu_time);

        save_general_matrix("output_cpu.txt", C, n);
    }

    // run gpu workload
    if (strcmp(mode, "gpu") == 0 || strcmp(mode, "both") == 0)
    {
        start_time = get_current_time();
        printf("\n--- Running GPU Version ---\n");

        // setup 1d arrays for cuda
        float *h_A_flat = (float *)malloc(n * n * sizeof(float));
        float *h_C_flat = (float *)malloc(n * n * sizeof(float));
        float h_K_flat[9];

        // flatten the 2d array A into 1d
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                h_A_flat[i * n + j] = (float)A[i][j];
            }
        }

        // flatten the kernel
        int idx = 0;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                h_K_flat[idx++] = (float)K[i][j];
            }
        }

        run_gpu_convolution(h_A_flat, h_K_flat, h_C_flat, n, n, 3);

        double gpu_time = get_current_time() - start_time;
        printf("GPU Version Elapsed Time: %.4f \n", gpu_time);

        save_gpu_matrix("output_gpu.txt", h_C_flat, n);

        free(h_A_flat);
        free(h_C_flat);
    }

    cleanup_matrices(n);

    return 0;
}