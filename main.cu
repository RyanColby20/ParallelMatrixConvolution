#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cpu_convolution.h"
#include "gpu_convolution.h"


/*
    ========================
    ====   MEMORY       ====
    ====   ALLOCATION   ====
    ========================
*/


// Dynamic pointers to support any size matrix
double **A;
double **C;

// 3x3 Kernel
double K[3][3] = {
    {0, -1, 0},
    {-1, 5, -1},
    {0, -1, 0}};



// Utility to dynamically allocate a 2D matrix
double **allocate_matrix(int size)
{
    // Allocate array of row pointers
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
    {
        // Allocate memory for each column in the row
        matrix[i] = (double *)malloc(size * sizeof(double));
    }
    return matrix;
}

/*
    ========================
    ==== INITIALIZATION ====
    ========================
*/

// check for device with cuda compatability
// return: 0 == no device; 1 == device
int has_gpu() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        return 0;
    }
    return 1;
}



// Initialization function
void initMatrices(int size)
{
    srand(time(NULL)); // Seed random number generator
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            A[i][j] = (double)(rand() % 10); // Random digits 0-9
            C[i][j] = 0; // Clear the output matrix
        }
    }
}

// Fct to parse command line arguments
int parse_arguments(int argc, char *argv[], int *n, int *num_threads, char *mode)
{
    // Grab inputs if the user provided them
    if (argc >= 2)
        *n = atoi(argv[1]);
    if (argc >= 3)
        *num_threads = atoi(argv[2]);
    if (argc >= 4)
    {
        strncpy(mode, argv[3], 9);
        mode[9] = '\0'; // Ensure string ends properly
    }

    // Matrix needs to be at least 3x3 for a 3x3 kernel
    if (*n < 3)
    {
        printf("Matrix size must be at least 3.\n");
        return 1;
    }

    // Make sure thread count makes sense
    if (*num_threads < 1)
        *num_threads = 1;
    // Don't use more threads than we have valid rows
    if (*num_threads > *n - 2)
        *num_threads = *n - 2;

    // Check for valid modes
    if (strcmp(mode, "cpu") != 0 && strcmp(mode, "gpu") != 0 && strcmp(mode, "both") != 0)
    {
        printf("Invalid mode. Please use 'cpu', 'gpu', or 'both'.\n");
        return 1;
    }

    return 0; // Success
}



/*
    ========================
    ====     UTILITY    ====
    ====     CLEANUP    ====
    ========================
*/

// returns current time
double get_current_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Print matrix for testing
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

void save_matrix(const char *filename, double **M, int size)
{
    FILE *fp = fopen(filename, "w");
    if(!fp) {
        perror("fopen");
        return;
    }

    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            fprintf(fp, "%.4f ", M[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}




// Free the 2D arrays to prevent memory leaks
void cleanup_matrices(int n)
{
    for (int i = 0; i < n; i++)
    {
        free(A[i]);
        free(C[i]);
    }
    free(A); // Free the row pointers
    free(C);
}

/*
    ========================
    ====      MAIN      ====
    ========================
*/


int main(int argc, char *argv[])
{
    int n = 500;
    int num_threads = 5;
    char mode[10] = "cpu"; // iand: why this?

    // Parse CLI inputs and exit if bad
    if (parse_arguments(argc, argv, &n, &num_threads, mode) != 0)
    {
        return 0;
    }

    // check for gpu
    if (strcmp(mode, "cpu") != 0){         
        if (has_gpu()) {
            printf("GPU available. Mode: %s\n", mode);
        } else{
            printf("No GPU detected, falling back to CPU only.\n");
            strcpy(mode, "cpu");
        }
    }   

    printf("Configuration: %dx%d matrix, %d threads.\n", n, n, num_threads);

    A = allocate_matrix(n);
    C = allocate_matrix(n);
    initMatrices(n);

    

    // TEST OUTPUT (only for small matrices)
    if (n <= 10)
    {
        print_matrix(A, n, "Initial Matrix A:");
    }

    // doubles to measure time
    double start_time;

    if (strcmp(mode, "cpu") == 0 || strcmp(mode, "both") == 0)
    {
        start_time = get_current_time();
        printf("\n--- Running CPU Version ---\n");
        run_convolution(n, num_threads);
        double cpu_time = get_current_time() - start_time;
        printf("CPU Version Elapsed Time: %.4f  \n", cpu_time);
    }

    if (strcmp(mode, "gpu") == 0 || strcmp(mode, "both") == 0)
    {
        // setting up vars for CUDA
        float *h_A_flat = (float *)malloc(n * n * sizeof(float));
        float *h_C_flat = (float *)malloc(n * n * sizeof(float));
        float h_K_flat[9];

        // pack A into h_A_flat
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                h_A_flat[i * n + j] = (float)A[i][j];
            }
        }

        // pack K into h_K_flat
        int idx = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                h_K_flat[idx++] = (float)K[i][j];
            }
        }

        start_time = get_current_time();
        printf("\n--- Running GPU Version ---\n");
        run_gpu_convolution(h_A_flat, h_K_flat, h_C_flat, n, n, 3);
        double gpu_time = get_current_time() - start_time;
        printf("GPU Version Elapsed Time: %.4f \n", gpu_time);

        // unpack C
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = h_C_flat[i * n + j];
            }
        }
    }

    save_matrix("output.txt", C, n);
    cleanup_matrices(n);

    return 0;
}