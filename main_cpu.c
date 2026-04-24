#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cpu_convolution.h"

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

// Initialization function
void initMatrices(int size)
{
    srand(time(NULL)); // Seed random number generator
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            A[i][j] = (double)(rand() % 10); // Random digits 0-9
            C[i][j] = 0;                     // Clear the output matrix
        }
    }
}

// Fct to parse command line arguments
int parse_arguments(int argc, char *argv[], int *n, int *num_threads)
{
    // Grab inputs if the user provided them
    if (argc >= 2)
        *n = atoi(argv[1]);
    if (argc >= 3)
        *num_threads = atoi(argv[2]);

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

    return 0; // Success
}

/*
    ========================
    ====     UTILITY    ====
    ====     CLEANUP    ====
    ========================
*/

void save_matrix(const char *filename, double **M, int size)
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

double get_current_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
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

    // Parse CLI inputs and exit if bad
    if (parse_arguments(argc, argv, &n, &num_threads) != 0)
    {
        return 1;
    }

    printf("Configuration: %dx%d matrix, %d threads.\n", n, n, num_threads);

    A = allocate_matrix(n);
    C = allocate_matrix(n);
    initMatrices(n);

    // doubles to measure time
    double start_time;

    start_time = get_current_time();
    printf("\n--- Running CPU Version ---\n");
    run_convolution(n, num_threads);
    double cpu_time = get_current_time() - start_time;
    printf("CPU Version Elapsed Time: %.4f  \n", cpu_time);

    save_matrix("output.txt", C, n);
    cleanup_matrices(n);

    return 0;
}