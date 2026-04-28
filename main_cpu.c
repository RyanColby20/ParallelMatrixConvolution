#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cpu_convolution.h"

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
int parse_arguments(int argc, char *argv[], int *n, int *num_threads)
{
    // grab inputs if provided
    if (argc >= 2)
        *n = atoi(argv[1]);
    if (argc >= 3)
        *num_threads = atoi(argv[2]);

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

    return 0;
}

// save cpu output to file
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

int main(int argc, char *argv[])
{
    int n = 500;
    int num_threads = 5;

    // parse cli inputs and exit if bad
    if (parse_arguments(argc, argv, &n, &num_threads) != 0)
    {
        return 1;
    }

    printf("Configuration: %dx%d matrix, %d threads.\n", n, n, num_threads);

    A = allocate_matrix(n);
    C = allocate_matrix(n);
    initMatrices(n);

    // timer variable
    double start_time;

    start_time = get_current_time();
    printf("\n--- Running CPU Version ---\n");
    run_convolution(n, num_threads);
    double cpu_time = get_current_time() - start_time;
    printf("CPU Version Elapsed Time: %.4f  \n", cpu_time);

    save_matrix("output_cpu.txt", C, n);
    cleanup_matrices(n);

    return 0;
}