// this file is named terribly and actually just contains the functions for the generator 
// as well as the functions to save and load matrices

#include <stdio.h>
#include <stdlib.h>
#include "matrix_generator.h"

void generate_test_matrices(double **A, double **B, int m, int n, int p)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = i + j;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            B[i][j] = i * j;
}

void generate_random_dense(double **A, double **B, int m, int n, int p)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = (rand() % 100) / 10.0;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            B[i][j] = (rand() % 100) / 10.0;
}

void generate_sparse(double **A, double **B, int m, int n, int p, double sparsity)
{
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            double r = (rand() % 10000) / 10000.0;
            A[i][j] = (r < sparsity) ? 0.0 : (rand() % 100) / 10.0;
        }
    }
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            double r = (rand() % 10000) / 10000.0;
            B[i][j] = (r < sparsity) ? 0.0 : (rand() % 100) / 10.0;
        }
    }
}

int load_matrix(const char *filename, double ***M, int *rows, int *cols)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    fscanf(fp, "%d %d", rows, cols);

    *M = malloc((*rows) * sizeof(double *));
    for (int i=0; i<*rows; i++){
        (*M)[i] = malloc((*cols) * sizeof(double));
        for (int j=0; j < *cols; j++){
            fscanf(fp, "%lf", &((*M)[i][j]));
        }
    }

    fclose(fp); 
    return 0;
}

void save_matrix(const char *filename, double **M, int rows, int cols)
{
    FILE *fp = fopen(filename, "w");
    if(!fp) {
        perror("fopen");
        return;
    }

    fprintf(fp, "%d %d \n", rows, cols);

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            fprintf(fp, "%.4f ", M[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

