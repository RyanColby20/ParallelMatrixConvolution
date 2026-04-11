// contains the code to generate matrices
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix_generator.h"

int main(int argc, char *argv[])
{
    int m = 5; 
    int n = 5;
    int p = 5;
    double sparsity = 0.9;
    char mode[32] = "test";

    // parse command line
    for (int i =1; i < argc; i++){
        if (strcmp(argv[i], "--m") == 0 && i + 1 < argc)
            m = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--p") == 0 && i + 1 < argc)
            p = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            strcpy(mode, argv[++i]);
        else if (strcmp(argv[i], "--sparsity") == 0 && i + 1 < argc)
            sparsity = atof(argv[++i]);
        else {
            printf("Unknown or incomplete argument: %s\n", argv[i]);
            printf("--m : matrix A dimenson m x n\n");
            printf("--n : matrix A dimenson m x n and matrix B dimension n x p\n");
            printf("--p : matrix B dimenson n x p\n");
            printf("--mode : mode selection\n");
            printf("         test : uses original simple test method\n");
            printf("         random : generates a dense matrix\n");
            printf("         sparse : generates a sparse matrix\n");
            printf("--sparsity : level of sparsity when using sparse mode\n");
            return 1;
        }
    }

    printf("Generating matrices with: \n");
    printf(" A: %d x %d \n", m, n);
    printf(" B: %d x %d \n", n, p);
    printf(" mode: %s \n", mode);
    if (strcmp(mode, "sparse") == 0)
        printf("  sparsity: %.2f\n", sparsity);



    // allocate
    double **A = malloc(m * sizeof(double *));
    for (int i=0; i<m; i++){
        A[i] = malloc(n * sizeof(double));
    }
    double **B = malloc(n * sizeof(double *));
    for (int i=0; i<n; i++){
        B[i] = malloc(p * sizeof(double));
    }    
    double **C = malloc(m * sizeof(double *));
    for (int i=0; i<m; i++){
        C[i] = malloc(p * sizeof(double));
    }


    // CHOOSE GENERATOR HERE
    if (strcmp(mode, "test") == 0) {
        generate_test_matrices(A, B, m, n, p);
    }
    else if (strcmp(mode, "random") == 0) {
        generate_random_dense(A, B, m, n, p);
    }
    else if (strcmp(mode, "sparse") == 0) {
        generate_sparse(A, B, m, n, p, sparsity);
    }
    else {
        printf("Unknown mode: %s\n", mode);
        return 1;
    }

    // generate_sparse(A, B, m, n, p, 0.95);
    // generate_random_dense(A, B, m, n, p);
    // generate_test_matrices(A, B, m, n, p);

    // save
    save_matrix("A.txt", A, m, n);
    save_matrix("B.txt", B, n, p);

    printf("Saved generated matrices to A.txt and B.txt\n");
    return 0;
}