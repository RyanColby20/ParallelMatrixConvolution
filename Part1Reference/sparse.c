#include <stdlib.h>
#include "sparse.h"


double compute_density(double **M, int rows, int cols)
{
    int nonzero = 0;
    int total = rows * cols;

    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            if (M[i][j] != 0.0)
            nonzero++;
        }
    }

    return (double)nonzero / (double)total;
}

sparse_row* build_sparse_rows(double **A, int m, int n)
{
    sparse_row *rows = malloc(m * sizeof(sparse_row));

    // count number of nonzero elements
    for (int i=0; i<m; i++){
        int count = 0;
        for (int j=0; j<n; j++){
            if (A[i][j] != 0.0)
                count++;
        }

    rows[i].num_nonzero = count;
    rows[i].entries = malloc(count * sizeof(entry_row));
    
    
        // go again, store non-zero entries
        int k = 0;
        for (int j=0; j<n; j++){
            if (A[i][j] != 0.0){
                rows[i].entries[k].col = j;
                rows[i].entries[k].val = A[i][j];
                k++;
            }
        }
    }
    return rows;

}

sparse_col* build_sparse_cols(double **B, int n, int p)
{
    sparse_col *cols = malloc(p * sizeof(sparse_col));

    // count number of nonzero elements
    for (int j=0; j<p; j++){
        int count = 0;
        for (int i=0; i<n; i++){
            if (B[i][j] != 0.0)
                count++;
        }

        cols[j].num_nonzero = count;
        cols[j].entries = malloc(count * sizeof(entry_col));
    
    
        // go again, store non-zero entries
        int k = 0;
        for (int i=0; i<n; i++){
            if (B[i][j] != 0.0){
                cols[j].entries[k].row = i;
                cols[j].entries[k].val = B[i][j];
                k++;
            }
        }
    }
    return cols;
}
