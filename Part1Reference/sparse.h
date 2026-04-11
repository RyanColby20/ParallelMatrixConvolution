// funcs and structs for sparse calculations



typedef struct  {
    int col;    // col index k
    double val; // A[row][k]
}entry_row;

typedef struct {
    int row;    // row index k
    double val; // B[k][col]
}entry_col;

typedef struct {
    int num_nonzero;
    entry_row *entries; // array of (col, vol) of non zero entries
}sparse_row;

typedef struct {
    int num_nonzero;
    entry_col *entries; // array of (col, vol) of non zero entries
}sparse_col;


double compute_density(double **M, int rows, int cols);
sparse_row* build_sparse_rows(double **A, int m, int n);
sparse_col* build_sparse_cols(double **B, int n, int p);
