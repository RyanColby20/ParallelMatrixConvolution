// matrix generator header file 

void generate_test_matrices(double **A, double **B, int m, int n, int p);
void generate_random_dense(double **A, double **B, int m, int n, int p);
void generate_sparse(double **A, double **B, int m, int n, int p, double sparsity);
int load_matrix(const char *filename, double ***M, int *rows, int *cols);
void save_matrix(const char *filename, double **M, int rows, int cols);
