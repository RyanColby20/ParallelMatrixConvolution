/**
 * Example client program that uses thread pool.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include "threadpool.h"
#include "matrix_generator.h"
#include "sparse.h"

int Total_no_of_partial_product = 5; //for make to work I have defined this, you should comupte this value from the dimension of the input matrices
pthread_mutex_t done_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t done_cond = PTHREAD_COND_INITIALIZER;

int tasks_remaining = 0;

struct data
{
    //information needed for one thread to process 
    //contains dimensions and pointers to arrays (matrices)

    // in matrix multi :
    // matrix A: with m x n 
    // matrix B: with n x p
    // resulting matrix C: m x p
    
    // row and col here represent current cell in product matrix being calculated
    int row;
    int col;

    int n;

    // matrices
    double **A;
    double **B;
    double **C;
};


struct data_sparse
{
    // row and col here represent current cell in product matrix being calculated
    int row;
    int col;

    sparse_row *A_rows; // array of m sparse rows
    sparse_col *B_cols;  // array of p sparse columns

    // output matrix
    double **C; 
};

void compute_partial_product(void *param)
{
    struct data *temp;
    temp = (struct data*)param;
    
    //printf("[client] Computer Cell (%d, %d)\n", temp->row, temp->col);

    // perform partial product
    double sum = 0.0;
    for (int i=0; i < temp->n; i++){
        sum += temp->A[temp->row][i] * temp->B[i][temp->col];
    }
    
    temp->C[temp->row][temp->col] = sum;

    // signal complete here
    pthread_mutex_lock(&done_mutex);
    tasks_remaining--;
    if (tasks_remaining == 0){
        pthread_cond_signal(&done_cond);
    }
    pthread_mutex_unlock(&done_mutex);

}

void compute_partial_product_sparse(void *param)
{
    struct data_sparse *temp;
    temp = (struct data_sparse*)param;

    sparse_row *srow = &temp->A_rows[temp->row];
    sparse_col *scol = &temp->B_cols[temp->col];

    double sum = 0;

    int i = 0, j = 0;
    while (i < srow->num_nonzero && j < scol->num_nonzero){
        int colA = srow->entries[i].col;
        int rowB = scol->entries[i].row;

        if (colA == rowB){
            sum += srow->entries[i].val * scol->entries[i].val;
            i++; j++;
        }
        else if (colA < rowB){
            i++;
        }
        else{
            j++;
        }
    }
    

    temp->C[temp->row][temp->col] = sum;

    // signal complete here
    pthread_mutex_lock(&done_mutex);
    tasks_remaining--;
    if (tasks_remaining == 0){
        pthread_cond_signal(&done_cond);
    }
    pthread_mutex_unlock(&done_mutex);

}

int main(void)
{
    // allocate A, B, C
    // fill A, B (LOAD)
    // build sparsity & check
    // allocate data array 
    // get start time
    // init threadpool 
    // assign work 
    // shutdown pool
    // get finish time
    // calculate elapsed time
    // output elapsed time and resulting matrix C to file


    // in matrix multi :
    // matrix A: with m x n 
    // matrix B: with n x p
    // resulting matrix C: m x p
    double **A, **B;

    int m, n, p;

    // assign data to A and B here
    load_matrix("A.txt", &A, &m, &n);
    load_matrix("B.txt", &B, &n, &p);

    
    // allocate C - we now know the size    
    double **C = malloc(m * sizeof(double *));
    for (int i=0; i<m; i++){
        C[i] = malloc(p * sizeof(double));
    }

    double densityA = compute_density(A, m, n);
    double densityB = compute_density(B, n, p);

    printf("Density A: %.4f \n", densityA);
    printf("Density B: %.4f \n", densityB);

    void (*worker_func)(void *);
    void *work_ptr = NULL;
    size_t task_size = 0;

    if ( (densityA < 0.2) || (densityB < 0.2) )
    {
        printf("Using Sparse Method: \n");

        // create data array req
        sparse_row *A_rows = build_sparse_rows(A, m, n);
        sparse_col *B_cols = build_sparse_cols(B, n, p);

        Total_no_of_partial_product = m * p;

        // create data array
        struct data_sparse *work_sparse = malloc(sizeof(struct data_sparse) * Total_no_of_partial_product);

        int i = 0;
        for (int row = 0; row<m; row++){
            for (int col = 0; col < p; col++){
                work_sparse[i].row = row;
                work_sparse[i].col = col;
                work_sparse[i].A_rows = A_rows;
                work_sparse[i].B_cols = B_cols;
                work_sparse[i].C = C;
                i++;
            }
        }

        worker_func = compute_partial_product_sparse;
        work_ptr = work_sparse;
        task_size = sizeof(struct data_sparse);

    }
    else
    {
        printf("Using Dense Method: \n");

        // create data array 
        Total_no_of_partial_product = m * p;
        int i = 0;
        struct data *work = malloc(sizeof(struct data) * Total_no_of_partial_product);

        for (int row = 0; row < m; row++){
            for (int col = 0; col < p; col++){
                work[i].row = row;
                work[i].col = col;
                work[i].n = n;
                work[i].A = A;
                work[i].B = B;
                work[i].C = C;
                i++;
            }
        }

        // assign func to later
        worker_func = compute_partial_product;
        work_ptr = work;
        task_size = sizeof(struct data);
    }


    // get start time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // initialize the thread pool
    pool_init();

    // submit the work to the queue
    tasks_remaining = Total_no_of_partial_product;
    printf("submitting %d tasks\n", tasks_remaining);
    for(int i=0;i<Total_no_of_partial_product;i++){
        void *task_ptr = (char*)work_ptr + i * task_size;
    	pool_submit(worker_func, task_ptr);
    }

    printf("All tasks submitted\n");    
    // may be helpful 
    //sleep(3);

    pthread_mutex_lock(&done_mutex);
    while (tasks_remaining > 0){
        pthread_cond_wait(&done_cond, &done_mutex);
    }
    pthread_mutex_unlock(&done_mutex);
    

    printf("All tasks done - pool shutdown\n");
    pool_shutdown();

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    FILE *fp = fopen("output.txt", "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    fprintf(fp, "Computation Time: %.4f seconds\n\n", elapsed);
    fprintf(fp, "resulting Matrix C: \n");
    printf("Computation Time: %.4f seconds\n\n", elapsed);
    printf("resulting Matrix C: \n");

    for (int j=0; j<m; j++){
        for (int k=0; k<p; k++){
            fprintf(fp, "%.2f ", C[j][k]);
            printf("%.2f ", C[j][k]);
        }
        fprintf(fp, "\n");
        printf("\n");
    }

    fclose(fp);


    // freeing memory
    for (int i=0; i<m; i++){
        free(A[i]);
    }
    free(A);

    for (int i=0; i<n; i++){
        free(B[i]);
    }
    free(B);
    
    for (int i=0; i<m; i++){
        free(C[i]);
    }
    free(C);


    /*
    
    if ((densityA < 0.2) || (densityB < 0.2)) {
        for (int r = 0; r < m; r++)
            free(A_rows[r].entries);
        free(A_rows);

        for (int c = 0; c < p; c++)
            free(B_cols[c].entries);
        free(B_cols);

        free(work_sparse);
    } else {
        free(work);
    }
    */

    //free(work);
    
    
    return 0;
}
