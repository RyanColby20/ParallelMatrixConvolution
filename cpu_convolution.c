#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "cpu_convolution.h"

// these defined in main
extern double **A;
extern double **C;
extern double K[3][3];

// Barrier to sync threads
pthread_barrier_t barrier;

// Struct to hold data for each thread
typedef struct {
    int id;
    int start;
    int end;
    int matrix_size;
} thread_data;

void *worker(void *arg) {
    thread_data *data = (thread_data *)arg;
    int start_row = data->start;
    int end_row   = data->end;
    int n         = data->matrix_size;

    for (int i = start_row; i <= end_row; i++) {
        for (int j = 1; j < n - 1; j++) {
            double sum = 0.0;
            for (int m = -1; m <= 1; m++) {
                for (int k = -1; k <= 1; k++) {
                    sum += A[i+m][j+k] * K[m+1][k+1];
                }
            }
            C[i][j] = sum;
        }
    }

    pthread_barrier_wait(&barrier);
    return NULL;
}

void run_convolution(int n, int num_threads) {
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data *tdata = malloc(num_threads * sizeof(thread_data));

    pthread_barrier_init(&barrier, NULL, num_threads);

    int validRows = n - 2;
    int base = validRows / num_threads;
    int extra = validRows % num_threads;

    int current = 1;

    for (int t = 0; t < num_threads; t++) {
        int rows = base + (extra-- > 0 ? 1 : 0);

        tdata[t].id = t;
        tdata[t].matrix_size = n;
        tdata[t].start = current;
        tdata[t].end = current + rows - 1;

        current = tdata[t].end + 1;

        pthread_create(&threads[t], NULL, worker, &tdata[t]);
    }

    for (int t = 0; t < num_threads; t++)
        pthread_join(threads[t], NULL);

    pthread_barrier_destroy(&barrier);
    free(threads);
    free(tdata);
}