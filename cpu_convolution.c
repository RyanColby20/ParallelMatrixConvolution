#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cpu_convolution.h"

// these defined in main
extern double **A;
extern double **C;
extern double K[3][3];

// Barrier to sync threads
pthread_barrier_t barrier;

// Struct to hold data for each thread
typedef struct
{
    int id;
    int start;
    int end;
    int matrix_size;
} thread_data;

// returns current time
double get_current_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void *worker(void *arg)
{
    // grab the thread info
    thread_data *data = (thread_data *)arg;
    int start_row = data->start;
    int end_row = data->end;
    int n = data->matrix_size;

    // loop through our assigned rows
    for (int i = start_row; i <= end_row; i++)
    {
        // loop through columns, skipping the edges
        for (int j = 1; j < n - 1; j++)
        {
            double sum = 0.0;

            // apply the 3x3 kernel to the neighbors
            for (int m = -1; m <= 1; m++)
            {
                for (int k = -1; k <= 1; k++)
                {
                    sum += A[i + m][j + k] * K[m + 1][k + 1];
                }
            }
            C[i][j] = sum;
        }
    }

    // wait for other threads to catch up
    pthread_barrier_wait(&barrier);

    // print completion
    printf("Thread %d finished convolution\n", data->id);
    return NULL;
}

void run_convolution(int n, int num_threads)
{
    // Allocate memory for threads and their data packets
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data *tdata = malloc(num_threads * sizeof(thread_data));

    // Initialize the barrier to wait for the exact number of threads
    pthread_barrier_init(&barrier, NULL, num_threads);

    // Calculate row distribution (skipping top and bottom edges)
    int validRows = n - 2;
    int base = validRows / num_threads;
    int extra = validRows % num_threads;

    int current = 1;

    // Distribute work and spawn threads
    for (int t = 0; t < num_threads; t++)
    {
        int rows = base + (extra-- > 0 ? 1 : 0);

        tdata[t].id = t;
        tdata[t].matrix_size = n;
        tdata[t].start = current;
        tdata[t].end = current + rows - 1;

        current = tdata[t].end + 1;

        // Create thread and make sure it actually spawned successfully
        if (pthread_create(&threads[t], NULL, worker, &tdata[t]) != 0)
        {
            perror("Failed to create thread");
            exit(1);
        }
    }

    // Wait for all threads to finish execution
    for (int t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    // Clean up synchronization primitives and memory
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(tdata);
}