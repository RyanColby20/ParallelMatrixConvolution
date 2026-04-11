
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define N 500
#define THREADS 5

double A[N][N];
double C[N][N];

pthread_barrier_t barrier;

typedef struct
{
    int id;
    int start;
    int end;
} thread_data;

void *worker(void *arg)
{
    thread_data *data = (thread_data *)arg;
    // TODO: perform convolution on assigned rows
    pthread_barrier_wait(&barrier);
    return NULL;
}