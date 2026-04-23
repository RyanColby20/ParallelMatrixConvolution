#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define N 5
#define THREADS 5

// NxN Matrices
double A[N][N];
double C[N][N];

// 3x3 Kernel
double K[3][3] = {
    {0,-1,0},
    {-1,5,-1},
    {0,-1,0}
};

pthread_barrier_t barrier;

typedef struct
{
    int id;
    int start;
    int end;
} thread_data;

void initMatrices(){

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (rand() % 10);
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = 0;
}

void *worker(void *arg)
{
    
    thread_data *data = (thread_data *)arg;

    // TO DO: perform convolution on assigned rows

    pthread_barrier_wait(&barrier);

    return NULL;
}

int main(void){

    initMatrices();

    // TEST OUTPUT 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.0f ", A[i][j]);
        }
        printf("\n");
    }

    // TO DO (Add restrctions so that the number of threads participating are = to what is passed)
    pthread_barrier_init(&barrier, NULL, THREADS);

    // Init the threads and the data they hold
    pthread_t threadArray[THREADS];
    thread_data data[THREADS];
    
    // Init the variables needed to split the rows up between threads
    int validRows = N - 2;
    int normalRows = validRows / THREADS;
    int oddRows = validRows % THREADS;
    int currentRow = 1;

    // Fct will assign a thread the normalRow amount unless there are still oddRows left
    // oddRows represents the # of rows that require + 1
    for (int count = 0; count < THREADS; ++count){
        int rowAmount = normalRows;

        if (oddRows > 0){
            rowAmount++;
            oddRows--;
        }

        // Setting the thread data
        data[count].id = count; 
        data[count].start = currentRow;
        data[count].end = currentRow + rowAmount - 1;

        currentRow = data[count].end + 1;
    }
// Creating and joinging the threads
    for (int i = 0; i < THREADS; i++){
        if (pthread_create(&threadArray[i],NULL,worker, &data[i]) != 0){
            perror("Failed to create thread");
            return 1; 
        }
    }

    for (int i = 0; i < THREADS; i++){
        if (pthread_join(threadArray[i],NULL) != 0){
            perror("Failed to shut down thread");
            return 1;
        }
    }

    printf("Convolution Complete");
    pthread_barrier_destroy(&barrier);
    return 0;
}
