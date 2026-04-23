#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>

// Dynamic pointers to support any size matrix
double **A;
double **C;

// 3x3 Sharpening Kernel
double K[3][3] = {
    {0, -1, 0},
    {-1, 5, -1},
    {0, -1, 0}
};

pthread_barrier_t barrier;

typedef struct {
    int id;
    int start;
    int end;
    int matrix_size; // Track boundaries in the worker
} thread_data;

// Utility to dynamically allocate a 2D matrix
double** allocate_matrix(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)malloc(size * sizeof(double));
    }
    return matrix;
}

// Initialization function
void initMatrices(int size) {
    srand(time(NULL));

    // Populate A with random values and zero out C
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i][j] = (double)(rand() % 10);
            C[i][j] = 0;
        }
    }
}

void* worker(void* arg){
    thread_data *data = (thread_data*)arg;
    int start_row = data->start;
    int end_row = data->end;
    int n = data->matrix_size;

    // Loop through assigned rows
    for (int i = start_row; i <= end_row; i++) {

        // Loop through columns (skipping edge columns 0 and n-1)
        for (int j = 1; j < n - 1; j++) {
            double sum = 0.0;

            // Apply the 3x3 sharpening kernel
            for (int m = -1; m <= 1; m++) {
                for (int n_k = -1; n_k <= 1; n_k++) {
                    // Map the -1, 0, 1 offsets to the 0, 1, 2 kernel indices
                    sum += A[i + m][j + n_k] * K[m + 1][n_k + 1];
                }
            }
            C[i][j] = sum;
        }
    }

    // Barrier synchronize convolution stages [cite: 67]
    pthread_barrier_wait(&barrier);

    // Print the expected output format [cite: 93]
    printf("Thread %d finished convolution\n", data->id);

    return NULL;
}

int main(int argc, char *argv[]) {
    // 1. CLI Parsing & Mode Selection
    int n = 500;           // Default size
    int num_threads = 5;   // Default threads
    char mode[10] = "cpu"; // Default mode

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);
    if (argc >= 4) strncpy(mode, argv[3], sizeof(mode) - 1);

    if (strcmp(mode, "gpu") == 0) {
        printf("GPU mode selected (Extra Credit - To Be Implemented)\n");
        return 0; // GPU logic will go here later
    }

    printf("Starting %s mode: %dx%d matrix, %d threads.\n", mode, n, n, num_threads);

    // 2. Memory Allocation & Initialization
    A = allocate_matrix(n);
    C = allocate_matrix(n);
    initMatrices(n);

    // Test output
    printf("Initial Matrix A:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.0f ", A[i][j]);
        }
        printf("\n");
    }

    // 3. Thread Setup
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data *t_data = malloc(num_threads * sizeof(thread_data));
    pthread_barrier_init(&barrier, NULL, num_threads);

    // 4. Timer Start
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // 5. Dynamic Work Distribution
    int validRows = n - 2;
    int normalRows = validRows / num_threads;
    int oddRows = validRows % num_threads;
    int currentRow = 1;

    for (int count = 0; count < num_threads; ++count) {
        int rowAmount = normalRows;

        if (oddRows > 0) {
            rowAmount++;
            oddRows--;
        }

        t_data[count].id = count;
        t_data[count].matrix_size = n;
        t_data[count].start = currentRow;
        t_data[count].end = currentRow + rowAmount - 1;

        currentRow = t_data[count].end + 1;

        // Create thread
        if (pthread_create(&threads[count], NULL, worker, &t_data[count]) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }

    // 6. Join Threads
    for (int i = 0; i < num_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("Failed to shut down thread");
            return 1;
        }
    }

    // 7. Timer End & Print
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double time_taken = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("Convolution complete.\n");
    printf("Execution time: %f seconds\n", time_taken);

    // 8. Cleanup
    pthread_barrier_destroy(&barrier);
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(C[i]);
    }
    free(A);
    free(C);
    free(threads);
    free(t_data);

    return 0;
}