#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>

// Dynamic pointers to support any size matrix
double **A;
double **C;

// 3x3 Kernel
double K[3][3] = {
    {0, -1, 0},
    {-1, 5, -1},
    {0, -1, 0}};

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

// Utility to dynamically allocate a 2D matrix
double **allocate_matrix(int size)
{
    // Allocate array of row pointers
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
    {
        // Allocate memory for each column in the row
        matrix[i] = (double *)malloc(size * sizeof(double));
    }
    return matrix;
}

// Initialization function
void initMatrices(int size)
{
    srand(time(NULL)); // Seed random number generator
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            A[i][j] = (double)(rand() % 10); // Random digits 0-9
            C[i][j] = 0; // Clear the output matrix
        }
    }
}

void *worker(void *arg)
{
    thread_data *data = (thread_data *)arg;
    int start_row = data->start;
    int end_row = data->end;
    int n = data->matrix_size;

    // Loop through the assigned rows for this thread
    for (int i = start_row; i <= end_row; i++)
    {
        // Loop through columns, skipping the left and right edges
        for (int j = 1; j < n - 1; j++)
        {
            double sum = 0.0;

            // Apply the 3x3 kernel
            for (int m = -1; m <= 1; m++)
            {
                for (int n_k = -1; n_k <= 1; n_k++)
                {
                    sum += A[i + m][j + n_k] * K[m + 1][n_k + 1];
                }
            }
            C[i][j] = sum;
        }
    }

    // Make sure all threads finish their math before moving on
    pthread_barrier_wait(&barrier);
    printf("Thread %d finished convolution\n", data->id);
    return NULL;
}

// Fct to parse command line arguments
int parse_arguments(int argc, char *argv[], int *n, int *num_threads, char *mode)
{
    // Grab inputs if the user provided them
    if (argc >= 2)
        *n = atoi(argv[1]);
    if (argc >= 3)
        *num_threads = atoi(argv[2]);
    if (argc >= 4)
    {
        strncpy(mode, argv[3], 9);
        mode[9] = '\0'; // Ensure string ends properly
    }

    // Matrix needs to be at least 3x3 for a 3x3 kernel
    if (*n < 3)
    {
        printf("Matrix size must be at least 3.\n");
        return 1;
    }

    // Make sure thread count makes sense
    if (*num_threads < 1)
        *num_threads = 1;
    // Don't use more threads than we have valid rows
    if (*num_threads > *n - 2)
        *num_threads = *n - 2;

    // Check for valid modes
    if (strcmp(mode, "cpu") != 0 && strcmp(mode, "gpu") != 0 && strcmp(mode, "both") != 0)
    {
        printf("Invalid mode. Please use 'cpu', 'gpu', or 'both'.\n");
        return 1;
    }

    return 0; // Success
}

// Print matrix for testing
void print_matrix(double **mat, int n, const char *title)
{
    printf("%s\n", title);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.0f ", mat[i][j]);
        }
        printf("\n");
    }
}

void run_convolution(int n, int num_threads)
{
    // Allocate thread and data arrays
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data *t_data = malloc(num_threads * sizeof(thread_data));
    pthread_barrier_init(&barrier, NULL, num_threads);

    // Start the stopwatch
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Init the variables needed to split the rows up between threads
    int validRows = n - 2; // Skip top and bottom edge
    int normalRows = validRows / num_threads;
    int oddRows = validRows % num_threads;
    int currentRow = 1; // Start at row 1 (skipping row 0)

    // Fct will assign a thread the normalRow amount unless there are still oddRows left
    // oddRows represents the # of rows that require + 1
    for (int count = 0; count < num_threads; ++count)
    {
        int rowAmount = normalRows;

        if (oddRows > 0)
        {
            rowAmount++;
            oddRows--;
        }

        // Setting the thread data
        t_data[count].id = count;
        t_data[count].matrix_size = n;
        t_data[count].start = currentRow;
        t_data[count].end = currentRow + rowAmount - 1;

        currentRow = t_data[count].end + 1; // Update for the next thread

        // Create the thread and check for errors
        if (pthread_create(&threads[count], NULL, worker, &t_data[count]) != 0)
        {
            perror("Failed to create thread");
            exit(1);
        }
    }

    // Creating and joining the threads
    for (int i = 0; i < num_threads; i++)
    {
        if (pthread_join(threads[i], NULL) != 0)
        {
            perror("Failed to join thread");
            exit(1);
        }
    }

    // Stop stopwatch and calculate time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double time_taken = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("CPU Convolution complete.\n");
    printf("CPU Execution time: %f seconds\n", time_taken);

    // Clean up thread stuff
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(t_data);
}

void run_gpu_convolution(int n)
{
    printf("GPU convolution executed (Placeholder).\n");
}

// Free the 2D arrays to prevent memory leaks
void cleanup_matrices(int n)
{
    for (int i = 0; i < n; i++)
    {
        free(A[i]);
        free(C[i]);
    }
    free(A); // Free the row pointers
    free(C);
}

int main(int argc, char *argv[])
{
    int n = 500;
    int num_threads = 5;
    char mode[10] = "cpu";

    // Parse CLI inputs and exit if bad
    if (parse_arguments(argc, argv, &n, &num_threads, mode) != 0)
    {
        return 0;
    }

    printf("Configuration: %dx%d matrix, %d threads.\n", n, n, num_threads);

    A = allocate_matrix(n);
    C = allocate_matrix(n);
    initMatrices(n);

    // TEST OUTPUT (only for small matrices)
    if (n <= 10)
    {
        print_matrix(A, n, "Initial Matrix A:");
    }

    if (strcmp(mode, "cpu") == 0 || strcmp(mode, "both") == 0)
    {
        printf("\n--- Running CPU Version ---\n");
        run_convolution(n, num_threads);
    }

    if (strcmp(mode, "gpu") == 0 || strcmp(mode, "both") == 0)
    {
        printf("\n--- Running GPU Version ---\n");
        run_gpu_convolution(n);
    }

    cleanup_matrices(n);

    return 0;
}