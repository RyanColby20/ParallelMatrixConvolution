#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>


/*
    ========================
    ====   MEMORY       ====
    ====   ALLOCATION   ====
    ========================
*/
// placeholders for functions so ian's eyes don't bleed.
void *worker(void *arg);
void run_gpu_convolution(const float *h_A, const float *h_K, float *h_C, int N, int M, int ksize);


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

/*
    ========================
    ==== INITIALIZATION ====
    ========================
*/

// check for device with cuda compatability
// return: 0 == no device; 1 == device
int has_gpu() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        return 0;
    }
    return 1;
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


/*
    ========================
    ====   CONVOLUTION  ====
    ====   FUNCTIONS    ====
    ========================
*/


void run_convolution(int n, int num_threads)
{
    // Allocate thread and data arrays
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data *t_data = malloc(num_threads * sizeof(thread_data));
    pthread_barrier_init(&barrier, NULL, num_threads);

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


    printf("CPU Convolution complete.\n");
    

    // Clean up thread stuff
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(t_data);
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

__global__
void convo_kernel(const float *A, const float *K, float *C, int N, int M, int ksize
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row

    int r = ksize / 2;

    if (i < r || i >= N - r || j < r || j >= M - r) {
        return; // skip edges
    }

    float sum = 0.0f;
    for (int ki = -r; ki <= r; ki++) {
        for (int kj = -r; kj <= r; kj++) {
            float a = A[(i + ki) * M + (j + kj)];
            float w = K[(ki + r) * ksize + (kj + r)];
            sum += a * w;
        }
    }
    C[i * M + j] = sum;
}


void run_gpu_convolution(const float *h_A, const float *h_K, float *h_C, int N, int M, int ksize)
{
    size_t bytes_A = N * M * sizeof(float);
    size_t bytes_C = N * M * sizeof(float);
    size_t bytes_K = ksize * ksize * sizeof(float);

    float *d_A, *d_C, *d_K;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_C, bytes_C);
    cudaMalloc(&d_K, bytes_K);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes_K, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid( (M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    convo_kernel<<<grid, block>>>(d_A, d_K, d_C, N, M, ksize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_K);

    printf("GPU Convolution complete.\n");

}



/*
    ========================
    ====     UTILITY    ====
    ====     CLEANUP    ====
    ========================
*/

// returns current time
double get_current_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

/*
    ========================
    ====      MAIN      ====
    ========================
*/


int main(int argc, char *argv[])
{
    // TODO: parse arguments
    int n = 500;
    int num_threads = 5;
    char mode[10] = "cpu"; // iand: why this?

    // Parse CLI inputs and exit if bad
    if (parse_arguments(argc, argv, &n, &num_threads, mode) != 0)
    {
        return 0;
    }

    // check for gpu
    if (strcmp(mode, "cpu") != 0){         
        if (has_gpu()) {
            printf("GPU available. Mode: %s\n", mode);
        } else{
            printf("No GPU detected, falling back to CPU only.\n");
            strcpy(mode, "cpu");
        }
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

    // doubles to measure time
    double start_time;

    if (strcmp(mode, "cpu") == 0 || strcmp(mode, "both") == 0)
    {
        start_time = get_current_time();
        printf("\n--- Running CPU Version ---\n");
        run_convolution(n, num_threads);
        double cpu_time = get_current_time() - start_time;
        printf("CPU Version Elapsed Time: %.4f", cpu_time);
    }

    if (strcmp(mode, "gpu") == 0 || strcmp(mode, "both") == 0)
    {
        // setting up vars for CUDA
        float *h_A_flat = malloc(n * n * sizeof(float));
        float *h_C_flat = malloc(n * n * sizeof(float));
        float h_K_flat[9]; // 3x3

        // pack A into h_A_flat
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                h_A_flat[i * n + j] = (float)A[i][j];
            }
        }

        // pack K into h_K_flat
        int idx = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                h_K_flat[idx++] = (float)K[i][j];
            }
        }

        start_time = get_current_time();
        printf("\n--- Running GPU Version ---\n");
        run_gpu_convolution(h_A_flat, h_K_flat, h_C_flat, n, n, 3);
        double gpu_time = get_current_time() - start_time;
        printf("GPU Version Elapsed Time: %.4f", gpu_time);

        // unpack C
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = h_C_flat[i * n + j];
            }
        }
    }

    cleanup_matrices(n);

    return 0;
}