// Benchmark wrapper for FIKNN_gpu_dense.
// Compiled from KNN/ with:
//   nvcc -I${CUDA_HOME}/samples/common/inc -I../pyrknn/GeMM/include \
//        ../pyrknn/GeMM/src/FIKNN_dense.cu cpp_bench.cu -O2 -o cpp_bench

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>

// Forward declaration — avoids pulling in helper_cuda.h from FIKNN_dense.h
void FIKNN_gpu_dense(float *data, int *G_Id, int M, int leaves, int k,
                     float *knn, int *knn_Id, int d);

static void suppress_stdout(int *saved_fd) {
    fflush(stdout);
    *saved_fd = dup(STDOUT_FILENO);
    int null_fd = open("/dev/null", O_WRONLY);
    dup2(null_fd, STDOUT_FILENO);
    close(null_fd);
}

static void restore_stdout(int saved_fd) {
    fflush(stdout);
    dup2(saved_fd, STDOUT_FILENO);
    close(saved_fd);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s N m d k\n", argv[0]);
        return 1;
    }
    int N     = atoi(argv[1]);
    int m     = atoi(argv[2]);
    int d_dim = atoi(argv[3]);
    int k     = atoi(argv[4]);
    int M     = N * m;

    float *h_data   = (float*)malloc(M * d_dim * sizeof(float));
    int   *h_G_Id   = (int*)  malloc(M * sizeof(int));
    float *h_knn    = (float*)malloc(M * k * sizeof(float));
    int   *h_knn_Id = (int*)  malloc(M * k * sizeof(int));

    srand(0);
    for (int i = 0; i < M * d_dim; i++) h_data[i] = (float)(rand() % 8);
    for (int i = 0; i < M; i++)         h_G_Id[i]  = i;

    float *d_data,   *d_knn;
    int   *d_G_Id,   *d_knn_Id;
    cudaMalloc(&d_data,   M * d_dim * sizeof(float));
    cudaMalloc(&d_G_Id,   M * sizeof(int));
    cudaMalloc(&d_knn,    M * k * sizeof(float));
    cudaMalloc(&d_knn_Id, M * k * sizeof(int));
    cudaMemcpy(d_data,  h_data,  M * d_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_G_Id,  h_G_Id,  M * sizeof(int),           cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    float ms = 0.0f;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    for (int iter = 0; iter < 3; iter++) {
        int saved;
        suppress_stdout(&saved);

        cudaEventRecord(t0);
        FIKNN_gpu_dense(d_data, d_G_Id, M, N, k, d_knn, d_knn_Id, d_dim);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        restore_stdout(saved);
        cudaEventElapsedTime(&ms, t0, t1);
    }

    printf("N=%d\n%.3f\n", N, ms);
    fflush(stdout);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_data); cudaFree(d_G_Id); cudaFree(d_knn); cudaFree(d_knn_Id);
    free(h_data); free(h_G_Id); free(h_knn); free(h_knn_Id);
    return 0;
}
