// Benchmark wrapper for dfi_leafknn across varying N (leaf count).
// Compiled from KNN/ with:
//   nvcc -I. -I../pyrknn/GeMM/pysrc/filknn/dense \
//        -gencode arch=compute_90,code=sm_90 \
//        ../pyrknn/GeMM/pysrc/filknn/dense/dfiknn_test.cu cpp_bench.cu \
//        -O2 -lcublas -o cpp_bench
//
// Timing comes from dfi_leafknn's own internal CUDA events; the binary
// prints "N=<n>\n<ms>\n" to stdout.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>
#include "dfiknn.h"

static char *capture_stdout(void (*fn)(float*, int*, int, int, int, float*, int*, int, int),
                             float *a, int *b, int M, int N, int k,
                             float *c, int *d, int dim, int dev) {
    int pipefd[2];
    pipe(pipefd);
    fflush(stdout);
    int saved = dup(STDOUT_FILENO);
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);

    fn(a, b, M, N, k, c, d, dim, dev);
    fflush(stdout);

    dup2(saved, STDOUT_FILENO);
    close(saved);

    char *buf = (char*)malloc(16384);
    ssize_t n = read(pipefd[0], buf, 16383);
    buf[n < 0 ? 0 : n] = '\0';
    close(pipefd[0]);
    return buf;
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

    srand(0);
    for (int i = 0; i < M * d_dim; i++) h_data[i] = (float)(rand() % 8);
    for (int i = 0; i < M; i++)         h_G_Id[i]  = i;

    size_t knn_bytes   = (size_t)M * k * sizeof(float);
    size_t knnid_bytes = (size_t)M * k * sizeof(int);

    float *d_data, *d_knn;
    int   *d_G_Id, *d_knn_Id;
    cudaMalloc(&d_data,   M * d_dim * sizeof(float));
    cudaMalloc(&d_G_Id,   M * sizeof(int));
    cudaMalloc(&d_knn,    knn_bytes);
    cudaMalloc(&d_knn_Id, knnid_bytes);
    cudaMemset(d_knn,    0, knn_bytes);
    cudaMemset(d_knn_Id, 0, knnid_bytes);
    cudaMemcpy(d_data, h_data, M * d_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_G_Id, h_G_Id, M * sizeof(int),           cudaMemcpyHostToDevice);

    // warm-up
    for (int iter = 0; iter < 2; iter++) {
        char *out = capture_stdout(dfi_leafknn,
                                   d_data, d_G_Id, M, N, k, d_knn, d_knn_Id, d_dim, 0);
        free(out);
        cudaMemset(d_knn,    0, knn_bytes);
        cudaMemset(d_knn_Id, 0, knnid_bytes);
    }

    // timed run — parse " Total = <seconds>" from dfi_leafknn's output
    char *out = capture_stdout(dfi_leafknn,
                               d_data, d_G_Id, M, N, k, d_knn, d_knn_Id, d_dim, 0);

    float elapsed_s = 0.0f;
    char *ptr = strstr(out, " Total = ");
    if (ptr)
        sscanf(ptr, " Total = %f", &elapsed_s);
    else
        fprintf(stderr, "Warning: could not parse Total time (N=%d)\nOutput:\n%s\n", N, out);
    free(out);

    float ms = elapsed_s * 1000.0f;
    printf("N=%d\n%.3f\n", N, ms);
    fflush(stdout);

    cudaFree(d_data); cudaFree(d_G_Id); cudaFree(d_knn); cudaFree(d_knn_Id);
    free(h_data); free(h_G_Id);
    return 0;
}
