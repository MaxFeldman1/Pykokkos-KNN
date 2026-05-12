// Reproduce Table 8 using dfi_leafknn from dfiknn_test.cu (the implementation
// actually used in the paper, with GPU-side sort precomputation via PrecompMergeNP2).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>
#include "dfiknn.h"

// Box-Muller: N(0,1) sample
static float randn_bm(void) {
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

static char *capture_stdout(void (*fn)(float*, int*, int, int, int, float*, int*, int, int),
                             float *a, int *b, int M, int N, int k, float *c, int *d, int dim, int dev) {
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

static float run_one(int M, int N, int k, int d) {
    size_t data_bytes  = (size_t)M * d * sizeof(float);
    size_t knn_bytes   = (size_t)M * k * sizeof(float);
    size_t knnid_bytes = (size_t)M * k * sizeof(int);
    size_t gid_bytes   = (size_t)M     * sizeof(int);

    float *h_data = (float*)malloc(data_bytes);
    int   *h_G_Id = (int*)  malloc(gid_bytes);
    if (!h_data || !h_G_Id) { fprintf(stderr, "host malloc failed\n"); exit(1); }

    srand(42);
    for (size_t i = 0; i < (size_t)M * d; i++) h_data[i] = randn_bm();
    for (int i = 0; i < M; i++) h_G_Id[i] = i;

    float *d_data, *d_knn;
    int   *d_G_Id, *d_knn_Id;
    cudaMalloc(&d_data,   data_bytes);
    cudaMalloc(&d_G_Id,   gid_bytes);
    cudaMalloc(&d_knn,    knn_bytes);
    cudaMalloc(&d_knn_Id, knnid_bytes);
    cudaMemset(d_knn,    0, knn_bytes);
    cudaMemset(d_knn_Id, 0, knnid_bytes);
    cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_G_Id, h_G_Id, gid_bytes,  cudaMemcpyHostToDevice);

    // warm-up
    for (int iter = 0; iter < 2; iter++) {
        char *out = capture_stdout(dfi_leafknn,
                                   d_data, d_G_Id, M, N, k, d_knn, d_knn_Id, d, 0);
        free(out);
        cudaMemset(d_knn,    0, knn_bytes);
        cudaMemset(d_knn_Id, 0, knnid_bytes);
    }

    // timed run — parse " Total = <seconds>" from dfi_leafknn's output
    char *out = capture_stdout(dfi_leafknn,
                               d_data, d_G_Id, M, N, k, d_knn, d_knn_Id, d, 0);

    float elapsed_s = -1.0f;
    char *ptr = strstr(out, " Total = ");
    if (ptr)
        sscanf(ptr, " Total = %f", &elapsed_s);
    else
        fprintf(stderr, "Warning: could not parse Total time (d=%d k=%d)\nOutput:\n%s\n", d, k, out);
    free(out);

    cudaFree(d_data); cudaFree(d_G_Id); cudaFree(d_knn); cudaFree(d_knn_Id);
    free(h_data); free(h_G_Id);
    return elapsed_s;
}

int main(void) {
    const int M = 4000000;
    const int N = 2000;

    const int ds[] = {4, 16, 64};
    const int ks[] = {16, 64};

    printf("Table 8 reproduction — dfi_leafknn (GPU-side sort precomputation)\n");
    printf("M=%d  N=%d leaves  m=%d per leaf\n\n", M, N, M / N);
    printf("%-10s %6s %14s\n", "Dataset", "k", "Total (s)");
    printf("%-10s %6s %14s\n", "-------", "-", "---------");

    for (int di = 0; di < 3; di++) {
        int d = ds[di];
        for (int ki = 0; ki < 2; ki++) {
            int k = ks[ki];
            fprintf(stderr, "Running d=%d k=%d ...\n", d, k);
            float t = run_one(M, N, k, d);
            char name[16];
            snprintf(name, sizeof(name), "Gauss%d", d);
            printf("%-10s %6d %14.2f\n", name, k, t);
            fflush(stdout);
        }
    }
    return 0;
}
