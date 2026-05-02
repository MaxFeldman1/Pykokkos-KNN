// Reproduce Table 8: Time Breakdown for Exact Search GPU Kernels on Dense Coordinates
// M = 4M points (N=2000 leaves, m=2000 per leaf), d in {4,16,64}, k in {16,64}
//
// Compile from KNN/ with:
//   nvcc -I. -I${CUDA_HOME}/samples/common/inc -I../pyrknn/GeMM/include \
//        ../pyrknn/GeMM/src/FIKNN_dense.cu table8_bench.cu -O2 -o table8_bench

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>

void FIKNN_gpu_dense(float *data, int *G_Id, int M, int leaves, int k,
                     float *knn, int *knn_Id, int d);

// Box-Muller: N(0,1) sample
static float randn_bm(void) {
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

static void suppress_stdout(int *saved) {
    fflush(stdout);
    *saved = dup(STDOUT_FILENO);
    int nfd = open("/dev/null", O_WRONLY);
    dup2(nfd, STDOUT_FILENO);
    close(nfd);
}

static void restore_stdout(int saved) {
    fflush(stdout);
    dup2(saved, STDOUT_FILENO);
    close(saved);
}

static char *capture_stdout(void (*fn)(float*, int*, int, int, int, float*, int*, int),
                             float *a, int *b, int M, int N, int k, float *c, int *d, int dim) {
    int pipefd[2];
    pipe(pipefd);
    fflush(stdout);
    int saved = dup(STDOUT_FILENO);
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);

    fn(a, b, M, N, k, c, d, dim);
    fflush(stdout);

    dup2(saved, STDOUT_FILENO);
    close(saved);

    char *buf = (char*)malloc(8192);
    ssize_t n = read(pipefd[0], buf, 8191);
    buf[n < 0 ? 0 : n] = '\0';
    close(pipefd[0]);
    return buf;
}

static float run_one(int M, int N, int k, int d) {
    size_t data_bytes   = (size_t)M * d * sizeof(float);
    size_t knn_bytes    = (size_t)M * k * sizeof(float);
    size_t knnid_bytes  = (size_t)M * k * sizeof(int);
    size_t gid_bytes    = (size_t)M     * sizeof(int);

    float *h_data = (float*)malloc(data_bytes);
    int   *h_G_Id = (int*)  malloc(gid_bytes);
    if (!h_data || !h_G_Id) { fprintf(stderr, "host malloc failed\n"); exit(1); }

    srand(42);
    for (size_t i = 0; i < (size_t)M * d; i++) h_data[i] = randn_bm();
    for (int i = 0; i < M; i++) h_G_Id[i] = i;

    float *d_data, *d_knn;
    int   *d_G_Id,  *d_knn_Id;
    cudaMalloc(&d_data,   data_bytes);
    cudaMalloc(&d_G_Id,   gid_bytes);
    cudaMalloc(&d_knn,    knn_bytes);
    cudaMalloc(&d_knn_Id, knnid_bytes);
    cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_G_Id, h_G_Id, gid_bytes,  cudaMemcpyHostToDevice);

    // warm-up iterations
    for (int iter = 0; iter < 2; iter++) {
        int sv;
        suppress_stdout(&sv);
        FIKNN_gpu_dense(d_data, d_G_Id, M, N, k, d_knn, d_knn_Id, d);
        restore_stdout(sv);
    }

    // timed iteration — parse FIKNN's internal CUDA-event elapsed time
    char *out = capture_stdout(FIKNN_gpu_dense,
                               d_data, d_G_Id, M, N, k, d_knn, d_knn_Id, d);

    float elapsed_s = -1.0f;
    char *ptr = strstr(out, "Elapsed time (s)");
    if (ptr)
        sscanf(ptr, "Elapsed time (s) : %f", &elapsed_s);
    else
        fprintf(stderr, "Warning: could not parse elapsed time (d=%d k=%d)\n", d, k);
    free(out);

    cudaFree(d_data); cudaFree(d_G_Id); cudaFree(d_knn); cudaFree(d_knn_Id);
    free(h_data); free(h_G_Id);
    return elapsed_s;
}

int main(void) {
    // Table 8: 4M total points split into 2000 leaves of 2000 points each
    const int M = 4000000;
    const int N = 2000;    // leaves; m = M/N = 2000

    const int ds[] = {4, 16, 64};
    const int ks[] = {16, 64};

    printf("Table 8 reproduction — FIKNN_gpu_dense (Row-Partitioned)\n");
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
