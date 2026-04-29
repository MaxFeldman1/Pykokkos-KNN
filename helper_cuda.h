#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void __checkCudaErrors(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s at %s:%d\n",
                cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) __checkCudaErrors((val), __FILE__, __LINE__)
