#include "mmul.h"
#include <iostream>
#include <cstddef>

// Uses a single cuBLAS call to perform the operation C := A B + C
// handle is a handle to an open cuBLAS instance
// A, B, and C are matrices with n rows and n columns stored in column-major
// NOTE: The cuBLAS call should be followed by a call to cudaDeviceSynchronize() for timing purposes


void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n){
    // intialization of values for gemm call
    float alpha = 1;
    float beta = 1;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    
    // gemm call
    cublasSgemm(handle, transA, transB, n, n, n, &alpha, A, n, B, n, &beta, C, n);
    
    cudaDeviceSynchronize();
}
