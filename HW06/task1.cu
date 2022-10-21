#include "mmul.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <chrono>


int main(int argc, char *argv[]){

    // Get Command Line Input
    int n = atoi(argv[1]);
    unsigned int test_n = atoi(argv[2]);

    // Initialize Managed Arrays
    float *A, *B, *C;
    cudaMallocManaged((void**)&A, n*n*sizeof(float));
    cudaMallocManaged((void**)&B, n*n*sizeof(float));
    cudaMallocManaged((void**)&C, n*n*sizeof(float));

    // Initialization for CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t_total, ms, t_avg;

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());

    // Generate random float values and populate arrays in Column Major Order
    std::uniform_real_distribution<float> RD(-1.0,1.0);
    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            A[j*n+i] = RD(generator);
            B[j*n+i] = RD(generator);
            C[j*n+1] = RD(generator);
        }
    }

    // Intitialization for cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Call and time mmul function test_n times, capturing the time for each call
    for (int i = 0; i < test_n; i++){
        cudaEventRecord(start);
        mmul(handle,A,B,C,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        // Get function elapsed time
        cudaEventElapsedTime(&ms, start, stop);
        t_total += ms;
    }
    // Calculate Total time taken
    t_avg = t_total/test_n;
    
    // Print Results
    prtinf("%f \n",t_avg);

    // Memory Cleanup
    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}