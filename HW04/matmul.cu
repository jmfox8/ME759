#include "matmul.cuh"
#include <iostream>
#include <cstddef>

using namespace std;

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    int thread = threadIdx.x;
    int block = blockIdx.x;
    int blocksize = blockDim.x;
    int c_index = block*blocksize + thread;
    if (c_index < (n*n)){
        for (int i = 0; i<n; i++){
            C[c_index] = C[c_index] + A[(c_index/n)*n+i]*B[(c_index % n) + (i*n)];
        }
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    int blocks = ((n*n)+threads_per_block-1)/threads_per_block;
    matmul_kernel<<<blocks,threads_per_block>>>(A,B,C,n);
}
