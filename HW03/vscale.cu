#include "vscale.cuh"
#include <iostream>
#include <cstddef>

using namespace std;

__global__ void vscale(const float *a, float *b, unsigned int n){
    int thread = threadIdx.x;
    int block = blockIdx.x;
    int blocksize = blockDim.x;
    int i = blocksize * block + thread;
    // ensure math only occurs within the actual bounds of the array
    if (i < n){
        b[i] = a[i]*b[i];
    }
}