#include "vscale.cuh"
#include <iostream>
#include <cstddef>

using namespace std;

__global__ void vscale(const float *a, float *b, unsigned int n){
    int x = threadIdx.x;
    int y = blockIdx.x;
    int blocksize = blockDim.x;
    b[blocksize*y+x] = a[blocksize*y+x]*b[blocksize*y+x];
}