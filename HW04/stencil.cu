#include "matmul.cuh"
#include <iostream>
#include <cstddef>

using namespace std;
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){

}

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){
   int blocks = (n+threads_per_block-1)/threads_per_block;
   stencil_kernel<<<blocks,threads_per_block>>>(image,mask,output,n,R);
                      }