#include "matmul.cuh"
#include <iostream>
#include <cstddef>

using namespace std;
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){

// Basic identifiers for the kernel that is running
int block = blockIdx.x;
int thread = threadIDx.x;
int blocksize = blockdim.x;

extern __shared__ float s[];
float *s_mask = s; // create shared array for the mask
float *s_output = (float*)&s_mask[2*R+1]; // determine size of the shared mask array
float *s_image = (float*)&s_output[blocksize]; // make shared output array as large as the number of threads in the block
// By consequence shared image array is also as large as the threads in the block based on the input to the kernel defined in the other function


// Load the mask array into shared memory, each thread loads a single entry    
if (thread <= 2*R+1){
     s_mask[thread] = mask[thread];
}
// Load the portions of the image array needed for the block into shared memory, each thread loads a single entry
if (block*blocksize+thread < n){
    s_image[thread] = image[thread];
}

// Make sure the shared arrays are fully loaded
__syncthreads();

// Determine the entry for this thread to calculate
int i = block*(blocksize-1) + thread;

for

float im;
for (int j = -R; j=R; j++){
    // Statements to deal with buffering the Image Array based on provided values
    if (i+j < 0 || i+j > n-1){
        im = 1;
    }
    else {
        im = s_image[i+j];
    }
    s_output[i] = s_ouput[i] + im *s_mask[j+R];
}
}
 
__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){
   int blocks = (n+threads_per_block-1)/threads_per_block;
   stencil_kernel<<<blocks,threads_per_block,(2*R+1)*sizeof(float)+threads_per_block*sizeof(float)+threads_per_block*sizeof(float)>>>(image,mask,output,n,R);
                      }