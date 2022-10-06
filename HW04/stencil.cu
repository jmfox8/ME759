#include "stencil.cuh"
#include <iostream>
#include <cstddef>

using namespace std;
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){

// Basic identifiers for the kernel that is running
int blocknum = blockIdx.x;
int threadnum = threadIdx.x;
int blocksize = blockDim.x;

extern __shared__ float s[];
float *s_mask = s; // create shared array for the mask
float *s_output = (float*)&s_mask[blocksize+(2*R)]; // determine size of the shared mask array
float *s_image = (float*)&s_output[blocksize]; // make shared output array as large as the number of threads in the block

// Load the mask array into shared memory, each thread loads a single entry    
if (threadnum < 2*R+1){
     s_mask[threadnum] = mask[threadnum];
}
 
// Load the portions of the image array needed for the block into shared memory, some threads may load more than one entry
for (int j = threadnum; j<blocksize+(2*R); j = j+blocksize){
    int imd = blocknum*blocksize+j-R;
    if(imd < 0 || imd > n-1){
    }
    else{
        s_image[j] = image[imd];
    }
}


// Make sure the shared arrays are fully loaded into shared memory
__syncthreads();

//Determine if thread is needed to calculate output array
int sizecheck = (threadnum + blocknum*blocksize);
if (sizecheck < n){
    // Calculate the output array value 
    float im = 0;
    for (int j = 0; j<2*R+1; j++){
        // Statements to deal with buffering the Image Array based on provided values
        int buffercheck = sizecheck+j-R;
        if (buffercheck < 0 || buffercheck > n-1){
            im = 1;
        }
        else {
            im = s_image[threadnum+j];
        }
        s_output[threadnum] += im * s_mask[j];
    }   
    // Ensure sync
    __syncthreads();

    // Write shared output to global memory
    output[sizecheck] = s_output[threadnum];
}

}
 
__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){
   int blocks = (n+threads_per_block-1)/threads_per_block;
   stencil_kernel<<<blocks,threads_per_block,
   ((2*R)+1)*sizeof(float)+threads_per_block*sizeof(float)+((2*R)+threads_per_block)*sizeof(float)>>>(image,mask,output,n,R);
                      }