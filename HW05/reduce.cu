#include "reduce.cuh"
#include <iostream>
#include <cstddef>

// implements the 'first add during global load' version (Kernel 4) for the
// parallel reduction g_idata is array to be reduced, and is available on
// the device. g_odata is the array that the reduced results will be written to,
// and is available on the device. expects a 1D configuration. uses only
// dynamically allocated shared memory.
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float sharedarray[];
    unsigned int threadnum = threadIdx.x;
    unsigned int i   = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    
    // check to make sure that thread has entry to pull into shared memory
    if (i >= n ){
        sharedarray[threadnum] = 0;
    }
    // check that there is a value to add to entry when putting into shared memory
    else if (i+blockDim.x >= n){
        sharedarray[threadnum] = g_idata[i];
    }
    // Add to shared memory and perform first level of reduction during pull in
    else {
        sharedarray[threadnum] = g_idata[i] + g_idata[i+blockDim.x];
    }
    __syncthreads();
    
    //Reduction of values in the shared array when exactly one or multiple blocks needed
    if (blockDim.x <= n){
        int reducingsize = blockDim.x;
        for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
            if(threadnum < s) {
                sharedarray[threadnum] += sharedarray[threadnum + s];
                if (threadnum == 0 && (reducingsize % 2) != 0){
                    sharedarray[threadnum] += sharedarray[2*s];
                }
            }
            reducingsize = s;
        }
    __syncthreads();
    }

    //Reduction of values in shared array when less than 1 full block needed
    else {
        int reducingsize = n;
        for(unsigned int s=n/2; s>0; s>>=1) {
            if(threadnum < s) {
                sharedarray[threadnum] += sharedarray[threadnum + s];
                if (threadnum == 0 && (reducingsize % 2) != 0){
                    sharedarray[threadnum] += sharedarray[2*s];
                }
            }
        reducingsize = s;
        }
    __syncthreads();
    }

    // write result for this block to global memory
    if(threadnum == 0){
     g_odata[blockIdx.x] = sharedarray[0];
     g_idata[blockIdx.x]=  sharedarray[0];
}
}


// the sum of all elements in the *input array should be written to the first
// element of the *input array. calls reduce_kernel repeatedly if needed. _No
// part_ of the sum should be computed on host. *input is an array of length N
// in device memory. *output is an array of length = (number of blocks needed
// for the first call of the reduce_kernel) in device memory. configures the
// kernel calls using threads_per_block threads per block. the function should
// end in a call to cudaDeviceSynchronize for timing purposes
__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block){
    int blocksneeded = (((N+threads_per_block-1)/threads_per_block)+1)/2;
    // Account for integers rounding down
    // Call kernel for first time
    reduce_kernel<<<blocksneeded,threads_per_block,threads_per_block*sizeof(float)>>>(*input,*output,N);
    // Call additional kernels if needed
    if (blocksneeded == 1){}
    else{
        //redefine n and blocksneeded for new input array from old output
        int n = blocksneeded;
        blocksneeded = (((n + threads_per_block -1)/threads_per_block)+1)/2;
        // Call Kernel and redefine input as many times as necessary
        while (n>threads_per_block * 2){
            reduce_kernel<<<blocksneeded,threads_per_block,threads_per_block*sizeof(float)>>>(*input,*output,n);
            n = blocksneeded;
            blocksneeded = (((n + threads_per_block -1)/threads_per_block)+1)/2;
        }
        // Final Kernel call for when reduction can occur within one block
        reduce_kernel<<<blocksneeded,threads_per_block,threads_per_block*sizeof(float)>>>(*input,*output,n);
    }
    cudaDeviceSynchronize();
}
