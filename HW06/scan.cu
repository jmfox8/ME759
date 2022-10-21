#include "scan.cuh"
#include <iostream>
#include <cstddef>
#include <cstdio>
#include <stdlib.h>


// Performs an *inclusive scan* on the array input and writes the results to the array output.
// The scan should be computed by making calls to your kernel hillis_steele with
// threads_per_block threads per block in a 1D configuration.
// input and output are arrays of length n allocated as managed memory.
//
// Assumptions:
// - n <= threads_per_block * threads_per_block

__global__ void hillis_steele(const float *input, float *output, float *sumarray, int n){
    extern volatile __shared__ float sharedarray[];
    // get thread information
    int thread = threadIdx.x; 
    int block = blockIdx.x;
    int blocksize = blockDim.x;
    int pout = 0, pin = 1;
    // Copy input to sharred array
    sharedarray[thread] = input[block*blocksize + thread];
    __syncthreads();
    // Begin iterating through offsets across dual buffers
    for (int offset = 1; offset<blocksize; offset *=2){
        pout = 1 - pout;
        pin = 1 - pout;
        if (thread >= offset) sharedarray[pout*blocksize+thread]= sharedarray[pin*blocksize+thread] + sharedarray[pin*blocksize + thread - offset];
        else sharedarray[pout*blocksize+thread] = sharedarray[pin*blocksize+thread];
        __syncthreads();
    }
    // write scan results of block to output array
    output[block*blocksize + thread] = sharedarray[pout*blocksize+thread];
    // write total sum of block to the blocksum array
    if (thread == blocksize - 1){
        sumarray[block] = output[block*blocksize + thread];
    }
}
__global__ void hillis_steele_odd(const float *input, float *output, float* sumarray,int n){
    extern volatile __shared__ float sharedarray[];
    // get thread information
    int thread = threadIdx.x;
    int blocksize = blockDim.x;
    int pout = 0, pin = 1;
    // Copy input to sharred array
    if (thread < n){
        sharedarray[thread] = input[(n/blocksize)*blocksize + thread];
        __syncthreads();
        for (int offset = 1; offset<n; offset *=2){
            pout = 1 - pout;
            pin = 1 - pout;
            if (thread >= offset) sharedarray[pout*blocksize+thread]= sharedarray[pin*blocksize+thread] + sharedarray[pin*blocksize + thread - offset];
            else sharedarray[pout*blocksize+thread] = sharedarray[pin*blocksize+thread];
            __syncthreads();
        }
         // write scanned block to output array
        output[(n/blocksize)*blocksize + thread] = sharedarray[pout*blocksize+thread];
        if (thread == (n%blocksize)- 1) sumarray[n/blocksize] = sharedarray[pout*blocksize+thread];
    }
}
__global__ void kernel_add(const float * scannedsumarray, float * output, int n){
    // get thread information
    int thread = threadIdx.x;
    int block = blockIdx.x;
    int blocksize = blockDim.x;
    if (block > 0 && block*blocksize + thread < n){
        output[block*blocksize + thread] += scannedsumarray[block - 1];
    }
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block){
    float *sumarray, *scannedsumarray;
    int fullblocksneeded = n/threads_per_block;
    int remainingvals = n - fullblocksneeded*threads_per_block;
// determine size of sumscan array and f helper kernel needed for first scan
if (n % threads_per_block > 0){
    //Allocate arrays for sum values
    cudaMallocManaged((void**)&sumarray, (fullblocksneeded + 1)*sizeof(float));
    cudaMallocManaged((void**)&scannedsumarray, (fullblocksneeded + 1)*sizeof(float));

    // Call kernel for complete thread blocks
    hillis_steele<<<fullblocksneeded,threads_per_block,2*threads_per_block*sizeof(float)>>>(input,output,sumarray, n);
    
    // Call kernel for final incomplete thread block
    hillis_steele_odd<<<1,threads_per_block,2*remainingvals*sizeof(float)>>>(input,output,sumarray,n);
}
else {
    cudaMallocManaged((void**)&sumarray, (fullblocksneeded)*sizeof(float));
    cudaMallocManaged((void**)&scannedsumarray, (fullblocksneeded)*sizeof(float));
    hillis_steele<<<fullblocksneeded,threads_per_block,2*threads_per_block*sizeof(float)>>>(input,output,sumarray, n);
}

// determine if the sumarray needs a full block kernel or helper kernel
if (n > threads_per_block*threads_per_block){
    hillis_steele<<<1,threads_per_block,2*threads_per_block*sizeof(float)>>>(sumarray,scannedsumarray,sumarray,fullblocksneeded);
    kernel_add<<<fullblocksneeded,threads_per_block>>>(scannedsumarray,output,n);
 
}
else if (remainingvals == 0) {
    hillis_steele_odd<<<1,threads_per_block,2*(fullblocksneeded)*sizeof(float)>>>(sumarray,scannedsumarray,sumarray,fullblocksneeded);
    kernel_add<<<fullblocksneeded,threads_per_block>>>(scannedsumarray,output,n);
}
else {
    hillis_steele_odd<<<1,threads_per_block,2*(fullblocksneeded+1)*sizeof(float)>>>(sumarray,scannedsumarray,sumarray,fullblocksneeded+1);
    kernel_add<<<fullblocksneeded+1,threads_per_block>>>(scannedsumarray,output,n);
}

// memory cleanup
cudaFree(sumarray);
cudaFree(scannedsumarray);
}