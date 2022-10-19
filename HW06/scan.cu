#include "scan.cuh"
#include <iostream>
#include <cstddef>



// Performs an *inclusive scan* on the array input and writes the results to the array output.
// The scan should be computed by making calls to your kernel hillis_steele with
// threads_per_block threads per block in a 1D configuration.
// input and output are arrays of length n allocated as managed memory.
//
// Assumptions:
// - n <= threads_per_block * threads_per_block

__global__ void hillis_steele(const float *input, float *output, float *temp, unsigned int n, int offset){
    extern __shared__ float sharedarray[];
    // get thread information
    thread = threadIdx.x;
    block = blockIdx.x;
    blocksize = blockDim.x;
    if (thread + block*blocksize > n){}
    else{
        if (block*blocksize+thread >= offset){
            output[block*blocksize + thread - offset] = input[block*blocksize + thread - offset] + input[block*blocksize + thread - offset];
        }
        else {
            output[block*blocksize + thread + offset - 1] = temp[block*blocksize + thread];
        }
    }

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block){

int blocksneeded = (n+threads_per_block-1)/threads_per_block;
for (offset = 1; offset < n; offset *=2){
    hillis_steele<<<blocksneeded,threads_per_block,threads_per_block*sizeof(float)>>>(input,output,temp,n, offset);
    blocksneeded = blocksneeded - offset;

}
}