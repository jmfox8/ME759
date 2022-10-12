

// implements the 'first add during global load' version (Kernel 4) for the
// parallel reduction g_idata is the array to be reduced, and is available on
// the device. g_odata is the array that the reduced results will be written to,
// and is available on the device. expects a 1D configuration. uses only
// dynamically allocated shared memory.
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
extern __shared__ int sharedarray[];
// perform first level of reduction upon reading from 
// global memory and writing to shared memory
unsigned int threadnum = threadIdx.x;
unsigned int i   = blockIdx.x*(blockDim.x*2) + threadIdx.x;
// check to make sure that thread can call an entry that is in bounds
if (blockIdx.x * blockDim.x +threadIDx.x >= n ){
}
// check that there is a value to add when putting into shared memory
else if (i+blockDim.x >= n){
    sharedarray[threadnum] = g_idata[i];
}
else {
    sharedarray[threadnum] = g_idata[i] + g_idata[i+blockDim.x];
}
__syncthreads();


for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if(threadnum < s) {
        sharedarray[threadnum] += sharedarray[threadnum + s];
    }
    __syncthreads();
}
// Check to account for left over entry in arrays with odd valued threads per block, 
// adds to final value after all else is complete
if (blockDim.x % 2 != 0 && threadnum == n-1) sharedarray[0] += sharedarray[blockDim.x-1];
    __syncthreads();

// write result for this block to global memory
if(threadnum == 0) g_odata[blockIdx.x] = sharedarray[0];
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
    if (blocksneeded == 0) blocksneeded == 1;
    float* g_idata = *input;
    float* g_odata = *output;
    int n = N;
    for (int i = 0; i<blocksneeded; i+=threads_per_block){
        reduce_kernel<<<blocksneeded,threads_per_block,threads_per_block*sizeof(float)>>>(g_idata,g_odata,N);
        g_idata = g_odata;
        n = blocksneeded;
        blocksneeded = (((n + threads_per_block -1)/threads_per_block)+1)/2;
    }
    cudaDeviceSynchronize();
}