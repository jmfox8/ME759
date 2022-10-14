#include "matmul.cuh"
#include <iostream>
#include <cstddef>


// You should implement Tiled Matrix Multiplication discussed in class
// Compute the matrix product C = AB.
// A, B, and C are row-major representations of nxn matrices in 'managed
// memory'. Configures the kernel call using a 2D configuration with blocks of
// dimensions block_dim x block_dim. The function should end in a call to
// cudaDeviceSynchronize for timing purposes.

// HINT: think about why we put '__host__' here
// matmul_1, matmul_2, and matmul_3 are doing the same thing except input data
// types, then what is the best way to handle it?

// You DO NOT have to follow the hint and you can do anything you what
// as long as you DO NOT add additional files, you DO NOT modify this header
// file, and your code CAN compile with provided compile command.
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, int n){
    
    // Delegate shared memory and identify data type
    extern __shared__ char sharedmem[];
    T *csub = sharedmem;
    T *As = (T*)&csub[1];
    T *Bs = (T*)&As[n*n];
    T * s_data = reinterpret_cast<T *>(sharedmem);

    // identify block index
    int blockx = blockIdx.x;
    int blocky = blockIdx.y;

    // identify Thread index
    int threadx = threadIdx.x;
    int thready = threadIdx.y;

// Index of the first sub-matrix of A processed by the block
    int Astart= w * blockDim.x * blocky;

// Index of the last sub-matrix of A processed by the block
    int Aend = Astart + n -1;

    int Astep = blockDim.x;
    int Bstart = blockDim.x * blockx;
    int Bstep = blockDim.x * n;
    


    for (int a = Astart, b = Bstart; a <= AEnd; a += aStep, b += bStep) {
        // Load tiles from global memory into shared memory; each
        // thread loads one element of the two tiles from A & B
        As[ty][tx] = A[a + wA* ty + tx];
        Bs[ty][tx] = B[b + wB* ty + tx];

    // Load tiles from global memory into shared memory; each
    // thread loads one element of the two tiles from A & B
    As[ty][tx] = A[a + wA* ty + tx];Bs[ty][tx] = B[b + wB* ty + tx];
    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    // Each thread in this block computes one element
    // of the block sub-matrix (tile).  Thread with indexes
    // ty and txcomputes in this tile the entry [ty][tx].
    for (int k = 0; k < BLOCK_SIZE; ++k){
    Csub+= As[ty][k] * Bs[k][tx];
    // Synchronize to make sure that the preceding
    // computation is done before loading two new// sub-matrices of A and B in the next iteration
    __syncthreads();}
    }
// Write the block sub-matrix to global memory;
// each thread writes one element
int c = wB* BLOCK_SIZE * by + BLOCK_SIZE * bx;
C[c + wB* ty + tx] = Csub;
    
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim){
    dim3 dimBlock(block_dim,block_dim);
    dim3 dimGrid( n/dimBlock.x, n/dimBlock.y);                    
    matmul_kernel<<<dimGrid,dimBlock,block_dim*block_dim*sizeof(int)>>>(A,B,C,);
                       }

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim){
    dim3 dimBlock(block_dim,block_dim);
    dim3 dimGrid( n/dimBlock.x, n/dimBlock.y);                    
    matmul_kernel<<<dimGrid,dimBlock,block_dim*block_dim*sizeof(float)>>>(A,B,C,n);

                       }

__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim,block_dim);
    dim3 dimGrid( n/dimBlock.x, n/dimBlock.y);                    
    matmul_kernel<<<dimGrid,dimBlock,block_dim*block_dim*sizeof(double)>>>(A,B,C,n);
                       }
