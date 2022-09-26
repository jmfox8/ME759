#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <chrono>

using namespace std;

__global__ void threadmath(int a, int* dA, int block_count, int threadcount){
    int x = threadIdx.x;
    int y = blockIdx.x;
    dA[threadcount*y+x] = a*x + y;
}

int main(int argc, char *argv[]){
    // Declare configuration values
    int threadcount = 8;
    int block_count = 2;
    int Asize = 16;

    // Allocate array on host
    int* hA = new int[Asize];
    
    // Allocate array on device
    int *dA = NULL;
    cudaMalloc(&dA, Asize*sizeof(int));
    
    // Initialization for randomization
    random_device entropy_source;
    mt19937 generator(entropy_source());
    
    // Generate random int value for a between 0 and 10
    uniform_real_distribution<float> distro(0,10);
    int a = distro(generator);

    // Call Kernel
    threadmath <<<2,8>>>(a,dA,block_count,threadcount);
    cudaDeviceSynchronize();

    // Copy values from device array to host array
    cudaMemcpy(hA,dA,Asize*sizeof(int),cudaMemcpyDeviceToHost);
        // Print array values from host array
    for (int j = 0; j < Asize; j++){
        printf("%d ",hA[j]);
    }
    cout << "\n";

// Deallocate arrays on host and device
    cudaFree(dA);
    delete[] hA;
    
    return 0;
}