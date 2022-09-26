#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <chrono>

using namespace std;

__global__ void threadmath(int a, int* dA){
    int x = threadIdx.x;
    int y = blockIdx.x;
    dA[8*y+x] = a*x + y;
}

int main(int argc, char *argv[]){
    // Allocate array on host
    int* hA = new int[16];
    
    // Allocate array on device
    int *dA = NULL;
    cudaMalloc(&dA, 16*sizeof(int));
    
    // Initialization for randomization
    random_device entropy_source;
    mt19937 generator(entropy_source());
    
    // Generate random int value for a between 0 and 10
    uniform_real_distribution<float> distro(0,10);
    int a = distro(generator);

    // Call Kernel
    threadmath <<<2,8>>>(a,dA);
    cudaDeviceSynchronize();

    // Copy values from device array to host array
    cudaMemcpy(hA,dA,16*sizeof(int),cudaMemcpyDeviceToHost);
        // Print array values from host array
    for (int j = 0; j < 15; j++){
        printf("%d ",hA[j]);
    }
    cout << "\n";

// Deallocate arrays on host and device
    cudaFree(dA);
    delete[] hA;
    
    return 0;
}