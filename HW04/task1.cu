#include "matmul.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <chrono>

using namespace std;
int main(int argc, char *argv[]){

    // Get Command Line Input
    int n = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    // Initialize Arrays on the Host
    float* Ah = new float[n*n];
    float* Bh = new float[n*n];
    float* Ch = new float[n*n];

    // Initialize Arrays on Device
    float *Ad = NULL;
    float *Bd = NULL;
    float *Cd = NULL;
    cudaMalloc(&Ad, n*n*sizeof(float));
    cudaMalloc(&Bd, n*n*sizeof(float));
    cudaMalloc(&Cd, n*n*sizeof(float));

    // Initialization for CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialization for randomization
    random_device entropy_source;
    mt19937 generator(entropy_source());
    
    // Generate random float values and populate host arrays
    uniform_real_distribution<float> RD(-1.0,1.0);
    for (int i = 0; i<n*n; i++){
        Ah[i] = RD(generator);
        Bh[i] = RD(generator);
    }

    // Copy Randomized Arrays from host to device
    cudaMemcpy(Ad,Ah,n*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(Bd,Bh,n*n*sizeof(float),cudaMemcpyHostToDevice);

    // Call and time matmul function
    cudaEventRecord(start);
    matmul(Ad, Ad, Cd, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get function elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy Resultant Array from device to Host
    cudaMemcpy(Ch,Cd,n*n*sizeof(float),cudaMemcpyDeviceToHost);

    // Print Results
    printf("%f \n",Ch[n*n-1]);
    printf("%f \n",ms);

    // Deallocate Memory
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    delete[] Ah;
    delete[] Bh;
    delete[] Ch;
}

