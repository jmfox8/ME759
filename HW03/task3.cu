#include "vscale.cuh"
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
    
    // Declare GPU configuration values
    int threadperblock = 512;
    int blocksneeded = (n+threadperblock-1)/threadperblock;

    // Allocate arrays on host
    float* hB = new float[n];
    float* hA = new float[n];
    float* hBout = new float[n];

    // Allocate arrays on device
    float *dA = NULL;
    float *dB = NULL;
    cudaMalloc(&dA, n*sizeof(float));
    cudaMalloc(&dB, n*sizeof(float));
    
    // Initialization for randomization
    random_device entropy_source;
    mt19937 generator(entropy_source());
    vector<float> random_val_A(n);
    vector<float> random_val_B(n);
    
    // Generate random float values for arrays
    uniform_real_distribution<float> distroA(-10.0,10.0);
    uniform_real_distribution<float> distroB(0.0,1.0);
    int i = 0;
    for (auto& value_A : random_val_A){
        value_A = distroA(generator);
        hA[i] = random_val_A[i];
        i++;
    }
    int j = 0;
    for (auto& value_B : random_val_B){
        value_B = distroB(generator);
        hB[j] = random_val_B[j];
        j++;
    }

    // Initialization for CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Transfer Arrays to Device from Host
    cudaMemcpy(dA,hA,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,n*sizeof(float),cudaMemcpyHostToDevice);
    
    // Call and time Kernel
    cudaEventRecord(start);
    vscale <<<blocksneeded,threadperblock>>>(dA,dB,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy values from device array to host array
    cudaMemcpy(hBout,dB,n*sizeof(float),cudaMemcpyDeviceToHost);
    
    // Get timing values for kernel execution
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    
    // Print output values
    cout << ms << "\n";
    cout << hBout[0] << "\n";
    cout << hBout[n-1] << "\n";

// Deallocate arrays on host and device
    cudaFree(dA);
    cudaFree(dB);
    delete[] hA;
    delete[] hB;
    delete[] hBout;
    return 0;
}