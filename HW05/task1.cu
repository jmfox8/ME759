#include "reduce.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <chrono>


int main(int argc, char *argv[]){

    // Get Command Line Input
    int n = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    // Initialize Arrays on the Host
    float* inputh = new float[n];
    float* outputh = new float[(((n+threads_per_block-1)/threads_per_block)+1)/2];

    // Initialize Arrays on Device
    float *inputd = NULL;
    float *outputd = NULL;
    cudaMalloc(&inputd, n*sizeof(float));
    cudaMalloc(&outputd, (((n+threads_per_block-1)/threads_per_block)+1)/2*sizeof(float));

    // Initialization for CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialization for randomization
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    
    // Generate random float values and populate host arrays
    std::uniform_real_distribution<float> RD(-1.0,1.0);
    for (int i = 0; i<n; i++){
        inputh[i] = RD(generator);
    }

    // Copy Randomized Array from host to device
    cudaMemcpy(inputd,inputh,n*sizeof(float),cudaMemcpyHostToDevice);

    // Call and time reduce function
    cudaEventRecord(start);
    reduce(&inputd, &outputd, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get function elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy output array from device to host
    cudaMemcpy(outputh,outputd,(((n+threads_per_block-1)/threads_per_block)+1)/2*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(inputh,inputd,n*sizeof(float),cudaMemcpyDeviceToHost);
    
    // Print Results
    std::cout << inputh[0]<<"\n";
    std::cout << ms<<"\n";

    // Deallocate Memory
    cudaFree(inputd);
    cudaFree(outputd);
    delete[] inputh;
    delete[] outputh;
}
