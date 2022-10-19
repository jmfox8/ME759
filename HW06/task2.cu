#include "scan.cuh"
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

    // Allocate managed memory
    float* input, output; 
    cudaMallocManaged((void**)&input, n*sizeof(float));
    cudaMallocManaged((void**)&output, n*sizeof(float));

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
        input[i] = RD(generator);
    }

    // Call and time scan function
    cudaEventRecord(start);
    scan(input, output, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Get function elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

