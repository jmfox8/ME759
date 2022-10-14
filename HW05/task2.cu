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

    // Initialize float Arrays on the Host
    float* Ah = new float[n*n];
    float* Bh = new float[n*n];
    float* Ch = new float[n*n];

    // Initialize float Arrays on Device
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
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    
    // Generate random float values and populate host arrays
    std::uniform_real_distribution<float> RD(0,10.0);
    for (int i = 0; i<n*n; i++){
        Ah[i] = RD(generator);
        Bh[i] = RD(generator);
        Ch[i] = 0;
    }

    // Copy Array from host to device
    cudaMemcpy(Ad,Ah,n*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(Bd,Bh,n*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(Cd,Ch,n*n*sizeof(float),cudaMemcpyHostToDevice);

    // Deallocate arrays on host
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    delete[] inputh;
    delete[] outputh;
}